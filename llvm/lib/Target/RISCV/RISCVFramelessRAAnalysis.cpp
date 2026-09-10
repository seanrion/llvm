//===-- RISCVFramelessRAAnalysis.cpp - Frameless path RA hints ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prefer caller-saved GPRs for values that are live on frameless paths from
// entry (notably vector finish/end pointer loads). That keeps callee-saved
// registers off the fast path so shrink-wrapping can delay the stack frame.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ShrinkFrameUtils.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

#define DEBUG_TYPE "frameless-ra"

extern cl::opt<bool> DisableRISCVFramelessRA;

namespace {
class RISCVFramelessRAAnalysis : public MachineFunctionPass {
public:
  static char ID;

  RISCVFramelessRAAnalysis() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "RISC-V Frameless RA Analysis";
  }
};
} // namespace

char RISCVFramelessRAAnalysis::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVFramelessRAAnalysis, DEBUG_TYPE,
                      "RISC-V Frameless RA Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(RISCVFramelessRAAnalysis, DEBUG_TYPE,
                    "RISC-V Frameless RA Analysis", false, false)

FunctionPass *llvm::createRISCVFramelessRAAnalysisPass() {
  return new RISCVFramelessRAAnalysis();
}

static bool isGPRVirtualReg(const MachineRegisterInfo &MRI, Register Reg) {
  if (!Reg.isVirtual())
    return false;
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  // Integer/pointer VRs use GPRRegClass. GPRCRegClass is only the compressed
  // subset (x8-x9, x10-x15) and must not be used here.
  return RISCV::GPRRegClass.hasSubClassEq(RC);
}

static bool isEntryBlock(const MachineFunction &MF,
                         const MachineBasicBlock *MBB) {
  return MBB && MBB == &MF.front();
}

static MachineBasicBlock *defBlock(const LiveIntervals &LIS, Register Reg) {
  if (!LIS.hasInterval(Reg))
    return nullptr;
  const LiveInterval &LI = LIS.getInterval(Reg);
  if (LI.empty())
    return nullptr;
  return LIS.getMBBFromIndex(LI.beginIndex());
}

static bool defInEntry(const MachineFunction &MF, const LiveIntervals &LIS,
                       Register Reg) {
  return isEntryBlock(MF, defBlock(LIS, Reg));
}

static bool defFrameless(const MachineFunction &MF, Register Reg,
                         const LiveIntervals &LIS,
                         const SmallPtrSetImpl<MachineBasicBlock *> &FR) {
  MachineBasicBlock *MBB = defBlock(LIS, Reg);
  if (!MBB)
    return false;
  return isEntryBlock(MF, MBB) || FR.contains(MBB);
}

static bool hasRegReadInBlock(const MachineBasicBlock &MBB, Register Reg) {
  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.getReg() == Reg && MO.readsReg())
        return true;
    }
  }
  return false;
}

static bool
usedInFramelessRegion(const SmallPtrSetImpl<MachineBasicBlock *> &FR,
                      Register Reg) {
  for (MachineBasicBlock *BB : FR) {
    if (hasRegReadInBlock(*BB, Reg))
      return true;
  }
  return false;
}

/// True if Reg is live across a non-tail call on a framed path and still read
/// after that call. Do not treat "live-in alone" as a use.
static bool liveAcrossColdCall(const MachineFunction &MF,
                               const LiveIntervals &LIS, Register Reg) {
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (const MachineBasicBlock &MBB : MF) {
    if (!blockNeedsFrame(MF, MBB) || !hasRegReadInBlock(MBB, Reg))
      continue;
    for (const MachineInstr &MI : MBB) {
      if (!MI.isCall() || TII->isTailCall(MI))
        continue;
      if (!LIS.hasInterval(Reg))
        return false;
      SlotIndex CallIdx = LIS.getInstructionIndex(MI);
      if (!CallIdx.isValid())
        continue;
      if (!LIS.getInterval(Reg).liveAt(CallIdx.getDeadSlot()))
        continue;
      for (auto I = std::next(MI.getIterator()); I != MBB.end(); ++I) {
        if (I->isDebugInstr())
          continue;
        for (const MachineOperand &MO : I->operands()) {
          if (MO.isReg() && MO.getReg() == Reg && MO.readsReg())
            return true;
        }
      }
    }
  }
  return false;
}

static bool shouldHintVReg(const MachineFunction &MF, Register Reg,
                           const LiveIntervals &LIS,
                           const SmallPtrSetImpl<MachineBasicBlock *> &FR) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  if (!isGPRVirtualReg(MRI, Reg))
    return false;

  std::pair<unsigned, Register> Hint = MRI.getRegAllocationHint(Reg);
  if (Hint.first == RISCVRI::RegPairEven || Hint.first == RISCVRI::RegPairOdd)
    return false;

  if (!defFrameless(MF, Reg, LIS, FR))
    return false;

  // Entry defs consumed on a frameless fast path (e.g. store through finish)
  // may still look live across cold calls due to imprecise LI / MIR. Prefer
  // the hint; the slow path typically reloads from memory.
  if (defInEntry(MF, LIS, Reg) && usedInFramelessRegion(FR, Reg))
    return true;

  return !liveAcrossColdCall(MF, LIS, Reg);
}

/// Skip VRs that only copy an incoming caller-saved physreg (a0/a1…).
static bool isEntryCopyFromCallerSavedPhys(const MachineFunction &MF,
                                           const LiveIntervals &LIS,
                                           const MachineRegisterInfo &MRI,
                                           const TargetRegisterInfo *TRI,
                                           Register Reg) {
  MachineBasicBlock *MBB = defBlock(LIS, Reg);
  if (!isEntryBlock(MF, MBB))
    return false;
  MachineInstr *MI =
      LIS.getInstructionFromIndex(LIS.getInterval(Reg).beginIndex());
  if (!MI || !MI->isCopy())
    return false;
  Register Src = MI->getOperand(1).getReg();
  if (!Src.isPhysical() || MRI.isReserved(Src))
    return false;
  return RISCVRI::isCallerSavedGPREncoding(TRI->getEncodingValue(Src));
}

static bool memOperandLooksLikeFinishOrEnd(const MachineInstr *MI) {
  for (const MachineMemOperand *MMO : MI->memoperands()) {
    const Value *V = MMO->getValue();
    if (!V)
      continue;
    StringRef Name = V->getName();
    if (Name.contains("finish") || Name.contains("end_p"))
      return true;
  }
  return false;
}

/// Entry loads of vector finish/end pointers (typically +8 / +16 from the
/// vector object). Name and offset heuristics are ABI-layout specific.
static bool isEntryFinishOrEndLoad(const MachineFunction &MF,
                                   const LiveIntervals &LIS, Register Reg) {
  if (!defInEntry(MF, LIS, Reg))
    return false;
  MachineInstr *MI =
      LIS.getInstructionFromIndex(LIS.getInterval(Reg).beginIndex());
  if (!MI || !MI->mayLoad())
    return false;

  if (memOperandLooksLikeFinishOrEnd(MI))
    return true;

  // Duplicate isel loads may lack IR names; fall back to vector layout offsets.
  switch (MI->getOpcode()) {
  case RISCV::LD:
  case RISCV::LD_RV32:
  case RISCV::LW:
  case RISCV::LWU:
    if (MI->getNumOperands() >= 3 && MI->getOperand(2).isImm()) {
      int64_t Off = MI->getOperand(2).getImm();
      return Off == 8 || Off == 16;
    }
    break;
  default:
    break;
  }
  return false;
}

static DebugLoc getHintDebugLoc(const LiveIntervals &LIS, Register Reg) {
  if (!LIS.hasInterval(Reg))
    return DebugLoc();
  const LiveInterval &LI = LIS.getInterval(Reg);
  if (LI.empty())
    return DebugLoc();
  if (MachineInstr *MI = LIS.getInstructionFromIndex(LI.beginIndex()))
    return MI->getDebugLoc();
  return DebugLoc();
}

bool RISCVFramelessRAAnalysis::runOnMachineFunction(MachineFunction &MF) {
  // Pass is only scheduled when dataflow shrink-wrapping is on.
  if (DisableRISCVFramelessRA)
    return false;
  if (skipFunction(MF.getFunction()))
    return false;
  if (shouldSkipFramelessRA(MF))
    return false;
  if (!coldPathExists(MF) || !framelessRegionExists(MF))
    return false;

  LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  RISCVMachineFunctionInfo *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  auto &ORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  SmallVector<MachineBasicBlock *, 8> FramelessBlocks;
  computeFramelessRegionBlocks(MF, FramelessBlocks);
  SmallPtrSet<MachineBasicBlock *, 16> FramelessSet(FramelessBlocks.begin(),
                                                    FramelessBlocks.end());

  bool Changed = false;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);

    // Only entry finish/end pointer loads unlock delayed frames for
    // push_back-like shapes; filter those first.
    if (!isEntryFinishOrEndLoad(MF, LIS, Reg))
      continue;
    if (isEntryCopyFromCallerSavedPhys(MF, LIS, MRI, TRI, Reg))
      continue;
    if (!shouldHintVReg(MF, Reg, LIS, FramelessSet))
      continue;

    MRI.setRegAllocationHint(Reg, RISCVRI::FramelessCallerSaved, Register());
    RVFI->setFramelessRACost(Reg, RISCVRI::EntryFramelessRACost);
    RVFI->setFramelessRAHardHint(Reg);
    Changed = true;

    DebugLoc DL = getHintDebugLoc(LIS, Reg);
    MachineBasicBlock *HintMBB = defBlock(LIS, Reg);
    if (!HintMBB)
      HintMBB = &MF.front();
    MachineOptimizationRemarkAnalysis Remark(DEBUG_TYPE, "FramelessRAHint",
                                             DiagnosticLocation(DL), HintMBB);
    std::string Msg;
    raw_string_ostream(Msg)
        << "Frameless-path hint for " << printReg(Reg, TRI)
        << " (cost=" << RISCVRI::EntryFramelessRACost << ")";
    Remark << Msg;
    ORE.emit(Remark);

    LLVM_DEBUG(dbgs() << "FramelessRA " << printReg(Reg, TRI) << " cost "
                      << RISCVRI::EntryFramelessRACost << '\n');
  }

  return Changed;
}
