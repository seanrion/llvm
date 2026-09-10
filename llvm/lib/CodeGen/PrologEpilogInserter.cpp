//===- PrologEpilogInserter.cpp - Insert Prolog/Epilog code in function ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PEI.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/ShrinkFrameUtils.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "prolog-epilog"

using MBBVector = SmallVector<MachineBasicBlock *, 4>;

STATISTIC(NumLeafFuncWithSpills, "Number of leaf functions with CSRs");
STATISTIC(NumFuncSeen, "Number of functions seen in PEI");

static void mergeCSIUnique(std::vector<CalleeSavedInfo> &DST,
                           const CalleeSavedInfo &CS);

namespace {

class PEIImpl {
  RegScavenger *RS = nullptr;

  // Prolog and Epilog blocks: stack allocation / deallocation.
  MBBVector PrologBlocks;
  MBBVector EpilogBlocks;

  // Save and Restore blocks: CSR spill / restore (may differ from prolog
  // when enableCSRSaveRestorePointsSplit is true).
  MBBVector SaveBlocks;
  MBBVector RestoreBlocks;

  // Flag to control whether to use the register scavenger to resolve
  // frame index materialization registers. Set according to
  // TRI->requiresFrameIndexScavenging() for the current function.
  bool FrameIndexVirtualScavenging = false;

  // Flag to control whether the scavenger should be passed even though
  // FrameIndexVirtualScavenging is used.
  bool FrameIndexEliminationScavenging = false;

  // Emit remarks.
  MachineOptimizationRemarkEmitter *ORE = nullptr;

  void calculateCallFrameInfo(MachineFunction &MF);
  void calculateSaveRestoreBlocks(MachineFunction &MF);
  void calculatePrologEpilogBlocks(MachineFunction &MF);
  void spillCalleeSavedRegs(MachineFunction &MF);

  void calculateFrameObjectOffsets(MachineFunction &MF);
  void replaceFrameIndices(MachineFunction &MF);
  void replaceFrameIndices(MachineBasicBlock *BB, MachineFunction &MF,
                           int &SPAdj);
  // Frame indices in debug values are encoded in a target independent
  // way with simply the frame index and offset rather than any
  // target-specific addressing mode.
  bool replaceFrameIndexDebugInstr(MachineFunction &MF, MachineInstr &MI,
                                   unsigned OpIdx, int SPAdj = 0);
  // Does same as replaceFrameIndices but using the backward MIR walk and
  // backward register scavenger walk.
  void replaceFrameIndicesBackward(MachineFunction &MF);
  void replaceFrameIndicesBackward(MachineBasicBlock *BB, MachineFunction &MF,
                                   int &SPAdj);

  void insertPrologEpilogCode(MachineFunction &MF);
  void insertZeroCallUsedRegs(MachineFunction &MF);

public:
  PEIImpl(MachineOptimizationRemarkEmitter *ORE) : ORE(ORE) {}
  bool run(MachineFunction &MF);
};

class PEILegacy : public MachineFunctionPass {
public:
  static char ID;

  PEILegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
  /// frame indexes with appropriate references.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char PEILegacy::ID = 0;

char &llvm::PrologEpilogCodeInserterID = PEILegacy::ID;

INITIALIZE_PASS_BEGIN(PEILegacy, DEBUG_TYPE, "Prologue/Epilogue Insertion",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(PEILegacy, DEBUG_TYPE,
                    "Prologue/Epilogue Insertion & Frame Finalization", false,
                    false)

MachineFunctionPass *llvm::createPrologEpilogInserterPass() {
  return new PEILegacy();
}

STATISTIC(NumBytesStackSpace,
          "Number of bytes used for stack in all functions");

void PEILegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineOptimizationRemarkEmitterPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// StackObjSet - A set of stack object indexes
using StackObjSet = SmallSetVector<int, 8>;

using SavedDbgValuesMap =
    SmallDenseMap<MachineBasicBlock *, SmallVector<MachineInstr *, 4>, 4>;

/// Stash DBG_VALUEs that describe parameters and which are placed at the start
/// of the block. Later on, after the prologue code has been emitted, the
/// stashed DBG_VALUEs will be reinserted at the start of the block.
static void stashEntryDbgValues(MachineBasicBlock &MBB,
                                SavedDbgValuesMap &EntryDbgValues) {
  SmallVector<const MachineInstr *, 4> FrameIndexValues;

  for (auto &MI : MBB) {
    if (!MI.isDebugInstr())
      break;
    if (!MI.isDebugValue() || !MI.getDebugVariable()->isParameter())
      continue;
    if (any_of(MI.debug_operands(),
               [](const MachineOperand &MO) { return MO.isFI(); })) {
      // We can only emit valid locations for frame indices after the frame
      // setup, so do not stash away them.
      FrameIndexValues.push_back(&MI);
      continue;
    }
    const DILocalVariable *Var = MI.getDebugVariable();
    const DIExpression *Expr = MI.getDebugExpression();
    auto Overlaps = [Var, Expr](const MachineInstr *DV) {
      return Var == DV->getDebugVariable() &&
             Expr->fragmentsOverlap(DV->getDebugExpression());
    };
    // See if the debug value overlaps with any preceding debug value that will
    // not be stashed. If that is the case, then we can't stash this value, as
    // we would then reorder the values at reinsertion.
    if (llvm::none_of(FrameIndexValues, Overlaps))
      EntryDbgValues[&MBB].push_back(&MI);
  }

  // Remove stashed debug values from the block.
  if (auto It = EntryDbgValues.find(&MBB); It != EntryDbgValues.end())
    for (auto *MI : It->second)
      MI->removeFromParent();
}

bool PEIImpl::run(MachineFunction &MF) {
  NumFuncSeen++;
  const Function &F = MF.getFunction();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();

  RS = TRI->requiresRegisterScavenging(MF) ? new RegScavenger() : nullptr;
  FrameIndexVirtualScavenging = TRI->requiresFrameIndexScavenging(MF);

  // Spill frame pointer and/or base pointer registers if they are clobbered.
  // It is placed before call frame instruction elimination so it will not mess
  // with stack arguments.
  TFI->spillFPBP(MF);

  // Calculate the MaxCallFrameSize value for the function's frame
  // information. Also eliminates call frame pseudo instructions.
  calculateCallFrameInfo(MF);

  // Determine placement of CSR spill/restore code:
  // place all spills in the entry block, all restores in return blocks
  // (or use shrink-wrapping / multi-point save-restore maps).
  calculateSaveRestoreBlocks(MF);

  // Determine placement of prolog/epilog (stack allocation) code.
  calculatePrologEpilogBlocks(MF);

  // Stash away DBG_VALUEs that should not be moved by insertion of prolog code.
  SavedDbgValuesMap EntryDbgValues;
  for (MachineBasicBlock *PrologBlock : PrologBlocks)
    stashEntryDbgValues(*PrologBlock, EntryDbgValues);

  // Handle CSR spilling and restoring, for targets that need it.
  if (MF.getTarget().usesPhysRegsForValues())
    spillCalleeSavedRegs(MF);

  // Allow the target machine to make final modifications to the function
  // before the frame layout is finalized.
  TFI->processFunctionBeforeFrameFinalized(MF, RS);

  // Calculate actual frame offsets for all abstract stack objects...
  calculateFrameObjectOffsets(MF);

  TFI->emitPostFrameLayoutCFI(MF);

  // Add prolog and epilog code to the function.  This function is required
  // to align the stack frame as necessary for any stack variables or
  // called functions.  Because of this, calculateCalleeSavedRegisters()
  // must be called before this function in order to set the AdjustsStack
  // and MaxCallFrameSize variables.
  if (!F.hasFnAttribute(Attribute::Naked))
    insertPrologEpilogCode(MF);

  // Reinsert stashed debug values at the start of the entry blocks.
  for (auto &I : EntryDbgValues)
    I.first->insert(I.first->begin(), I.second.begin(), I.second.end());

  // Allow the target machine to make final modifications to the function
  // before the frame layout is finalized.
  TFI->processFunctionBeforeFrameIndicesReplaced(MF, RS);

  // Replace all MO_FrameIndex operands with physical register references
  // and actual offsets.
  if (TFI->needsFrameIndexResolution(MF)) {
    // Allow the target to determine this after knowing the frame size.
    FrameIndexEliminationScavenging =
        (RS && !FrameIndexVirtualScavenging) ||
        TRI->requiresFrameIndexReplacementScavenging(MF);

    if (TRI->eliminateFrameIndicesBackwards())
      replaceFrameIndicesBackward(MF);
    else
      replaceFrameIndices(MF);
  }

  // If register scavenging is needed, as we've enabled doing it as a
  // post-pass, scavenge the virtual registers that frame index elimination
  // inserted.
  if (TRI->requiresRegisterScavenging(MF) && FrameIndexVirtualScavenging)
    scavengeFrameVirtualRegs(MF, *RS);

  // Warn on stack size when we exceeds the given limit.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t StackSize = MFI.getStackSize();

  uint64_t Threshold = TFI->getStackThreshold();
  if (MF.getFunction().hasFnAttribute("warn-stack-size")) {
    bool Failed = MF.getFunction()
                      .getFnAttribute("warn-stack-size")
                      .getValueAsString()
                      .getAsInteger(10, Threshold);
    // Verifier should have caught this.
    assert(!Failed && "Invalid warn-stack-size fn attr value");
    (void)Failed;
  }
  uint64_t UnsafeStackSize = MFI.getUnsafeStackSize();
  if (MF.getFunction().hasFnAttribute(Attribute::SafeStack))
    StackSize += UnsafeStackSize;

  if (StackSize > Threshold) {
    DiagnosticInfoStackSize DiagStackSize(F, StackSize, Threshold, DS_Warning);
    F.getContext().diagnose(DiagStackSize);
    int64_t SpillSize = 0;
    for (int Idx = MFI.getObjectIndexBegin(), End = MFI.getObjectIndexEnd();
         Idx != End; ++Idx) {
      if (MFI.isSpillSlotObjectIndex(Idx))
        SpillSize += MFI.getObjectSize(Idx);
    }

    [[maybe_unused]] float SpillPct =
        static_cast<float>(SpillSize) / static_cast<float>(StackSize);
    LLVM_DEBUG(
        dbgs() << formatv("{0}/{1} ({3:P}) spills, {2}/{1} ({4:P}) variables",
                          SpillSize, StackSize, StackSize - SpillSize, SpillPct,
                          1.0f - SpillPct));
    if (UnsafeStackSize != 0) {
      LLVM_DEBUG(dbgs() << formatv(", {0}/{2} ({1:P}) unsafe stack",
                                   UnsafeStackSize,
                                   static_cast<float>(UnsafeStackSize) /
                                       static_cast<float>(StackSize),
                                   StackSize));
    }
    LLVM_DEBUG(dbgs() << "\n");
  }

  ORE->emit([&]() {
    return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, "StackSize",
                                             MF.getFunction().getSubprogram(),
                                             &MF.front())
           << ore::NV("NumStackBytes", StackSize)
           << " stack bytes in function '"
           << ore::NV("Function", MF.getFunction().getName()) << "'";
  });

  // Emit any remarks implemented for the target, based on final frame layout.
  TFI->emitRemarks(MF, ORE);

  delete RS;
  SaveBlocks.clear();
  RestoreBlocks.clear();
  PrologBlocks.clear();
  EpilogBlocks.clear();
  MFI.clearSavePoints();
  MFI.clearRestorePoints();
  return true;
}

/// runOnMachineFunction - Insert prolog/epilog code and replace abstract
/// frame indexes with appropriate references.
bool PEILegacy::runOnMachineFunction(MachineFunction &MF) {
  MachineOptimizationRemarkEmitter *ORE =
      &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
  return PEIImpl(ORE).run(MF);
}

PreservedAnalyses
PrologEpilogInserterPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  MachineOptimizationRemarkEmitter &ORE =
      MFAM.getResult<MachineOptimizationRemarkEmitterAnalysis>(MF);
  if (!PEIImpl(&ORE).run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<MachineDominatorTreeAnalysis>()
      .preserve<MachineLoopAnalysis>();
}

/// Calculate the MaxCallFrameSize variable for the function's frame
/// information and eliminate call frame pseudo instructions.
void PEIImpl::calculateCallFrameInfo(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Get the function call frame set-up and tear-down instruction opcode
  unsigned FrameSetupOpcode = TII.getCallFrameSetupOpcode();
  unsigned FrameDestroyOpcode = TII.getCallFrameDestroyOpcode();

  // Early exit for targets which have no call frame setup/destroy pseudo
  // instructions.
  if (FrameSetupOpcode == ~0u && FrameDestroyOpcode == ~0u)
    return;

  // (Re-)Compute the MaxCallFrameSize.
  [[maybe_unused]] uint64_t MaxCFSIn =
      MFI.isMaxCallFrameSizeComputed() ? MFI.getMaxCallFrameSize() : UINT64_MAX;
  std::vector<MachineBasicBlock::iterator> FrameSDOps;
  MFI.computeMaxCallFrameSize(MF, &FrameSDOps);
  assert(MFI.getMaxCallFrameSize() <= MaxCFSIn &&
         "Recomputing MaxCFS gave a larger value.");
  assert((FrameSDOps.empty() || MF.getFrameInfo().adjustsStack()) &&
         "AdjustsStack not set in presence of a frame pseudo instruction.");

  if (TFI->canSimplifyCallFramePseudos(MF)) {
    // If call frames are not being included as part of the stack frame, and
    // the target doesn't indicate otherwise, remove the call frame pseudos
    // here. The sub/add sp instruction pairs are still inserted, but we don't
    // need to track the SP adjustment for frame index elimination.
    for (MachineBasicBlock::iterator I : FrameSDOps)
      TFI->eliminateCallFramePseudoInstr(MF, *I->getParent(), I);

    // We can't track the call frame size after call frame pseudos have been
    // eliminated. Set it to zero everywhere to keep MachineVerifier happy.
    for (MachineBasicBlock &MBB : MF)
      MBB.setCallFrameSize(0);
  }
}

/// Compute blocks for placing prolog and epilog (stack allocation) code.
void PEIImpl::calculatePrologEpilogBlocks(MachineFunction &MF) {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // If shrink-frame has placed a Prolog, honor it regardless of split mode.
  // Frame-bound CSRs (ra / FP) travel with the main prologue (GCC policy).
  // When a delayed Prolog is unsafe (mixed frame/frameless return, CSR save or
  // stack touch before P0, frame-bound clobber before P0), hoist Prolog to
  // Entry and pin frame-bound spills there — keep s* multi-point delay.
  if (MFI.getProlog()) {
    MachineDominatorTree DT(MF);
    MachineBasicBlock *Prolog = MFI.getProlog();
    MachineBasicBlock *Entry = &MF.front();
    SmallVector<Register, 4> FrameBound;
    TFI->getFrameBoundCalleeSaves(MF, FrameBound);

    auto PinFrameBoundToEntry = [&]() {
      if (FrameBound.empty())
        return;
      if (!MFI.getSavePoints().empty()) {
        SaveRestorePoints NewSP = MFI.getSavePoints();
        SmallDenseSet<unsigned, 4> FB;
        for (Register R : FrameBound)
          if (R)
            FB.insert(R.id());
        std::vector<CalleeSavedInfo> Moved;
        SmallVector<MachineBasicBlock *, 4> Empty;
        for (auto &KV : NewSP) {
          if (KV.first == Entry)
            continue;
          std::vector<CalleeSavedInfo> Keep;
          for (const CalleeSavedInfo &CS : KV.second) {
            if (FB.contains(CS.getReg()))
              Moved.push_back(CS);
            else
              Keep.push_back(CS);
          }
          KV.second = std::move(Keep);
          if (KV.second.empty())
            Empty.push_back(KV.first);
        }
        for (MachineBasicBlock *BB : Empty)
          NewSP.erase(BB);
        for (const CalleeSavedInfo &CS : Moved)
          mergeCSIUnique(NewSP[Entry], CS);
        MFI.setSavePoints(std::move(NewSP));
      }
      if (!MFI.getRestorePoints().empty()) {
        SaveRestorePoints NewRP = MFI.getRestorePoints();
        SmallDenseSet<unsigned, 4> FB;
        for (Register R : FrameBound)
          if (R)
            FB.insert(R.id());
        std::vector<CalleeSavedInfo> FBCSIs;
        for (auto &KV : NewRP) {
          for (const CalleeSavedInfo &CS : KV.second)
            if (FB.contains(CS.getReg()))
              mergeCSIUnique(FBCSIs, CS);
          llvm::erase_if(KV.second, [&](const CalleeSavedInfo &CS) {
            return FB.contains(CS.getReg());
          });
        }
        if (!FBCSIs.empty()) {
          for (MachineBasicBlock &MBB : MF) {
            if (!MBB.isReturnBlock())
              continue;
            for (const CalleeSavedInfo &CS : FBCSIs)
              mergeCSIUnique(NewRP[&MBB], CS);
          }
        }
        SmallVector<MachineBasicBlock *, 4> Empty;
        for (auto &KV : NewRP)
          if (KV.second.empty())
            Empty.push_back(KV.first);
        for (MachineBasicBlock *BB : Empty)
          NewRP.erase(BB);
        MFI.setRestorePoints(std::move(NewRP));
      }
    };

    if (Prolog != Entry && mustHoistShrinkFramePrologToEntry(MF, Prolog, DT)) {
      // ShrinkWrapping abandons clones when mustHoist&&clones. PEI must not
      // undo clones while keeping stale RestorePoints. If clones remain, do not
      // hoist to Entry (would dominate frameless clones).
      if (MFI.hasShrinkFrameClones()) {
        LLVM_DEBUG(dbgs() << "PEI: mustHoist with clones still present "
                             "(ShrinkWrapping should have undone them) — "
                             "skip hoist\n");
      } else {
        LLVM_DEBUG(
            dbgs() << "PEI: hoist Prolog from " << printMBBReference(*Prolog)
                   << " to Entry (main prologue before stack/returns)\n");
        MFI.setProlog(Entry);
        Prolog = Entry;
        PinFrameBoundToEntry();
      }
    }

    if (MFI.getProlog()) {
      Prolog = MFI.getProlog();
      PrologBlocks.push_back(Prolog);
      for (MachineBasicBlock &MBB : MF) {
        if (MBB.isReturnBlock() && DT.dominates(Prolog, &MBB) &&
            !MFI.isShrinkFrameClone(&MBB))
          EpilogBlocks.push_back(&MBB);
      }
      // Shrink-frame clones must not run emitEpilogue(); CSR ld only via
      // appendShrinkFrameCloneRestorePoints.
      if (EpilogBlocks.empty()) {
        LLVM_DEBUG(dbgs() << "PEI: shrink-frame Prolog "
                          << printMBBReference(*Prolog)
                          << " dominates no return — ignoring Prolog\n");
        PrologBlocks.clear();
        abandonShrinkFrame(MF);
        // Fall through to non-shrink-frame placement below.
      } else {
        return;
      }
    }
  }

  // Data-flow CSR split (GCC-like separate shrink-wrapping): always allocate
  // and free the full frame at entry / returns. Only CSR save/restore points
  // may be multi-point; never shrink addi sp onto a cold path or loop latch.
  if (TFI->enableCSRSaveRestorePointsSplit()) {
    PrologBlocks.push_back(&MF.front());
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.isEHFuncletEntry())
        PrologBlocks.push_back(&MBB);
      if (MBB.isReturnBlock())
        EpilogBlocks.push_back(&MBB);
    }
    return;
  }

  if (MFI.getProlog()) {
    PrologBlocks.push_back(MFI.getProlog());
  } else if (!MFI.getSavePoints().empty()) {
    // Legacy shrink-wrap colocates prolog with the (single) save point.
    PrologBlocks = SaveBlocks;
  } else {
    PrologBlocks.push_back(&MF.front());
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.isEHFuncletEntry())
        PrologBlocks.push_back(&MBB);
    }
  }

  if (MFI.getEpilog()) {
    EpilogBlocks.push_back(MFI.getEpilog());
  } else if (!MFI.getRestorePoints().empty()) {
    // Legacy shrink-wrap colocates epilog with the (single) restore point.
    EpilogBlocks = RestoreBlocks;
  } else {
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.isReturnBlock())
        EpilogBlocks.push_back(&MBB);
    }
  }
}

/// Compute the sets of blocks for saving and restoring callee-saved registers.
void PEIImpl::calculateSaveRestoreBlocks(MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  // Even when we do not change any CSR, we still want to insert the
  // prologue and epilogue of the function.
  // So set the save points for those.

  // Use the points found by shrink-wrapping, if any.
  if (!MFI.getSavePoints().empty()) {
    if (!TFI->enableCSRSaveRestorePointsSplit()) {
      assert(MFI.getSavePoints().size() == 1 &&
             "Multiple save points require enableCSRSaveRestorePointsSplit!");
      const auto &SavePoint = *MFI.getSavePoints().begin();
      SaveBlocks.push_back(SavePoint.first);
      assert(MFI.getRestorePoints().size() == 1 &&
             "Multiple restore points require enableCSRSaveRestorePointsSplit!");
      const auto &RestorePoint = *MFI.getRestorePoints().begin();
      MachineBasicBlock *RestoreBlock = RestorePoint.first;
      // If RestoreBlock does not have any successor and is not a return block
      // then the end point is unreachable and we do not need to insert any
      // epilogue.
      if (!RestoreBlock->succ_empty() || RestoreBlock->isReturnBlock())
        RestoreBlocks.push_back(RestoreBlock);
      return;
    }

    assert(!MFI.getRestorePoints().empty() &&
           "Both restore and save must be set");
    for (const auto &Item : MFI.getSavePoints())
      SaveBlocks.push_back(Item.first);

    for (const auto &Item : MFI.getRestorePoints()) {
      MachineBasicBlock *RestoreBlock = Item.first;
      // If RestoreBlock does not have any successor and is not a return block
      // then the end point is unreachable and we do not need to insert any
      // epilogue.
      if (!RestoreBlock->succ_empty() || RestoreBlock->isReturnBlock())
        RestoreBlocks.push_back(RestoreBlock);
    }
    return;
  }

  // Save refs to entry and return blocks.
  SaveBlocks.push_back(&MF.front());
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.isEHFuncletEntry())
      SaveBlocks.push_back(&MBB);
    if (MBB.isReturnBlock())
      RestoreBlocks.push_back(&MBB);
  }
}

static void assignCalleeSavedSpillSlots(MachineFunction &F,
                                        const BitVector &SavedRegs) {
  if (SavedRegs.empty())
    return;

  const TargetRegisterInfo *RegInfo = F.getSubtarget().getRegisterInfo();
  const MCPhysReg *CSRegs = F.getRegInfo().getCalleeSavedRegs();
  BitVector CSMask(SavedRegs.size());

  for (unsigned i = 0; CSRegs[i]; ++i)
    CSMask.set(CSRegs[i]);

  std::vector<CalleeSavedInfo> CSI;
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (SavedRegs.test(Reg)) {
      bool SavedSuper = false;
      for (const MCPhysReg &SuperReg : RegInfo->superregs(Reg)) {
        // Some backends set all aliases for some registers as saved, such as
        // Mips's $fp, so they appear in SavedRegs but not CSRegs.
        if (SavedRegs.test(SuperReg) && CSMask.test(SuperReg)) {
          SavedSuper = true;
          break;
        }
      }

      if (!SavedSuper)
        CSI.push_back(CalleeSavedInfo(Reg));
    }
  }

  const TargetFrameLowering *TFI = F.getSubtarget().getFrameLowering();
  MachineFrameInfo &MFI = F.getFrameInfo();
  if (!TFI->assignCalleeSavedSpillSlots(F, RegInfo, CSI)) {
    // If target doesn't implement this, use generic code.

    if (CSI.empty())
      return; // Early exit if no callee saved registers are modified!

    unsigned NumFixedSpillSlots;
    const TargetFrameLowering::SpillSlot *FixedSpillSlots =
        TFI->getCalleeSavedSpillSlots(NumFixedSpillSlots);

    // Now that we know which registers need to be saved and restored, allocate
    // stack slots for them.
    for (auto &CS : CSI) {
      // If the target has spilled this register to another register or already
      // handled it , we don't need to allocate a stack slot.
      if (CS.isSpilledToReg())
        continue;

      MCRegister Reg = CS.getReg();
      const TargetRegisterClass *RC = RegInfo->getMinimalPhysRegClass(Reg);

      int FrameIdx;
      if (RegInfo->hasReservedSpillSlot(F, Reg, FrameIdx)) {
        CS.setFrameIdx(FrameIdx);
        continue;
      }

      // Check to see if this physreg must be spilled to a particular stack slot
      // on this target.
      const TargetFrameLowering::SpillSlot *FixedSlot = FixedSpillSlots;
      while (FixedSlot != FixedSpillSlots + NumFixedSpillSlots &&
             FixedSlot->Reg != Reg)
        ++FixedSlot;

      unsigned Size = RegInfo->getSpillSize(*RC);
      if (FixedSlot == FixedSpillSlots + NumFixedSpillSlots) {
        // Nope, just spill it anywhere convenient.
        Align Alignment = RegInfo->getSpillAlign(*RC);
        // We may not be able to satisfy the desired alignment specification of
        // the TargetRegisterClass if the stack alignment is smaller. Use the
        // min.
        Alignment = std::min(Alignment, TFI->getStackAlign());
        FrameIdx = MFI.CreateStackObject(Size, Alignment, true, nullptr,
                                         RegInfo->getSpillStackID(*RC));
        MFI.setIsCalleeSavedObjectIndex(FrameIdx, true);
      } else {
        // Spill it to the stack where we must.
        FrameIdx = MFI.CreateFixedSpillStackObject(Size, FixedSlot->Offset);
      }

      CS.setFrameIdx(FrameIdx);
    }
  }

  MFI.setCalleeSavedInfo(CSI);
}

/// Helper function to update the liveness information for one callee-saved
/// register under multi-point save/restore maps.
static void updateLivenessSplit(MachineFunction &MF, CalleeSavedInfo &Info) {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  // Visited contains blocks where the CSR is still live (outside the
  // save→restore region for this register).
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> WorkList;
  MachineBasicBlock *Entry = &MF.front();

  WorkList.push_back(Entry);
  Visited.insert(Entry);
  while (!WorkList.empty()) {
    bool AddSucc = true;
    const MachineBasicBlock *CurBB = WorkList.pop_back_val();
    auto SaveIt = MFI.getSavePoints().find(CurBB);
    auto RestoreIt = MFI.getRestorePoints().find(CurBB);
    if (SaveIt != MFI.getSavePoints().end() &&
        count_if(SaveIt->second, [&Info](const CalleeSavedInfo &Other) {
          return Other.getReg() == Info.getReg();
        })) {
      AddSucc = false;
      if (RestoreIt != MFI.getRestorePoints().end() &&
          count_if(RestoreIt->second, [&Info](const CalleeSavedInfo &Other) {
            return Other.getReg() == Info.getReg();
          }))
        AddSucc = true;
    }

    if (AddSucc) {
      for (MachineBasicBlock *SuccBB : CurBB->successors())
        if (Visited.insert(SuccBB).second)
          WorkList.push_back(SuccBB);
    }
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock *MBB : Visited) {
    MCRegister Reg = Info.getReg();
    if (!MRI.isReserved(Reg) && !MBB->isLiveIn(Reg))
      MBB->addLiveIn(Reg);
  }
  if (Info.isSpilledToReg()) {
    for (MachineBasicBlock &MBB : MF) {
      if (Visited.count(&MBB))
        continue;
      MCRegister DstReg = Info.getDstReg();
      if (!MBB.isLiveIn(DstReg))
        MBB.addLiveIn(DstReg);
    }
  }
}

/// Helper function to update the liveness information for the callee-saved
/// registers (legacy single save/restore point).
static void updateLivenessLegacy(MachineFunction &MF) {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  // Visited will contain all the basic blocks that are in the region
  // where the callee saved registers are alive:
  // - Anything that is not Save or Restore -> LiveThrough.
  // - Save -> LiveIn.
  // - Restore -> LiveOut.
  // The live-out is not attached to the block, so no need to keep
  // Restore in this set.
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> WorkList;
  MachineBasicBlock *Entry = &MF.front();

  assert(MFI.getSavePoints().size() < 2 &&
         "Multiple save points not yet supported!");
  MachineBasicBlock *Save = MFI.getSavePoints().empty()
                                ? nullptr
                                : (*MFI.getSavePoints().begin()).first;

  if (!Save)
    Save = Entry;

  if (Entry != Save) {
    WorkList.push_back(Entry);
    Visited.insert(Entry);
  }
  Visited.insert(Save);

  assert(MFI.getRestorePoints().size() < 2 &&
         "Multiple restore points not yet supported!");
  MachineBasicBlock *Restore = MFI.getRestorePoints().empty()
                                   ? nullptr
                                   : (*MFI.getRestorePoints().begin()).first;
  if (Restore)
    // By construction Restore cannot be visited, otherwise it
    // means there exists a path to Restore that does not go
    // through Save.
    WorkList.push_back(Restore);

  while (!WorkList.empty()) {
    const MachineBasicBlock *CurBB = WorkList.pop_back_val();
    // By construction, the region that is after the save point is
    // dominated by the Save and post-dominated by the Restore.
    if (CurBB == Save && Save != Restore)
      continue;
    // Enqueue all the successors not already visited.
    // Those are by construction either before Save or after Restore.
    for (MachineBasicBlock *SuccBB : CurBB->successors())
      if (Visited.insert(SuccBB).second)
        WorkList.push_back(SuccBB);
  }

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (const CalleeSavedInfo &I : CSI) {
    for (MachineBasicBlock *MBB : Visited) {
      MCRegister Reg = I.getReg();
      // Add the callee-saved register as live-in.
      // It's killed at the spill.
      if (!MRI.isReserved(Reg) && !MBB->isLiveIn(Reg))
        MBB->addLiveIn(Reg);
    }
    // If callee-saved register is spilled to another register rather than
    // spilling to stack, the destination register has to be marked as live for
    // each MBB between the prologue and epilogue so that it is not clobbered
    // before it is reloaded in the epilogue. The Visited set contains all
    // blocks outside of the region delimited by prologue/epilogue.
    if (I.isSpilledToReg()) {
      for (MachineBasicBlock &MBB : MF) {
        if (Visited.count(&MBB))
          continue;
        MCRegister DstReg = I.getDstReg();
        if (!MBB.isLiveIn(DstReg))
          MBB.addLiveIn(DstReg);
      }
    }
  }
}

/// Insert spill code for the callee-saved registers used in the function.
static void insertCSRSaves(MachineBasicBlock &SaveBlock,
                           ArrayRef<CalleeSavedInfo> CSI) {
  if (CSI.empty())
    return;

  MachineFunction &MF = *SaveBlock.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  MachineBasicBlock::iterator I = SaveBlock.begin();
  if (!TFI->spillCalleeSavedRegisters(SaveBlock, I, CSI, TRI)) {
    for (const CalleeSavedInfo &CS : CSI) {
      TFI->spillCalleeSavedRegister(SaveBlock, I, CS, TII, TRI);
    }
  }
}

/// True if \p MBB has no non-EH successor (noreturn / trap / throw-only exit).
static bool isNoreturnOrEHOnlyExit(const MachineBasicBlock &MBB) {
  if (MBB.isReturnBlock())
    return false;
  for (const MachineBasicBlock *Succ : MBB.successors()) {
    if (!Succ->isEHPad())
      return false;
  }
  return true;
}

static bool isNoReturnCallMI(const MachineInstr &MI) {
  if (!MI.isCall())
    return false;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isGlobal()) {
      if (const Function *F = dyn_cast<Function>(MO.getGlobal()))
        if (F->doesNotReturn())
          return true;
    } else if (MO.isSymbol()) {
      // Libcalls may appear as external symbols without an IR Function*.
      StringRef N = MO.getSymbolName();
      if (N == "exit" || N == "_exit" || N == "abort" || N == "__cxa_throw" ||
          N == "__cxa_rethrow" || N == "_Unwind_Resume" ||
          N == "__assert_fail")
        return true;
    }
  }
  return false;
}

static bool blockHasNoReturnCall(const MachineBasicBlock &MBB) {
  return any_of(MBB, isNoReturnCallMI);
}

/// Collect return/epilog blocks reachable from \p Start without walking past
/// a noreturn call. Used so restores are not placed on early exits that never
/// executed the corresponding CSR save.
/// Collect epilog blocks that are dominated by \p SaveBB.
/// Dominance (not mere reachability) is required: a return reachable from a
/// save via the slow path may also be reachable from an early-exit that never
/// saved (Sphere::intersect / ray).
static void collectDominatedEpilogs(MachineBasicBlock *SaveBB,
                                    ArrayRef<MachineBasicBlock *> EpilogBlocks,
                                    const MachineDominatorTree &DT,
                                    SmallPtrSetImpl<MachineBasicBlock *> &Out) {
  for (MachineBasicBlock *EB : EpilogBlocks) {
    if (blockHasNoReturnCall(*EB))
      continue;
    if (DT.dominates(SaveBB, EB))
      Out.insert(EB);
  }
}

static void mergeCSIUnique(std::vector<CalleeSavedInfo> &DST,
                           const CalleeSavedInfo &CS) {
  if (any_of(DST, [&](const CalleeSavedInfo &E) {
        return E.getReg() == CS.getReg();
      }))
    return;
  DST.push_back(CS);
  sort(DST, [](const CalleeSavedInfo &L, const CalleeSavedInfo &R) {
    return L.getFrameIdx() < R.getFrameIdx();
  });
}

/// Move RestorePoints off noreturn-tainted blocks onto epilogs dominated by
/// the CSR's save points only (not every return — that broke SIBsim4/PENNANT).
static void rehomeNoreturnRestorePoints(MachineFrameInfo &MFI,
                                        MBBVector &EpilogBlocks) {
  const SaveRestorePoints &Cur = MFI.getRestorePoints();
  if (Cur.empty())
    return;

  SaveRestorePoints New;
  // Reg -> CSIs cleared from noreturn blocks.
  DenseMap<MCRegister, SmallVector<CalleeSavedInfo, 2>> ClearedByReg;
  bool Changed = false;
  for (const auto &[BB, Regs] : Cur) {
    if (Regs.empty())
      continue;

    // Pure return blocks keep their restores.
    if (BB->isReturnBlock() && !blockHasNoReturnCall(*BB)) {
      New.try_emplace(BB, Regs);
      continue;
    }

    if (isNoreturnOrEHOnlyExit(*BB) || blockHasNoReturnCall(*BB)) {
      Changed = true;
      LLVM_DEBUG(dbgs() << "PEI: rehome CSR restores from noreturn-tainted "
                        << printMBBReference(*BB) << '\n');
      for (const CalleeSavedInfo &CS : Regs)
        ClearedByReg[CS.getReg()].push_back(CS);
      continue;
    }

    New.try_emplace(BB, Regs);
  }
  if (!Changed)
    return;

  MachineFunction &MF = *EpilogBlocks.front()->getParent();
  MachineDominatorTree DT(MF);
  for (auto &[Reg, CSList] : ClearedByReg) {
    SmallPtrSet<MachineBasicBlock *, 8> Targets;
    bool AnySave = false;
    for (const auto &[SaveBB, SaveRegs] : MFI.getSavePoints()) {
      if (!any_of(SaveRegs, [&](const CalleeSavedInfo &S) {
            return S.getReg() == Reg;
          }))
        continue;
      AnySave = true;
      collectDominatedEpilogs(SaveBB, EpilogBlocks, DT, Targets);
    }
    if (!AnySave) {
      for (MachineBasicBlock *EB : EpilogBlocks)
        Targets.insert(EB);
    }
    // Use the first CSI entry (same FI/reg for a given CSR).
    const CalleeSavedInfo &CS = CSList.front();
    for (MachineBasicBlock *EB : Targets)
      mergeCSIUnique(New[EB], CS);
  }

  MFI.setRestorePoints(std::move(New));
}

/// Insert restore code for the callee-saved registers used in the function.
static void insertCSRRestores(MachineBasicBlock &RestoreBlock,
                              std::vector<CalleeSavedInfo> CSI) {
  if (CSI.empty())
    return;

  // Belt-and-suspenders: never emit restore(+CFI) on noreturn-only exits.
  // ShrinkWrapping/PEI rehome should have emptied these already.
  if (isNoreturnOrEHOnlyExit(RestoreBlock)) {
    LLVM_DEBUG(dbgs() << "Skip CSR restores in noreturn/EH-only exit "
                      << printMBBReference(RestoreBlock) << '\n');
    return;
  }

  MachineFunction &MF = *RestoreBlock.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  MachineBasicBlock::iterator I = RestoreBlock.getFirstTerminator();

  if (!TFI->restoreCalleeSavedRegisters(RestoreBlock, I, CSI, TRI)) {
    for (const CalleeSavedInfo &CI : reverse(CSI)) {
      TFI->restoreCalleeSavedRegister(RestoreBlock, I, CI, TII, TRI);
    }
  }
}

/// Insert CSR restores on frameless shared-return clone blocks. The save may
/// be on an ancestor of the frameless predecessor (not necessarily on the pred
/// itself). Only clone returns are targeted — never all non-epilog returns.
static void appendShrinkFrameCloneRestorePoints(
    MachineFunction &MF, MachineFrameInfo &MFI,
    DenseMap<MCRegister, CalleeSavedInfo *> &RegToInfo,
    SaveRestorePoints &Inner) {
  if (!MFI.hasShrinkFrameClones())
    return;

  MachineBasicBlock *SFProlog = MFI.getProlog();
  MachineDominatorTree DT(MF);

  for (const auto &KV : MFI.getShrinkFrameClones()) {
    MachineBasicBlock *Clone = KV.second;
    if (!Clone->isReturnBlock() || blockHasNoReturnCall(*Clone))
      continue;

    for (MachineBasicBlock *Pred : Clone->predecessors()) {
      if (SFProlog && DT.dominates(SFProlog, Pred))
        continue;

      for (const auto &[SaveBB, SaveRegs] : MFI.getSavePoints()) {
        if (SFProlog && DT.dominates(SFProlog, SaveBB))
          continue;
        if (SaveBB != Pred && !DT.dominates(SaveBB, Pred))
          continue;

        for (const CalleeSavedInfo &S : SaveRegs) {
          if (!isLegalOffPrologCSRSaveForCloneRestore(MF, SFProlog, SaveBB,
                                                      S.getReg(), DT))
            continue;
          auto It = RegToInfo.find(S.getReg());
          if (It == RegToInfo.end())
            continue;
          mergeCSIUnique(Inner[Clone], *It->second);
        }
      }
    }
  }
}

/// Fill per-BB CalleeSavedInfo vectors for multi-point save/restore maps.
/// Registers listed in SavePoints/RestorePoints are resolved against
/// \p RegToInfo. Any CSR missing from those lists is spilled/restored in
/// the prolog/epilog blocks instead — for restores, only epilogs dominated
/// by a save of that CSR (avoid early-exit ld without a prior sd).
static void fillCSInfoPerBB(MachineFrameInfo &MFI,
                            DenseMap<MCRegister, CalleeSavedInfo *> &RegToInfo,
                            MBBVector &PrologEpilogBlocks, bool IsSave) {
  SmallSet<MCRegister, 16> AssignedRegs;
  const SaveRestorePoints &SRPoints =
      IsSave ? MFI.getSavePoints() : MFI.getRestorePoints();
  SaveRestorePoints Inner;

  MachineDominatorTree DT;
  if (!IsSave && !PrologEpilogBlocks.empty())
    DT.recalculate(*PrologEpilogBlocks.front()->getParent());

  for (const auto &[BB, Regs] : SRPoints) {
    std::vector<CalleeSavedInfo> CSIV;
    for (const CalleeSavedInfo &Reg : Regs) {
      auto It = RegToInfo.find(Reg.getReg());
      if (It == RegToInfo.end())
        continue;
      // Drop restores on returns not dominated by any save of this CSR.
      // Never drop frame-bound CSRs (ra/FP): they are pinned to the frame
      // prolog and must be restored on returns (GCC separate policy / pr51933).
      // Clone returns get non-frame-bound restores from
      // appendShrinkFrameCloneRestorePoints below.
      if (!IsSave && BB->isReturnBlock() && !MFI.isShrinkFrameClone(BB)) {
        bool DominatedBySave = false;
        unsigned NumSaves = 0;
        for (const auto &[SaveBB, SaveRegs] : MFI.getSavePoints()) {
          if (!any_of(SaveRegs, [&](const CalleeSavedInfo &S) {
                return S.getReg() == Reg.getReg();
              }))
            continue;
          ++NumSaves;
          if (DT.dominates(SaveBB, BB)) {
            DominatedBySave = true;
            break;
          }
        }
        if (!DominatedBySave && NumSaves == 1) {
          SmallVector<Register, 4> FrameBound;
          BB->getParent()
              ->getSubtarget()
              .getFrameLowering()
              ->getFrameBoundCalleeSaves(*BB->getParent(), FrameBound);
          if (!any_of(FrameBound,
                      [&](Register R) { return R.id() == Reg.getReg(); }))
            continue;
        }
      }
      CSIV.push_back(*It->second);
      AssignedRegs.insert(Reg.getReg());
    }
    // AArch64 expects CSI sorted by frame index.
    sort(CSIV, [](const CalleeSavedInfo &Lhs, const CalleeSavedInfo &Rhs) {
      return Lhs.getFrameIdx() < Rhs.getFrameIdx();
    });
    if (!CSIV.empty())
      Inner.try_emplace(BB, std::move(CSIV));
  }

  // Spill/restore any CSR not assigned to a split point in prolog/epilog.
  if (AssignedRegs.size() < RegToInfo.size()) {
    for (auto &RTI : RegToInfo) {
      if (AssignedRegs.contains(RTI.first))
        continue;

      SmallVector<MachineBasicBlock *, 4> Targets;
      if (IsSave) {
        Targets.assign(PrologEpilogBlocks.begin(), PrologEpilogBlocks.end());
      } else {
        // Path-sensitive: only epilogs dominated by a save of this CSR.
        // Do not rehome onto the save block when Dominated is empty — that
        // restores too early if the CSR is live in successors.
        SmallPtrSet<MachineBasicBlock *, 8> Dominated;
        bool AnySave = false;
        for (const auto &[SaveBB, SaveRegs] : MFI.getSavePoints()) {
          if (!any_of(SaveRegs, [&](const CalleeSavedInfo &S) {
                return S.getReg() == RTI.first;
              }))
            continue;
          AnySave = true;
          collectDominatedEpilogs(SaveBB, PrologEpilogBlocks, DT, Dominated);
        }
        if (!AnySave) {
          Targets.assign(PrologEpilogBlocks.begin(), PrologEpilogBlocks.end());
        } else if (!Dominated.empty()) {
          Targets.assign(Dominated.begin(), Dominated.end());
        } else {
          // Shared-return diamond (pr78622): restore was listed on a join
          // return that is not dominated by the save, so it was dropped above.
          // No epilog is dominated either. Restore on each predecessor of a
          // return that lies on the save side (insertCSRRestores places ld
          // before the terminator — after the hot-path work, before the join).
          //
          // Only Pred with a unique successor == Epilog: a multi-exit Pred
          // would restore on non-return paths too.
          SmallPtrSet<MachineBasicBlock *, 8> PredTargets;
          for (MachineBasicBlock *Epilog : PrologEpilogBlocks) {
            if (!Epilog->isReturnBlock() || blockHasNoReturnCall(*Epilog))
              continue;
            for (MachineBasicBlock *Pred : Epilog->predecessors()) {
              if (Pred->succ_size() != 1 || *Pred->succ_begin() != Epilog)
                continue;
              for (const auto &[SaveBB, SaveRegs] : MFI.getSavePoints()) {
                if (!any_of(SaveRegs, [&](const CalleeSavedInfo &S) {
                      return S.getReg() == RTI.first;
                    }))
                  continue;
                if (Pred == SaveBB || DT.dominates(SaveBB, Pred))
                  PredTargets.insert(Pred);
              }
            }
          }
          Targets.assign(PredTargets.begin(), PredTargets.end());
        }
      }

      for (MachineBasicBlock *BB : Targets) {
        if (auto Entry = Inner.find(BB); Entry != Inner.end()) {
          auto &CSI = Entry->second;
          CSI.push_back(*RTI.second);
          sort(CSI, [](const CalleeSavedInfo &Lhs, const CalleeSavedInfo &Rhs) {
            return Lhs.getFrameIdx() < Rhs.getFrameIdx();
          });
          continue;
        }
        Inner.try_emplace(BB, std::vector<CalleeSavedInfo>{*RTI.second});
      }
    }
  }

  if (!IsSave && MFI.hasShrinkFrameClones()) {
    MachineFunction *MFF = nullptr;
    if (!PrologEpilogBlocks.empty())
      MFF = PrologEpilogBlocks.front()->getParent();
    else if (!MFI.getSavePoints().empty())
      MFF = MFI.getSavePoints().begin()->first->getParent();
    else if (!MFI.getShrinkFrameClones().empty())
      MFF = MFI.getShrinkFrameClones().begin()->second->getParent();
    if (MFF)
      appendShrinkFrameCloneRestorePoints(*MFF, MFI, RegToInfo, Inner);
  }

  if (IsSave)
    MFI.setSavePoints(std::move(Inner));
  else
    MFI.setRestorePoints(std::move(Inner));
}

void PEIImpl::spillCalleeSavedRegs(MachineFunction &MF) {
  // We can't list this requirement in getRequiredProperties because some
  // targets (WebAssembly) use virtual registers past this point, and the pass
  // pipeline is set up without giving the passes a chance to look at the
  // TargetMachine.
  // FIXME: Find a way to express this in getRequiredProperties.
  assert(MF.getProperties().hasNoVRegs());

  const Function &F = MF.getFunction();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const bool SplitPoints = TFI->enableCSRSaveRestorePointsSplit();

  // Determine which of the registers in the callee save list should be saved.
  BitVector SavedRegs;
  TFI->determineCalleeSaves(MF, SavedRegs, RS);

  // Assign stack slots for any callee-saved registers that must be spilled.
  assignCalleeSavedSpillSlots(MF, SavedRegs);

  // Add the code to save and restore the callee saved registers.
  if (!F.hasFnAttribute(Attribute::Naked)) {
    MFI.setCalleeSavedInfoValid(true);

    std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

    if (SplitPoints) {
      DenseMap<MCRegister, CalleeSavedInfo *> RegToInfo;
      for (CalleeSavedInfo &CS : CSI)
        RegToInfo.insert({CS.getReg(), &CS});

      // Before fillCSInfoPerBB: move restores off noreturn-tainted blocks so
      // those CSRs are not treated as "already assigned" then dropped.
      if (!MFI.getRestorePoints().empty())
        rehomeNoreturnRestorePoints(MFI, EpilogBlocks);

      if (!MFI.getSavePoints().empty()) {
        fillCSInfoPerBB(MFI, RegToInfo, PrologBlocks, /*IsSave=*/true);

        // SP-relative CSR spills must not run before addi sp (bc_modulo /
        // pr78622):
        //   sd s1, 0x18(sp); ...; addi sp, -N; sd ra...
        // writes into the caller's frame. Fallthrough-merge (save -> prolog) is
        // not enough when the save is earlier on the path. Hoist the frame
        // prolog to dominate those saves, and move CSR saves onto it.
        if (!PrologBlocks.empty()) {
          MachineFunction &MFRef = *PrologBlocks.front()->getParent();
          MachineDominatorTree DT(MFRef);
          MachineBasicBlock *CurProlog = PrologBlocks.front();
          MachineBasicBlock *NewProlog = CurProlog;
          for (const auto &KV : MFI.getSavePoints()) {
            MachineBasicBlock *SaveBB = KV.first;
            if (SaveBB == NewProlog || DT.dominates(NewProlog, SaveBB))
              continue;
            // Save is not after the current prolog. If SaveBB dominates the
            // prolog (linear prefix) or they share an NCD, hoist there.
            MachineBasicBlock *Cand =
                DT.findNearestCommonDominator(SaveBB, NewProlog);
            if (Cand)
              NewProlog = Cand;
          }
          if (NewProlog != CurProlog) {
            LLVM_DEBUG(dbgs() << "PEI: hoist prolog from "
                              << printMBBReference(*CurProlog) << " to "
                              << printMBBReference(*NewProlog)
                              << " to cover early CSR saves\n");
            PrologBlocks.clear();
            PrologBlocks.push_back(NewProlog);
            if (MFI.getProlog() == CurProlog)
              MFI.setProlog(NewProlog);

            // Move CSR saves that are not dominated by the new prolog onto it
            // (they would otherwise run before addi sp). Saves already after
            // NewProlog stay (legal delayed CSR).
            SmallVector<MachineBasicBlock *, 4> Erase;
            SaveRestorePoints NewSP = MFI.getSavePoints();
            for (const auto &KV : MFI.getSavePoints()) {
              MachineBasicBlock *BB = KV.first;
              if (BB == NewProlog || DT.dominates(NewProlog, BB))
                continue;
              Erase.push_back(BB);
            }
            for (MachineBasicBlock *BB : Erase) {
              auto It = NewSP.find(BB);
              if (It == NewSP.end())
                continue;
              auto &Dest = NewSP[NewProlog];
              for (const CalleeSavedInfo &CS : It->second)
                mergeCSIUnique(Dest, CS);
              NewSP.erase(It);
            }
            MFI.setSavePoints(std::move(NewSP));
          } else {
            // Same prolog BB: still merge immediate fallthrough save preds
            // (pr78622 diamond edge).
            SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock *>, 4>
                Moves;
            for (const auto &KV : MFI.getSavePoints()) {
              MachineBasicBlock *BB = KV.first;
              if (is_contained(PrologBlocks, BB))
                continue;
              if (BB->succ_size() != 1)
                continue;
              MachineBasicBlock *Succ = *BB->succ_begin();
              if (!is_contained(PrologBlocks, Succ))
                continue;
              Moves.emplace_back(BB, Succ);
            }
            if (!Moves.empty()) {
              SaveRestorePoints NewSP = MFI.getSavePoints();
              for (auto [From, To] : Moves) {
                auto It = NewSP.find(From);
                if (It == NewSP.end())
                  continue;
                auto &Dest = NewSP[To];
                for (const CalleeSavedInfo &CS : It->second)
                  mergeCSIUnique(Dest, CS);
                NewSP.erase(It);
              }
              MFI.setSavePoints(std::move(NewSP));
            }
          }
        }

        fillCSInfoPerBB(MFI, RegToInfo, EpilogBlocks, /*IsSave=*/false);
      } else {
        SaveRestorePoints SavePts;
        for (MachineBasicBlock *PrologBlock : PrologBlocks)
          SavePts.insert({PrologBlock, MFI.getCalleeSavedInfo()});
        MFI.setSavePoints(std::move(SavePts));
      }

      if (!CSI.empty()) {
        if (!MFI.hasCalls())
          NumLeafFuncWithSpills++;

        for (MachineBasicBlock &MBB : MF)
          insertCSRSaves(MBB, MFI.getSaveCSInfo(&MBB));

        for (CalleeSavedInfo &CS : CSI)
          updateLivenessSplit(MF, CS);

        if (MFI.getRestorePoints().empty()) {
          SaveRestorePoints RestorePts;
          for (MachineBasicBlock *EpilogBlock : EpilogBlocks)
            RestorePts.insert({EpilogBlock, MFI.getCalleeSavedInfo()});
          MFI.setRestorePoints(std::move(RestorePts));
        }

        for (MachineBasicBlock &MBB : MF)
          insertCSRRestores(MBB, MFI.getRestoreCSInfo(&MBB));
      }
    } else {
      // Legacy path: single save/restore point (or entry/returns).
      if (!MFI.getSavePoints().empty()) {
        SaveRestorePoints SaveRestorePts;
        for (const auto &SavePoint : MFI.getSavePoints())
          SaveRestorePts.insert({SavePoint.first, CSI});
        MFI.setSavePoints(std::move(SaveRestorePts));

        SaveRestorePts.clear();
        for (const auto &RestorePoint : MFI.getRestorePoints())
          SaveRestorePts.insert({RestorePoint.first, CSI});
        MFI.setRestorePoints(std::move(SaveRestorePts));
      }

      if (!CSI.empty()) {
        if (!MFI.hasCalls())
          NumLeafFuncWithSpills++;

        for (MachineBasicBlock *SaveBlock : SaveBlocks)
          insertCSRSaves(*SaveBlock, CSI);

        // Update the live-in information of all the blocks up to the save point.
        updateLivenessLegacy(MF);

        for (MachineBasicBlock *RestoreBlock : RestoreBlocks)
          insertCSRRestores(*RestoreBlock, CSI);
      }
    }
  }
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
static inline void AdjustStackOffset(MachineFrameInfo &MFI, int FrameIdx,
                                     bool StackGrowsDown, int64_t &Offset,
                                     Align &MaxAlign) {
  // If the stack grows down, add the object size to find the lowest address.
  if (StackGrowsDown)
    Offset += MFI.getObjectSize(FrameIdx);

  Align Alignment = MFI.getObjectAlign(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Alignment);

  // Adjust to alignment boundary.
  Offset = alignTo(Offset, Alignment);

  if (StackGrowsDown) {
    LLVM_DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") at SP[" << -Offset
                      << "]\n");
    MFI.setObjectOffset(FrameIdx, -Offset); // Set the computed offset
  } else {
    LLVM_DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") at SP[" << Offset
                      << "]\n");
    MFI.setObjectOffset(FrameIdx, Offset);
    Offset += MFI.getObjectSize(FrameIdx);
  }
}

/// Compute which bytes of fixed and callee-save stack area are unused and keep
/// track of them in StackBytesFree.
static inline void computeFreeStackSlots(MachineFrameInfo &MFI,
                                         bool StackGrowsDown,
                                         int64_t FixedCSEnd,
                                         BitVector &StackBytesFree) {
  // Avoid undefined int64_t -> int conversion below in extreme case.
  if (FixedCSEnd > std::numeric_limits<int>::max())
    return;

  StackBytesFree.resize(FixedCSEnd, true);

  SmallVector<int, 16> AllocatedFrameSlots;
  // Add fixed objects.
  for (int i = MFI.getObjectIndexBegin(); i != 0; ++i)
    // StackSlot scavenging is only implemented for the default stack.
    if (MFI.getStackID(i) == TargetStackID::Default)
      AllocatedFrameSlots.push_back(i);
  // Add callee-save objects if there are any.
  for (int i = MFI.getObjectIndexBegin(); i < MFI.getObjectIndexEnd(); i++)
    if (MFI.isCalleeSavedObjectIndex(i) &&
        MFI.getStackID(i) == TargetStackID::Default)
      AllocatedFrameSlots.push_back(i);

  for (int i : AllocatedFrameSlots) {
    // These are converted from int64_t, but they should always fit in int
    // because of the FixedCSEnd check above.
    int ObjOffset = MFI.getObjectOffset(i);
    int ObjSize = MFI.getObjectSize(i);
    int ObjStart, ObjEnd;
    if (StackGrowsDown) {
      // ObjOffset is negative when StackGrowsDown is true.
      ObjStart = -ObjOffset - ObjSize;
      ObjEnd = -ObjOffset;
    } else {
      ObjStart = ObjOffset;
      ObjEnd = ObjOffset + ObjSize;
    }
    // Ignore fixed holes that are in the previous stack frame.
    if (ObjEnd > 0)
      StackBytesFree.reset(ObjStart, ObjEnd);
  }
}

/// Assign frame object to an unused portion of the stack in the fixed stack
/// object range.  Return true if the allocation was successful.
static inline bool scavengeStackSlot(MachineFrameInfo &MFI, int FrameIdx,
                                     bool StackGrowsDown, Align MaxAlign,
                                     BitVector &StackBytesFree) {
  if (MFI.isVariableSizedObjectIndex(FrameIdx))
    return false;

  if (StackBytesFree.none()) {
    // clear it to speed up later scavengeStackSlot calls to
    // StackBytesFree.none()
    StackBytesFree.clear();
    return false;
  }

  Align ObjAlign = MFI.getObjectAlign(FrameIdx);
  if (ObjAlign > MaxAlign)
    return false;

  int64_t ObjSize = MFI.getObjectSize(FrameIdx);
  int FreeStart;
  for (FreeStart = StackBytesFree.find_first(); FreeStart != -1;
       FreeStart = StackBytesFree.find_next(FreeStart)) {

    // Check that free space has suitable alignment.
    unsigned ObjStart = StackGrowsDown ? FreeStart + ObjSize : FreeStart;
    if (alignTo(ObjStart, ObjAlign) != ObjStart)
      continue;

    if (FreeStart + ObjSize > StackBytesFree.size())
      return false;

    bool AllBytesFree = true;
    for (unsigned Byte = 0; Byte < ObjSize; ++Byte)
      if (!StackBytesFree.test(FreeStart + Byte)) {
        AllBytesFree = false;
        break;
      }
    if (AllBytesFree)
      break;
  }

  if (FreeStart == -1)
    return false;

  if (StackGrowsDown) {
    int ObjStart = -(FreeStart + ObjSize);
    LLVM_DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") scavenged at SP["
                      << ObjStart << "]\n");
    MFI.setObjectOffset(FrameIdx, ObjStart);
  } else {
    LLVM_DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") scavenged at SP["
                      << FreeStart << "]\n");
    MFI.setObjectOffset(FrameIdx, FreeStart);
  }

  StackBytesFree.reset(FreeStart, FreeStart + ObjSize);
  return true;
}

/// AssignProtectedObjSet - Helper function to assign large stack objects (i.e.,
/// those required to be close to the Stack Protector) to stack offsets.
static void AssignProtectedObjSet(const StackObjSet &UnassignedObjs,
                                  SmallSet<int, 16> &ProtectedObjs,
                                  MachineFrameInfo &MFI, bool StackGrowsDown,
                                  int64_t &Offset, Align &MaxAlign) {

  for (int i : UnassignedObjs) {
    AdjustStackOffset(MFI, i, StackGrowsDown, Offset, MaxAlign);
    ProtectedObjs.insert(i);
  }
}

/// After frame-index offsets are finalized, verify default-stack objects do not
/// overlap. Only used with data-flow CSR save/restore splitting.
static void verifyDefaultStackObjectLayout(const MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto objectInterval =
      [&MFI](unsigned FI) -> std::optional<std::pair<int64_t, int64_t>> {
    if (MFI.isDeadObjectIndex(FI))
      return std::nullopt;
    if (MFI.getStackID(FI) != TargetStackID::Default)
      return std::nullopt;
    if (MFI.isVariableSizedObjectIndex(FI))
      return std::nullopt;
    int64_t Start = MFI.getObjectOffset(FI);
    return std::pair{Start, Start + (int64_t)MFI.getObjectSize(FI)};
  };

  int64_t MinBound = 0;
  int64_t MaxBound = 0;
  bool HaveBounds = false;

  for (unsigned I = 0, E = MFI.getObjectIndexEnd(); I != E; ++I) {
    auto A = objectInterval(I);
    if (!A)
      continue;

    if (!HaveBounds) {
      MinBound = A->first;
      MaxBound = A->second;
      HaveBounds = true;
    } else {
      MinBound = std::min(MinBound, A->first);
      MaxBound = std::max(MaxBound, A->second);
    }

    for (unsigned J = I + 1; J != E; ++J) {
      auto B = objectInterval(J);
      if (!B)
        continue;
      if (A->first < B->second && B->first < A->second) {
        report_fatal_error(Twine(MF.getName()) + ": stack object FI#" +
                           Twine(I) +
                           (MFI.isCalleeSavedObjectIndex(I) ? " (CSR)" : "") +
                           " overlaps FI#" + Twine(J) +
                           (MFI.isCalleeSavedObjectIndex(J) ? " (CSR)" : ""));
      }
    }
  }

  if (HaveBounds && MFI.getStackSize() &&
      MaxBound - MinBound > (int64_t)MFI.getStackSize())
    report_fatal_error(Twine(MF.getName()) +
                       ": default stack objects span " +
                       Twine(MaxBound - MinBound) + " bytes but StackSize is " +
                       Twine(MFI.getStackSize()));
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
void PEIImpl::calculateFrameObjectOffsets(MachineFunction &MF) {
  const TargetFrameLowering &TFI = *MF.getSubtarget().getFrameLowering();

  bool StackGrowsDown =
    TFI.getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown;

  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Start at the beginning of the local area.
  // The Offset is the distance from the stack top in the direction
  // of stack growth -- so it's always nonnegative.
  int LocalAreaOffset = TFI.getOffsetOfLocalArea();
  if (StackGrowsDown)
    LocalAreaOffset = -LocalAreaOffset;
  assert(LocalAreaOffset >= 0
         && "Local area offset should be in direction of stack growth");
  int64_t Offset = LocalAreaOffset;

#ifdef EXPENSIVE_CHECKS
  for (unsigned i = 0, e = MFI.getObjectIndexEnd(); i != e; ++i)
    if (!MFI.isDeadObjectIndex(i) &&
        MFI.getStackID(i) == TargetStackID::Default)
      assert(MFI.getObjectAlign(i) <= MFI.getMaxAlign() &&
             "MaxAlignment is invalid");
#endif

  // If there are fixed sized objects that are preallocated in the local area,
  // non-fixed objects can't be allocated right at the start of local area.
  // Adjust 'Offset' to point to the end of last fixed sized preallocated
  // object.
  for (int i = MFI.getObjectIndexBegin(); i != 0; ++i) {
    // Only allocate objects on the default stack.
    if (MFI.getStackID(i) != TargetStackID::Default)
      continue;

    int64_t FixedOff;
    if (StackGrowsDown) {
      // The maximum distance from the stack pointer is at lower address of
      // the object -- which is given by offset. For down growing stack
      // the offset is negative, so we negate the offset to get the distance.
      FixedOff = -MFI.getObjectOffset(i);
    } else {
      // The maximum distance from the start pointer is at the upper
      // address of the object.
      FixedOff = MFI.getObjectOffset(i) + MFI.getObjectSize(i);
    }
    if (FixedOff > Offset) Offset = FixedOff;
  }

  Align MaxAlign = MFI.getMaxAlign();
  // First assign frame offsets to stack objects that are used to spill
  // callee saved registers.
  auto AllFIs = seq(MFI.getObjectIndexBegin(), MFI.getObjectIndexEnd());
  for (int FI : reverse_conditionally(AllFIs, /*Reverse=*/!StackGrowsDown)) {
    // Only allocate objects on the default stack.
    if (!MFI.isCalleeSavedObjectIndex(FI) ||
        MFI.getStackID(FI) != TargetStackID::Default)
      continue;

    // TODO: should this just be if (MFI.isDeadObjectIndex(FI))
    if (!StackGrowsDown && MFI.isDeadObjectIndex(FI))
      continue;

    AdjustStackOffset(MFI, FI, StackGrowsDown, Offset, MaxAlign);
  }

  assert(MaxAlign == MFI.getMaxAlign() &&
         "MFI.getMaxAlign should already account for all callee-saved "
         "registers without a fixed stack slot");

  // FixedCSEnd is the stack offset to the end of the fixed and callee-save
  // stack area.
  int64_t FixedCSEnd = Offset;

  // Make sure the special register scavenging spill slot is closest to the
  // incoming stack pointer if a frame pointer is required and is closer
  // to the incoming rather than the final stack pointer.
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  bool EarlyScavengingSlots = TFI.allocateScavengingFrameIndexesNearIncomingSP(MF);
  if (RS && EarlyScavengingSlots) {
    SmallVector<int, 2> SFIs;
    RS->getScavengingFrameIndices(SFIs);
    for (int SFI : SFIs)
      AdjustStackOffset(MFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  // FIXME: Once this is working, then enable flag will change to a target
  // check for whether the frame is large enough to want to use virtual
  // frame index registers. Functions which don't want/need this optimization
  // will continue to use the existing code path.
  if (MFI.getUseLocalStackAllocationBlock()) {
    Align Alignment = MFI.getLocalFrameMaxAlign();

    // Adjust to alignment boundary.
    Offset = alignTo(Offset, Alignment);

    LLVM_DEBUG(dbgs() << "Local frame base offset: " << Offset << "\n");

    // Resolve offsets for objects in the local block.
    for (unsigned i = 0, e = MFI.getLocalFrameObjectCount(); i != e; ++i) {
      std::pair<int, int64_t> Entry = MFI.getLocalFrameObjectMap(i);
      int64_t FIOffset = (StackGrowsDown ? -Offset : Offset) + Entry.second;
      LLVM_DEBUG(dbgs() << "alloc FI(" << Entry.first << ") at SP[" << FIOffset
                        << "]\n");
      MFI.setObjectOffset(Entry.first, FIOffset);
    }
    // Allocate the local block
    Offset += MFI.getLocalFrameSize();

    MaxAlign = std::max(Alignment, MaxAlign);
  }

  // Retrieve the Exception Handler registration node.
  int EHRegNodeFrameIndex = std::numeric_limits<int>::max();
  if (const WinEHFuncInfo *FuncInfo = MF.getWinEHFuncInfo())
    EHRegNodeFrameIndex = FuncInfo->EHRegNodeFrameIndex;

  // Make sure that the stack protector comes before the local variables on the
  // stack.
  SmallSet<int, 16> ProtectedObjs;
  if (MFI.hasStackProtectorIndex()) {
    int StackProtectorFI = MFI.getStackProtectorIndex();
    StackObjSet LargeArrayObjs;
    StackObjSet SmallArrayObjs;
    StackObjSet AddrOfObjs;

    // If we need a stack protector, we need to make sure that
    // LocalStackSlotPass didn't already allocate a slot for it.
    // If we are told to use the LocalStackAllocationBlock, the stack protector
    // is expected to be already pre-allocated.
    if (MFI.getStackID(StackProtectorFI) != TargetStackID::Default) {
      // If the stack protector isn't on the default stack then it's up to the
      // target to set the stack offset.
      assert(MFI.getObjectOffset(StackProtectorFI) != 0 &&
             "Offset of stack protector on non-default stack expected to be "
             "already set.");
      assert(!MFI.isObjectPreAllocated(MFI.getStackProtectorIndex()) &&
             "Stack protector on non-default stack expected to not be "
             "pre-allocated by LocalStackSlotPass.");
    } else if (!MFI.getUseLocalStackAllocationBlock()) {
      AdjustStackOffset(MFI, StackProtectorFI, StackGrowsDown, Offset,
                        MaxAlign);
    } else if (!MFI.isObjectPreAllocated(MFI.getStackProtectorIndex())) {
      llvm_unreachable(
          "Stack protector not pre-allocated by LocalStackSlotPass.");
    }

    // Assign large stack objects first.
    for (unsigned i = 0, e = MFI.getObjectIndexEnd(); i != e; ++i) {
      if (MFI.isObjectPreAllocated(i) && MFI.getUseLocalStackAllocationBlock())
        continue;
      if (MFI.isCalleeSavedObjectIndex(i))
        continue;
      if (RS && RS->isScavengingFrameIndex((int)i))
        continue;
      if (MFI.isDeadObjectIndex(i))
        continue;
      if (StackProtectorFI == (int)i || EHRegNodeFrameIndex == (int)i)
        continue;
      // Only allocate objects on the default stack.
      if (MFI.getStackID(i) != TargetStackID::Default)
        continue;

      switch (MFI.getObjectSSPLayout(i)) {
      case MachineFrameInfo::SSPLK_None:
        continue;
      case MachineFrameInfo::SSPLK_SmallArray:
        SmallArrayObjs.insert(i);
        continue;
      case MachineFrameInfo::SSPLK_AddrOf:
        AddrOfObjs.insert(i);
        continue;
      case MachineFrameInfo::SSPLK_LargeArray:
        LargeArrayObjs.insert(i);
        continue;
      }
      llvm_unreachable("Unexpected SSPLayoutKind.");
    }

    // We expect **all** the protected stack objects to be pre-allocated by
    // LocalStackSlotPass. If it turns out that PEI still has to allocate some
    // of them, we may end up messing up the expected order of the objects.
    if (MFI.getUseLocalStackAllocationBlock() &&
        !(LargeArrayObjs.empty() && SmallArrayObjs.empty() &&
          AddrOfObjs.empty()))
      llvm_unreachable("Found protected stack objects not pre-allocated by "
                       "LocalStackSlotPass.");

    AssignProtectedObjSet(LargeArrayObjs, ProtectedObjs, MFI, StackGrowsDown,
                          Offset, MaxAlign);
    AssignProtectedObjSet(SmallArrayObjs, ProtectedObjs, MFI, StackGrowsDown,
                          Offset, MaxAlign);
    AssignProtectedObjSet(AddrOfObjs, ProtectedObjs, MFI, StackGrowsDown,
                          Offset, MaxAlign);
  }

  SmallVector<int, 8> ObjectsToAllocate;

  // Then prepare to assign frame offsets to stack objects that are not used to
  // spill callee saved registers.
  for (unsigned i = 0, e = MFI.getObjectIndexEnd(); i != e; ++i) {
    if (MFI.isObjectPreAllocated(i) && MFI.getUseLocalStackAllocationBlock())
      continue;
    if (MFI.isCalleeSavedObjectIndex(i))
      continue;
    if (RS && RS->isScavengingFrameIndex((int)i))
      continue;
    if (MFI.isDeadObjectIndex(i))
      continue;
    if (MFI.getStackProtectorIndex() == (int)i || EHRegNodeFrameIndex == (int)i)
      continue;
    if (ProtectedObjs.count(i))
      continue;
    // Only allocate objects on the default stack.
    if (MFI.getStackID(i) != TargetStackID::Default)
      continue;

    // Add the objects that we need to allocate to our working set.
    ObjectsToAllocate.push_back(i);
  }

  // Allocate the EH registration node first if one is present.
  if (EHRegNodeFrameIndex != std::numeric_limits<int>::max())
    AdjustStackOffset(MFI, EHRegNodeFrameIndex, StackGrowsDown, Offset,
                      MaxAlign);

  // Give the targets a chance to order the objects the way they like it.
  if (MF.getTarget().getOptLevel() != CodeGenOptLevel::None &&
      MF.getTarget().Options.StackSymbolOrdering)
    TFI.orderFrameObjects(MF, ObjectsToAllocate);

  // Keep track of which bytes in the fixed and callee-save range are used so we
  // can use the holes when allocating later stack objects.  Only do this if
  // stack protector isn't being used and the target requests it and we're
  // optimizing.
  BitVector StackBytesFree;
  if (!ObjectsToAllocate.empty() &&
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None &&
      MFI.getStackProtectorIndex() < 0 && TFI.enableStackSlotScavenging(MF))
    computeFreeStackSlots(MFI, StackGrowsDown, FixedCSEnd, StackBytesFree);

  // Now walk the objects and actually assign base offsets to them.
  for (auto &Object : ObjectsToAllocate)
    if (!scavengeStackSlot(MFI, Object, StackGrowsDown, MaxAlign,
                           StackBytesFree))
      AdjustStackOffset(MFI, Object, StackGrowsDown, Offset, MaxAlign);

  // Make sure the special register scavenging spill slot is closest to the
  // stack pointer.
  if (RS && !EarlyScavengingSlots) {
    SmallVector<int, 2> SFIs;
    RS->getScavengingFrameIndices(SFIs);
    for (int SFI : SFIs)
      AdjustStackOffset(MFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  if (!TFI.targetHandlesStackFrameRounding()) {
    // If we have reserved argument space for call sites in the function
    // immediately on entry to the current function, count it as part of the
    // overall stack size.
    if (MFI.adjustsStack() && TFI.hasReservedCallFrame(MF))
      Offset += MFI.getMaxCallFrameSize();

    // Round up the size to a multiple of the alignment.  If the function has
    // any calls or alloca's, align to the target's StackAlignment value to
    // ensure that the callee's frame or the alloca data is suitably aligned;
    // otherwise, for leaf functions, align to the TransientStackAlignment
    // value.
    Align StackAlign;
    if (MFI.adjustsStack() || MFI.hasVarSizedObjects() ||
        (RegInfo->hasStackRealignment(MF) && MFI.getObjectIndexEnd() != 0))
      StackAlign = TFI.getStackAlign();
    else
      StackAlign = TFI.getTransientStackAlign();

    // If the frame pointer is eliminated, all frame offsets will be relative to
    // SP not FP. Align to MaxAlign so this works.
    StackAlign = std::max(StackAlign, MaxAlign);
    int64_t OffsetBeforeAlignment = Offset;
    Offset = alignTo(Offset, StackAlign);

    // If we have increased the offset to fulfill the alignment constrants,
    // then the scavenging spill slots may become harder to reach from the
    // stack pointer, float them so they stay close.
    if (StackGrowsDown && OffsetBeforeAlignment != Offset && RS &&
        !EarlyScavengingSlots) {
      SmallVector<int, 2> SFIs;
      RS->getScavengingFrameIndices(SFIs);
      LLVM_DEBUG(if (!SFIs.empty()) llvm::dbgs()
                     << "Adjusting emergency spill slots!\n";);
      int64_t Delta = Offset - OffsetBeforeAlignment;
      for (int SFI : SFIs) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Adjusting offset of emergency spill slot #" << SFI
                       << " from " << MFI.getObjectOffset(SFI););
        MFI.setObjectOffset(SFI, MFI.getObjectOffset(SFI) - Delta);
        LLVM_DEBUG(llvm::dbgs() << " to " << MFI.getObjectOffset(SFI) << "\n";);
      }
    }
  }

  // Update frame info to pretend that this is part of the stack...
  int64_t StackSize = Offset - LocalAreaOffset;
  MFI.setStackSize(StackSize);
  NumBytesStackSpace += StackSize;

  if (TFI.enableCSRSaveRestorePointsSplit())
    verifyDefaultStackObjectLayout(MF);
}

/// insertPrologEpilogCode - Scan the function for modified callee saved
/// registers, insert spill code for these callee saved registers, then add
/// prolog and epilog code to the function.
void PEIImpl::insertPrologEpilogCode(MachineFunction &MF) {
  const TargetFrameLowering &TFI = *MF.getSubtarget().getFrameLowering();

  // Add prologue to the function...
  for (MachineBasicBlock *PrologBlock : PrologBlocks)
    TFI.emitPrologue(MF, *PrologBlock);

  // Add epilogue to restore the callee-save registers in each exiting block.
  for (MachineBasicBlock *EpilogBlock : EpilogBlocks)
    TFI.emitEpilogue(MF, *EpilogBlock);

  // Zero call used registers before restoring callee-saved registers.
  insertZeroCallUsedRegs(MF);

  for (MachineBasicBlock *PrologBlock : PrologBlocks)
    TFI.inlineStackProbe(MF, *PrologBlock);

  // Emit additional code that is required to support segmented stacks, if
  // we've been asked for it.  This, when linked with a runtime with support
  // for segmented stacks (libgcc is one), will result in allocating stack
  // space in small chunks instead of one large contiguous block.
  if (MF.shouldSplitStack()) {
    for (MachineBasicBlock *PrologBlock : PrologBlocks)
      TFI.adjustForSegmentedStacks(MF, *PrologBlock);
  }

  // Emit additional code that is required to explicitly handle the stack in
  // HiPE native code (if needed) when loaded in the Erlang/OTP runtime. The
  // approach is rather similar to that of Segmented Stacks, but it uses a
  // different conditional check and another BIF for allocating more stack
  // space.
  if (MF.getFunction().getCallingConv() == CallingConv::HiPE)
    for (MachineBasicBlock *PrologBlock : PrologBlocks)
      TFI.adjustForHiPEPrologue(MF, *PrologBlock);
}

/// insertZeroCallUsedRegs - Zero out call used registers.
void PEIImpl::insertZeroCallUsedRegs(MachineFunction &MF) {
  const Function &F = MF.getFunction();

  if (!F.hasFnAttribute("zero-call-used-regs"))
    return;

  using namespace ZeroCallUsedRegs;

  ZeroCallUsedRegsKind ZeroRegsKind =
      StringSwitch<ZeroCallUsedRegsKind>(
          F.getFnAttribute("zero-call-used-regs").getValueAsString())
          .Case("skip", ZeroCallUsedRegsKind::Skip)
          .Case("used-gpr-arg", ZeroCallUsedRegsKind::UsedGPRArg)
          .Case("used-gpr", ZeroCallUsedRegsKind::UsedGPR)
          .Case("used-arg", ZeroCallUsedRegsKind::UsedArg)
          .Case("used", ZeroCallUsedRegsKind::Used)
          .Case("all-gpr-arg", ZeroCallUsedRegsKind::AllGPRArg)
          .Case("all-gpr", ZeroCallUsedRegsKind::AllGPR)
          .Case("all-arg", ZeroCallUsedRegsKind::AllArg)
          .Case("all", ZeroCallUsedRegsKind::All);

  if (ZeroRegsKind == ZeroCallUsedRegsKind::Skip)
    return;

  const bool OnlyGPR = static_cast<unsigned>(ZeroRegsKind) & ONLY_GPR;
  const bool OnlyUsed = static_cast<unsigned>(ZeroRegsKind) & ONLY_USED;
  const bool OnlyArg = static_cast<unsigned>(ZeroRegsKind) & ONLY_ARG;

  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const BitVector AllocatableSet(TRI.getAllocatableSet(MF));

  // Mark all used registers.
  BitVector UsedRegs(TRI.getNumRegs());
  if (OnlyUsed)
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB) {
        // skip debug instructions
        if (MI.isDebugInstr())
          continue;

        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg())
            continue;

          MCRegister Reg = MO.getReg();
          if (AllocatableSet[Reg.id()] && !MO.isImplicit() &&
              (MO.isDef() || MO.isUse()))
            UsedRegs.set(Reg.id());
        }
      }

  // Get a list of registers that are used.
  BitVector LiveIns(TRI.getNumRegs());
  for (const MachineBasicBlock::RegisterMaskPair &LI : MF.front().liveins())
    LiveIns.set(LI.PhysReg);

  BitVector RegsToZero(TRI.getNumRegs());
  for (MCRegister Reg : AllocatableSet.set_bits()) {
    // Skip over fixed registers.
    if (TRI.isFixedRegister(MF, Reg))
      continue;

    // Want only general purpose registers.
    if (OnlyGPR && !TRI.isGeneralPurposeRegister(MF, Reg))
      continue;

    // Want only used registers.
    if (OnlyUsed && !UsedRegs[Reg.id()])
      continue;

    // Want only registers used for arguments.
    if (OnlyArg) {
      if (OnlyUsed) {
        for (MCRegister LiveReg : LiveIns.set_bits()) {
          if (TRI.regsOverlap(Reg, LiveReg))
            RegsToZero.set(LiveReg);
        }
        continue;
      } else if (!TRI.isArgumentRegister(MF, Reg)) {
        continue;
      }
    }

    RegsToZero.set(Reg.id());
  }

  // Don't clear registers that are live when leaving the function.
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB.terminators()) {
      if (!MI.isReturn())
        continue;

      for (const auto &MO : MI.operands()) {
        if (!MO.isReg())
          continue;

        MCRegister Reg = MO.getReg();
        if (!Reg)
          continue;

        // This picks up sibling registers (e.q. %al -> %ah).
        // FIXME: Mixing physical registers and register units is likely a bug.
        for (MCRegUnit Unit : TRI.regunits(Reg))
          RegsToZero.reset(static_cast<unsigned>(Unit));

        for (MCPhysReg SReg : TRI.sub_and_superregs_inclusive(Reg))
          RegsToZero.reset(SReg);
      }
    }

  // Don't need to clear registers that are used/clobbered by terminating
  // instructions.
  for (const MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;

    MachineBasicBlock::const_iterator MBBI = MBB.getFirstTerminator();
    for (MachineBasicBlock::const_iterator I = MBBI, E = MBB.end(); I != E;
         ++I) {
      for (const MachineOperand &MO : I->operands()) {
        if (!MO.isReg())
          continue;

        MCRegister Reg = MO.getReg();
        if (!Reg)
          continue;

        for (const MCPhysReg Reg : TRI.sub_and_superregs_inclusive(Reg))
          RegsToZero.reset(Reg);
      }
    }
  }

  // Don't clear registers that must be preserved.
  for (const MCPhysReg *CSRegs = TRI.getCalleeSavedRegs(&MF);
       MCPhysReg CSReg = *CSRegs; ++CSRegs)
    for (MCRegister Reg : TRI.sub_and_superregs_inclusive(CSReg))
      RegsToZero.reset(Reg.id());

  const TargetFrameLowering &TFI = *MF.getSubtarget().getFrameLowering();
  for (MachineBasicBlock &MBB : MF)
    if (MBB.isReturnBlock())
      TFI.emitZeroCallUsedRegs(RegsToZero, MBB);
}

/// Replace all FrameIndex operands with physical register references and actual
/// offsets.
void PEIImpl::replaceFrameIndicesBackward(MachineFunction &MF) {
  const TargetFrameLowering &TFI = *MF.getSubtarget().getFrameLowering();

  for (auto &MBB : MF) {
    int SPAdj = 0;
    if (!MBB.succ_empty()) {
      // Get the SP adjustment for the end of MBB from the start of any of its
      // successors. They should all be the same.
      assert(all_of(MBB.successors(), [&MBB](const MachineBasicBlock *Succ) {
        return Succ->getCallFrameSize() ==
               (*MBB.succ_begin())->getCallFrameSize();
      }));
      const MachineBasicBlock &FirstSucc = **MBB.succ_begin();
      SPAdj = TFI.alignSPAdjust(FirstSucc.getCallFrameSize());
      if (TFI.getStackGrowthDirection() == TargetFrameLowering::StackGrowsUp)
        SPAdj = -SPAdj;
    }

    replaceFrameIndicesBackward(&MBB, MF, SPAdj);

    // We can't track the call frame size after call frame pseudos have been
    // eliminated. Set it to zero everywhere to keep MachineVerifier happy.
    MBB.setCallFrameSize(0);
  }
}

/// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
/// register references and actual offsets.
void PEIImpl::replaceFrameIndices(MachineFunction &MF) {
  const TargetFrameLowering &TFI = *MF.getSubtarget().getFrameLowering();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

#ifndef NDEBUG
  // When shrink-frame is active, verify no FI in blocks outside the frame
  // region.
  if (MachineBasicBlock *Prolog = MFI.getProlog()) {
    MachineDominatorTree DomTree(MF);
    for (auto &MBB : MF) {
      if (DomTree.dominates(Prolog, &MBB))
        continue;
      for (auto &MI : MBB) {
        if (MI.isDebugInstr())
          continue;
        for (const MachineOperand &MO : MI.operands())
          assert(!MO.isFI() &&
                 "Frame index in block not dominated by shrink-frame Prolog");
      }
    }
  }
#endif

  for (auto &MBB : MF) {
    int SPAdj = TFI.alignSPAdjust(MBB.getCallFrameSize());
    if (TFI.getStackGrowthDirection() == TargetFrameLowering::StackGrowsUp)
      SPAdj = -SPAdj;

    replaceFrameIndices(&MBB, MF, SPAdj);

    // We can't track the call frame size after call frame pseudos have been
    // eliminated. Set it to zero everywhere to keep MachineVerifier happy.
    MBB.setCallFrameSize(0);
  }
}

bool PEIImpl::replaceFrameIndexDebugInstr(MachineFunction &MF, MachineInstr &MI,
                                          unsigned OpIdx, int SPAdj) {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  if (MI.isDebugValue()) {

    MachineOperand &Op = MI.getOperand(OpIdx);
    assert(MI.isDebugOperand(&Op) &&
           "Frame indices can only appear as a debug operand in a DBG_VALUE*"
           " machine instruction");
    Register Reg;
    unsigned FrameIdx = Op.getIndex();
    unsigned Size = MF.getFrameInfo().getObjectSize(FrameIdx);

    StackOffset Offset = TFI->getFrameIndexReference(MF, FrameIdx, Reg);
    Op.ChangeToRegister(Reg, false /*isDef*/);

    const DIExpression *DIExpr = MI.getDebugExpression();

    // If we have a direct DBG_VALUE, and its location expression isn't
    // currently complex, then adding an offset will morph it into a
    // complex location that is interpreted as being a memory address.
    // This changes a pointer-valued variable to dereference that pointer,
    // which is incorrect. Fix by adding DW_OP_stack_value.

    if (MI.isNonListDebugValue()) {
      unsigned PrependFlags = DIExpression::ApplyOffset;
      if (!MI.isIndirectDebugValue() && !DIExpr->isComplex())
        PrependFlags |= DIExpression::StackValue;

      // If we have DBG_VALUE that is indirect and has a Implicit location
      // expression need to insert a deref before prepending a Memory
      // location expression. Also after doing this we change the DBG_VALUE
      // to be direct.
      if (MI.isIndirectDebugValue() && DIExpr->isImplicit()) {
        SmallVector<uint64_t, 2> Ops = {dwarf::DW_OP_deref_size, Size};
        bool WithStackValue = true;
        DIExpr = DIExpression::prependOpcodes(DIExpr, Ops, WithStackValue);
        // Make the DBG_VALUE direct.
        MI.getDebugOffset().ChangeToRegister(0, false);
      }
      DIExpr = TRI.prependOffsetExpression(DIExpr, PrependFlags, Offset);
    } else {
      // The debug operand at DebugOpIndex was a frame index at offset
      // `Offset`; now the operand has been replaced with the frame
      // register, we must add Offset with `register x, plus Offset`.
      unsigned DebugOpIndex = MI.getDebugOperandIndex(&Op);
      SmallVector<uint64_t, 3> Ops;
      TRI.getOffsetOpcodes(Offset, Ops);
      DIExpr = DIExpression::appendOpsToArg(DIExpr, Ops, DebugOpIndex);
    }
    MI.getDebugExpressionOp().setMetadata(DIExpr);
    return true;
  }

  if (MI.isDebugPHI()) {
    // Allow stack ref to continue onwards.
    return true;
  }

  // TODO: This code should be commoned with the code for
  // PATCHPOINT. There's no good reason for the difference in
  // implementation other than historical accident.  The only
  // remaining difference is the unconditional use of the stack
  // pointer as the base register.
  if (MI.getOpcode() == TargetOpcode::STATEPOINT) {
    assert((!MI.isDebugValue() || OpIdx == 0) &&
           "Frame indices can only appear as the first operand of a "
           "DBG_VALUE machine instruction");
    Register Reg;
    MachineOperand &Offset = MI.getOperand(OpIdx + 1);
    StackOffset refOffset = TFI->getFrameIndexReferencePreferSP(
        MF, MI.getOperand(OpIdx).getIndex(), Reg, /*IgnoreSPUpdates*/ false);
    assert(!refOffset.getScalable() &&
           "Frame offsets with a scalable component are not supported");
    Offset.setImm(Offset.getImm() + refOffset.getFixed() + SPAdj);
    MI.getOperand(OpIdx).ChangeToRegister(Reg, false /*isDef*/);
    return true;
  }
  return false;
}

void PEIImpl::replaceFrameIndicesBackward(MachineBasicBlock *BB,
                                          MachineFunction &MF, int &SPAdj) {
  assert(MF.getSubtarget().getRegisterInfo() &&
         "getRegisterInfo() must be implemented!");

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const TargetFrameLowering &TFI = *MF.getSubtarget().getFrameLowering();

  RegScavenger *LocalRS = FrameIndexEliminationScavenging ? RS : nullptr;
  if (LocalRS)
    LocalRS->enterBasicBlockEnd(*BB);

  for (MachineBasicBlock::iterator I = BB->end(); I != BB->begin();) {
    MachineInstr &MI = *std::prev(I);

    if (TII.isFrameInstr(MI)) {
      SPAdj -= TII.getSPAdjust(MI);
      TFI.eliminateCallFramePseudoInstr(MF, *BB, &MI);
      continue;
    }

    // Step backwards to get the liveness state at (immedately after) MI.
    if (LocalRS)
      LocalRS->backward(I);

    bool RemovedMI = false;
    for (const auto &[Idx, Op] : enumerate(MI.operands())) {
      if (!Op.isFI())
        continue;

      if (replaceFrameIndexDebugInstr(MF, MI, Idx, SPAdj))
        continue;

      int FrameIndex = MI.getOperand(Idx).getIndex();
      int SPA = SPAdj;
      // Only under multi-point CSR shrink-wrapping do CSR spills outside the
      // prolog/epilog need an extra SP adjustment (split SP allocation).
      if (TFI.enableCSRSaveRestorePointsSplit() &&
          TRI.isCSIFrameIndex(&MF, FrameIndex) &&
          !(is_contained(PrologBlocks, BB) || is_contained(EpilogBlocks, BB))) {
        int CSIOff = TRI.getCSIFrameOffset(&MF);
        if (CSIOff != 0)
          SPA = CSIOff;
      }

      // Eliminate this FrameIndex operand.
      RemovedMI = TRI.eliminateFrameIndex(MI, SPA, Idx, LocalRS);
      if (RemovedMI)
        break;
    }

    if (!RemovedMI)
      --I;
  }
}

void PEIImpl::replaceFrameIndices(MachineBasicBlock *BB, MachineFunction &MF,
                                  int &SPAdj) {
  assert(MF.getSubtarget().getRegisterInfo() &&
         "getRegisterInfo() must be implemented!");
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();

  bool InsideCallSequence = false;

  for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
    if (TII.isFrameInstr(*I)) {
      InsideCallSequence = TII.isFrameSetup(*I);
      SPAdj += TII.getSPAdjust(*I);
      I = TFI->eliminateCallFramePseudoInstr(MF, *BB, I);
      continue;
    }

    MachineInstr &MI = *I;
    bool DoIncr = true;
    bool DidFinishLoop = true;
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      if (!MI.getOperand(i).isFI())
        continue;

      if (replaceFrameIndexDebugInstr(MF, MI, i, SPAdj))
        continue;

      // Some instructions (e.g. inline asm instructions) can have
      // multiple frame indices and/or cause eliminateFrameIndex
      // to insert more than one instruction. We need the register
      // scavenger to go through all of these instructions so that
      // it can update its register information. We keep the
      // iterator at the point before insertion so that we can
      // revisit them in full.
      bool AtBeginning = (I == BB->begin());
      if (!AtBeginning) --I;

      // If this instruction has a FrameIndex operand, we need to
      // use that target machine register info object to eliminate
      // it.
      TRI.eliminateFrameIndex(MI, SPAdj, i, RS);

      // Reset the iterator if we were at the beginning of the BB.
      if (AtBeginning) {
        I = BB->begin();
        DoIncr = false;
      }

      DidFinishLoop = false;
      break;
    }

    // If we are looking at a call sequence, we need to keep track of
    // the SP adjustment made by each instruction in the sequence.
    // This includes both the frame setup/destroy pseudos (handled above),
    // as well as other instructions that have side effects w.r.t the SP.
    // Note that this must come after eliminateFrameIndex, because
    // if I itself referred to a frame index, we shouldn't count its own
    // adjustment.
    if (DidFinishLoop && InsideCallSequence)
      SPAdj += TII.getSPAdjust(MI);

    if (DoIncr && I != BB->end())
      ++I;
  }
}
