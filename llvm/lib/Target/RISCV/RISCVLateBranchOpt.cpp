//===-- RISCVLateBranchOpt.cpp - Late Stage Branch Optimization -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides RISC-V specific target optimizations, currently it's
/// limited to convert conditional branches into unconditional branches when
/// the condition can be statically evaluated.
///
//===----------------------------------------------------------------------===//

#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/ShrinkFrameUtils.h"
#include <optional>

using namespace llvm;

#define RISCV_LATE_BRANCH_OPT_NAME "RISC-V Late Branch Optimisation Pass"

namespace {

struct RISCVLateBranchOpt : public MachineFunctionPass {
  static char ID;

  RISCVLateBranchOpt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return RISCV_LATE_BRANCH_OPT_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  bool runOnBasicBlock(MachineBasicBlock &MBB) const;

  const RISCVInstrInfo *RII = nullptr;
  /// Non-null when shrink-frame is active; used to reject folds that would
  /// leave a cross-region CFG edge.
  const MachineDominatorTree *SFDomTree = nullptr;
};
} // namespace

char RISCVLateBranchOpt::ID = 0;
INITIALIZE_PASS(RISCVLateBranchOpt, "riscv-late-branch-opt",
                RISCV_LATE_BRANCH_OPT_NAME, false, false)

bool RISCVLateBranchOpt::runOnBasicBlock(MachineBasicBlock &MBB) const {
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;
  if (RII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return false;

  if (!TBB || Cond.size() != 3)
    return false;

  RISCVCC::CondCode CC = RISCVInstrInfo::getCondFromBranchOpc(Cond[0].getImm());
  assert(CC != RISCVCC::COND_INVALID);

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // Try and convert a conditional branch that can be evaluated statically
  // into an unconditional branch.
  int64_t C0, C1;
  if (!RISCVInstrInfo::isFromLoadImm(MRI, Cond[1], C0) ||
      !RISCVInstrInfo::isFromLoadImm(MRI, Cond[2], C1))
    return false;

  MachineBasicBlock *Folded =
      RISCVInstrInfo::evaluateCondBranch(CC, C0, C1) ? TBB : FBB;

  // Resolve the CFG successor this fold would leave. Prefer an existing CFG
  // successor over layout order: after block placement, ++MBB may not be the
  // fallthrough edge analyzeBranch described (FBB == nullptr).
  MachineBasicBlock *NewSucc = Folded;
  if (!NewSucc) {
    for (MachineBasicBlock *Succ : MBB.successors()) {
      if (Succ != TBB) {
        NewSucc = Succ;
        break;
      }
    }
    if (!NewSucc)
      return false;
  }

  // Shrink-frame: do not commit a fold that leaves a cross-region edge.
  if (SFDomTree &&
      shrinkFrameGuardRejectEdge(*MBB.getParent(), MBB, *NewSucc, *SFDomTree))
    return false;

  // At this point, its legal to optimize.
  RII->removeBranch(MBB);

  // Only need to insert a branch if the taken path is not layout fallthrough.
  // When Folded is null, the false path already falls through in layout/CFG.
  if (Folded) {
    DebugLoc DL = MBB.findBranchDebugLoc();
    RII->insertBranch(MBB, Folded, nullptr, {}, DL);
  }

  // Update the successors. Remove them all and add back the correct one.
  while (!MBB.succ_empty())
    MBB.removeSuccessor(MBB.succ_end() - 1);

  MBB.addSuccessor(NewSucc);

  return true;
}

bool RISCVLateBranchOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  auto &ST = Fn.getSubtarget<RISCVSubtarget>();
  RII = ST.getInstrInfo();

  std::optional<MachineDominatorTree> SFDomTreeStorage;
  if (hasShrinkFrame(Fn))
    SFDomTreeStorage.emplace(Fn);
  SFDomTree = SFDomTreeStorage ? &*SFDomTreeStorage : nullptr;

  bool Changed = false;
  for (MachineBasicBlock &MBB : Fn)
    Changed |= runOnBasicBlock(MBB);

  SFDomTree = nullptr;
  return Changed;
}

FunctionPass *llvm::createRISCVLateBranchOptPass() {
  return new RISCVLateBranchOpt();
}
