//===- ShrinkFrameUtils.cpp - Shrink-frame region helpers -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ShrinkFrameUtils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

bool llvm::hasShrinkFrame(const MachineFunction &MF) {
  return MF.getFrameInfo().getProlog() != nullptr;
}

MachineBasicBlock *llvm::getShrinkFrameProlog(const MachineFunction &MF) {
  return MF.getFrameInfo().getProlog();
}

bool llvm::isInFrameRegion(const MachineBasicBlock &BB,
                           const MachineDominatorTree &DT) {
  const MachineFunction *MF = BB.getParent();
  MachineBasicBlock *Prolog = getShrinkFrameProlog(*MF);
  if (!Prolog)
    return true;
  return DT.dominates(Prolog, &BB);
}

bool llvm::sameFrameRegion(const MachineBasicBlock *A,
                           const MachineBasicBlock *B,
                           const MachineDominatorTree &DT) {
  if (!A || !B)
    return false;
  return isInFrameRegion(*A, DT) == isInFrameRegion(*B, DT);
}

bool llvm::crossesFrameRegionBoundary(
    ArrayRef<const MachineBasicBlock *> Blocks,
    const MachineDominatorTree &DT) {
  if (Blocks.empty())
    return false;
  const MachineFunction *MF = Blocks.front()->getParent();
  if (!hasShrinkFrame(*MF))
    return false;
  bool First = isInFrameRegion(*Blocks.front(), DT);
  for (const MachineBasicBlock *BB : Blocks)
    if (isInFrameRegion(*BB, DT) != First)
      return true;
  return false;
}

bool llvm::crossesFrameRegionEdge(const MachineBasicBlock &Pred,
                                  const MachineBasicBlock &Succ,
                                  const MachineDominatorTree &DT) {
  const MachineFunction &MF = *Pred.getParent();
  if (!hasShrinkFrame(MF))
    return false;
  const bool PredF = isInFrameRegion(Pred, DT);
  const bool SuccF = isInFrameRegion(Succ, DT);
  if (PredF == SuccF)
    return false;
  if (!PredF && SuccF)
    return &Succ != getShrinkFrameProlog(MF);
  return true;
}

bool llvm::isShrinkFrameEpilogBlock(const MachineBasicBlock &BB,
                                    const MachineFunction &MF,
                                    const MachineDominatorTree &DT) {
  if (!hasShrinkFrame(MF))
    return false;
  return BB.isReturnBlock() && isInFrameRegion(BB, DT);
}

bool llvm::blockHasEpiloguePattern(const MachineBasicBlock &BB,
                                   const MachineFunction &MF) {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  for (const MachineInstr &MI : BB)
    if (TFI->isShrinkFrameEpiloguePattern(MI, MF))
      return true;
  return false;
}

bool llvm::isFrameRelatedForGuard(const MachineInstr &MI,
                                  const MachineFunction &MF) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isFI())
      return true;
  return MF.getSubtarget().getFrameLowering()->isShrinkFrameFrameRelatedMI(MI,
                                                                          MF);
}

bool llvm::shrinkFrameGuardRejectEdge(const MachineFunction &MF,
                                      const MachineBasicBlock &Pred,
                                      const MachineBasicBlock &Succ,
                                      const MachineDominatorTree &DT) {
  return hasShrinkFrame(MF) && crossesFrameRegionEdge(Pred, Succ, DT);
}

bool llvm::shrinkFrameGuardRejectBlocks(
    const MachineFunction &MF, ArrayRef<const MachineBasicBlock *> Blocks,
    const MachineDominatorTree &DT) {
  return hasShrinkFrame(MF) && crossesFrameRegionBoundary(Blocks, DT);
}

bool llvm::shrinkFrameGuardRejectTailDuplicate(
    const MachineFunction &MF, const MachineBasicBlock &Src,
    const MachineBasicBlock &Pred, const MachineDominatorTree &DT) {
  if (!hasShrinkFrame(MF))
    return false;
  if (&Src == getShrinkFrameProlog(MF))
    return true;
  if (!sameFrameRegion(&Src, &Pred, DT))
    return true;
  if (blockHasEpiloguePattern(Src, MF) && !isInFrameRegion(Pred, DT))
    return true;
  return false;
}

bool llvm::shrinkFrameGuardRejectTailMerge(const MachineFunction &MF,
                                           const MachineBasicBlock &M1,
                                           const MachineBasicBlock &M2,
                                           const MachineBasicBlock &Dest,
                                           const MachineDominatorTree &DT) {
  if (!hasShrinkFrame(MF))
    return false;
  const MachineBasicBlock *Blocks[] = {&M1, &M2, &Dest};
  if (crossesFrameRegionBoundary(Blocks, DT))
    return true;
  const bool Epilogue =
      blockHasEpiloguePattern(M1, MF) || blockHasEpiloguePattern(M2, MF) ||
      blockHasEpiloguePattern(Dest, MF);
  if (Epilogue && !isInFrameRegion(Dest, DT))
    return true;
  if (Epilogue && (!isInFrameRegion(M1, DT) || !isInFrameRegion(M2, DT)))
    return true;
  return false;
}

bool llvm::shrinkFrameGuardRejectHoist(const MachineFunction &MF,
                                       const MachineBasicBlock &Dest,
                                       bool HoistingFrameRelated,
                                       const MachineDominatorTree &DT) {
  if (!hasShrinkFrame(MF))
    return false;
  if (&Dest == getShrinkFrameProlog(MF))
    return true;
  if (HoistingFrameRelated && isShrinkFrameEpilogBlock(Dest, MF, DT))
    return true;
  return false;
}
