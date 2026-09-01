//===- ShrinkFrameUtils.cpp - Shrink-frame region helpers -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ShrinkFrameUtils.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "shrink-frame"

using namespace llvm;

bool llvm::hasShrinkFrame(const MachineFunction &MF) {
  return MF.getFrameInfo().getProlog() != nullptr;
}

MachineBasicBlock *llvm::getShrinkFrameProlog(const MachineFunction &MF) {
  return MF.getFrameInfo().getProlog();
}

bool llvm::isShrinkFrameClone(const MachineBasicBlock &BB) {
  const MachineFunction *MF = BB.getParent();
  return MF && MF->getFrameInfo().isShrinkFrameClone(&BB);
}

bool llvm::areShrinkFrameClonePair(const MachineBasicBlock &A,
                                   const MachineBasicBlock &B) {
  if (A.getParent() != B.getParent())
    return false;
  MachineBasicBlock *Partner =
      A.getParent()->getFrameInfo().getShrinkFrameClonePartner(&A);
  return Partner == &B;
}

void llvm::undoShrinkFrameClones(MachineFunction &MF) {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.hasShrinkFrameClones())
    return;

  LLVM_DEBUG(dbgs() << "Shrink-frame: undo shared-return clones (keep Prolog/CSR "
                    << "maps) in " << MF.getName() << '\n');

  SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock *>, 4> Pairs;
  for (const auto &KV : MFI.getShrinkFrameClones())
    Pairs.emplace_back(KV.first, KV.second);

  // Drop Save/Restore entries keyed on clones before the MBBs are erased.
  // Keep maps on other blocks (multi-point CSR early exits without dup).
  if (!MFI.getSavePoints().empty() || !MFI.getRestorePoints().empty()) {
    SaveRestorePoints NewSP = MFI.getSavePoints();
    SaveRestorePoints NewRP = MFI.getRestorePoints();
    for (auto [Orig, Clone] : Pairs) {
      (void)Orig;
      NewSP.erase(Clone);
      NewRP.erase(Clone);
    }
    MFI.setSavePoints(std::move(NewSP));
    MFI.setRestorePoints(std::move(NewRP));
  }

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (auto [Orig, Clone] : Pairs) {
    SmallVector<MachineBasicBlock *, 4> Preds(Clone->pred_begin(),
                                               Clone->pred_end());
    for (MachineBasicBlock *Pred : Preds) {
      MachineBasicBlock *PrevFT = Pred->getNextNode();
      const bool WasCloneFallthrough = PrevFT == Clone;
      Pred->ReplaceUsesOfBlockWith(Clone, Orig);
      MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
      SmallVector<MachineOperand, 4> Cond;
      if (!TII->analyzeBranch(*Pred, TBB, FBB, Cond)) {
        Pred->updateTerminator(WasCloneFallthrough ? Orig : PrevFT);
      } else if (!Pred->isLayoutSuccessor(Orig)) {
        // Unanalyzable branch: ensure CFG edge is realized with an explicit
        // jump when Orig is not the layout fallthrough.
        Cond.clear();
        TII->removeBranch(*Pred);
        TII->insertBranch(*Pred, Orig, nullptr, Cond, DebugLoc());
      }
    }
    Clone->eraseFromParent();
  }

  MFI.clearShrinkFrameClones();

  if (MF.getRegInfo().tracksLiveness()) {
    SmallVector<MachineBasicBlock *, 8> ToRecompute;
    for (auto [Orig, Clone] : Pairs)
      ToRecompute.push_back(Orig);
    fullyRecomputeLiveIns(ToRecompute);
  }
}

void llvm::abandonShrinkFrame(MachineFunction &MF) {
  undoShrinkFrameClones(MF);
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setProlog(nullptr);
  MFI.setEpilog(nullptr);
  MFI.clearSavePoints();
  MFI.clearRestorePoints();
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
  // Do not TailDup a shrink-frame original/clone body (would erase the pair
  // distinction).
  // Also reject duplicating one side of a pair into the other.
  if (MF.getFrameInfo().getShrinkFrameClonePartner(&Src) ||
      areShrinkFrameClonePair(Src, Pred))
    return true;
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
  // Never TailMerge a registered original or frameless clone with anything
  // (including a split common-tail that is not the registered partner).
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (MFI.getShrinkFrameClonePartner(&M1) ||
      MFI.getShrinkFrameClonePartner(&M2) ||
      MFI.getShrinkFrameClonePartner(&Dest))
    return true;
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

static bool isReachableMBB(const MachineBasicBlock *From,
                           const MachineBasicBlock *To) {
  for (auto I = df_begin(From), E = df_end(From); I != E; ++I)
    if (*I == To)
      return true;
  return false;
}

static bool blockMayTouchStackFrame(const MachineBasicBlock &MBB,
                                    const TargetRegisterInfo *TRI,
                                    Register SP) {
  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isFI())
        return true;
    }
    if (SP && MI.modifiesRegister(SP, TRI))
      return true;
    if (MI.isCall())
      return true;
  }
  return false;
}

bool llvm::mustHoistShrinkFramePrologToEntry(const MachineFunction &MF,
                                             MachineBasicBlock *Prolog,
                                             const MachineDominatorTree &DT) {
  if (!Prolog || Prolog == &MF.front())
    return false;

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  Register SP = MF.getSubtarget()
                      .getTargetLowering()
                      ->getStackPointerRegisterToSaveRestore();
  MachineBasicBlock *Entry = const_cast<MachineBasicBlock *>(&MF.front());

  SmallVector<Register, 4> FrameBound;
  TFI->getFrameBoundCalleeSaves(MF, FrameBound);
  for (const MachineBasicBlock &MBB : MF) {
    if (DT.dominates(Prolog, &MBB))
      continue;
    for (const MachineInstr &MI : MBB) {
      for (Register R : FrameBound) {
        if (R && MI.modifiesRegister(R, TRI))
          return true;
      }
    }
  }

  for (const auto &[SaveBB, _] : MFI.getSavePoints()) {
    if (SaveBB != Prolog && !DT.dominates(Prolog, SaveBB))
      return true;
  }

  for (const MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;
    bool HasFramedPred = false;
    bool HasFramelessPred = false;
    for (const MachineBasicBlock *Pred : MBB.predecessors()) {
      if (DT.dominates(Prolog, Pred))
        HasFramedPred = true;
      else
        HasFramelessPred = true;
    }
    if (HasFramedPred && HasFramelessPred)
      return true;
  }

  for (const MachineBasicBlock &MBB : MF) {
    if (DT.dominates(Prolog, &MBB))
      continue;
    if (!isReachableMBB(Entry, &MBB))
      continue;
    if (blockMayTouchStackFrame(MBB, TRI, SP))
      return true;
  }

  return false;
}

bool llvm::isLegalOffPrologCSRSaveForCloneRestore(
    const MachineFunction &MF, MachineBasicBlock *Prolog,
    MachineBasicBlock *SaveBB, MCRegister Reg,
    const MachineDominatorTree &DT) {
  if (!Prolog || DT.dominates(Prolog, SaveBB))
    return true;

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.hasShrinkFrameClones() || isReachableMBB(Prolog, SaveBB))
    return false;

  SmallVector<Register, 4> FrameBound;
  MF.getSubtarget().getFrameLowering()->getFrameBoundCalleeSaves(MF, FrameBound);
  for (Register R : FrameBound) {
    if (R && R.id() == Reg)
      return false;
  }

  for (const auto &KV : MFI.getShrinkFrameClones()) {
    MachineBasicBlock *Clone = KV.second;
    if (!Clone->isReturnBlock())
      continue;
    for (MachineBasicBlock *Pred : Clone->predecessors()) {
      if (DT.dominates(Prolog, Pred))
        continue;
      if (SaveBB == Pred || DT.dominates(SaveBB, Pred))
        return true;
    }
  }
  return false;
}

bool llvm::isLegalOffPrologCSRSavePoint(
    const MachineFunction &MF, MachineBasicBlock *Prolog,
    MachineBasicBlock *SaveBB, ArrayRef<MCRegister> Regs,
    const MachineDominatorTree &DT) {
  if (!Prolog || DT.dominates(Prolog, SaveBB))
    return true;
  for (MCRegister Reg : Regs) {
    if (!isLegalOffPrologCSRSaveForCloneRestore(MF, Prolog, SaveBB, Reg, DT))
      return false;
  }
  return true;
}
