//===-- ShrinkWrapCFIFixup.cpp - remember/restore for multi-point CFI -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After PEI + block placement, FDE state follows *layout* order. A return
// epilogue's .cfi_restore / def_cfa_offset 0 therefore poisons later blocks
// (e.g. throw paths) that still have a live frame on every CFG path.
//
// GCC dwarf2cfi inserts DW_CFA_remember_state / restore_state when connecting
// such traces. This pass does the same for enableCSRSaveRestorePointsSplit():
//
//   * CFG: after prologue and before any return epilogue => "has frame"
//   * Layout: previous block's FDE state may say "no frame"
//   * Fix: remember after real prologue; restore_state at the block that
//     needs a frame again
//
// Prologue end is the last FrameSetup CFI in the *entry* block only — not a
// reverse scan of the whole function (delayed CSR .cfi_offset is FrameSetup).
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ShrinkWrapCFIFixup.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "shrink-wrap-cfi-fixup"

char ShrinkWrapCFIFixup::ID = 0;

INITIALIZE_PASS(ShrinkWrapCFIFixup, DEBUG_TYPE,
                "Insert CFI remember/restore for multi-point shrink-wrapping",
                false, false)

ShrinkWrapCFIFixup::ShrinkWrapCFIFixup() : MachineFunctionPass(ID) {
  initializeShrinkWrapCFIFixupPass(*PassRegistry::getPassRegistry());
}

FunctionPass *llvm::createShrinkWrapCFIFixup() {
  return new ShrinkWrapCFIFixup();
}

static bool isPrologueCFIInstruction(const MachineInstr &MI) {
  return MI.getOpcode() == TargetOpcode::CFI_INSTRUCTION &&
         MI.getFlag(MachineInstr::FrameSetup);
}

/// Prologue end = last FrameSetup CFI in the entry block (GCC-like full frame
/// at entry under data-flow shrink-wrapping).
static MachineBasicBlock *
findEntryPrologueEnd(MachineFunction &MF,
                     MachineBasicBlock::iterator &PrologueEnd) {
  MachineBasicBlock &Entry = MF.front();
  for (MachineInstr &MI : reverse(Entry.instrs())) {
    if (!isPrologueCFIInstruction(MI))
      continue;
    PrologueEnd = std::next(MI.getIterator());
    return &Entry;
  }
  return nullptr;
}

struct BlockFlags {
  bool Reachable : 1;
  bool StrongNoFrameOnEntry : 1;
  bool HasFrameOnEntry : 1;
  bool HasFrameOnExit : 1;
  BlockFlags()
      : Reachable(false), StrongNoFrameOnEntry(false), HasFrameOnEntry(false),
        HasFrameOnExit(false) {}
};

using BlockFlagsVector = SmallVector<BlockFlags, 32>;

static BlockFlagsVector
computeBlockInfo(const MachineFunction &MF,
                 const MachineBasicBlock *PrologueBlock) {
  BlockFlagsVector BlockInfo(MF.getNumBlockIDs());
  BlockInfo[0].Reachable = true;
  BlockInfo[0].StrongNoFrameOnEntry = true;

  ReversePostOrderTraversal<const MachineBasicBlock *> RPOT(&*MF.begin());
  for (const MachineBasicBlock *MBB : RPOT) {
    BlockFlags &Info = BlockInfo[MBB->getNumber()];

    bool HasPrologue = MBB == PrologueBlock;
    // With GCC-like policy, the full frame is torn down only on returns.
    bool HasEpilogue = false;
    if (Info.HasFrameOnEntry || HasPrologue)
      HasEpilogue = MBB->isReturnBlock();

    Info.HasFrameOnExit = (Info.HasFrameOnEntry || HasPrologue) && !HasEpilogue;

    for (MachineBasicBlock *Succ : MBB->successors()) {
      BlockFlags &SuccInfo = BlockInfo[Succ->getNumber()];
      SuccInfo.Reachable = true;
      SuccInfo.StrongNoFrameOnEntry |=
          Info.StrongNoFrameOnEntry && !HasPrologue;
      SuccInfo.HasFrameOnEntry = Info.HasFrameOnExit;
    }
  }

  return BlockInfo;
}

struct InsertionPoint {
  MachineBasicBlock *MBB = nullptr;
  MachineBasicBlock::iterator Iterator;
};

static InsertionPoint
insertRememberRestorePair(const InsertionPoint &RememberInsertPt,
                          const InsertionPoint &RestoreInsertPt) {
  MachineFunction &MF = *RememberInsertPt.MBB->getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  unsigned CFIIndex =
      MF.addFrameInst(MCCFIInstruction::createRememberState(nullptr));
  BuildMI(*RememberInsertPt.MBB, RememberInsertPt.Iterator, DebugLoc(),
          TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);

  CFIIndex = MF.addFrameInst(MCCFIInstruction::createRestoreState(nullptr));
  return {RestoreInsertPt.MBB,
          std::next(BuildMI(*RestoreInsertPt.MBB, RestoreInsertPt.Iterator,
                            DebugLoc(), TII.get(TargetOpcode::CFI_INSTRUCTION))
                        .addCFIIndex(CFIIndex)
                        ->getIterator())};
}

static InsertionPoint cloneCfiPrologue(const InsertionPoint &PrologueEnd,
                                       const InsertionPoint &DstInsertPt) {
  MachineFunction &MF = *DstInsertPt.MBB->getParent();

  auto cloneCfiInstructions = [&](MachineBasicBlock::iterator Begin,
                                  MachineBasicBlock::iterator End) {
    auto ToClone = map_range(
        make_filter_range(make_range(Begin, End), isPrologueCFIInstruction),
        [&](const MachineInstr &MI) { return MF.CloneMachineInstr(&MI); });
    DstInsertPt.MBB->insert(DstInsertPt.Iterator, ToClone.begin(),
                            ToClone.end());
  };

  for (auto &MBB : make_range(MF.begin(), PrologueEnd.MBB->getIterator()))
    cloneCfiInstructions(MBB.begin(), MBB.end());
  cloneCfiInstructions(PrologueEnd.MBB->begin(), PrologueEnd.Iterator);
  return DstInsertPt;
}

static bool
fixupBlock(MachineBasicBlock &CurrBB, const BlockFlagsVector &BlockInfo,
           SmallDenseMap<MBBSectionID, InsertionPoint> &InsertionPts,
           const InsertionPoint &Prologue) {
  const MachineFunction &MF = *CurrBB.getParent();
  const TargetFrameLowering &TFL = *MF.getSubtarget().getFrameLowering();
  const BlockFlags &Info = BlockInfo[CurrBB.getNumber()];

  if (!Info.Reachable)
    return false;

  // Always fix every reachable block. Do not consult enableFullCFIFixup():
  // under data-flow split we disable stock CFIFixup via enableCFIFixup()==false,
  // and the default enableFullCFIFixup() mirrors that — which would skip all
  // non-section-start blocks and never insert remember/restore.

  const BlockFlags &PrevInfo =
      BlockInfo[std::prev(CurrBB.getIterator())->getNumber()];
  bool HasFrame = PrevInfo.HasFrameOnExit && !CurrBB.isBeginSection();
  bool NeedsFrame = Info.HasFrameOnEntry && !Info.StrongNoFrameOnEntry;

  if (HasFrame == NeedsFrame)
    return false;

  if (!NeedsFrame) {
    TFL.resetCFIToInitialState(CurrBB);
    return true;
  }

  InsertionPoint &InsertPt = InsertionPts[CurrBB.getSectionID()];
  if (InsertPt.MBB == nullptr) {
    InsertPt = cloneCfiPrologue(Prologue, {&CurrBB, CurrBB.begin()});
  } else {
    InsertPt = insertRememberRestorePair(InsertPt, {&CurrBB, CurrBB.begin()});
  }
  return true;
}

bool ShrinkWrapCFIFixup::runOnMachineFunction(MachineFunction &MF) {
  const TargetFrameLowering *TFL = MF.getSubtarget().getFrameLowering();
  if (!TFL->enableCSRSaveRestorePointsSplit())
    return false;
  if (!MF.needsFrameMoves() ||
      MF.getTarget().getMCAsmInfo()->usesWindowsCFI())
    return false;
  if (MF.getNumBlockIDs() < 2)
    return false;

  MachineBasicBlock::iterator PrologueEnd;
  MachineBasicBlock *PrologueBlock = findEntryPrologueEnd(MF, PrologueEnd);
  if (!PrologueBlock)
    return false;

  BlockFlagsVector BlockInfo = computeBlockInfo(MF, PrologueBlock);

  bool Change = false;
  SmallDenseMap<MBBSectionID, InsertionPoint> InsertionPts;
  InsertionPts[PrologueBlock->getSectionID()] = {PrologueBlock, PrologueEnd};

  assert(PrologueEnd != PrologueBlock->begin() &&
         "Inconsistent notion of \"prologue block\"");

  for (MachineBasicBlock &MBB :
       make_range(std::next(PrologueBlock->getIterator()), MF.end()))
    Change |=
        fixupBlock(MBB, BlockInfo, InsertionPts, {PrologueBlock, PrologueEnd});

  return Change;
}
