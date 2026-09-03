//===- ShrinkFrameUtils.h - Shrink-frame region helpers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared predicates for shrink-frame (getProlog() != nullptr): frame vs
// no-frame regions, CFG-edge legality, and reject helpers for later passes.
// Target-specific epilogue / frame-related MI checks live on
// TargetFrameLowering (default false).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SHRINKFRAMEUTILS_H
#define LLVM_CODEGEN_SHRINKFRAMEUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineBasicBlock;
class MachineDominatorTree;
class MachineFunction;
class MachineInstr;

LLVM_ABI bool hasShrinkFrame(const MachineFunction &MF);
LLVM_ABI MachineBasicBlock *getShrinkFrameProlog(const MachineFunction &MF);

/// True if \p BB is a frameless clone of a shared return (shrink-frame).
LLVM_ABI bool isShrinkFrameClone(const MachineBasicBlock &BB);

/// True if A and B are a registered original/clone pair (either order).
LLVM_ABI bool areShrinkFrameClonePair(const MachineBasicBlock &A,
                                      const MachineBasicBlock &B);

/// Redirect frameless preds back to originals, erase clones, clear pair maps.
/// Keeps Prolog and Save/RestorePoints so shrink-frame without shared-return
/// duplication can continue. Use when clones must be removed but multi-point
/// CSR early-exit maps should survive (e.g. prologue hoisting with clones).
LLVM_ABI void undoShrinkFrameClones(MachineFunction &MF);

/// Full shrink-frame teardown: undo shared-return clones and clear Prolog /
/// Epilog / SavePoints / RestorePoints. Only use when the whole function must
/// leave shrink-frame (e.g. Prolog dominates no return). Do NOT use merely
/// because shared-return duplication is incompatible with prologue hoisting —
/// that regresses to entry-full CSR placement.
LLVM_ABI void abandonShrinkFrame(MachineFunction &MF);

/// Dominated by Prolog. If there is no Prolog, the whole function is treated
/// as in-frame (entry-full frame with multi-point CSR save/restore).
LLVM_ABI bool isInFrameRegion(const MachineBasicBlock &BB,
                              const MachineDominatorTree &DT);

LLVM_ABI bool sameFrameRegion(const MachineBasicBlock *A,
                              const MachineBasicBlock *B,
                              const MachineDominatorTree &DT);

/// True if the blocks are not all on the same side of the frame boundary.
LLVM_ABI bool
crossesFrameRegionBoundary(ArrayRef<const MachineBasicBlock *> Blocks,
                           const MachineDominatorTree &DT);

/// CFG edge check. Not "different regions ⇒ reject":
/// same side: allow; no-frame → Prolog: allow; no-frame → other framed: reject;
/// framed → no-frame: reject.
LLVM_ABI bool crossesFrameRegionEdge(const MachineBasicBlock &Pred,
                                     const MachineBasicBlock &Succ,
                                     const MachineDominatorTree &DT);

/// PEI EpilogBlocks: isReturnBlock && dom(Prolog).
LLVM_ABI bool isShrinkFrameEpilogBlock(const MachineBasicBlock &BB,
                                       const MachineFunction &MF,
                                       const MachineDominatorTree &DT);

LLVM_ABI bool blockHasEpiloguePattern(const MachineBasicBlock &BB,
                                      const MachineFunction &MF);

LLVM_ABI bool isFrameRelatedForGuard(const MachineInstr &MI,
                                     const MachineFunction &MF);

/// True = reject the edge (no-op if !hasShrinkFrame).
LLVM_ABI bool
shrinkFrameGuardRejectEdge(const MachineFunction &MF,
                           const MachineBasicBlock &Pred,
                           const MachineBasicBlock &Succ,
                           const MachineDominatorTree &DT);

/// Common subset: crossesFrameRegionBoundary. TailMerge/Dup/Hoist need extra
/// checks below.
LLVM_ABI bool
shrinkFrameGuardRejectBlocks(const MachineFunction &MF,
                             ArrayRef<const MachineBasicBlock *> Blocks,
                             const MachineDominatorTree &DT);

LLVM_ABI bool
shrinkFrameGuardRejectTailDuplicate(const MachineFunction &MF,
                                    const MachineBasicBlock &Src,
                                    const MachineBasicBlock &Pred,
                                    const MachineDominatorTree &DT);

/// Dest is the merge target (may be Prolog). Rejects epilogue merged into a
/// no-frame dest/pred, and shrink-frame original/clone pairs.
LLVM_ABI bool shrinkFrameGuardRejectTailMerge(
    const MachineFunction &MF, const MachineBasicBlock &M1,
    const MachineBasicBlock &M2, const MachineBasicBlock &Dest,
    const MachineDominatorTree &DT);

/// Reject hoist into Prolog (before FrameSetup) or frame-related MI into Epilog.
LLVM_ABI bool shrinkFrameGuardRejectHoist(const MachineFunction &MF,
                                          const MachineBasicBlock &Dest,
                                          bool HoistingFrameRelated,
                                          const MachineDominatorTree &DT);

/// True when a delayed shrink-frame Prolog cannot be used safely and must be
/// placed at the function entry (GCC: main prologue before any stack touch or
/// mixed frame/frameless return join). Covers frame-bound clobber before
/// Prolog, CSR save points outside dom(Prolog), shared returns with
/// framed+frameless preds, and stack-touching code reachable before dom(Prolog).
LLVM_ABI bool mustHoistShrinkFramePrologToEntry(
    const MachineFunction &MF, MachineBasicBlock *Prolog,
    const MachineDominatorTree &DT);

/// True when \p Reg may be saved at \p SaveBB even though SaveBB is not
/// dominated by \p Prolog while shared-return clones are active: SaveBB must be
/// unreachable from Prolog, \p Reg must not be frame-bound, and PrologEpilog
/// can insert a matching restore on a frameless clone predecessor (\p SaveBB
/// equals that pred or dominates it). On-Prolog saves always return true.
LLVM_ABI bool isLegalOffPrologCSRSaveForCloneRestore(
    const MachineFunction &MF, MachineBasicBlock *Prolog,
    MachineBasicBlock *SaveBB, MCRegister Reg,
    const MachineDominatorTree &DT);

/// True when every register in \p Regs satisfies
/// isLegalOffPrologCSRSaveForCloneRestore.
LLVM_ABI bool isLegalOffPrologCSRSavePoint(
    const MachineFunction &MF, MachineBasicBlock *Prolog,
    MachineBasicBlock *SaveBB, ArrayRef<MCRegister> Regs,
    const MachineDominatorTree &DT);

/// True if \p MBB needs a stack frame (non-tail call, FI, SP/FP modify, etc.).
/// Matches ShrinkWrapping::blockNeedsFrame.
LLVM_ABI bool blockNeedsFrame(const MachineFunction &MF,
                              const MachineBasicBlock &MBB);

/// Shared bail for Frameless RA with shrink-wrapping (naked, EH, sanitizer, …).
/// Does not check FramelessExists/ColdExists (caller computes those).
LLVM_ABI bool shouldSkipFramelessRA(const MachineFunction &MF);

/// Blocks reachable from entry without passing a NeedsFrame block (except entry).
LLVM_ABI void computeFramelessRegionBlocks(
    const MachineFunction &MF, SmallVectorImpl<MachineBasicBlock *> &Out);

LLVM_ABI bool framelessRegionExists(const MachineFunction &MF);
LLVM_ABI bool coldPathExists(const MachineFunction &MF);

} // namespace llvm

#endif
