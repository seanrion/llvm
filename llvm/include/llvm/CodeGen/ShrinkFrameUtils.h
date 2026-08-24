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
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineBasicBlock;
class MachineDominatorTree;
class MachineFunction;
class MachineInstr;

LLVM_ABI bool hasShrinkFrame(const MachineFunction &MF);
LLVM_ABI MachineBasicBlock *getShrinkFrameProlog(const MachineFunction &MF);

/// Dominated by Prolog. If there is no Prolog, the whole function is treated
/// as in-frame (M5c).
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
/// no-frame dest/pred.
LLVM_ABI bool shrinkFrameGuardRejectTailMerge(
    const MachineFunction &MF, const MachineBasicBlock &M1,
    const MachineBasicBlock &M2, const MachineBasicBlock &Dest,
    const MachineDominatorTree &DT);

/// Reject hoist into Prolog (before FrameSetup) or frame-related MI into Epilog.
LLVM_ABI bool shrinkFrameGuardRejectHoist(const MachineFunction &MF,
                                          const MachineBasicBlock &Dest,
                                          bool HoistingFrameRelated,
                                          const MachineDominatorTree &DT);

} // namespace llvm

#endif
