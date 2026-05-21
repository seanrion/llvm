//===-- RISCVBTBBranchRelaxation.h - RISC-V BTB branch relaxation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command-line options for RISC-V BTB fetch-line limits and MC-layer shrink /
// verify helpers. Including this header in a translation unit registers the
// options with the global llvm::cl registry.
//
// Typical consumers:
//   - lib/Target/RISCV: RISCVAsmBackend, RISCVAsmPrinter,
//     RISCVBTBBranchRelaxation.cpp
//   - lib/CodeGen: MachineBlockPlacement (skip MI alignBlocks when enabled)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_RISCV_RISCVBTBBRANCHRELAXATION_H
#define LLVM_TARGET_RISCV_RISCVBTBBRANCHRELAXATION_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <cstdint>

namespace llvm {

inline void validateBTBFetchLineSizeOption(const unsigned &V) {
  if (V == 0)
    report_fatal_error(
        "-riscv-fetchline-size: value must be greater than zero");
  if (!isPowerOf2_64(static_cast<uint64_t>(V)))
    report_fatal_error("-riscv-fetchline-size: value must be a power of two "
                       "(required for llvm::Align / emitCodeAlignment)");
}

inline void validateBTBMaxBranchesPerFetchLineOption(const unsigned &V) {
  if (V == 0)
    report_fatal_error("-riscv-btb-max-branches-per-fetchline: value must be "
                       "greater than zero");
}

/// Pass options; also used by MC backend for post-layout verification.
inline cl::opt<bool> EnableRISCVBTBFetchLineBranchRelaxation(
    "riscv-btb-fetchline-branch-relaxation",
    cl::desc("Limit number of branches per fetch line to reduce BTB pressure"),
    cl::init(false), cl::Hidden);

inline cl::opt<unsigned> BTBFetchLineSize(
    "riscv-fetchline-size",
    cl::desc("Fetch line size in bytes; must be a power of two (e.g. 32)"),
    cl::init(32), cl::Hidden,
    cl::callback([](const unsigned &V) { validateBTBFetchLineSizeOption(V); }));

inline cl::opt<unsigned> BTBMaxBranchesPerFetchLine(
    "riscv-btb-max-branches-per-fetchline",
    cl::desc("Max branch instructions per fetch line (e.g. 4)"),
    cl::init(4), cl::Hidden,
    cl::callback(
        [](const unsigned &V) { validateBTBMaxBranchesPerFetchLineOption(V); }));

/// When true, post-layout BTB fetch-line verification reports a warning instead
/// of an error if a window exceeds the branch limit.
inline cl::opt<bool> RISCVBTBFetchLinePostLayoutViolationAsWarning(
    "riscv-btb-fetchline-violation-as-warning",
    cl::desc("Emit a warning instead of an error when BTB fetch-line branch "
             "limit is exceeded after MC layout"),
    cl::init(true), cl::Hidden);

/// When true, shrinkSection runs in debug-only pre-shrink dump mode and returns
/// without modifying any NBF.
inline cl::opt<bool> RISCVBTBShrinkSectionDiagOnly(
    "riscv-btb-shrink-section-diag-only",
    cl::desc("Debug: print shrinkSection pre-shrink layout diagnostics only; "
             "do not shrink NOP pools beside branches"),
    cl::init(false), cl::Hidden);

class MCAssembler;
class MCFragment;
class MCInst;
class MCSection;
class MCNopsBesideBranchFragment;

/// Branch site used by BTB shrink / verify (offset + encoding bytes).
struct BTBBranchInfo {
  uint64_t Offset = 0;
  uint8_t Size = 0;
  std::array<uint8_t, 4> Bytes = {0, 0, 0, 0};
};

/// Align region layout for a text section (padding start + region end).
struct BTBAlignLayout {
  SmallVector<uint64_t, 16> PaddingStart;
  SmallVector<uint64_t, 16> RegionEnd;
};

/// On/Off NBF pools collected for shrink.
struct BTBNopPools {
  SmallVector<std::pair<uint64_t, MCNopsBesideBranchFragment *>, 32> On;
  SmallVector<std::pair<uint64_t, MCNopsBesideBranchFragment *>, 32> Off;
};

namespace RISCVBTB {

/// True if \p Inst gets an OffExecPath NOP pool at emit (unconditional jump,
/// tail, or JAL/JALR with rd==x0).
bool isOffExecPathJump(const MCInst &Inst);

/// True if \p Inst gets an OnExecPath NOP pool (fall-through or call return).
bool isOnExecPathJump(const MCInst &Inst);

/// True if \p Inst is a conditional branch (may relax to PseudoLong*).
bool isConditionalBranch(const MCInst &Inst);

/// After MC relax, reclassify On-path NBF to Off when the preceding code
/// fragment's last instruction encodes as an off-execution-path jump.
void refreshNBFInsertKindsAfterRelax(MCSection &Sec);

/// Read \p Size bytes at \p OffsetInFragment from fixed or variable fragment
/// data into \p Out. Used by refresh and branch collection.
bool readBranchBytesFromFragment(const MCFragment &Frag,
                                 uint64_t OffsetInFragment, uint8_t Size,
                                 std::array<uint8_t, 4> &Out);

/// Collect align region boundaries: padding start and region end per FT_Align.
BTBAlignLayout collectBTBAlignLayout(const MCAssembler &Asm,
                                     const MCSection &Sec);

/// Sorted unique branch offsets in \p Sec with offset \>= \p MinOffset (fixups
/// plus byte-scanned JALR / compressed branches without fixups).
void collectBTBBranchOffsets(const MCAssembler &Asm, const MCSection &Sec,
                             uint64_t MinOffset,
                             SmallVectorImpl<uint64_t> &Out);

/// Like \c collectBTBBranchOffsets but retains instruction encodings for
/// diagnostics.
void collectBTBBranches(const MCAssembler &Asm, const MCSection &Sec,
                        SmallVectorImpl<BTBBranchInfo> &Out);

/// Non-empty On/Off \c MCNopsBesideBranchFragment pools with end offset \>
/// \p MinOffset, for shrink.
void collectBTBNopPools(const MCAssembler &Asm, MCSection &Sec,
                        uint64_t MinOffset, BTBNopPools &Out);

/// Shrink NBF pools so each fetch-line window has at most N control-flow
/// instructions. Sets \p RestartWinBase when a change is made; returns true if
/// layout must be re-run.
bool shrinkSection(MCAssembler &Asm, MCSection &Sec, uint64_t &RestartWinBase);

/// Post-layout check: each half-open window \c [w, w+F) has at most N branches.
/// Violations are reported as warning or error per command-line options.
void verifyBTBFetchLineBranchLimits(const MCAssembler &Asm);

} // namespace RISCVBTB
} // namespace llvm

#endif // LLVM_TARGET_RISCV_RISCVBTBBRANCHRELAXATION_H
