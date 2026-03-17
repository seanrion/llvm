//===-- RISCVBTBFetchLineBranchRelaxation.h - BTB fetch line branch relax -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options and helpers for the BTB fetch-line limit pass and for verifying
// the limit. Canonical verification runs after MC layout is computed
// (RISCVAsmBackend::performPostLayout).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVBTBFETCHLINEBRANCHRELAXATION_H
#define LLVM_LIB_TARGET_RISCV_RISCVBTBFETCHLINEBRANCHRELAXATION_H

#include "llvm/Support/CommandLine.h"

namespace llvm {

/// Pass options; also used by MC backend for post-layout verification.
inline cl::opt<bool> EnableRISCVBTBFetchLineBranchRelaxation(
    "riscv-btb-fetchline-branch-relaxation",
    cl::desc("Limit number of branches per fetch line to reduce BTB pressure"),
    cl::init(false), cl::Hidden);

inline cl::opt<unsigned> BTBFetchLineSize(
    "riscv-fetchline-size",
    cl::desc("Fetch line size in bytes (e.g. 32 for 32B)"),
    cl::init(32), cl::Hidden);

inline cl::opt<unsigned> BTBMaxBranchesPerFetchLine(
    "riscv-btb-max-branches-per-fetchline",
    cl::desc("Max branch instructions per fetch line (e.g. 4)"),
    cl::init(4), cl::Hidden);


} // end namespace llvm

#endif
