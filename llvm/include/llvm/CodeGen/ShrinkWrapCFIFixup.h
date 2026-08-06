//===-- ShrinkWrapCFIFixup.h - remember/restore for multi-point CFI -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Insert .cfi_remember_state / .cfi_restore_state after final block layout
/// for targets using multi-point CSR save/restore (data-flow shrink-wrapping).
///
/// Unlike CFIFixup, prologue end is taken from the *entry* block only. Delayed
/// spill CFI on cold paths is also FrameSetup and must not be treated as the
/// real prologue (that bug poisoned throw paths after return epilogues).
///
/// Mirrors the intent of GCC dwarf2cfi remember/restore when connecting an
/// epilogue trace to a later block that still needs a call frame.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SHRINKWRAPCFIFIXUP_H
#define LLVM_CODEGEN_SHRINKWRAPCFIFIXUP_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class ShrinkWrapCFIFixup : public MachineFunctionPass {
public:
  static char ID;

  ShrinkWrapCFIFixup();

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

FunctionPass *createShrinkWrapCFIFixup();

} // namespace llvm

#endif // LLVM_CODEGEN_SHRINKWRAPCFIFIXUP_H
