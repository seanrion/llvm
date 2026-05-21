//===-- RISCVAsmBackend.h - RISC-V Assembler Backend ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVASMBACKEND_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVASMBACKEND_H

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCObjectStreamer.h"

namespace llvm {
class MCAssembler;
class MCObjectTargetWriter;
class MCSection;
class raw_ostream;

class RISCVAsmBackend : public MCAsmBackend {
protected:
  const MCSubtargetInfo &STI;
  uint8_t OSABI;
  bool Is64Bit;
  const MCTargetOptions &TargetOptions;

  unsigned BranchSpacingValue = 0;
  MCBranchSpacingFragment *LastBA = nullptr;
  MCBranchSpacingFragment *PendingBA = nullptr;

  // Temporary symbol used to check whether a PC-relative fixup is resolved.
  MCSymbol *PCRelTemp = nullptr;

  /// Cached §4 post-pool sizes for this backend (F, N, M(STI) fixed at construction).
  unsigned BTBOffPoolInitialBytes = 0;
  unsigned BTBOnCallPoolInitialBytes = 0;
  unsigned BTBOnCondPoolInitialBytes = 0;

  unsigned getBTBPostPoolInitialBytes(NopsBesideBranchKind Kind,
                                      bool LastWasConditional) const;

  /// Initialize \p BTB*PoolInitialBytes from CLI (F, N) and \p STI (Zca → M).
  void initBTBPostPoolInitialBytes();

  bool isPCRelFixupResolved(const MCSymbol *SymA, const MCFragment &F);

  StringMap<MCSymbol *> VendorSymbols;

public:
  RISCVAsmBackend(const MCSubtargetInfo &STI, uint8_t OSABI, bool Is64Bit,
                  bool IsLittleEndian, const MCTargetOptions &Options);
  ~RISCVAsmBackend() override = default;

  std::optional<bool> evaluateFixup(const MCFragment &, MCFixup &, MCValue &,
                                    uint64_t &) override;
  bool addReloc(const MCFragment &, const MCFixup &, const MCValue &,
                uint64_t &FixedValue, bool IsResolved);

  void maybeAddVendorReloc(const MCFragment &, const MCFixup &);

  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &Target,
                  uint8_t *Data, uint64_t Value, bool IsResolved) override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;

  bool fixupNeedsRelaxationAdvanced(const MCFragment &, const MCFixup &,
                                    const MCValue &, uint64_t,
                                    bool) const override;

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override;

  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override;

  bool mayNeedRelaxation(unsigned Opcode, ArrayRef<MCOperand> Operands,
                         const MCSubtargetInfo &STI) const override;
  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override;

  bool relaxAlign(MCFragment &F, unsigned &Size) override;
  bool relaxDwarfLineAddr(MCFragment &) const override;
  bool relaxDwarfCFA(MCFragment &) const override;
  std::pair<bool, bool> relaxLEB128(MCFragment &LF,
                                    int64_t &Value) const override;

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  void performPostLayout(const MCAssembler &Asm) const override;

  bool shrinkSection(MCAssembler &Asm, MCSection &Sec,
                     uint64_t &RestartWinBase) override;

  const MCTargetOptions &getTargetOptions() const { return TargetOptions; }

  bool needBranchSpacing(const MCInst &Inst)const;
  void BranchSpacing();
  void emitInstructionBegin(MCObjectStreamer &S, const MCInst &Inst,
                            const MCSubtargetInfo &STI);
  void emitInstructionEnd(MCObjectStreamer &S, const MCInst &Inst,
                          const MCSubtargetInfo &STI);

  void refreshNBFInsertKindsAfterRelax(const MCAssembler &Asm,
                                     MCSection &Sec) override;
};
}

#endif
