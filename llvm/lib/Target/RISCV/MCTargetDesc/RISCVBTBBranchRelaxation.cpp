//===-- RISCVBTBBranchRelaxation.cpp - RISC-V BTB branch relaxation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVBaseInfo.h"
#include "RISCVFixupKinds.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/Target/RISCV/RISCVBTBBranchRelaxation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "riscv-btb-branch-relaxation"

namespace {
const MCSection *ShrinkDiagTrackedSection = nullptr;
unsigned ShrinkDiagCallIdx = 0;
} // namespace

// JALR opcode (bits [6:0])
static constexpr uint8_t OPC_JALR = 0b1100111;

namespace llvm {
namespace RISCVBTB {

bool isOffExecPathJump(const MCInst &Inst) {
  unsigned Opcode = Inst.getOpcode();
  switch (Opcode) {
  case RISCV::C_J:
  case RISCV::C_JR:
  case RISCV::PseudoBR:
  case RISCV::PseudoBRIND:
  case RISCV::PseudoBRINDNonX7:
  case RISCV::PseudoBRINDX7:
  case RISCV::PseudoTAIL:
  case RISCV::PseudoTAILIndirect:
  case RISCV::PseudoTAILIndirectNonX7:
  case RISCV::PseudoTAILIndirectX7:
    return true;
  case RISCV::JAL:
  case RISCV::JALR:
    return Inst.getOperand(0).getReg() == RISCV::X0;
  default:
    return false;
  }
}

bool isOnExecPathJump(const MCInst &Inst) {
  unsigned Opcode = Inst.getOpcode();
  switch (Opcode) {
  case RISCV::BEQ:
  case RISCV::BNE:
  case RISCV::BLT:
  case RISCV::BGE:
  case RISCV::BLTU:
  case RISCV::BGEU:
  case RISCV::BEQI:
  case RISCV::BNEI:
  case RISCV::C_BEQZ:
  case RISCV::C_BNEZ:
  case RISCV::C_JAL:
  case RISCV::C_JALR:
    return true;
  case RISCV::PseudoCALL:
  case RISCV::PseudoCALLReg:
  case RISCV::PseudoJump:
  case RISCV::PseudoCALLIndirect:
  case RISCV::PseudoCALLIndirectNonX7:
  case RISCV::PseudoCALLIndirectX7:
    return true;
  case RISCV::JAL:
  case RISCV::JALR:
    return Inst.getOperand(0).getReg() != RISCV::X0;
  default:
    return false;
  }
}

bool isConditionalBranch(const MCInst &Inst) {
  switch (Inst.getOpcode()) {
  case RISCV::BEQ:
  case RISCV::BNE:
  case RISCV::BLT:
  case RISCV::BGE:
  case RISCV::BLTU:
  case RISCV::BGEU:
  case RISCV::BEQI:
  case RISCV::BNEI:
  case RISCV::C_BEQZ:
  case RISCV::C_BNEZ:
    return true;
  default:
    return false;
  }
}

/// Recognize off-path jump encodings in machine code (JAL/JALR x0, C.J, C.JR).
/// Used after relax when MCInst is no longer available on the fragment.
static bool isOffExecPathInsnEncoding(const MCFragment &F, uint64_t Rel) {
  std::array<uint8_t, 4> Bytes = {};
  if (!readBranchBytesFromFragment(F, Rel, 2, Bytes))
    return false;
  if ((Bytes[0] & 3) != 3) {
    uint16_t Half = uint16_t(Bytes[0]) | (uint16_t(Bytes[1]) << 8);
    if ((Bytes[0] & 3) == 1) {
      unsigned Funct3 = (Half >> 13) & 7;
      if (Funct3 == 5)
        return true;
    }
    if ((Bytes[0] & 3) == 2 && ((Bytes[0] >> 2) & 0x1F) == 0) {
      unsigned Funct4 = (Bytes[1] >> 4) & 0xF;
      unsigned Rs1 = ((Bytes[1] & 0xF) << 1) | (Bytes[0] >> 7);
      if (Rs1 != 0 && Funct4 == 8)
        return true;
    }
    return false;
  }
  if (!readBranchBytesFromFragment(F, Rel, 4, Bytes))
    return false;
  uint32_t Insn = uint32_t(Bytes[0]) | (uint32_t(Bytes[1]) << 8) |
                  (uint32_t(Bytes[2]) << 16) | (uint32_t(Bytes[3]) << 24);
  const uint8_t Opc = Insn & 0x7F;
  if (Opc == 0x6F || Opc == OPC_JALR)
    return ((Insn >> 7) & 0x1F) == 0;
  return false;
}

/// Fragment kinds that may contain instructions used as an NBF predecessor.
static bool isRefreshPrevCodeFragment(const MCFragment &F) {
  return F.getKind() == MCFragment::FT_Data ||
         F.getKind() == MCFragment::FT_Relaxable ||
         (F.getKind() == MCFragment::FT_Align && F.getFixedSize() > 0);
}

/// True if the last instruction in \p F is Off-path (emit invariant: NBF follows
/// the branch). Handles [32-bit][16-bit] tails after relax.
static bool fragmentEndsWithOffExecPathJump(const MCFragment &F) {
  if (!isRefreshPrevCodeFragment(F))
    return false;

  const size_t FixedSize = F.getContents().size();
  const bool CodeInFixedOnly = F.getKind() == MCFragment::FT_Align ||
                               F.getKind() == MCFragment::FT_Data;
  const size_t Total =
      CodeInFixedOnly ? FixedSize : FixedSize + F.getVarContents().size();
  if (Total < 2)
    return false;

  std::array<uint8_t, 4> B = {};
  std::optional<uint64_t> Rel;
  if (Total >= 4 && readBranchBytesFromFragment(F, Total - 4, 2, B) &&
      (B[0] & 3) == 3 && readBranchBytesFromFragment(F, Total - 2, 2, B) &&
      (B[0] & 3) == 3)
    Rel = Total - 4;
  else if (readBranchBytesFromFragment(F, Total - 2, 2, B) && (B[0] & 3) != 3)
    Rel = Total - 2;
  else if (Total >= 4 && readBranchBytesFromFragment(F, Total - 4, 2, B) &&
           (B[0] & 3) == 3)
    Rel = Total - 4;
  if (!Rel)
    return false;

  if (!readBranchBytesFromFragment(F, *Rel, 2, B))
    return false;
  const unsigned InsnSize = ((B[0] & 3) == 3) ? 4u : 2u;
  if (*Rel + InsnSize != Total)
    return false;
  return isOffExecPathInsnEncoding(F, *Rel);
}

void refreshNBFInsertKindsAfterRelax(MCSection &Sec) {
  // On-path NBF emitted before relax may become Off-path (e.g. PseudoLong* → jal x0).
  if (!EnableRISCVBTBFetchLineBranchRelaxation)
    return;

  const MCFragment *PrevCode = nullptr;
  for (MCFragment &F : Sec) {
    if (auto *NBF = dyn_cast<MCNopsBesideBranchFragment>(&F)) {
      if (NBF->getInsertKind() == NopsBesideBranchKind::OnExecPath &&
          NBF->getNumBytes() > 0 && PrevCode != nullptr &&
          fragmentEndsWithOffExecPathJump(*PrevCode))
        NBF->setInsertKind(NopsBesideBranchKind::OffExecPath);
    }
    if (isRefreshPrevCodeFragment(F))
      PrevCode = &F;
  }
}

bool readBranchBytesFromFragment(const MCFragment &Frag,
                                 uint64_t OffsetInFragment, uint8_t Size,
                                 std::array<uint8_t, 4> &Out) {
  ArrayRef<char> Fixed = Frag.getContents();
  ArrayRef<char> Variable = Frag.getVarContents();
  const size_t FixedSize = Fixed.size();
  if (OffsetInFragment + Size <= FixedSize) {
    for (uint8_t I = 0; I != Size; ++I)
      Out[I] = static_cast<uint8_t>(Fixed[OffsetInFragment + I]);
    return true;
  }
  if (OffsetInFragment >= FixedSize &&
      OffsetInFragment - FixedSize + Size <= Variable.size()) {
    uint64_t VarOffset = OffsetInFragment - FixedSize;
    for (uint8_t I = 0; I != Size; ++I)
      Out[I] = static_cast<uint8_t>(Variable[VarOffset + I]);
    return true;
  }
  return false;
}

static bool fragmentHasScannableCode(const MCFragment &F) {
  // Same kinds as refresh predecessor and byte scan (FT_Align fixed prefix).
  return F.getKind() == MCFragment::FT_Data ||
         F.getKind() == MCFragment::FT_Relaxable ||
         (F.getKind() == MCFragment::FT_Align && F.getFixedSize() > 0);
}

/// Byte-scan fragment for branches that may lack fixups after relax (JALR,
/// C.BEQZ/C.BNEZ, C.JR/C.JALR).
static void collectJalrBranchInfosFromBytes(const MCFragment &Frag,
                                            uint64_t FragOffset,
                                            SmallVectorImpl<BTBBranchInfo> &Out) {
  ArrayRef<char> Fixed = Frag.getContents();
  ArrayRef<char> Var = Frag.getVarContents();
  const size_t FixedSize = Fixed.size();
  const bool ScanFixedOnly = Frag.getKind() == MCFragment::FT_Align ||
                             Frag.getKind() == MCFragment::FT_Data;
  const size_t Total = ScanFixedOnly ? FixedSize : (FixedSize + Var.size());

  auto readU8FromFixed = [&](size_t Idx) -> uint8_t {
    assert(Idx < FixedSize && "readU8FromFixed index out of fixed range");
    return static_cast<uint8_t>(Fixed[Idx]);
  };
  auto readU8FromFixedAndVar = [&](size_t Idx) -> uint8_t {
    assert(Idx < Total && "readU8 index out of fragment range");
    return static_cast<uint8_t>(Idx < FixedSize ? Fixed[Idx]
                                                : Var[Idx - FixedSize]);
  };
  auto readU8 = [&](size_t Idx) -> uint8_t {
    return ScanFixedOnly ? readU8FromFixed(Idx) : readU8FromFixedAndVar(Idx);
  };

  size_t Pos = 0;
  while (Pos + 2 <= Total) {
    uint8_t b0 = readU8(Pos);
    uint8_t b1 = readU8(Pos + 1);
    if ((b0 & 3) == 3) {
      if ((b0 & 0x7F) == OPC_JALR && Pos + 4 <= Total) {
        BTBBranchInfo Info;
        Info.Offset = FragOffset + Pos;
        Info.Size = 4;
        for (uint8_t I = 0; I != 4; ++I)
          Info.Bytes[I] = readU8(Pos + I);
        Out.push_back(Info);
        LLVM_DEBUG({
          dbgs() << "collectJalrBranchInfosFromBytes: JALR @0x"
                 << format_hex(Info.Offset, 8) << " enc";
          for (uint8_t I = 0; I != 4; ++I)
            dbgs() << ' ' << format_hex_no_prefix(Info.Bytes[I], 2,
                                                  /*Upper=*/false);
          dbgs() << "\n";
        });
      }
      Pos += 4;
    } else {
      if ((b0 & 3) == 1) {
        uint16_t Half = uint16_t(b0) | (uint16_t(b1) << 8);
        unsigned Funct3 = (Half >> 13) & 7;
        if (Funct3 == 6 || Funct3 == 7) {
          BTBBranchInfo Info;
          Info.Offset = FragOffset + Pos;
          Info.Size = 2;
          Info.Bytes[0] = b0;
          Info.Bytes[1] = b1;
          Out.push_back(Info);
          LLVM_DEBUG({
            dbgs() << "collectJalrBranchInfosFromBytes: "
                   << (Funct3 == 7 ? "C.BNEZ" : "C.BEQZ") << " @0x"
                   << format_hex(Info.Offset, 8) << " enc"
                   << ' ' << format_hex_no_prefix(Info.Bytes[0], 2,
                                                  /*Upper=*/false)
                   << ' ' << format_hex_no_prefix(Info.Bytes[1], 2,
                                                  /*Upper=*/false)
                   << "\n";
          });
        }
      }
      if ((b0 & 3) == 2 && ((b0 >> 2) & 0x1F) == 0) {
        unsigned funct4 = (b1 >> 4) & 0xF;
        unsigned rs1 = ((b1 & 0xF) << 1) | (b0 >> 7);
        if (rs1 != 0 && (funct4 == 8 || funct4 == 9)) {
          BTBBranchInfo Info;
          Info.Offset = FragOffset + Pos;
          Info.Size = 2;
          Info.Bytes[0] = b0;
          Info.Bytes[1] = b1;
          Out.push_back(Info);
          LLVM_DEBUG({
            dbgs() << "collectJalrBranchInfosFromBytes: "
                   << (funct4 == 9 ? "C.JALR" : "C.JR") << " @0x"
                   << format_hex(Info.Offset, 8) << " enc"
                   << ' ' << format_hex_no_prefix(Info.Bytes[0], 2,
                                                  /*Upper=*/false)
                   << ' ' << format_hex_no_prefix(Info.Bytes[1], 2,
                                                  /*Upper=*/false)
                   << "\n";
          });
        }
      }
      Pos += 2;
    }
  }
}

static bool isBTBBranchFixup(MCFixupKind Kind) {
  return Kind == RISCV::fixup_riscv_jal || Kind == RISCV::fixup_riscv_branch ||
         Kind == RISCV::fixup_riscv_rvc_jump ||
         Kind == RISCV::fixup_riscv_rvc_branch;
}

static bool isBTBCallFixup(MCFixupKind Kind) {
  return Kind == RISCV::fixup_riscv_call ||
         Kind == RISCV::fixup_riscv_call_plt;
}

static uint8_t fixupBranchInsnSize(MCFixupKind Kind) {
  return Kind == RISCV::fixup_riscv_rvc_jump ||
                 Kind == RISCV::fixup_riscv_rvc_branch
             ? 2
             : 4;
}

static void noteFixupBranch(uint64_t FragOffset, const MCFixup &Fixup,
                            uint64_t MinOffset,
                            SmallVectorImpl<uint64_t> *OffsetsOut,
                            SmallVectorImpl<BTBBranchInfo> *BranchesOut,
                            const MCFragment &F) {
  MCFixupKind Kind = Fixup.getKind();
  if (isBTBBranchFixup(Kind)) {
    uint64_t Off = FragOffset + Fixup.getOffset();
    if (Off < MinOffset)
      return;
    if (OffsetsOut)
      OffsetsOut->push_back(Off);
    else if (BranchesOut) {
      BTBBranchInfo Info;
      Info.Offset = Off;
      Info.Size = fixupBranchInsnSize(Kind);
      readBranchBytesFromFragment(F, Fixup.getOffset(), Info.Size, Info.Bytes);
      BranchesOut->push_back(Info);
    }
    return;
  }
  if (isBTBCallFixup(Kind)) {
    uint64_t Off = FragOffset + Fixup.getOffset() + 4;
    if (Off < MinOffset)
      return;
    if (OffsetsOut)
      OffsetsOut->push_back(Off);
    else if (BranchesOut) {
      BTBBranchInfo Info;
      Info.Offset = Off;
      Info.Size = 4;
      readBranchBytesFromFragment(F, Fixup.getOffset() + 4, 4, Info.Bytes);
      BranchesOut->push_back(Info);
    }
  }
}

/// Record branches from one fragment via fixups and optional byte scan.
/// Exactly one of \p OffsetsOut or \p BranchesOut must be non-null.
static void appendBranchesFromFragment(const MCAssembler &Asm,
                                       const MCFragment &F, uint64_t FragOffset,
                                       uint64_t MinOffset,
                                       SmallVectorImpl<uint64_t> *OffsetsOut,
                                       SmallVectorImpl<BTBBranchInfo> *BranchesOut) {
  assert((OffsetsOut != nullptr) ^ (BranchesOut != nullptr));

  for (const MCFixup &Fixup : F.getFixups())
    noteFixupBranch(FragOffset, Fixup, MinOffset, OffsetsOut, BranchesOut, F);
  for (const MCFixup &Fixup : F.getVarFixups())
    noteFixupBranch(FragOffset, Fixup, MinOffset, OffsetsOut, BranchesOut, F);

  if (!fragmentHasScannableCode(F))
    return;

  SmallVector<BTBBranchInfo, 32> Scanned;
  collectJalrBranchInfosFromBytes(F, FragOffset, Scanned);
  for (const BTBBranchInfo &BI : Scanned) {
    if (BI.Offset < MinOffset)
      continue;
    if (OffsetsOut)
      OffsetsOut->push_back(BI.Offset);
    else
      BranchesOut->push_back(BI);
  }
}

BTBAlignLayout collectBTBAlignLayout(const MCAssembler &Asm,
                                      const MCSection &Sec) {
  BTBAlignLayout Layout;
  for (const MCFragment &F : Sec) {
    if (F.getKind() == MCFragment::FT_Align) {
      uint64_t FragOffset = Asm.getFragmentOffset(F);
      uint64_t FragEnd = FragOffset + Asm.computeFragmentSize(F);
      Layout.PaddingStart.push_back(FragOffset + F.getFixedSize());
      Layout.RegionEnd.push_back(FragEnd);
    }
  }
  return Layout;
}

void collectBTBBranchOffsets(const MCAssembler &Asm, const MCSection &Sec,
                             uint64_t MinOffset,
                             SmallVectorImpl<uint64_t> &Out) {
  for (const MCFragment &F : Sec)
    appendBranchesFromFragment(Asm, F, Asm.getFragmentOffset(F), MinOffset,
                               &Out, nullptr);
  llvm::sort(Out);
  Out.erase(std::unique(Out.begin(), Out.end()), Out.end());
}

void collectBTBBranches(const MCAssembler &Asm, const MCSection &Sec,
                        SmallVectorImpl<BTBBranchInfo> &Out) {
  for (const MCFragment &F : Sec)
    appendBranchesFromFragment(Asm, F, Asm.getFragmentOffset(F), 0, nullptr,
                               &Out);
  llvm::sort(Out, [](const BTBBranchInfo &A, const BTBBranchInfo &B) {
    return A.Offset < B.Offset;
  });
  Out.erase(std::unique(Out.begin(), Out.end(),
                        [](const BTBBranchInfo &A, const BTBBranchInfo &B) {
                          return A.Offset == B.Offset;
                        }),
            Out.end());
}

void collectBTBNopPools(const MCAssembler &Asm, MCSection &Sec,
                        uint64_t MinOffset, BTBNopPools &Pools) {
  for (MCFragment &F : Sec) {
    uint64_t FragOffset = Asm.getFragmentOffset(F);
    auto *NBF = dyn_cast<MCNopsBesideBranchFragment>(&F);
    if (!NBF || NBF->getNumBytes() == 0 ||
        FragOffset + NBF->getNumBytes() <= MinOffset)
      continue;
    LLVM_DEBUG(dbgs() << "  NBF at " << format_hex(FragOffset, 8)
                      << " size=" << NBF->getNumBytes()
                      << " kind="
                      << (NBF->getInsertKind() ==
                                  NopsBesideBranchKind::OnExecPath
                              ? "OnExecPath"
                              : "OffExecPath")
                      << "\n");
    if (NBF->getInsertKind() == NopsBesideBranchKind::OnExecPath)
      Pools.On.push_back({FragOffset, NBF});
    else
      Pools.Off.push_back({FragOffset, NBF});
  }
}

/// Debug-only dump of distance between branch \p i and \p i+N vs fetch-line size.
static void emitPreShrinkPairwiseDebug(ArrayRef<uint64_t> BranchOffsets,
                                       unsigned N, unsigned FetchLineSize) {
  LLVM_DEBUG({
    dbgs() << "  branch -> (N+1)th@+N (N=" << N
           << "), dist vs FetchLineSize=" << FetchLineSize << ":\n";
    unsigned NumGapLtFetch = 0;
    for (size_t i = 0; i + N < BranchOffsets.size(); ++i) {
      uint64_t B0 = BranchOffsets[i];
      uint64_t BN = BranchOffsets[i + N];
      uint64_t Dist = BN - B0;
      bool LtFetch = Dist < FetchLineSize;
      if (LtFetch)
        ++NumGapLtFetch;
      dbgs() << "    i=" << i << " @" << format_hex(B0, 8) << " -> @"
             << format_hex(BN, 8) << " dist=" << Dist
             << (LtFetch ? "  (<FetchLineSize)\n" : "\n");
    }
    dbgs() << "  pairs with dist < FetchLineSize: " << NumGapLtFetch << " / "
           << (BranchOffsets.size() > N ? BranchOffsets.size() - N : 0)
           << "\n";
  });
}

/// Code segment start and align index for \p RestartWinBase (skip finished align
/// regions).
static std::pair<uint64_t, size_t>
findSegmentStart(const BTBAlignLayout &Align, uint64_t RestartWinBase) {
  size_t AlignIdx = 0;
  uint64_t SegmentStart = 0;
  while (AlignIdx < Align.RegionEnd.size() &&
         Align.RegionEnd[AlignIdx] <= RestartWinBase) {
    SegmentStart = Align.RegionEnd[AlignIdx];
    ++AlignIdx;
  }
  return {SegmentStart, AlignIdx};
}

/// If \p WinBase lies inside align padding, skip to the end of that region.
static uint64_t advanceWinBasePastPadding(const BTBAlignLayout &Align,
                                          size_t AlignIdx, uint64_t WinBase) {
  if (AlignIdx < Align.PaddingStart.size() &&
      WinBase >= Align.PaddingStart[AlignIdx] &&
      WinBase < Align.RegionEnd[AlignIdx])
    return Align.RegionEnd[AlignIdx];
  return WinBase;
}

/// Halve removable NOP bytes for one NBF, capped by \p BCritical - \p WinEnd.
/// Returns true if \p Frag was modified.
static bool tryShrinkOneNBF(uint64_t FragOff, MCNopsBesideBranchFragment *Frag,
                            uint64_t WinEnd, uint64_t BCritical) {
  int64_t max_removal = Frag->getNumBytes();
  if (BCritical != UINT64_MAX) {
    if (BCritical < WinEnd)
      return false;
    int64_t limit = BCritical - WinEnd;
    max_removal = std::min(max_removal, limit);
  }
  int64_t removal = (max_removal / 2) * 2;
  if (removal <= 0)
    return false;

  LLVM_DEBUG(dbgs() << "    shrink: FragOff=" << format_hex(FragOff, 8)
                    << " oldBytes=" << Frag->getNumBytes()
                    << " removal=" << removal << "\n");
  if (removal >= Frag->getNumBytes())
    Frag->setNumBytes(0);
  else
    Frag->setNumBytes(Frag->getNumBytes() - removal);
  return true;
}

/// Try to shrink the next eligible NBF in \p NopsList (On before Off at call site).
static bool tryShrinkNBFList(
    SmallVectorImpl<std::pair<uint64_t, MCNopsBesideBranchFragment *>> &NopsList,
    size_t &NopIdx, uint64_t LoopWinBase, uint64_t WinEnd, uint64_t SegmentEnd,
    uint64_t BCritical) {
  LLVM_DEBUG(dbgs() << "      tryShrink: NopIdx=" << NopIdx
                    << " NopsList.size()=" << NopsList.size() << "\n");
  for (; NopIdx < NopsList.size(); ++NopIdx) {
    auto [FragOff, Frag] = NopsList[NopIdx];
    if (FragOff > SegmentEnd)
      break;
    if (BCritical != UINT64_MAX && FragOff > BCritical)
      break;
    if (FragOff < LoopWinBase)
      continue;
    if (FragOff >= WinEnd)
      break;
    if (tryShrinkOneNBF(FragOff, Frag, WinEnd, BCritical))
      return true;
  }
  return false;
}

/// Advance branch/NBF cursors after processing window \c [w, WinEnd).
static void advanceIndicesPastWindow(ArrayRef<uint64_t> BranchOffsets,
                                     size_t &BranchIdx, uint64_t WinEnd,
                                     BTBNopPools &Pools, size_t &NopOnIdx,
                                     size_t &NopOffIdx, uint64_t w) {
  while (BranchIdx < BranchOffsets.size() &&
         BranchOffsets[BranchIdx] < WinEnd)
    ++BranchIdx;
  auto advanceNop = [&](auto &NopsList, size_t &NopIdx) {
    while (NopIdx < NopsList.size()) {
      auto [Off, Frag] = NopsList[NopIdx];
      uint64_t End = Off + Frag->getNumBytes();
      if (Off >= w && End <= WinEnd)
        ++NopIdx;
      else
        break;
    }
  };
  advanceNop(Pools.On, NopOnIdx);
  advanceNop(Pools.Off, NopOffIdx);
}

/// Walk fetch-line windows; shrink one NBF per invocation when needed.
/// \p BCritical is the offset of the (N+1)th branch from the segment cursor.
static bool runShrinkWindowLoop(MCAssembler &Asm, MCSection &Sec,
                                const BTBAlignLayout &Align,
                                ArrayRef<uint64_t> BranchOffsets,
                                BTBNopPools &Pools, uint64_t &RestartWinBase,
                                uint64_t WinBase, size_t AlignIdx,
                                unsigned N, unsigned FetchLineSize) {
  const uint64_t SectionSize = Asm.getSectionAddressSize(Sec);
  size_t BranchIdx = 0;
  size_t NopOnIdx = 0;
  size_t NopOffIdx = 0;

  while (BranchIdx < BranchOffsets.size() && BranchOffsets[BranchIdx] < WinBase)
    ++BranchIdx;

  bool Changed = false;
  while (WinBase < SectionSize) {
    uint64_t SegmentEnd = AlignIdx < Align.PaddingStart.size()
                             ? Align.PaddingStart[AlignIdx]
                             : SectionSize;
    uint64_t AlignEnd = AlignIdx < Align.RegionEnd.size()
                            ? Align.RegionEnd[AlignIdx]
                            : SectionSize;

    for (uint64_t w = WinBase; w < SegmentEnd; w += FetchLineSize) {
      uint64_t WinEnd = w + FetchLineSize;

      LLVM_DEBUG(dbgs() << "  window [" << format_hex(w, 8) << ", "
                        << format_hex(WinEnd, 8) << ") segmentEnd="
                        << format_hex(SegmentEnd, 8) << " alignEnd="
                        << format_hex(AlignEnd, 8) << "\n");

      for (size_t i = BranchIdx; i < BranchOffsets.size(); ++i) {
        uint64_t Off = BranchOffsets[i];
        if (Off >= WinEnd)
          break;
        if (Off >= w)
          LLVM_DEBUG(dbgs() << "    branch @ " << format_hex(Off, 8) << "\n");
      }

      size_t Count = 0;
      size_t CriticalIdx = BranchOffsets.size();
      for (size_t i = BranchIdx;
           i < BranchOffsets.size() && BranchOffsets[i] < SegmentEnd; ++i) {
        ++Count;
        if (Count == N + 1) {
          CriticalIdx = i;
          break;
        }
      }

      uint64_t BCritical = CriticalIdx < BranchOffsets.size()
                               ? BranchOffsets[CriticalIdx]
                               : UINT64_MAX;

      LLVM_DEBUG({
        dbgs() << "    BCritical=";
        if (BCritical != UINT64_MAX)
          dbgs() << format_hex(BCritical, 8);
        else
          dbgs() << "none";
        dbgs() << "      BranchIdx=" << BranchIdx
               << " BranchOffsets.size()=" << BranchOffsets.size() << "\n";
      });

      if (tryShrinkNBFList(Pools.On, NopOnIdx, WinBase, WinEnd, SegmentEnd,
                           BCritical) ||
          tryShrinkNBFList(Pools.Off, NopOffIdx, WinBase, WinEnd, SegmentEnd,
                           BCritical)) {
        LLVM_DEBUG(dbgs() << "    made change, break to re-layout\n");
        RestartWinBase = w;
        return true;
      }

      advanceIndicesPastWindow(BranchOffsets, BranchIdx, WinEnd, Pools,
                               NopOnIdx, NopOffIdx, w);
    }

    ++AlignIdx;
    WinBase = AlignEnd;
  }
  return Changed;
}

/// Drop NBF fragments with \c NumBytes == 0 after shrink.
static bool removeEmptyNBFFragments(MCSection &Sec) {
  SmallVector<MCNopsBesideBranchFragment *, 8> ToRemove;
  for (MCFragment &F : Sec) {
    auto *NBF = dyn_cast<MCNopsBesideBranchFragment>(&F);
    if (NBF && NBF->getNumBytes() == 0)
      ToRemove.push_back(NBF);
  }
  LLVM_DEBUG(dbgs() << "  remove " << ToRemove.size()
                    << " empty NOP fragment(s)\n");
  for (MCNopsBesideBranchFragment *NBF : ToRemove)
    Sec.removeFragment(*NBF);
  return !ToRemove.empty();
}

/// MC \c shrinkSection hook: collect branches/NBF, optionally shrink one pool per
/// call, set \p RestartWinBase for the next layout iteration.
bool shrinkSection(MCAssembler &Asm, MCSection &Sec, uint64_t &RestartWinBase) {
  LLVM_DEBUG(dbgs() << "shrinkSection: section " << Sec.getName()
                    << " RestartWinBase=" << format_hex(RestartWinBase, 8)
                    << "\n");

  if (!EnableRISCVBTBFetchLineBranchRelaxation || BTBFetchLineSize == 0 ||
      BTBMaxBranchesPerFetchLine == 0) {
    LLVM_DEBUG(dbgs() << "  early return (disabled or N=0)\n");
    return false;
  }

  if (ShrinkDiagTrackedSection != &Sec) {
    ShrinkDiagTrackedSection = &Sec;
    ShrinkDiagCallIdx = 0;
  }
  const bool EmitPreShrinkPairwiseDebug = (ShrinkDiagCallIdx == 0);
  ++ShrinkDiagCallIdx;

  const unsigned N = BTBMaxBranchesPerFetchLine;
  const unsigned FetchLineSize = BTBFetchLineSize;

  uint64_t SectionSize = Asm.getSectionAddressSize(Sec);
  if (RestartWinBase >= SectionSize)
    RestartWinBase = 0;

  BTBAlignLayout Align = collectBTBAlignLayout(Asm, Sec);
  auto [SegmentStart, AlignIdx] = findSegmentStart(Align, RestartWinBase);
  uint64_t WinBase = advanceWinBasePastPadding(Align, AlignIdx, RestartWinBase);

  SmallVector<uint64_t, 64> BranchOffsets;
  collectBTBBranchOffsets(Asm, Sec, SegmentStart, BranchOffsets);
  BTBNopPools Pools;
  collectBTBNopPools(Asm, Sec, SegmentStart, Pools);

  LLVM_DEBUG(dbgs() << "  collected: branches=" << BranchOffsets.size()
                    << " nopsOnExecPath=" << Pools.On.size()
                    << " nopsOffExecPath=" << Pools.Off.size()
                    << " segmentStart=" << format_hex(SegmentStart, 8)
                    << " winBase=" << format_hex(WinBase, 8) << "\n");

  if (EmitPreShrinkPairwiseDebug)
    emitPreShrinkPairwiseDebug(BranchOffsets, N, FetchLineSize);

  if (RISCVBTBShrinkSectionDiagOnly)
    return false;

  bool Changed = runShrinkWindowLoop(Asm, Sec, Align, BranchOffsets, Pools,
                                     RestartWinBase, WinBase, AlignIdx, N,
                                     FetchLineSize);
  Changed |= removeEmptyNBFFragments(Sec);

  LLVM_DEBUG(dbgs() << "  shrinkSection returns " << (Changed ? "true" : "false")
                    << "\n");
  return Changed;
}

void verifyBTBFetchLineBranchLimits(const MCAssembler &Asm) {
  if (!EnableRISCVBTBFetchLineBranchRelaxation || BTBFetchLineSize == 0 ||
      BTBMaxBranchesPerFetchLine == 0)
    return;

  for (const MCSection &Sec : Asm) {
    if (!Sec.isText())
      continue;

    SmallVector<BTBBranchInfo, 64> Branches;
    collectBTBBranches(Asm, Sec, Branches);
    BTBAlignLayout Align = collectBTBAlignLayout(Asm, Sec);

    if (Branches.empty())
      continue;

    uint64_t SectionSize = Asm.getSectionAddressSize(Sec);
    auto ItLow = Branches.begin();
    size_t AlignIdx = 0;
    uint64_t WinBase = 0;

    while (WinBase < SectionSize) {
      uint64_t NextAlignEnd = AlignIdx < Align.RegionEnd.size()
                                  ? Align.RegionEnd[AlignIdx]
                                  : SectionSize;

      for (uint64_t w = WinBase; w < NextAlignEnd; w += BTBFetchLineSize) {
        uint64_t WinEnd = w + BTBFetchLineSize;
        auto ItHigh = llvm::lower_bound(
            llvm::make_range(ItLow, Branches.end()), WinEnd,
            [](const BTBBranchInfo &B, uint64_t End) {
              return B.Offset < End;
            });
        unsigned Count = ItHigh - ItLow;
        if (Count > BTBMaxBranchesPerFetchLine) {
          SmallString<512> Buf;
          raw_svector_ostream OS(Buf);
          OS << "BTB fetch-line limit violated in section '" << Sec.getName()
             << "': [0x";
          OS.write_hex(w);
          OS << ", 0x";
          OS.write_hex(WinEnd);
          OS << ") has " << Count << " branches (max "
             << BTBMaxBranchesPerFetchLine << "). Branches: ";
          for (auto I = ItLow; I != ItHigh; ++I) {
            if (I != ItLow)
              OS << ", ";
            OS << "0x";
            OS.write_hex(I->Offset);
            OS << "[";
            for (uint8_t B = 0; B != I->Size; ++B) {
              if (B != 0)
                OS << " ";
              OS << format_hex_no_prefix(I->Bytes[B], 2, /*Upper=*/false);
            }
            OS << "]";
          }
          if (RISCVBTBFetchLinePostLayoutViolationAsWarning)
            Asm.getContext().reportWarning(SMLoc(), Buf);
          else
            Asm.getContext().reportError(SMLoc(), Buf);
        }
        ItLow = ItHigh;
      }
      ++AlignIdx;
      WinBase = NextAlignEnd;
    }

    LLVM_DEBUG({
      dbgs() << "verifyBTBFetchLineBranchLimits: section '" << Sec.getName()
             << "' all branches (count=" << Branches.size() << "):\n";
      for (const BTBBranchInfo &BI : Branches) {
        dbgs() << "  @0x" << format_hex(BI.Offset, 8)
               << " sz=" << unsigned(BI.Size) << " enc";
        for (uint8_t I = 0; I != BI.Size; ++I)
          dbgs() << ' ' << format_hex_no_prefix(BI.Bytes[I], 2,
                                                /*Upper=*/false);
        dbgs() << '\n';
      }
    });
  }
}

} // namespace RISCVBTB
} // namespace llvm
