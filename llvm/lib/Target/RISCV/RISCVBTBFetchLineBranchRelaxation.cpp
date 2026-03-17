//===-- RISCVBTBFetchLineBranchRelaxation.cpp - BTB fetch line branch relax-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contributed by zardeth (shuyun.liang@rivai.ai), RiVAI Technologies Ltd.
//===----------------------------------------------------------------------===//
//
// This pass limits the number of branch instructions per fetch line (e.g. 64B)
// to avoid BTB (Branch Target Buffer) slot overflow. When a fetch-line-sized
// window contains more than MaxBranches branches, it inserts NOPs so that
// either (1) after the nearest unconditional jump above the overflow branch,
// or (2) after the previous branch (the branch before the overflow branch),
// to push the overflow branch into the next fetch line. Prefer (1) to avoid
// adding executed instructions. (2) uses NOPs after a conditional branch to
// help CPU recovery on branch misprediction.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/RISCV/RISCVBTBFetchLineBranchRelaxation.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-btb-fetchline-opt"

STATISTIC(NumNopBytesAfterUncondJump,
          "Number of NOP bytes inserted after unconditional jump");
STATISTIC(NumNopBytesAfterCondJump,
          "Number of NOP bytes inserted after conditional branch (fallback)");

#define RISCV_BTB_FETCHLINE_BRANCH_RELAXATION_NAME "RISC-V BTB Fetch Line Branch Relaxation"

static cl::opt<std::string> PrintBTBFetchLineLayout(
    "riscv-btb-fetchline-print-layout",
    cl::desc("In debug build, print each instruction and offset in layout order "
             "for the function whose name matches this option (e.g. =foo)"),
    cl::init(""), cl::Hidden);

namespace {

class RISCVBTBFetchLineBranchRelaxation : public MachineFunctionPass {
public:
  static char ID;

  RISCVBTBFetchLineBranchRelaxation() : MachineFunctionPass(ID) {
    initializeRISCVBTBFetchLineBranchRelaxationPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // Branch relaxation may create new blocks and edges.
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_BTB_FETCHLINE_BRANCH_RELAXATION_NAME;
  }

private:
  const RISCVInstrInfo *TII = nullptr;
  const RISCVSubtarget *STI = nullptr;

  /// Minimum instruction alignment (bytes). RISC-V has 2-byte min.
  static constexpr unsigned InstrAlign = 2;

  /// Return true if MI is a branch for BTB purposes.
  bool isBranch(const MachineInstr &MI) const;

  /// Return minimum possible size in bytes for layout modeling. For
  /// instructions whose size is not accurately known (e.g. pseudos that expand
  /// to compressible form in MC), returns the minimum so the modeled address
  /// space is conservative and we may insert more NOPs but guarantee the BTB
  /// constraint.
  unsigned getMinInstSizeInBytes(const MachineInstr &MI) const;

  /// Insert NOP(s) of total Size bytes at InsertPos in MBB. Returns number of
  /// bytes inserted.
  unsigned insertNops(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPos, unsigned Size,
                     DebugLoc DL);

  /// Build branch list (MI*, offset) in layout order and return total size.
  uint64_t buildLayout(const MachineFunction &MF,
                      SmallVectorImpl<std::pair<MachineInstr *, uint64_t>> &Branches);

  /// Fix overflow in window [Base, Base+FetchLineSize): insert NOPs only
  /// within this window. Returns true if any change was made. Updates Branches
  /// offsets for instructions at or after the insertion point.
  bool fixWindow(MachineFunction &MF, uint64_t Base,
                 SmallVectorImpl<std::pair<MachineInstr *, uint64_t>> &Branches);

  /// Run one round: build layout, scan fetch-line windows, fix first overflow
  /// if any. Returns true if a fix was applied (caller should repeat).
  bool runOneRound(MachineFunction &MF, unsigned &Round);

  //=== Branch range relaxation (maintain jump range after NOP insert)

  struct BasicBlockInfo {
    unsigned Offset = 0;
    unsigned Size = 0;
    unsigned postOffset(const MachineBasicBlock &MBB) const;
  };

  SmallVector<BasicBlockInfo, 16> BlockInfo;
  MachineBasicBlock *TrampolineInsertionPoint = nullptr;
  SmallDenseSet<std::pair<MachineBasicBlock *, MachineBasicBlock *>>
      RelaxedUnconditionals;
  std::unique_ptr<RegScavenger> RS;
  LivePhysRegs LiveRegs;
  MachineFunction *RelaxMF = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetMachine *TM = nullptr;

  void scanBlockInfo(MachineFunction &MF);
  uint64_t computeBlockSizeRelax(const MachineBasicBlock &MBB) const;
  unsigned getInstrOffsetRelax(const MachineInstr &MI) const;
  void adjustBlockOffsets(MachineBasicBlock &Start);
  void adjustBlockOffsets(MachineBasicBlock &Start,
                          MachineFunction::iterator End);
  bool isBlockInRange(const MachineInstr &MI,
                      const MachineBasicBlock &DestBB) const;
  MachineBasicBlock *createNewBlockAfter(MachineBasicBlock &OrigMBB);
  MachineBasicBlock *createNewBlockAfter(MachineBasicBlock &OrigMBB,
                                        const BasicBlock *BB);
  MachineBasicBlock *splitBlockBeforeInstr(MachineInstr &MI,
                                          MachineBasicBlock *DestBB);
  bool fixupConditionalBranch(MachineInstr &MI);
  bool fixupUnconditionalBranch(MachineInstr &MI);
  bool relaxBranchInstructions();
  /// Run relaxation until no out-of-range branches remain. Returns true if any
  /// fixup was applied.
  bool runRelaxationRounds(MachineFunction &MF);
};

} // end anonymous namespace

char RISCVBTBFetchLineBranchRelaxation::ID = 0;

INITIALIZE_PASS(RISCVBTBFetchLineBranchRelaxation, DEBUG_TYPE,
  RISCV_BTB_FETCHLINE_BRANCH_RELAXATION_NAME, false, false)

bool RISCVBTBFetchLineBranchRelaxation::isBranch(const MachineInstr &MI) const {
  if (MI.isMetaInstruction())
    return false;
  return MI.isBranch() || MI.isCall() || MI.isReturn();
}

unsigned RISCVBTBFetchLineBranchRelaxation::getMinInstSizeInBytes(
    const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();
  // Pseudos that expand in MC to a compressible instruction: use minimum size
  // so layout is conservative.
  switch (Opcode) {
  case RISCV::PseudoBR:
  case RISCV::PseudoBRIND:
  case RISCV::PseudoBRINDNonX7:
  case RISCV::PseudoBRINDX7:
  case RISCV::PseudoCALLIndirect:
  case RISCV::PseudoCALLIndirectNonX7:
  case RISCV::PseudoCALLIndirectX7:
  case RISCV::PseudoRET:
  case RISCV::PseudoTAILIndirect:
  case RISCV::PseudoTAILIndirectNonX7:
  case RISCV::PseudoTAILIndirectX7:
    // Expand to JAL or JALR; with Zca can be C_J/c.jr/c.jalr (2), else 4.
    return STI->hasStdExtZca() ? 2 : 4;
  default: {
    unsigned Size = TII->getInstSizeInBytes(MI);
    // For BTB layout use minimum size. With Zca, BEQ/BNE/JAL/JALR may compress
    // to 2 bytes in MC; assume 2 so we insert enough NOPs for the BTB constraint.
    if (STI->hasStdExtZca() && isBranch(MI) && Size > 2 &&
        (Opcode == RISCV::BEQ || Opcode == RISCV::BNE || Opcode == RISCV::JAL ||
         Opcode == RISCV::JALR))
      Size = 2;
    return Size;
  }
  }
}

unsigned RISCVBTBFetchLineBranchRelaxation::insertNops(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPos,
    unsigned Size, DebugLoc DL) {
  const unsigned NopSize = STI->hasStdExtZca() ? 2 : 4;
  unsigned Inserted = 0;
  while (Inserted < Size) {
    if (STI->hasStdExtZca())
      BuildMI(MBB, InsertPos, DL, TII->get(RISCV::C_NOP));
    else
      BuildMI(MBB, InsertPos, DL, TII->get(RISCV::ADDI), RISCV::X0)
          .addReg(RISCV::X0)
          .addImm(0);
    Inserted += NopSize;
  }
  return Inserted;
}

uint64_t RISCVBTBFetchLineBranchRelaxation::buildLayout(
    const MachineFunction &MF,
    SmallVectorImpl<std::pair<MachineInstr *, uint64_t>> &Branches) {
  Branches.clear();
  uint64_t Offset = 0;
  bool PrintLayout = !PrintBTBFetchLineLayout.empty() &&
                     MF.getName() == PrintBTBFetchLineLayout;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction())
        continue;
      LLVM_DEBUG(
        if (PrintLayout) {
          dbgs() << "    " << format_hex(Offset, 8) << ": ";
          MI.print(dbgs(), TII);
          if (isBranch(MI))
            dbgs() << " (branch)";
        }
      );
      if (isBranch(MI))
        Branches.push_back({const_cast<MachineInstr *>(&MI), Offset});
      Offset += getMinInstSizeInBytes(MI);
    }
  }
  LLVM_DEBUG(dbgs() << "  buildLayout: total size=" << format_hex(Offset, 8)
                    << " branches=" << Branches.size() << "\n");
  return Offset;
}

bool RISCVBTBFetchLineBranchRelaxation::fixWindow(
    MachineFunction &MF, uint64_t Base,
    SmallVectorImpl<std::pair<MachineInstr *, uint64_t>> &Branches) {
  const unsigned FetchLineSize = BTBFetchLineSize;
  const unsigned MaxBranches = BTBMaxBranchesPerFetchLine;

  // Collect branches in this window [Base, Base+FetchLineSize).
  SmallVector<std::pair<MachineInstr *, uint64_t>, 8> InWindow;
  for (const auto &P : Branches) {
    if (P.second >= Base && P.second < Base + FetchLineSize)
      InWindow.push_back(P);
  }
  if (InWindow.size() <= MaxBranches)
    return false;

  // Overflow: the (MaxBranches+1)-th branch is the "trigger".
  uint64_t OverflowOffset = InWindow[MaxBranches].second;
  unsigned NopBytesNeeded =
      alignTo((Base + FetchLineSize) - OverflowOffset, InstrAlign);
  if (NopBytesNeeded == 0)
    return false;

  LLVM_DEBUG(dbgs() << "  fixWindow: overflow at window ["
                    << format_hex(Base, 8) << ", "
                    << format_hex(Base + FetchLineSize, 8)
                    << ") count=" << InWindow.size()
                    << " overflow branch offset=" << format_hex(OverflowOffset, 8)
                    << " NopBytesNeeded=" << NopBytesNeeded << "\n");

  // Find the nearest unconditional jump above OverflowBranch that lies inside
  // the current window [Base, Base+FetchLineSize). We only insert NOPs within
  // this window.
  MachineInstr *UncondJmp = nullptr;
  uint64_t UncondJmpOffset = 0;
  for (size_t j = Branches.size(); j > 0; --j) {
    const auto &P = Branches[j - 1];
    if (P.second >= OverflowOffset)
      continue;
    if (P.second < Base)
      break; // Outside current window; no need to look further back.
    if (P.first->getDesc().isUnconditionalBranch() ||
        P.first->getDesc().isReturn()) {
      UncondJmp = P.first;
      UncondJmpOffset = P.second;
      break;
    }
  }

  auto insertAfter = [this, NopBytesNeeded](MachineInstr *U) {
    MachineBasicBlock &MBB = *U->getParent();
    MachineBasicBlock::iterator Next = std::next(U->getIterator());
    DebugLoc DL = U->getDebugLoc();
    insertNops(MBB, Next, NopBytesNeeded, DL);
  };

  auto updateBranchesAfterInsert = [NopBytesNeeded](uint64_t InsertEnd,
      SmallVectorImpl<std::pair<MachineInstr *, uint64_t>> &BrList) {
    for (auto &P : BrList)
      if (P.second >= InsertEnd)
        P.second += NopBytesNeeded;
  };

  if (UncondJmp) {
    LLVM_DEBUG({
      dbgs() << "    insert after UncondJmp at offset="
             << format_hex(UncondJmpOffset, 8) << " (within window): ";
      UncondJmp->print(dbgs(), TII);
      dbgs() << "\n";
    });
    NumNopBytesAfterUncondJump += NopBytesNeeded;
    uint64_t InsertEnd = UncondJmpOffset + getMinInstSizeInBytes(*UncondJmp);
    insertAfter(UncondJmp);
    updateBranchesAfterInsert(InsertEnd, Branches);
    return true;
  }

  // No unconditional jump in window: insert after the previous branch (the
  // branch immediately before the overflow branch). NOPs after a conditional
  // branch help CPU recovery on branch misprediction.
  MachineInstr *PrevBranch = InWindow[MaxBranches - 1].first;
  uint64_t PrevBranchOffset = InWindow[MaxBranches - 1].second;
  LLVM_DEBUG({
    dbgs() << "    insert after previous branch at offset="
           << format_hex(PrevBranchOffset, 8) << ": ";
    PrevBranch->print(dbgs(), TII);
    dbgs() << "\n";
  });
  NumNopBytesAfterCondJump += NopBytesNeeded;
  insertAfter(PrevBranch);
  uint64_t InsertEnd = PrevBranchOffset + getMinInstSizeInBytes(*PrevBranch);
  updateBranchesAfterInsert(InsertEnd, Branches);
  return true;
}

bool RISCVBTBFetchLineBranchRelaxation::runOneRound(MachineFunction &MF,
                                                   unsigned &Round) {
  SmallVector<std::pair<MachineInstr *, uint64_t>, 256> Branches;
  buildLayout(MF, Branches);

  LLVM_DEBUG(dbgs() << "  round " << ++Round << " branches=" << Branches.size()
                    << "\n");

  // Sliding window: for each branch, use its offset as the start of a
  // BTBFetchLineSize-sized window. If that window exceeds
  // BTBMaxBranchesPerFetchLine branches, fix it by inserting NOPs only
  // within the window.
  for (const auto &Br : Branches) {
    uint64_t Base = Br.second;

    // Count branches in window [Base, Base+BTBFetchLineSize).
    unsigned Count = 0;
    for (const auto &P : Branches) {
      if (P.second >= Base && P.second < Base + BTBFetchLineSize)
        ++Count;
    }

    LLVM_DEBUG(dbgs() << "  window [" << format_hex(Base, 8) << ", "
                      << format_hex(Base + BTBFetchLineSize, 8)
                      << ") branches=" << Count
                      << (Count > BTBMaxBranchesPerFetchLine ? " OVERFLOW" : "")
                      << "\n");

    if (Count <= BTBMaxBranchesPerFetchLine)
      continue;

    if (fixWindow(MF, Base, Branches)) {
      LLVM_DEBUG(dbgs() << "  fixWindow made change, restart round\n");
      return true;
    }
  }
  return false;
}

//=== BasicBlockInfo::postOffset (for branch relaxation)

unsigned RISCVBTBFetchLineBranchRelaxation::BasicBlockInfo::postOffset(
    const MachineBasicBlock &MBB) const {
  const unsigned PO = Offset + Size;
  const Align Alignment = MBB.getAlignment();
  const Align ParentAlign = MBB.getParent()->getAlignment();
  if (Alignment <= ParentAlign)
    return alignTo(PO, Alignment);
  return alignTo(PO, Alignment) + Alignment.value() - ParentAlign.value();
}

//=== Branch relaxation: scan and block offset

void RISCVBTBFetchLineBranchRelaxation::scanBlockInfo(MachineFunction &MF) {
  BlockInfo.clear();
  BlockInfo.resize(MF.getNumBlockIDs());

  TrampolineInsertionPoint = nullptr;
  RelaxedUnconditionals.clear();

  for (MachineBasicBlock &MBB : MF) {
    BlockInfo[MBB.getNumber()].Size = computeBlockSizeRelax(MBB);
    if (MBB.getSectionID() != MBBSectionID::ColdSectionID)
      TrampolineInsertionPoint = &MBB;
  }

  adjustBlockOffsets(*MF.begin());
}

uint64_t RISCVBTBFetchLineBranchRelaxation::computeBlockSizeRelax(
    const MachineBasicBlock &MBB) const {
  uint64_t Size = 0;
  for (const MachineInstr &MI : MBB)
    Size += TII->getInstSizeInBytes(MI);
  return Size;
}

unsigned RISCVBTBFetchLineBranchRelaxation::getInstrOffsetRelax(
    const MachineInstr &MI) const {
  const MachineBasicBlock *MBB = MI.getParent();
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;
  for (MachineBasicBlock::const_iterator I = MBB->begin(); &*I != &MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->getInstSizeInBytes(*I);
  }
  return Offset;
}

void RISCVBTBFetchLineBranchRelaxation::adjustBlockOffsets(
    MachineBasicBlock &Start) {
  adjustBlockOffsets(Start, RelaxMF->end());
}

void RISCVBTBFetchLineBranchRelaxation::adjustBlockOffsets(
    MachineBasicBlock &Start, MachineFunction::iterator End) {
  unsigned PrevNum = Start.getNumber();
  for (auto &MBB :
       make_range(std::next(MachineFunction::iterator(Start)), End)) {
    unsigned Num = MBB.getNumber();
    BlockInfo[Num].Offset = BlockInfo[PrevNum].postOffset(MBB);
    PrevNum = Num;
  }
}

bool RISCVBTBFetchLineBranchRelaxation::isBlockInRange(
    const MachineInstr &MI, const MachineBasicBlock &DestBB) const {
  int64_t BrOffset = getInstrOffsetRelax(MI);
  int64_t DestOffset = BlockInfo[DestBB.getNumber()].Offset;
  const MachineBasicBlock *SrcBB = MI.getParent();

  if (TII->isBranchOffsetInRange(
          MI.getOpcode(),
          SrcBB->getSectionID() != DestBB.getSectionID()
              ? TM->getMaxCodeSize()
              : DestOffset - BrOffset))
    return true;
  LLVM_DEBUG(dbgs() << "Out of range branch to destination "
                    << printMBBReference(DestBB) << " from "
                    << printMBBReference(*MI.getParent()) << " to "
                    << format_hex(DestOffset, 8) << " offset "
                    << format_hex(DestOffset - BrOffset, 8) << '\t' << MI);
  return false;
}

//=== Branch relaxation: block creation

MachineBasicBlock *RISCVBTBFetchLineBranchRelaxation::createNewBlockAfter(
    MachineBasicBlock &OrigMBB) {
  return createNewBlockAfter(OrigMBB, OrigMBB.getBasicBlock());
}

MachineBasicBlock *RISCVBTBFetchLineBranchRelaxation::createNewBlockAfter(
    MachineBasicBlock &OrigMBB, const BasicBlock *BB) {
  MachineBasicBlock *NewBB = RelaxMF->CreateMachineBasicBlock(BB);
  RelaxMF->insert(++OrigMBB.getIterator(), NewBB);

  NewBB->setSectionID(OrigMBB.getSectionID());
  NewBB->setIsEndSection(OrigMBB.isEndSection());
  OrigMBB.setIsEndSection(false);

  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());
  return NewBB;
}

MachineBasicBlock *RISCVBTBFetchLineBranchRelaxation::splitBlockBeforeInstr(
    MachineInstr &MI, MachineBasicBlock *DestBB) {
  MachineBasicBlock *OrigBB = MI.getParent();

  MachineBasicBlock *NewBB =
      RelaxMF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  RelaxMF->insert(++OrigBB->getIterator(), NewBB);

  NewBB->setSectionID(OrigBB->getSectionID());
  NewBB->setIsEndSection(OrigBB->isEndSection());
  OrigBB->setIsEndSection(false);

  NewBB->splice(NewBB->end(), OrigBB, MI.getIterator(), OrigBB->end());
  TII->insertUnconditionalBranch(*OrigBB, NewBB, DebugLoc());

  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  NewBB->transferSuccessors(OrigBB);
  OrigBB->addSuccessor(NewBB);
  OrigBB->addSuccessor(DestBB);

  OrigBB->updateTerminator(NewBB);

  BlockInfo[OrigBB->getNumber()].Size = computeBlockSizeRelax(*OrigBB);
  BlockInfo[NewBB->getNumber()].Size = computeBlockSizeRelax(*NewBB);
  adjustBlockOffsets(*OrigBB, std::next(NewBB->getIterator()));

  if (TRI->trackLivenessAfterRegAlloc(*RelaxMF))
    computeAndAddLiveIns(LiveRegs, *NewBB);

  return NewBB;
}

//=== Branch relaxation: fixup conditional branch

bool RISCVBTBFetchLineBranchRelaxation::fixupConditionalBranch(
    MachineInstr &MI) {
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  MachineBasicBlock *NewBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;

  auto insertUncondBranch = [this, DL](MachineBasicBlock *MBB,
                                       MachineBasicBlock *DestBB) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int NewBrSize = 0;
    TII->insertUnconditionalBranch(*MBB, DestBB, DL, &NewBrSize);
    BBSize += NewBrSize;
  };
  auto insertBranch = [this, DL](MachineBasicBlock *MBB, MachineBasicBlock *TBB,
                                 MachineBasicBlock *FBB,
                                 SmallVectorImpl<MachineOperand> &Cond) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int NewBrSize = 0;
    TII->insertBranch(*MBB, TBB, FBB, Cond, DL, &NewBrSize);
    BBSize += NewBrSize;
  };
  auto removeBranch = [this](MachineBasicBlock *MBB) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int RemovedSize = 0;
    TII->removeBranch(*MBB, &RemovedSize);
    BBSize -= RemovedSize;
  };

  auto updateOffsetAndLiveness = [this](MachineBasicBlock *NewBB) {
    assert(NewBB != nullptr);
    adjustBlockOffsets(*std::prev(NewBB->getIterator()),
                       std::next(NewBB->getIterator()));
    if (TRI->trackLivenessAfterRegAlloc(*RelaxMF))
      computeAndAddLiveIns(LiveRegs, *NewBB);
  };

  bool Fail =
      static_cast<const TargetInstrInfo *>(TII)->analyzeBranch(*MBB, TBB, FBB,
                                                               Cond);
  assert(!Fail && "branches to be relaxed must be analyzable");
  (void)Fail;

  if (MBB->getSectionID() != TBB->getSectionID() &&
      TBB->getSectionID() == MBBSectionID::ColdSectionID &&
      TrampolineInsertionPoint != nullptr) {
    NewBB =
        createNewBlockAfter(*TrampolineInsertionPoint, MBB->getBasicBlock());

    if (isBlockInRange(MI, *NewBB)) {
      insertUncondBranch(NewBB, TBB);
      MBB->replaceSuccessor(TBB, NewBB);
      NewBB->addSuccessor(TBB);
      removeBranch(MBB);
      insertBranch(MBB, NewBB, FBB, Cond);
      TrampolineInsertionPoint = NewBB;
      updateOffsetAndLiveness(NewBB);
      return true;
    }

    TrampolineInsertionPoint->setIsEndSection(NewBB->isEndSection());
    RelaxMF->erase(NewBB);
    NewBB = nullptr;
  }

  bool ReversedCond = !TII->reverseBranchCondition(Cond);
  if (ReversedCond) {
    if (FBB && isBlockInRange(MI, *FBB)) {
      removeBranch(MBB);
      insertBranch(MBB, FBB, TBB, Cond);
      return true;
    }
    if (FBB) {
      NewBB = createNewBlockAfter(*MBB);
      insertUncondBranch(NewBB, FBB);
      MBB->replaceSuccessor(FBB, NewBB);
      NewBB->addSuccessor(FBB);
      updateOffsetAndLiveness(NewBB);
    }

    MachineBasicBlock &NextBB = *std::next(MachineFunction::iterator(MBB));
    removeBranch(MBB);
    insertBranch(MBB, &NextBB, TBB, Cond);
    return true;
  }

  if (!FBB)
    FBB = &(*std::next(MachineFunction::iterator(MBB)));

  NewBB = createNewBlockAfter(*MBB);
  insertUncondBranch(NewBB, TBB);

  MBB->replaceSuccessor(TBB, NewBB);
  NewBB->addSuccessor(TBB);
  removeBranch(MBB);
  insertBranch(MBB, NewBB, FBB, Cond);

  updateOffsetAndLiveness(NewBB);
  return true;
}

//=== Branch relaxation: fixup unconditional branch

bool RISCVBTBFetchLineBranchRelaxation::fixupUnconditionalBranch(
    MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();
  unsigned OldBrSize = TII->getInstSizeInBytes(MI);
  MachineBasicBlock *DestBB = TII->getBranchDestBlock(MI);

  int64_t DestOffset = BlockInfo[DestBB->getNumber()].Offset;
  int64_t SrcOffset = getInstrOffsetRelax(MI);

  assert(!TII->isBranchOffsetInRange(
      MI.getOpcode(),
      MBB->getSectionID() != DestBB->getSectionID()
          ? TM->getMaxCodeSize()
          : DestOffset - SrcOffset));

  BlockInfo[MBB->getNumber()].Size -= OldBrSize;

  MachineBasicBlock *BranchBB = MBB;

  if (!MBB->empty()) {
    BranchBB = createNewBlockAfter(*MBB);

    for (const MachineBasicBlock *Succ : MBB->successors()) {
      for (const MachineBasicBlock::RegisterMaskPair &LiveIn : Succ->liveins())
        BranchBB->addLiveIn(LiveIn);
    }
    BranchBB->sortUniqueLiveIns();
    BranchBB->addSuccessor(DestBB);
    MBB->replaceSuccessor(DestBB, BranchBB);
    if (TrampolineInsertionPoint == MBB)
      TrampolineInsertionPoint = BranchBB;
  }

  DebugLoc DL = MI.getDebugLoc();
  MI.eraseFromParent();

  MachineBasicBlock *RestoreBB =
      createNewBlockAfter(RelaxMF->back(), DestBB->getBasicBlock());
  std::prev(RestoreBB->getIterator())->setIsEndSection(RestoreBB->isEndSection());
  RestoreBB->setIsEndSection(false);

  TII->insertIndirectBranch(
      *BranchBB, *DestBB, *RestoreBB, DL,
      BranchBB->getSectionID() != DestBB->getSectionID()
          ? TM->getMaxCodeSize()
          : DestOffset - SrcOffset,
      RS.get());

  BlockInfo[BranchBB->getNumber()].Size = computeBlockSizeRelax(*BranchBB);
  adjustBlockOffsets(*MBB, std::next(BranchBB->getIterator()));

  if (!RestoreBB->empty()) {
    if (MBB->getSectionID() == MBBSectionID::ColdSectionID &&
        DestBB->getSectionID() != MBBSectionID::ColdSectionID) {
      MachineBasicBlock *NewBB =
          createNewBlockAfter(*TrampolineInsertionPoint);
      TII->insertUnconditionalBranch(*NewBB, DestBB, DebugLoc());
      BlockInfo[NewBB->getNumber()].Size = computeBlockSizeRelax(*NewBB);
      adjustBlockOffsets(*TrampolineInsertionPoint,
                         std::next(NewBB->getIterator()));
      TrampolineInsertionPoint = NewBB;
      BranchBB->replaceSuccessor(DestBB, NewBB);
      NewBB->addSuccessor(DestBB);
      DestBB = NewBB;
    }

    assert(!DestBB->isEntryBlock() &&
           "restore block cannot be placed before entry");
    MachineBasicBlock *PrevBB = &*std::prev(DestBB->getIterator());
    if (auto *FT = PrevBB->getLogicalFallThrough()) {
      assert(FT == DestBB);
      TII->insertUnconditionalBranch(*PrevBB, FT, DebugLoc());
      BlockInfo[PrevBB->getNumber()].Size = computeBlockSizeRelax(*PrevBB);
    }
    RelaxMF->splice(DestBB->getIterator(), RestoreBB->getIterator());
    RestoreBB->addSuccessor(DestBB);
    BranchBB->replaceSuccessor(DestBB, RestoreBB);
    if (TRI->trackLivenessAfterRegAlloc(*RelaxMF))
      computeAndAddLiveIns(LiveRegs, *RestoreBB);
    BlockInfo[RestoreBB->getNumber()].Size = computeBlockSizeRelax(*RestoreBB);
    adjustBlockOffsets(*PrevBB, DestBB->getIterator());
    RestoreBB->setSectionID(DestBB->getSectionID());
    RestoreBB->setIsBeginSection(DestBB->isBeginSection());
    DestBB->setIsBeginSection(false);
    RelaxedUnconditionals.insert({BranchBB, RestoreBB});
  } else {
    RelaxMF->erase(RestoreBB);
    RelaxedUnconditionals.insert({BranchBB, DestBB});
  }

  return true;
}

//=== Branch relaxation: main loop

bool RISCVBTBFetchLineBranchRelaxation::relaxBranchInstructions() {
  bool Changed = false;

  for (MachineBasicBlock &MBB : *RelaxMF) {
    MachineBasicBlock::iterator Last = MBB.getLastNonDebugInstr();
    if (Last == MBB.end())
      continue;

    if (Last->isUnconditionalBranch()) {
      if (MachineBasicBlock *DestBB = TII->getBranchDestBlock(*Last)) {
        if (!isBlockInRange(*Last, *DestBB) && !TII->isTailCall(*Last) &&
            !RelaxedUnconditionals.contains({&MBB, DestBB})) {
          fixupUnconditionalBranch(*Last);
          Changed = true;
        }
      }
    }

    MachineBasicBlock::iterator Next;
    for (MachineBasicBlock::iterator J = MBB.getFirstTerminator();
         J != MBB.end(); J = Next) {
      Next = std::next(J);
      MachineInstr &MI = *J;

      if (!MI.isConditionalBranch())
        continue;
      if (MI.getOpcode() == TargetOpcode::FAULTING_OP)
        continue;

      MachineBasicBlock *DestBB = TII->getBranchDestBlock(MI);
      if (!DestBB || isBlockInRange(MI, *DestBB))
        continue;

      if (Next != MBB.end() && Next->isConditionalBranch()) {
        splitBlockBeforeInstr(*Next, DestBB);
      } else {
        fixupConditionalBranch(MI);
      }
      Changed = true;
      Next = MBB.getFirstTerminator();
    }
  }

  if (Changed)
    adjustBlockOffsets(RelaxMF->front());

  return Changed;
}

bool RISCVBTBFetchLineBranchRelaxation::runRelaxationRounds(
    MachineFunction &MF) {
  RelaxMF = &MF;
  TRI = MF.getSubtarget().getRegisterInfo();
  TM = &MF.getTarget();
  // RISC-V insertIndirectBranch requires a non-null RegScavenger.
  RS = std::make_unique<RegScavenger>();

  MF.RenumberBlocks();
  scanBlockInfo(MF);

  bool MadeChange = false;
  while (relaxBranchInstructions())
    MadeChange = true;

  BlockInfo.clear();
  RelaxedUnconditionals.clear();
  RelaxMF = nullptr;
  return MadeChange;
}

bool RISCVBTBFetchLineBranchRelaxation::runOnMachineFunction(
    MachineFunction &MF) {
  if (!EnableRISCVBTBFetchLineBranchRelaxation || BTBFetchLineSize == 0 ||
      BTBMaxBranchesPerFetchLine == 0)
    return false;

  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = static_cast<const RISCVInstrInfo *>(STI->getInstrInfo());

  LLVM_DEBUG(dbgs() << "RISCVBTBFetchLineBranchRelaxation: " << MF.getName()
                    << " FetchLineSize=" << BTBFetchLineSize
                    << " MaxBranches=" << BTBMaxBranchesPerFetchLine << "\n");

  bool MadeChange = false;
  unsigned Round = 0;
  for (;;) {
    bool BTBChanged = false;
    while (runOneRound(MF, Round))
      BTBChanged = true;

    MadeChange |= BTBChanged;
    if (!BTBChanged)
      break; // No NOPs inserted; no need to re-check branch ranges.

    bool RelaxChanged = runRelaxationRounds(MF);
    if (!RelaxChanged)
      break; // Branch ranges OK; no need to re-run BTB.
    MadeChange = true; // Relaxation changed layout; run BTB again.
  }

  // Align function *address* to fetch-line boundary so our function-relative
  // windows [0, FetchLineSize), [FetchLineSize, 2*FetchLineSize), ... match
  // actual fetch lines in the binary. AsmPrinter emits alignment before the
  // function (padding before the function label), so no NOPs inside the body.
  if (BTBFetchLineSize > 0 && isPowerOf2_32(BTBFetchLineSize)) {
    Align FnAlign(BTBFetchLineSize);
    if (MF.getAlignment() < FnAlign)
      MF.setAlignment(FnAlign);
  }

  LLVM_DEBUG(dbgs() << "RISCVBTBFetchLineBranchRelaxation: " << MF.getName()
                    << " done MadeChange=" << MadeChange << "\n");
  return MadeChange;
}

FunctionPass *llvm::createRISCVBTBFetchLineBranchRelaxationPass() {
  return new RISCVBTBFetchLineBranchRelaxation();
}
