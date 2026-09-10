//===-- ShrinkWrapping.cpp - Reduce spills/restores of callee-saved regs --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a shrink wrapping variant of prolog/epilog insertion:
// - Spills and restores of callee-saved registers (CSRs) are placed in the
//   machine CFG to tightly surround their uses so that execution paths that
//   do not use CSRs do not pay the spill/restore penalty.
//
// - Avoiding placment of spills/restores in loops: if a CSR is used inside a
//   loop the spills are placed in the loop preheader, and restores are
//   placed in the loop exit nodes (the successors of loop _exiting_ nodes).
//
// - Covering paths without CSR uses:
//   If a region in a CFG uses CSRs and has multiple entry and/or exit points,
//   the use info for the CSRs inside the region is propagated outward in the
//   CFG to ensure validity of the spill/restore placements. This decreases
//   the effectiveness of shrink wrapping but does not require edge splitting
//   in the machine CFG.
//
// This shrink wrapping implementation uses an iterative analysis to determine
// which basic blocks require spills and restores for CSRs.
//
// This pass uses MachineDominators and MachineLoopInfo. Loop information
// is used to prevent placement of callee-saved register spills/restores
// in the bodies of loops.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "shrink-wrapping"

#include "llvm/CodeGen/ShrinkWrapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include <sstream>

using namespace llvm;

STATISTIC(numSRReduced, "Number of CSR spills+restores reduced.");
STATISTIC(NumFunc, "Number of functions");
STATISTIC(NumShrinkFrame, "Number of functions with frame shrunk to non-entry");
STATISTIC(NumNotSaveOrRestore,
          "Number of cases, in which one of the sets are empty.");
STATISTIC(NumCandidates, "Number of shrink-wrapping candidates");
STATISTIC(
    NumFuncWithSplitting,
    "Number of functions, for which we managed to split Save/Restore points");

static cl::opt<cl::boolOrDefault>
    EnableShrinkWrappingOpt("enable-shrink-wrapping", cl::Hidden,
                            cl::desc("enable the shrink-wrapping pass"));
static cl::opt<bool> EnableShrinkWrappingSplitOpt(
    "enable-shrink-wrapping-into-multiple-points", cl::init(false), cl::Hidden,
    cl::desc("enable splitting of the save and restore blocks if possible"));

static cl::opt<bool> DisableShrinkFrame(
    "disable-shrink-frame", cl::init(false), cl::Hidden,
    cl::desc("Disable moving stack frame setup from entry to a later block"));

// Debugging level for shrink wrapping.
enum ShrinkWrappingDebugLevel { Disabled, BasicInfo, Iterations, Details };

static cl::opt<enum ShrinkWrappingDebugLevel> ShrinkWrappingDebugging(
    "shrink-wrapping-dbg", cl::Hidden,
    cl::desc("Print shrink wrapping debugging information"),
    cl::values(clEnumValN(Disabled, "disable", "disable debug output"),
               clEnumValN(BasicInfo, "basic", "print basic DF sets"),
               clEnumValN(Iterations, "iters",
                          "print SR sets for each iteration"),
               clEnumValN(Details, "details", "print all DF sets")));

struct AuxGraphNode {
  std::string Name;
  MachineBasicBlock *MatchMBB;
  std::vector<AuxGraphNode *> Successors;
  std::vector<AuxGraphNode *> Predecessors;

  AuxGraphNode(MachineBasicBlock *MBB) : MatchMBB(MBB) {}

  void addSuccessor(AuxGraphNode *S) {
    Successors.push_back(S);
    S->Predecessors.push_back(this);
  }

  std::string getBBName(const MachineBasicBlock *MBB) {
    if (!MBB)
      return "";

    if (MBB->getBasicBlock())
      return MBB->getBasicBlock()->getName().str();

    std::ostringstream name;
    name << "_MBB_" << MBB->getNumber();
    return name.str();
  }

  auto succ_size() { return Successors.size(); }

  auto succ_begin() { return Successors.begin(); }

  auto succ_end() { return Successors.end(); }

  auto pred_size() { return Predecessors.size(); }

  auto pred_begin() { return Predecessors.begin(); }

  auto pred_end() { return Predecessors.end(); }

  void setupName() {
    if (MatchMBB) {
      Name = getBBName(MatchMBB);
    } else {
      assert(succ_size() == 1 && "Auxillary node has more than one successor!");
      assert(pred_size() == 1 &&
             "Auxillary node has more than one predecessor!");
      MachineBasicBlock *Succ = (*succ_begin())->MatchMBB;
      MachineBasicBlock *Pred = (*pred_begin())->MatchMBB;
      assert(Succ && "Auxillary node should have real successor!");
      assert(Pred && "Auxillary node should have real predecessor!");
      std::string SuccName = getBBName(Succ);
      std::string PredName = getBBName(Pred);
      Name = PredName + "->" + SuccName;
    }
  }
};

struct AuxGraph {
  std::vector<AuxGraphNode *> Nodes;
  AuxGraphNode *Entry = nullptr;

  void addNode(AuxGraphNode *N) {
    Nodes.push_back(N);
    if (!Entry)
      Entry = N;
  }

  AuxGraphNode *getNode(MachineBasicBlock *MBB) {
    auto It =
        find_if(Nodes, [&MBB](AuxGraphNode *N) { return N->MatchMBB == MBB; });
    if (It != Nodes.end())
      return *It;
    return nullptr;
  }
};

namespace llvm {
template <> struct GraphTraits<AuxGraphNode *> {
  using NodeRef = AuxGraphNode *;
  using ChildIteratorType = typename std::vector<AuxGraphNode *>::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return N->Successors.begin();
  }

  static ChildIteratorType child_end(NodeRef N) { return N->Successors.end(); }
};

// Reverse traversal traits
template <> struct GraphTraits<Inverse<AuxGraphNode *>> {
  using NodeRef = AuxGraphNode *;
  using ChildIteratorType = typename std::vector<AuxGraphNode *>::iterator;

  static NodeRef getEntryNode(Inverse<NodeRef> N) { return N.Graph; }

  // NOTE: You need a way to get predecessors!
  // Simple approach: store them explicitly or compute on demand
  static ChildIteratorType child_begin(NodeRef N) {
    // In real code, maintain a Predecessors vector or compute via graph scan
    return N->Predecessors.begin(); // Assume Predecessors exists
  }

  static ChildIteratorType child_end(NodeRef N) {
    return N->Predecessors.end();
  }
};

template <>
struct GraphTraits<AuxGraph *> : public GraphTraits<AuxGraphNode *> {
  using NodeRef = AuxGraphNode *;
  using nodes_iterator = typename std::vector<AuxGraphNode *>::iterator;

  static NodeRef getEntryNode(AuxGraph *G) { return G->Entry; }

  static nodes_iterator nodes_begin(AuxGraph *G) { return G->Nodes.begin(); }

  static nodes_iterator nodes_end(AuxGraph *G) { return G->Nodes.end(); }

  static unsigned size(AuxGraph *G) { return G->Nodes.size(); }
};
} // namespace llvm

namespace {

/// Class to determine where the safe point to insert the
/// prologue and epilogue are.
/// Unlike the paper from Fred C. Chow, PLDI'88, that introduces the
/// shrink-wrapping term for prologue/epilogue placement, this pass
/// does not rely on expensive data-flow analysis. Instead we use the
/// dominance properties and loop information to decide which point
/// are safe for such insertion.
class ShrinkWrappingImpl {
  /// Hold callee-saved information.
  RegisterClassInfo RCI;
  MachineDominatorTree *MDT = nullptr;
  MachinePostDominatorTree *MPDT = nullptr;

  /// Hash table, mapping register with its corresponding spill and restore
  /// basic block.
  //  DenseMap<Register, std::pair<MachineBasicBlock *, MachineBasicBlock *>>
  //    SavedRegs;
  typedef SparseBitVector<> CSRegSet;
  typedef DenseMap<MachineBasicBlock *, CSRegSet> CSRegBlockMap;
  typedef DenseMap<AuxGraphNode *, CSRegSet> CSRegNodeMap;
  CSRegSet UsedCSRegs;
  CSRegBlockMap CSRUsed;
  CSRegNodeMap AnticIn, AnticOut;
  CSRegNodeMap AvailIn, AvailOut;
  CSRegNodeMap CSRSave;
  CSRegNodeMap CSRRestore;

  /// Current opcode for frame setup.
  unsigned FrameSetupOpcode = ~0u;

  /// Current opcode for frame destroy.
  unsigned FrameDestroyOpcode = ~0u;

  /// Stack pointer register, used by llvm.{savestack,restorestack}
  Register SP;

  class SaveRestorePoints {
    llvm::SaveRestorePoints SRPoints;

  public:
    llvm::SaveRestorePoints &get() { return SRPoints; }

    void set(llvm::SaveRestorePoints &Rhs) { SRPoints = std::move(Rhs); }

    void clear() { SRPoints.clear(); }

    bool areMultiple() const { return SRPoints.size() > 1; }

    MachineBasicBlock *getFirst() {
      return SRPoints.empty() ? nullptr : SRPoints.begin()->first;
    }

    void insert(const std::pair<MachineBasicBlock *,
                                std::vector<CalleeSavedInfo>> &Point) {
      SRPoints.insert(Point);
    }

    void insert(
        std::pair<MachineBasicBlock *, std::vector<CalleeSavedInfo>> &&Point) {
      SRPoints.insert(Point);
    }

    /// Record that \p Reg is saved/restored in \p MBB. When
    /// \p SaveRestoreBlockList is given, newly seen blocks are appended to it so
    /// that the prolog/epilog placement sees every save/restore point.
    void insertReg(Register Reg, MachineBasicBlock *MBB,
                   std::vector<MachineBasicBlock *> *SaveRestoreBlockList) {
      assert(MBB && "MBB is nullptr");
      if (SRPoints.contains(MBB)) {
        SRPoints[MBB].push_back(CalleeSavedInfo(Reg));
        return;
      }
      std::vector CSInfos{CalleeSavedInfo(Reg)};
      SRPoints.insert(std::make_pair(MBB, CSInfos));
      if (SaveRestoreBlockList)
        SaveRestoreBlockList->push_back(MBB);
    }

    void print(raw_ostream &OS, const TargetRegisterInfo *TRI) const {
      for (auto [BB, CSIV] : SRPoints) {
        OS << printMBBReference(*BB) << ": ";
        for (auto &CSI : CSIV) {
          OS << printReg(CSI.getReg(), TRI) << " ";
        }
        OS << "\n";
      }
    }

    void dump(const TargetRegisterInfo *TRI) const { print(dbgs(), TRI); }
  };

  /// Class, wrapping hash table contained safe points, found for register spill
  /// mapped to the list of corresponding registers. Register spill will be
  /// inserted before the first instruction in this basic block.
  SaveRestorePoints SavePoints;

  /// Class, wrapping hash table contained safe points, found for register
  /// restore mapped to the list of corresponding registers. Register restore
  /// will be inserted before the first terminator instruction in this basic
  /// block.
  SaveRestorePoints RestorePoints;

  std::vector<MachineBasicBlock *> SaveBlocks;
  std::vector<MachineBasicBlock *> RestoreBlocks;

  MachineBasicBlock *Prolog = nullptr;
  MachineBasicBlock *Epilog = nullptr;

  // Entry and return blocks of the current function.
  SmallVector<MachineBasicBlock *, 4> ReturnBlocks;

  /// Hold the loop information. Used to determine if Save and Restore
  /// are in the same loop.
  MachineLoopInfo *MLI = nullptr;

  // Emit remarks.
  MachineOptimizationRemarkEmitter *ORE = nullptr;

  /// Entry block.
  MachineBasicBlock *Entry = nullptr;

  bool HasFastExitPath = false;

  using SetOfRegs = SmallSetVector<unsigned, 16>;

  /// Registers that need to be saved for the current function.
  mutable SparseBitVector<> CurrentCSRsBitVec;

  AuxGraph AuxillaryCFG;

  /// Current MachineFunction.
  MachineFunction *MachineFunc = nullptr;

  /// Is `true` for the block numbers where we assume possible stack accesses
  /// or computation of stack-relative addresses on any CFG path including the
  /// block itself. Is `false` for basic blocks where we can guarantee the
  /// opposite. False positives won't lead to incorrect analysis results,
  /// therefore this approach is fair.
  BitVector StackAddressUsedBlockInfo;

  bool useOrDefCSR(const MachineInstr &MI, RegScavenger *RS,
                   CSRegSet *RegsToSave) const;

  /// Check if \p MI uses or defines a frame index.
  /// If this is the case, this means \p MI must happen
  /// after Save and before Restore.
  bool useOrDefFI(const MachineInstr &MI, RegScavenger *RS,
                  bool StackAddressUsed) const;

  void createAuxillaryCFG();

  bool calculateSets(MachineFunction &MF,
                     const ReversePostOrderTraversal<MachineBasicBlock *> &RPOT,
                     RegScavenger *RS);

  std::string getBasicBlockName(const MachineBasicBlock *MBB) {
    if (!MBB)
      return "";

    if (MBB->getBasicBlock())
      return MBB->getBasicBlock()->getName().str();

    std::ostringstream name;
    name << "_MBB_" << MBB->getNumber();
    return name.str();
  }

  const CSRegSet &getCurrentCSRsBitVec(RegScavenger *RS) const {
    if (CurrentCSRsBitVec.empty()) {
      BitVector SavedRegs;
      const TargetFrameLowering *TFI =
          MachineFunc->getSubtarget().getFrameLowering();

      TFI->determineCalleeSaves(*MachineFunc, SavedRegs, RS);
      for (unsigned Bit : SavedRegs.set_bits()) {
        CurrentCSRsBitVec.set(Bit);
      }
    }
    return CurrentCSRsBitVec;
  }

  void propagateUsesAroundLoop(MachineBasicBlock *MBB, MachineLoop *LP);
  void verifySpillRestorePlacement();
  void dumpUsed(MachineBasicBlock *MBB);

  void dumpSet(const CSRegSet &s);

  void dumpAllUsed();

  void dumpSets(AuxGraphNode *Node);

  void dumpAllSets();

  void dumpSRSets();

  std::string stringifyCSRegSet(const CSRegSet &s);

  MachineBasicBlock *splitEdge(MachineBasicBlock *Pred,
                               MachineBasicBlock *Succ);

  /// Split edges for save/restore points that need their own block. Returns
  /// true if any new block was created, in which case the dominator trees are
  /// stale and must be recomputed.
  bool setupCFG();

  void setupSaveRestorePoints();

  /// Determine whether a block requires a stack frame (has calls, FI uses,
  /// or defs of ra/FP/SP).
  bool blockNeedsFrame(const MachineBasicBlock &MBB) const;

  /// Move frame setup from entry to the nearest common dominator of all
  /// frame-requiring blocks. Returns true if a non-entry Prolog was chosen.
  bool runShrinkFrame(MachineFunction &MF);

  void findFastExitPath();

  void clearAnticAvailSets(RegScavenger *RS);

  void clearAllSets(RegScavenger *RS);

  void dumpSets1(AuxGraphNode *MBB);

  bool calcAnticInOut(AuxGraphNode *Node);

  bool calcAvailInOut(AuxGraphNode *Node);

  void calculateAnticAvail(MachineFunction &Fn, RegScavenger *RS);

  bool calcSpillPlacements(AuxGraphNode *Node,
                           SmallVectorImpl<AuxGraphNode *> &blks,
                           CSRegNodeMap &prevSpills);

  bool calcRestorePlacements(AuxGraphNode *Node,
                             SmallVectorImpl<AuxGraphNode *> &blks,
                             CSRegNodeMap &prevRestores);

  void placeSpillsAndRestores(MachineFunction &Fn);

  /// Initialize the pass for \p MF.
  void init(MachineFunction &MF, RegScavenger *RS) {
    const TargetSubtargetInfo &Subtarget = MF.getSubtarget();
    const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
    FrameSetupOpcode = TII.getCallFrameSetupOpcode();
    FrameDestroyOpcode = TII.getCallFrameDestroyOpcode();
    SP = Subtarget.getTargetLowering()->getStackPointerRegisterToSaveRestore();
    RCI.runOnMachineFunction(MF);
    SavePoints.clear();
    RestorePoints.clear();
    Prolog = nullptr;
    Epilog = nullptr;
    SaveBlocks.clear();
    RestoreBlocks.clear();
    Entry = &MF.front();
    CurrentCSRsBitVec.clear();
    MachineFunc = &MF;
    clearAllSets(RS);
    HasFastExitPath = false;
    ++NumFunc;
  }

public:
  ShrinkWrappingImpl(MachineDominatorTree *MDT, MachinePostDominatorTree *MPDT,
                     MachineBlockFrequencyInfo *MBFI, MachineLoopInfo *MLI,
                     MachineOptimizationRemarkEmitter *ORE)
      : MDT(MDT), MPDT(MPDT), MLI(MLI), ORE(ORE) {}

  /// Check if shrink wrapping is enabled for this target and function.
  static bool isShrinkWrappingEnabled(const MachineFunction &MF);

  bool run(MachineFunction &MF);
};

class ShrinkWrappingLegacy : public MachineFunctionPass {
public:
  static char ID;

  ShrinkWrappingLegacy() : MachineFunctionPass(ID) {
    initializeShrinkWrappingLegacyPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  StringRef getPassName() const override { return "Shrink Wrapping analysis"; }

  /// Perform the shrink-wrapping analysis and update
  /// the MachineFrameInfo attached to \p MF with the results.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char ShrinkWrappingLegacy::ID = 0;

char &llvm::ShrinkWrappingID = ShrinkWrappingLegacy::ID;

INITIALIZE_PASS_BEGIN(ShrinkWrappingLegacy, DEBUG_TYPE, "Shrink Wrapping Pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(ShrinkWrappingLegacy, DEBUG_TYPE, "Shrink Wrapping Pass",
                    false, false)

static bool giveUpWithRemarks(MachineOptimizationRemarkEmitter *ORE,
                              StringRef RemarkName, StringRef RemarkMessage,
                              const DiagnosticLocation &Loc,
                              const MachineBasicBlock *MBB) {
  ORE->emit([&]() {
    return MachineOptimizationRemarkMissed(DEBUG_TYPE, RemarkName, Loc, MBB)
           << RemarkMessage;
  });

  LLVM_DEBUG(dbgs() << RemarkMessage << '\n');
  return false;
}

void ShrinkWrappingImpl::createAuxillaryCFG() {
  std::set<MachineBasicBlock *> Visited;
  SmallVector<MachineBasicBlock *, 8> WorkList;
  WorkList.push_back(Entry);
  Visited.insert(Entry);
  auto *EntryNode = new AuxGraphNode(Entry);
  AuxillaryCFG.addNode(EntryNode);
  while (!WorkList.empty()) {
    MachineBasicBlock *MBB = WorkList.pop_back_val();
    AuxGraphNode *Node = nullptr;
    if (MBB == Entry) {
      Node = EntryNode;
    } else {
      Node = AuxillaryCFG.getNode(MBB);
    }
    assert(Node && "Node is null!");
    for (MachineBasicBlock *Succ : MBB->successors()) {
      auto *AuxNode = new AuxGraphNode(nullptr);
      Node->addSuccessor(AuxNode);
      AuxGraphNode *SuccNode = nullptr;
      if (!Visited.insert(Succ).second) {
        SuccNode = AuxillaryCFG.getNode(Succ);
      } else {
        SuccNode = new AuxGraphNode(Succ);
        AuxillaryCFG.addNode(SuccNode);
        WorkList.push_back(Succ);
      }
      AuxNode->addSuccessor(SuccNode);
      AuxillaryCFG.addNode(AuxNode);
    }
  }

  for (auto &Node : AuxillaryCFG.Nodes)
    Node->setupName();
}

/// findFastExitPath - debugging method used to detect functions
/// with at least one path from the entry block to a return block
/// directly or which has a very small number of edges.
///
void ShrinkWrappingImpl::findFastExitPath() {
  if (!Entry)
    return;
  // Fina a path from EntryBlock to any return block that does not branch:
  //        Entry
  //          |     ...
  //          v      |
  //         B1<-----+
  //          |
  //          v
  //       Return
  for (MachineBasicBlock::succ_iterator SI = Entry->succ_begin(),
                                        SE = Entry->succ_end();
       SI != SE; ++SI) {
    MachineBasicBlock *SUCC = *SI;

    // Assume positive, disprove existence of fast path.
    HasFastExitPath = true;

    // Check the immediate successors.
    if (SUCC->isReturnBlock()) {
      if (ShrinkWrappingDebugging >= BasicInfo)
        dbgs() << "Fast exit path: " << printMBBReference(*Entry) << "->"
               << printMBBReference(*SUCC) << "\n";
      break;
    }
    // Traverse df from SUCC, look for a branch block.
    std::string exitPath = getBasicBlockName(SUCC);
    for (df_iterator<MachineBasicBlock *> BI = df_begin(SUCC),
                                          BE = df_end(SUCC);
         BI != BE; ++BI) {
      MachineBasicBlock *SBB = *BI;
      // Reject paths with branch nodes.
      if (SBB->succ_size() > 1) {
        HasFastExitPath = false;
        break;
      }
      exitPath += "->" + getBasicBlockName(SBB);
    }
    if (HasFastExitPath) {
      if (ShrinkWrappingDebugging >= BasicInfo)
        dbgs() << "Fast exit path: " << getBasicBlockName(Entry) << "->"
               << exitPath << "\n";
      break;
    }
  }
}

/// verifySpillRestorePlacement - check the current spill/restore
/// sets for safety. Attempt to find spills without restores or
/// restores without spills.
/// Spills: walk df from each MBB in spill set ensuring that
///         all CSRs spilled at MMBB are restored on all paths
///         from MBB to all exit blocks.
/// Restores: walk idf from each MBB in restore set ensuring that
///           all CSRs restored at MBB are spilled on all paths
///           reaching MBB.
///
void ShrinkWrappingImpl::verifySpillRestorePlacement() {
  for (CSRegNodeMap::iterator BI = CSRSave.begin(), BE = CSRSave.end();
       BI != BE; ++BI) {
    MachineBasicBlock *MBB = BI->first->MatchMBB;
    AuxGraphNode *Node = BI->first;
    CSRegSet spilled = BI->second;
    CSRegSet restored;

    if (spilled.empty())
      continue;

    LLVM_DEBUG(dbgs() << "SAVE[" << getBasicBlockName(MBB)
                      << "] = " << stringifyCSRegSet(spilled) << "  RESTORE["
                      << getBasicBlockName(MBB)
                      << "] = " << stringifyCSRegSet(CSRRestore[Node]) << "\n");

    if (CSRRestore[Node].intersects(spilled)) {
      restored |= (CSRRestore[Node] & spilled);
    }

    // Walk depth first from MBB to find restores of all CSRs spilled at MBB:
    // we must find restores for all spills w/no intervening spills on all
    // paths from MBB to all return blocks.
    for (df_iterator<MachineBasicBlock *> BI = df_begin(MBB), BE = df_end(MBB);
         BI != BE; ++BI) {
      MachineBasicBlock *SBB = *BI;
      if (SBB == MBB)
        continue;
      // Stop when we encounter spills of any CSRs spilled at MBB that
      // have not yet been seen to be restored.
      AuxGraphNode *SBBNode = AuxillaryCFG.getNode(SBB);
      if (CSRSave[SBBNode].intersects(spilled) &&
          !restored.contains(CSRSave[SBBNode] & spilled))
        break;
      // Collect the CSRs spilled at MBB that are restored
      // at this DF successor of MBB.

      if (CSRRestore[SBBNode].intersects(spilled))
        restored |= (CSRRestore[SBBNode] & spilled);
      // If we are at a retun block, check that the restores
      // we have seen so far exhaust the spills at MBB, then
      // reset the restores.
      if (SBB->isReturnBlock()) {
        if (restored != spilled) {
          CSRegSet notRestored = (spilled - restored);
          LLVM_DEBUG(dbgs() << MachineFunc->getName() << ": "
                            << stringifyCSRegSet(notRestored) << " spilled at "
                            << getBasicBlockName(MBB)
                            << " are never restored on path to return "
                            << getBasicBlockName(SBB) << "\n");
        }
        SparseBitVector<> SBBRestored = CSRRestore[SBBNode] & spilled;
        for (unsigned Bit : SBBRestored)
          restored.reset(Bit);
      }
    }
  }

  // Check restore placements.
  for (CSRegNodeMap::iterator BI = CSRRestore.begin(), BE = CSRRestore.end();
       BI != BE; ++BI) {
    MachineBasicBlock *MBB = BI->first->MatchMBB;
    AuxGraphNode *Node = BI->first;
    CSRegSet restored = BI->second;
    CSRegSet spilled;

    if (restored.empty())
      continue;

    LLVM_DEBUG(dbgs() << "SAVE[" << getBasicBlockName(MBB)
                      << "] = " << stringifyCSRegSet(CSRSave[Node])
                      << "  RESTORE[" << getBasicBlockName(MBB)
                      << "] = " << stringifyCSRegSet(restored) << "\n");

    if (CSRSave[Node].intersects(restored)) {
      spilled |= (CSRSave[Node] & restored);
    }
    // Walk inverse depth first from MBB to find spills of all
    // CSRs restored at MBB:
    for (idf_iterator<MachineBasicBlock *> BI = idf_begin(MBB),
                                           BE = idf_end(MBB);
         BI != BE; ++BI) {
      MachineBasicBlock *PBB = *BI;
      if (PBB == MBB)
        continue;
      AuxGraphNode *PBBNode = AuxillaryCFG.getNode(PBB);
      // Stop when we encounter restores of any CSRs restored at MBB that
      // have not yet been seen to be spilled.
      if (CSRRestore[PBBNode].intersects(restored) &&
          !spilled.contains(CSRRestore[PBBNode] & restored))
        break;
      // Collect the CSRs restored at MBB that are spilled
      // at this DF predecessor of MBB.
      if (CSRSave[PBBNode].intersects(restored))
        spilled |= (CSRSave[PBBNode] & restored);
    }
    if (spilled != restored) {
      CSRegSet notSpilled = (restored - spilled);
      LLVM_DEBUG(dbgs() << MachineFunc->getName() << ": "
                        << stringifyCSRegSet(notSpilled) << " restored at "
                        << printMBBReference(*MBB) << " are never spilled\n");
    }
  }
}

std::string ShrinkWrappingImpl::stringifyCSRegSet(const CSRegSet &s) {
  const TargetRegisterInfo *TRI = MachineFunc->getSubtarget().getRegisterInfo();
  std::ostringstream srep;

  if (s.empty()) {
    srep << "[]";
    return srep.str();
  }

  srep << "[";
  for (unsigned Reg : s) {
    srep << ",";
    srep << TRI->getName(Reg);
  }
  srep << "]";
  return srep.str();
}

void ShrinkWrappingImpl::dumpSet(const CSRegSet &s) {
  LLVM_DEBUG(dbgs() << stringifyCSRegSet(s) << "\n");
}

void ShrinkWrappingImpl::dumpUsed(MachineBasicBlock *MBB) {
  LLVM_DEBUG({
    if (MBB)
      dbgs() << "CSRUsed[" << getBasicBlockName(MBB)
             << "] = " << stringifyCSRegSet(CSRUsed[MBB]) << "\n";
  });
}

void ShrinkWrappingImpl::dumpAllUsed() {
  for (MachineFunction::iterator MBBI = MachineFunc->begin(),
                                 MBBE = MachineFunc->end();
       MBBI != MBBE; ++MBBI) {
    dumpUsed(&(*MBBI));
  }
}

void ShrinkWrappingImpl::dumpSets(AuxGraphNode *Node) {
  LLVM_DEBUG({
    if (Node) {
      CSRegSet Used;
      if (Node->MatchMBB)
        Used = CSRUsed[Node->MatchMBB];
      dbgs() << Node->Name << " | " << stringifyCSRegSet(Used) << " | "
             << stringifyCSRegSet(AnticIn[Node]) << " | "
             << stringifyCSRegSet(AnticOut[Node]) << " | "
             << stringifyCSRegSet(AvailIn[Node]) << " | "
             << stringifyCSRegSet(AvailOut[Node]) << "\n";
    }
  });
}

void ShrinkWrappingImpl::dumpSets1(AuxGraphNode *Node) {
  LLVM_DEBUG({
    if (Node) {
      CSRegSet Used;
      if (Node->MatchMBB)
        Used = CSRUsed[Node->MatchMBB];
      dbgs() << Node->Name << " | " << stringifyCSRegSet(Used) << " | "
             << stringifyCSRegSet(AnticIn[Node]) << " | "
             << stringifyCSRegSet(AnticOut[Node]) << " | "
             << stringifyCSRegSet(AvailIn[Node]) << " | "
             << stringifyCSRegSet(AvailOut[Node]) << " | "
             << stringifyCSRegSet(CSRSave[Node]) << " | "
             << stringifyCSRegSet(CSRRestore[Node]) << "\n";
    }
  });
}

void ShrinkWrappingImpl::dumpAllSets() {
  for (auto &Node : AuxillaryCFG.Nodes) {
    dumpSets1(Node);
  }
}

void ShrinkWrappingImpl::dumpSRSets() {
  LLVM_DEBUG({
    for (auto &Node : AuxillaryCFG.Nodes) {
      if (!CSRSave[Node].empty()) {
        dbgs() << "SAVE[" << Node->Name
               << "] = " << stringifyCSRegSet(CSRSave[Node]);
        dbgs() << '\n';
      }

      if (!CSRRestore[Node].empty())
        dbgs() << "RESTORE[" << Node->Name
               << "] = " << stringifyCSRegSet(CSRRestore[Node]) << "\n";
    }
  });
}

// Initialize shrink wrapping DFA sets, called before iterations.
void ShrinkWrappingImpl::clearAnticAvailSets(RegScavenger *RS) {
  for (auto &Node : AuxillaryCFG.Nodes) {
    AnticIn[Node] = getCurrentCSRsBitVec(RS);
    AvailOut[Node] = getCurrentCSRsBitVec(RS);
  }
  AnticOut.clear();
  AvailIn.clear();
}

// Clear all sets constructed by shrink wrapping.
void ShrinkWrappingImpl::clearAllSets(RegScavenger *RS) {
  ReturnBlocks.clear();
  clearAnticAvailSets(RS);
  UsedCSRegs.clear();
  CSRUsed.clear();
  CSRSave.clear();
  CSRRestore.clear();
}

/// propagateUsesAroundLoop - copy used register info from MBB to all blocks
/// of the loop given by LP and its parent loops. This prevents spills/restores
/// from being placed in the bodies of loops.
///
void ShrinkWrappingImpl::propagateUsesAroundLoop(MachineBasicBlock *MBB,
                                                 MachineLoop *LP) {
  if (!MBB || !LP)
    return;

  std::vector<MachineBasicBlock *> loopBlocks = LP->getBlocks();
  for (unsigned i = 0, e = loopBlocks.size(); i != e; ++i) {
    MachineBasicBlock *LBB = loopBlocks[i];
    if (LBB == MBB)
      continue;
    if (CSRUsed[LBB].contains(CSRUsed[MBB]))
      continue;
    CSRUsed[LBB] |= CSRUsed[MBB];
  }
}

/// calcAnticInOut - calculate the anticipated in/out reg sets
/// for the given MBB by looking forward in the MCFG at MBB's
/// successors.
///
bool ShrinkWrappingImpl::calcAnticInOut(AuxGraphNode *Node) {
  bool changed = false;

  // AnticOut[MBB] = INTERSECT(AnticIn[S] for S in SUCCESSORS(MBB))
  SmallVector<AuxGraphNode *, 4> successors;
  for (auto SI = Node->succ_begin(), SE = Node->succ_end(); SI != SE; ++SI) {
    AuxGraphNode *SUCC = *SI;
    if (SUCC != Node)
      successors.push_back(SUCC);
  }

  unsigned i = 0, e = successors.size();
  if (i != e) {
    CSRegSet prevAnticOut = AnticOut[Node];
    AuxGraphNode *SUCC = successors[i];

    AnticOut[Node] = AnticIn[SUCC];
    for (++i; i != e; ++i) {
      SUCC = successors[i];
      AnticOut[Node] &= AnticIn[SUCC];
    }
    if (prevAnticOut != AnticOut[Node])
      changed = true;
  }

  // AnticIn[MBB] = UNION(CSRUsed[MBB], AnticOut[MBB]);
  CSRegSet prevAnticIn = AnticIn[Node];
  CSRegSet UsedRegSet;
  if (Node->MatchMBB)
    UsedRegSet = CSRUsed[Node->MatchMBB];

  AnticIn[Node] = UsedRegSet | AnticOut[Node];
  if (prevAnticIn != AnticIn[Node])
    changed = true;
  return changed;
}

/// calcAvailInOut - calculate the available in/out reg sets
/// for the given MBB by looking backward in the MCFG at MBB's
/// predecessors.
///
bool ShrinkWrappingImpl::calcAvailInOut(AuxGraphNode *Node) {
  bool changed = false;

  // AvailIn[MBB] = INTERSECT(AvailOut[P] for P in PREDECESSORS(MBB))
  SmallVector<AuxGraphNode *, 4> predecessors;
  for (auto PI = Node->pred_begin(), PE = Node->pred_end(); PI != PE; ++PI) {
    AuxGraphNode *PRED = *PI;
    if (PRED != Node)
      predecessors.push_back(PRED);
  }

  unsigned i = 0, e = predecessors.size();
  if (i != e) {
    CSRegSet prevAvailIn = AvailIn[Node];
    AuxGraphNode *PRED = predecessors[i];

    AvailIn[Node] = AvailOut[PRED];
    for (++i; i != e; ++i) {
      PRED = predecessors[i];
      AvailIn[Node] &= AvailOut[PRED];
    }
    if (prevAvailIn != AvailIn[Node])
      changed = true;
  }

  // AvailOut[MBB] = UNION(CSRUsed[MBB], AvailIn[MBB]);
  CSRegSet prevAvailOut = AvailOut[Node];
  CSRegSet UsedRegSet;
  if (Node->MatchMBB)
    UsedRegSet = CSRUsed[Node->MatchMBB];
  AvailOut[Node] = UsedRegSet | AvailIn[Node];
  if (prevAvailOut != AvailOut[Node])
    changed = true;
  return changed;
}

/// calculateAnticAvail - build the sets anticipated and available
/// registers in the MCFG of the current function iteratively,
/// doing a combined forward and backward analysis.
///
void ShrinkWrappingImpl::calculateAnticAvail(MachineFunction &Fn,
                                             RegScavenger *RS) {
  // Initialize data flow sets.
  clearAnticAvailSets(RS);

  // Calculate Antic{In,Out} and Avail{In,Out} iteratively on the MCFG.
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;
    ++iterations;
    for (auto &Node : AuxillaryCFG.Nodes) {
      // Calculate anticipated in, out regs at MBB from
      // anticipated at successors of MBB.
      changed |= calcAnticInOut(Node);

      // Calculate available in, out regs at MBB from
      // available at predecessors of MBB.
      changed |= calcAvailInOut(Node);
    }
  }

  LLVM_DEBUG({
    if (ShrinkWrappingDebugging >= Details) {
      dbgs() << "-----------------------------------------------------------\n"
             << " Antic/Avail Sets:\n"
             << "-----------------------------------------------------------\n"
             << "iterations = " << iterations << "\n"
             << "-----------------------------------------------------------\n"
             << "MBB | USED | ANTIC_IN | ANTIC_OUT | AVAIL_IN | AVAIL_OUT\n"
             << "-----------------------------------------------------------\n";

      for (auto &Node : AuxillaryCFG.Nodes)
        dumpSets(Node);

      dbgs() << "-----------------------------------------------------------\n";
    }
  });
}

bool ShrinkWrappingImpl::useOrDefCSR(const MachineInstr &MI, RegScavenger *RS,
                                     CSRegSet *RegsToSave) const {
  const MachineFunction *MF = MI.getParent()->getParent();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg()) {
      // Ignore instructions like DBG_VALUE which don't read/def the register.
      if (!MO.isDef() && !MO.readsReg())
        continue;
      Register PhysReg = MO.getReg();
      if (!PhysReg)
        continue;
      assert(PhysReg.isPhysical() && "Unallocated register?!");
      // The stack pointer is not normally described as a callee-saved register
      // in calling convention definitions, so we need to watch for it
      // separately. An SP mentioned by a call instruction, we can ignore,
      // though, as it's harmless and we do not want to effectively disable tail
      // calls by forcing the restore point to post-dominate them.
      // PPC's LR is also not normally described as a callee-saved register in
      // calling convention definitions, so we need to watch for it, too. An LR
      // mentioned implicitly by a return (or "branch to link register")
      // instruction we can ignore, otherwise we may pessimize shrinkwrapping.
      // PPC's Frame pointer (FP) is also not described as a callee-saved
      // register. Until the FP is assigned a Physical Register PPC's FP needs
      // to be checked separately.
      if ((!MI.isCall() && PhysReg == SP) ||
          RCI.getLastCalleeSavedAlias(PhysReg) ||
          (!MI.isReturn() &&
           TRI->isNonallocatableRegisterCalleeSave(PhysReg)) ||
          TRI->isVirtualFrameRegister(PhysReg)) {
        LLVM_DEBUG(dbgs() << MI << " uses or defines CSR: "
                          << RCI.getLastCalleeSavedAlias(PhysReg) << "\n");
        if (!RegsToSave)
          return true;

        RegsToSave->set(RCI.getLastCalleeSavedAlias(PhysReg));
      }
    } else if (MO.isRegMask()) {
      // Check if this regmask clobbers any of the CSRs.
      for (unsigned Reg : getCurrentCSRsBitVec(RS)) {
        if (MO.clobbersPhysReg(Reg)) {
          if (!RegsToSave)
            return true;
          RegsToSave->set(RCI.getLastCalleeSavedAlias(Reg));
        }
      }
    }
  }

  // Skip FrameIndex operands in DBG_VALUE instructions.
  if (RegsToSave && !RegsToSave->empty()) {
    return true;
  }
  return false;
}

bool ShrinkWrappingImpl::useOrDefFI(const MachineInstr &MI, RegScavenger *RS,
                                    bool StackAddressUsed) const {
  /// Check if \p Op is known to access an address not on the function's stack .
  /// At the moment, accesses where the underlying object is a global, function
  /// argument, or jump table are considered non-stack accesses. Note that the
  /// caller's stack may get accessed when passing an argument via the stack,
  /// but not the stack of the current function.
  ///
  auto IsKnownNonStackPtr = [](MachineMemOperand *Op) {
    if (Op->getValue()) {
      const Value *UO = getUnderlyingObject(Op->getValue());
      if (!UO)
        return false;
      if (auto *Arg = dyn_cast<Argument>(UO))
        return !Arg->hasPassPointeeByValueCopyAttr();
      return isa<GlobalValue>(UO);
    }
    if (const PseudoSourceValue *PSV = Op->getPseudoValue())
      return PSV->isJumpTable() || PSV->isConstantPool();
    return false;
  };
  // Load/store operations may access the stack indirectly when we previously
  // computed an address to a stack location.
  if (StackAddressUsed && MI.mayLoadOrStore() &&
      (MI.isCall() || MI.hasUnmodeledSideEffects() || MI.memoperands_empty() ||
       !all_of(MI.memoperands(), IsKnownNonStackPtr)))
    return true;

  if (MI.getOpcode() == FrameSetupOpcode ||
      MI.getOpcode() == FrameDestroyOpcode) {
    LLVM_DEBUG(dbgs() << "Frame instruction: " << MI << '\n');
    return true;
  }

  if (MI.isDebugValue())
    return false;

  const auto &Ops = MI.operands();

  auto FIOpIt = std::find_if(Ops.begin(), Ops.end(),
                             [](const auto &MO) { return MO.isFI(); });
  if (FIOpIt == Ops.end())
    return false;

  LLVM_DEBUG(dbgs() << "Use or define FI( " << FIOpIt->isFI() << "): " << MI
                    << '\n');

  return true;
}

/// calculateSets - collect the CSRs used in this function, compute
/// the DF sets that describe the initial minimal regions in the
/// Machine CFG around which CSR spills and restores must be placed.
///
/// Additionally, this function decides if shrink wrapping should
/// be disabled for the current function, checking the following:
///  1. the current function has more than 500 MBBs: heuristic limit
///     on function size to reduce compile time impact of the current
///     iterative algorithm.
///  2. all CSRs are used in the entry block.
///  3. all CSRs are used in all immediate successors of the entry block.
///  4. all CSRs are used in a subset of blocks, each of which dominates
///     all return blocks. These blocks, taken as a subgraph of the MCFG,
///     are equivalent to the entry block since all execution paths pass
///     through them.
///
bool ShrinkWrappingImpl::calculateSets(
    MachineFunction &MF,
    const ReversePostOrderTraversal<MachineBasicBlock *> &RPOT,
    RegScavenger *RS) {
  bool ShrinkWrappingEnabled = true;

  // Sets used to compute spill, restore placement sets.
  UsedCSRegs = getCurrentCSRsBitVec(RS);

  SetOfRegs CurrentCSRs;
  for (unsigned Reg : UsedCSRegs) {
    CurrentCSRs.insert((unsigned)Reg);
  }

  findFastExitPath();

  // If no CSRs used, we are done.
  if (UsedCSRegs.empty()) {
    LLVM_DEBUG(dbgs() << "DISABLED: " << MF.getName()
                      << ": uses no callee-saved registers\n");
    return false;
  }

  // Limit shrink wrapping via the current iterative bit vector
  // implementation to functions with <= 500 MBBs.
  if (MF.size() > 500) {
    LLVM_DEBUG(dbgs() << "DISABLED: " << MF.getName() << ": too large ("
                      << MF.size() << " MBBs)\n");
    return false;
  }

  // Walk instructions in all MBBs, create CSRUsed[] sets, choose
  // whether or not to shrink wrap this function.
  // MachineLoopInfo &LI = getAnalysis<MachineLoopInfo>();
  // MachineDominatorTree &DT = getAnalysis<MachineDominatorTree>();

  for (MachineFunction::iterator MBB = MF.begin(), E = MF.end(); MBB != E;
       ++MBB)
    if (MBB->isReturnBlock())
      ReturnBlocks.push_back(&(*MBB));

  CSRegSet RegsAccessed;
  bool allCSRUsesInEntryBlock = true;
  for (MachineFunction::iterator MBBI = MF.begin(), MBBE = MF.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock *MBB = &(*MBBI);
    bool StackAddressUsed = false;
    // Check if we found any stack accesses in the predecessors. We are not
    // doing a full dataflow analysis here to keep things simple but just
    // rely on a reverse portorder traversal (RPOT) to guarantee predecessors
    // are already processed except for loops (and accept the conservative
    // result for loops).
    for (const MachineBasicBlock *Pred : MBB->predecessors()) {
      if (StackAddressUsedBlockInfo.test(Pred->getNumber())) {
        StackAddressUsed = true;
        break;
      }
    }
    for (MachineBasicBlock::iterator MI = MBB->begin(); MI != MBB->end();
         ++MI) {
      RegsAccessed.clear();
      if (useOrDefFI(*MI, RS, StackAddressUsed)) {
        SaveBlocks.push_back(MBB);
        RestoreBlocks.push_back(MBB);
        StackAddressUsed = true;
      }
      if (useOrDefCSR(*MI, RS, &RegsAccessed)) {
        CSRUsed[MBB] |= RegsAccessed;
        StackAddressUsed = true;
        if (MBB != Entry && !CSRUsed[MBB].empty())
          allCSRUsesInEntryBlock = false;
      }
    }
    StackAddressUsedBlockInfo[MBB->getNumber()] = StackAddressUsed;
    if (CSRUsed[MBB].empty())
      continue;

    // Propagate CSRUsed[MBB] in loops
    if (MachineLoop *LP = MLI->getLoopFor(MBB)) {
      // Push uses from inside loop to its parent loops,
      // or to all other MBBs in its loop.
      if (LP->getLoopDepth() > 1) {
        for (MachineLoop *PLP = LP->getParentLoop(); PLP;
             PLP = PLP->getParentLoop()) {
          propagateUsesAroundLoop(MBB, PLP);
        }
      } else {
        propagateUsesAroundLoop(MBB, LP);
      }
    }
  }

  if (allCSRUsesInEntryBlock) {
    LLVM_DEBUG(dbgs() << "DISABLED: " << MF.getName()
                      << ": all CSRs used in EntryBlock\n");
    ShrinkWrappingEnabled = false;
  } else {
    bool allCSRsUsedInEntryFanout = true;
    for (MachineBasicBlock::succ_iterator SI = Entry->succ_begin(),
                                          SE = Entry->succ_end();
         SI != SE; ++SI) {
      MachineBasicBlock *SUCC = *SI;
      if (CSRUsed[SUCC] != UsedCSRegs)
        allCSRsUsedInEntryFanout = false;
    }
    if (allCSRsUsedInEntryFanout) {
      LLVM_DEBUG(dbgs() << "DISABLED: " << MF.getName()
                        << ": all CSRs used in imm successors of EntryBlock\n");
      ShrinkWrappingEnabled = false;
    }
  }

  if (ShrinkWrappingEnabled) {
    // Check if MBB uses CSRs and dominates all exit nodes.
    // Such nodes are equiv. to the entry node w.r.t.
    // CSR uses: every path through the function must
    // pass through this node. If each CSR is used at least
    // once by these nodes, shrink wrapping is disabled.
    CSRegSet CSRUsedInChokePoints;
    for (MachineFunction::iterator MBBI = MF.begin(), MBBE = MF.end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock *MBB = &(*MBBI);
      if (MBB == Entry || CSRUsed[MBB].empty() || MBB->succ_size() < 1)
        continue;
      bool dominatesExitNodes = true;
      for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
        if (!MDT->dominates(MBB, ReturnBlocks[ri])) {
          dominatesExitNodes = false;
          break;
        }
      if (dominatesExitNodes) {
        CSRUsedInChokePoints |= CSRUsed[MBB];
        if (CSRUsedInChokePoints == UsedCSRegs) {
          LLVM_DEBUG(dbgs() << "DISABLED: " << MF.getName()
                            << ": all CSRs used in choke point(s) at "
                            << getBasicBlockName(MBB) << "\n");
          ShrinkWrappingEnabled = false;
          break;
        }
      }
    }
  }

  // Return now if we have decided not to apply shrink wrapping
  // to the current function.
  if (!ShrinkWrappingEnabled)
    return false;

  LLVM_DEBUG({
    dbgs() << "ENABLED: " << MF.getName();
    if (HasFastExitPath)
      dbgs() << " (fast exit path)";
    dbgs() << "\n";
    if (ShrinkWrappingDebugging >= BasicInfo) {
      dbgs() << "------------------------------"
             << "-----------------------------\n";
      dbgs() << "UsedCSRegs = " << stringifyCSRegSet(UsedCSRegs) << "\n";
      if (ShrinkWrappingDebugging >= Details) {
        dbgs() << "------------------------------"
               << "-----------------------------\n";
        dumpAllUsed();
      }
    }
  });

  // Build initial DF sets to determine minimal regions in the
  // Machine CFG around which CSRs must be spilled and restored.
  calculateAnticAvail(MF, RS);

  return true;
}

/// calcSpillPlacements - determine which CSRs should be spilled
/// in MBB using AnticIn sets of MBB's predecessors, keeping track
/// of changes to spilled reg sets. Add MBB to the set of blocks
/// that need to be processed for propagating use info to cover
/// multi-entry/exit regions.
///
bool ShrinkWrappingImpl::calcSpillPlacements(
    AuxGraphNode *Node, SmallVectorImpl<AuxGraphNode *> &blks,
    CSRegNodeMap &prevSpills) {
  bool placedSpills = false;
  // Intersect (CSRegs - AnticIn[P]) for P in Predecessors(MBB)
  CSRegSet anticInPreds;
  SmallVector<AuxGraphNode *, 4> predecessors;
  for (auto PI = Node->pred_begin(), PE = Node->pred_end(); PI != PE; ++PI) {
    AuxGraphNode *PRED = *PI;
    if (PRED != Node)
      predecessors.push_back(PRED);
  }
  unsigned i = 0, e = predecessors.size();
  if (i != e) {
    AuxGraphNode *PRED = predecessors[i];
    anticInPreds = UsedCSRegs - AnticIn[PRED];
    for (++i; i != e; ++i) {
      PRED = predecessors[i];
      anticInPreds &= (UsedCSRegs - AnticIn[PRED]);
    }
  } else {
    // Handle uses in entry blocks (which have no predecessors).
    // This is necessary because the DFA formulation assumes the
    // entry and (multiple) exit nodes cannot have CSR uses, which
    // is not the case in the real world.
    anticInPreds = UsedCSRegs;
  }
  // Compute spills required at MBB:
  CSRSave[Node] |= (AnticIn[Node] - AvailIn[Node]) & anticInPreds;

  if (!CSRSave[Node].empty()) {
    if (Node == AuxillaryCFG.getNode(Entry)) {
      for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
        CSRRestore[AuxillaryCFG.getNode(ReturnBlocks[ri])] |= CSRSave[Node];
    } else {
      // Reset all regs spilled in MBB that are also spilled in EntryBlock.
      if (CSRSave[AuxillaryCFG.getNode(Entry)].intersects(CSRSave[Node])) {
        CSRSave[Node] = CSRSave[Node] - CSRSave[AuxillaryCFG.getNode(Entry)];
      }
    }
  }
  placedSpills = (CSRSave[Node] != prevSpills[Node]);
  prevSpills[Node] = CSRSave[Node];
  // Remember this block for adding restores to successor
  // blocks for multi-entry region.
  if (placedSpills)
    blks.push_back(Node);

  LLVM_DEBUG(if (!CSRSave[Node].empty() &&
                 ShrinkWrappingDebugging >= Iterations) dbgs()
             << "SAVE[" << Node->Name
             << "] = " << stringifyCSRegSet(CSRSave[Node]) << "\n");

  return placedSpills;
}

/// calcRestorePlacements - determine which CSRs should be restored
/// in MBB using AvailOut sets of MBB's succcessors, keeping track
/// of changes to restored reg sets. Add MBB to the set of blocks
/// that need to be processed for propagating use info to cover
/// multi-entry/exit regions.
///
bool ShrinkWrappingImpl::calcRestorePlacements(
    AuxGraphNode *Node, SmallVectorImpl<AuxGraphNode *> &blks,
    CSRegNodeMap &prevRestores) {
  bool placedRestores = false;
  // Intersect (CSRegs - AvailOut[S]) for S in Successors(MBB)
  CSRegSet availOutSucc;
  SmallVector<AuxGraphNode *, 4> successors;
  for (auto SI = Node->succ_begin(), SE = Node->succ_end(); SI != SE; ++SI) {
    AuxGraphNode *SUCC = *SI;
    if (SUCC != Node)
      successors.push_back(SUCC);
  }
  unsigned i = 0, e = successors.size();
  if (i != e) {
    AuxGraphNode *SUCC = successors[i];
    availOutSucc = UsedCSRegs - AvailOut[SUCC];
    for (++i; i != e; ++i) {
      SUCC = successors[i];
      availOutSucc &= (UsedCSRegs - AvailOut[SUCC]);
    }
  } else if (Node->MatchMBB && Node->MatchMBB->isReturnBlock()) {
    // Leaf return blocks: place restores (DFA assumes exits have no uses).
    // Do NOT treat noreturn exits (e.g. call __cxa_throw + barrier) as
    // returns — restore+CFI after noreturn poisons FDE at the call RA
    // (libunwind uses pc after the call). Unwind recovers CSRs via save
    // CFI instead. Matches GCC separate shrink-wrap: epi only on paths
    // that leave the framed region toward a normal EXIT.
    CSRegSet Used = CSRUsed[Node->MatchMBB];
    if (!Used.empty() || !AvailOut[Node].empty())
      availOutSucc = UsedCSRegs;
  }
  // Compute restores required at MBB:
  CSRRestore[Node] |= (AvailOut[Node] - AnticOut[Node]) & availOutSucc;

  // Postprocess restore placements at MBB.
  // Remove the CSRs that are restored in the return blocks.
  // Lest this be confusing, note that:
  // CSRSave[EntryBlock] == CSRRestore[B] for all B in ReturnBlocks.
  if (Node->succ_size() && !CSRRestore[Node].empty()) {
    if (!CSRSave[AuxillaryCFG.getNode(Entry)].empty())
      CSRRestore[Node] =
          CSRRestore[Node] - CSRSave[AuxillaryCFG.getNode(Entry)];
  }
  placedRestores = (CSRRestore[Node] != prevRestores[Node]);
  prevRestores[Node] = CSRRestore[Node];
  // Remember this block for adding saves to predecessor
  // blocks for multi-entry region.
  if (placedRestores)
    blks.push_back(Node);

  LLVM_DEBUG(if (!CSRRestore[Node].empty() &&
                 ShrinkWrappingDebugging >= Iterations) dbgs()
             << "RESTORE[" << Node->Name
             << "] = " << stringifyCSRegSet(CSRRestore[Node]) << "\n");

  return placedRestores;
}

/// placeSpillsAndRestores - place spills and restores of CSRs
/// used in MBBs in minimal regions that contain the uses.
///
void ShrinkWrappingImpl::placeSpillsAndRestores(MachineFunction &Fn) {
  CSRegNodeMap prevCSRSave;
  CSRegNodeMap prevCSRRestore;
  SmallVector<AuxGraphNode *, 4> cvBlocks, ncvBlocks;
  bool changed = true;
  unsigned iterations = 0;

  // Iterate computation of spill and restore placements in the MCFG until:
  //   1. CSR use info has been fully propagated around the MCFG, and
  //   2. computation of CSRSave[], CSRRestore[] reach fixed points.
  while (changed) {
    changed = false;
    ++iterations;

    LLVM_DEBUG(if (ShrinkWrappingDebugging >= Iterations) dbgs()
               << "iter " << iterations
               << " --------------------------------------------------\n");

    // Calculate CSR{Save,Restore} sets using Antic, Avail on the MCFG,
    // which determines the placements of spills and restores.
    // Keep track of changes to spills, restores in each iteration to
    // minimize the total iterations.
    for (auto &Node : AuxillaryCFG.Nodes) {
      // Place spills for CSRs in MBB.
      calcSpillPlacements(Node, cvBlocks, prevCSRSave);

      // Place restores for CSRs in MBB.
      calcRestorePlacements(Node, cvBlocks, prevCSRRestore);
    }
  }

  // Check for effectiveness:
  //  SR0 = {r | r in CSRSave[EntryBlock], CSRRestore[RB], RB in ReturnBlocks}
  //  numSRReduced = |(UsedCSRegs - SR0)|, approx. SR0 by CSRSave[EntryBlock]
  // Gives a measure of how many CSR spills have been moved from EntryBlock
  // to minimal regions enclosing their uses.
  CSRegSet notSpilledInEntryBlock =
      (UsedCSRegs - CSRSave[AuxillaryCFG.getNode(Entry)]);
  unsigned numSRReducedThisFunc = notSpilledInEntryBlock.count();
  numSRReduced += numSRReducedThisFunc;
  LLVM_DEBUG(if (ShrinkWrappingDebugging >= BasicInfo) {
    dbgs() << "-----------------------------------------------------------\n";
    dbgs() << "total iterations = " << iterations << " ( " << Fn.getName()
           << " " << numSRReducedThisFunc << " " << Fn.size() << " )\n";
    dbgs() << "-----------------------------------------------------------\n";
    dumpSRSets();
    dbgs() << "-----------------------------------------------------------\n";
  });
}

bool ShrinkWrappingImpl::blockNeedsFrame(const MachineBasicBlock &MBB) const {
  const TargetInstrInfo *TII = MachineFunc->getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MachineFunc->getSubtarget().getRegisterInfo();
  Register StackPtr = MachineFunc->getSubtarget()
                          .getTargetLowering()
                          ->getStackPointerRegisterToSaveRestore();
  Register FramePtr = TRI->getFrameRegister(*MachineFunc);

  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;

    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isFI())
        return true;
    }

    if (MI.isCall() && !TII->isTailCall(MI))
      return true;

    if (StackPtr && MI.modifiesRegister(StackPtr, TRI))
      return true;
    if (FramePtr && MI.modifiesRegister(FramePtr, TRI))
      return true;

    if (MI.getOpcode() == TargetOpcode::STATEPOINT ||
        MI.getOpcode() == TargetOpcode::STACKMAP ||
        MI.getOpcode() == TargetOpcode::PATCHPOINT)
      return true;
  }
  return false;
}

bool ShrinkWrappingImpl::runShrinkFrame(MachineFunction &MF) {
  if (DisableShrinkFrame)
    return false;

  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Bail-out conditions: these functions cannot have their frame shrunk.
  const Function &F = MF.getFunction();
  if (F.hasFnAttribute(Attribute::Naked) ||
      F.hasFnAttribute(Attribute::NoReturn) ||
      F.hasFnAttribute(Attribute::OptimizeNone) || MFI.hasVarSizedObjects() ||
      MFI.hasOpaqueSPAdjustment() ||
      MF.getFunction().getCallingConv() == CallingConv::GHC ||
      MF.hasEHFunclets() || F.hasFnAttribute(Attribute::SanitizeAddress) ||
      F.hasFnAttribute(Attribute::SanitizeThread) ||
      F.hasFnAttribute(Attribute::SanitizeMemory) ||
      F.hasFnAttribute(Attribute::SanitizeHWAddress))
    return false;

  // Check for EH pads (already checked by caller for irreducible CFG).
  for (const MachineBasicBlock &MBB : MF) {
    if (MBB.isEHPad() || MBB.isEHFuncletEntry())
      return false;
  }

  // Collect all blocks that need a frame.
  SmallVector<MachineBasicBlock *, 16> FrameBlocks;
  for (MachineBasicBlock &MBB : MF) {
    if (blockNeedsFrame(MBB))
      FrameBlocks.push_back(&MBB);
  }

  // If no block needs a frame, nothing to do.
  if (FrameBlocks.empty())
    return false;

  // If entry itself needs a frame, no benefit from shrinking.
  if (FrameBlocks.size() == 1 && FrameBlocks[0] == Entry)
    return false;

  // Compute NCD (nearest common dominator) of all frame-needing blocks.
  MachineBasicBlock *P = FrameBlocks[0];
  for (unsigned I = 1, E = FrameBlocks.size(); I < E; ++I) {
    P = MDT->findNearestCommonDominator(P, FrameBlocks[I]);
    if (!P)
      break;
  }

  if (!P || P == Entry)
    return false;

  // Prolog must not be inside a loop — lift to preheader, repeating for nested
  // loops until P is outside every loop (a single lift can leave P inside an
  // outer loop's body).
  while (MachineLoop *L = MLI->getLoopFor(P)) {
    MachineBasicBlock *PH = L->getLoopPreheader();
    if (!PH || PH == Entry)
      return false;
    P = PH;
    if (P == Entry)
      return false;
  }

  // Verify no "mixed SP" convergence: a return reachable from Prolog but not
  // dominated by it means a framed path shares a frameless epilogue (no ld ra /
  // addi). Classic symptom: call clobbers ra, ret jumps to the post-call insn
  // → infinite loop (502.gcc is_gimple_invariant_address). Checking only
  // "P dominates a predecessor of the return" misses joins where the
  // immediate predecessor is itself a merge of framed and frameless paths.
  for (MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;
    if (MDT->dominates(P, &MBB))
      continue;
    // Frameless return: must not be reachable from P.
    for (auto DI = df_begin(P), DE = df_end(P); DI != DE; ++DI) {
      if (*DI == &MBB) {
        LLVM_DEBUG(dbgs() << "Shrink-frame bail: return "
                          << printMBBReference(MBB) << " reachable from Prolog "
                          << printMBBReference(*P)
                          << " but not dominated by it\n");
        return false;
      }
    }
  }

  // Check canUseAsPrologue (target hook).
  if (!TFI->canUseAsPrologue(*P))
    return false;

  LLVM_DEBUG(dbgs() << "Shrink-frame: setting Prolog to "
                    << printMBBReference(*P) << "\n");

  MFI.setProlog(P);
  ++NumShrinkFrame;
  return true;
}

bool ShrinkWrappingImpl::run(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "**** Analysing " << MF.getName() << '\n');

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  std::unique_ptr<RegScavenger> RS(
      TRI->requiresRegisterScavenging(MF) ? new RegScavenger() : nullptr);

  init(MF, RS.get());

  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF.begin());
  if (containsIrreducibleCFG<MachineBasicBlock *>(RPOT, *MLI)) {
    // If MF is irreducible, a block may be in a loop without
    // MachineLoopInfo reporting it. I.e., we may use the
    // post-dominance property in loops, which lead to incorrect
    // results. Moreover, we may miss that the prologue and
    // epilogue are not in the same loop, leading to unbalanced
    // construction/deconstruction of the stack frame.
    return giveUpWithRemarks(ORE, "UnsupportedIrreducibleCFG",
                             "Irreducible CFGs are not supported yet.",
                             MF.getFunction().getSubprogram(), &MF.front());
  }

  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (MFI.hasVarSizedObjects()) {
    LLVM_DEBUG(dbgs() << "Can't split save/restore points, because frame "
                         "contains var sized objects\n");
    return false;
  }

  for (MachineBasicBlock *MBB : RPOT) {
    if (MBB->isEHFuncletEntry())
      return giveUpWithRemarks(ORE, "UnsupportedEHFunclets",
                               "EH Funclets are not supported yet.",
                               MBB->front().getDebugLoc(), MBB);

    if (MBB->isEHPad() || MBB->isInlineAsmBrIndirectTarget())
      return giveUpWithRemarks(
          ORE, "EHPads and isInlineAsmBrIndirectTargets",
          "EHPads and isInlineAsmBrIndirectTargets are not supported yet.",
          MBB->front().getDebugLoc(), MBB);
  }

  const MachineJumpTableInfo *MJTI = MachineFunc->getJumpTableInfo();
  if (MJTI)
    return giveUpWithRemarks(ORE, "UnsupportedMJTI",
                             "JumpTables are not supported yet.",
                             MF.getFunction().getSubprogram(), &MF.front());

  // Attempt to shrink the stack frame setup to a later block.
  // This must run before calculateSets/Chow since it sets the Prolog.
  runShrinkFrame(MF);

  createAuxillaryCFG();

  // Initially, conservatively assume that stack addresses can be used in each
  // basic block and change the state only for those basic blocks for which we
  // were able to prove the opposite.
  StackAddressUsedBlockInfo.resize(MF.getNumBlockIDs(), true);
  bool HasCandidates = calculateSets(MF, RPOT, RS.get());
  StackAddressUsedBlockInfo.clear();
  if (!HasCandidates) {
    // If shrink-frame placed a Prolog but Chow found no CSR candidates,
    // check if there are delayed (non-frame-bound) CSRs. If so, PEI would
    // place saves in entry (no frame) → rollback.
    if (MFI.getProlog()) {
      SmallVector<Register, 4> FrameBound;
      MF.getSubtarget().getFrameLowering()->getFrameBoundCalleeSaves(
          MF, FrameBound);
      SmallDenseSet<unsigned, 4> FrameBoundSet;
      for (Register R : FrameBound)
        if (R)
          FrameBoundSet.insert(R.id());

      bool HasDelayedCSR = false;
      for (unsigned Reg : UsedCSRegs) {
        if (!FrameBoundSet.count(Reg)) {
          HasDelayedCSR = true;
          break;
        }
      }
      if (HasDelayedCSR) {
        LLVM_DEBUG(dbgs() << "Shrink-frame rollback: Chow failed with "
                             "delayed CSRs present\n");
        MFI.setProlog(nullptr);
      }
    }
    return MFI.getProlog() != nullptr;
  }

  placeSpillsAndRestores(MF);

  // Frame-bound CSRs (ra / hard FP on RISC-V): never delay as separate
  // components — match GCC TARGET_SHRINK_WRAP_GET_SEPARATE_COMPONENTS.
  // Important: MRI.isPhysRegModified(ra) can be false around calls because the
  // ABI regmask treats ra as preserved; PEI still spills ra when hasCalls.
  // Leaf functions may still use ra as a scratch GPR after a conditional save
  // (scale_bitcount) — pin whenever Chow already placed a save, or hasCalls /
  // determineCalleeSaves / isPhysRegModified says so.
  auto PinFrameBoundCSRs = [&]() {
    const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
    SmallVector<Register, 4> FrameBound;
    TFI->getFrameBoundCalleeSaves(MF, FrameBound);
    if (FrameBound.empty())
      return;

    MachineBasicBlock *SFProlog = MFI.getProlog();
    MachineBasicBlock *PinBlock = SFProlog ? SFProlog : Entry;
    AuxGraphNode *PinNode = AuxillaryCFG.getNode(PinBlock);
    const MachineRegisterInfo &MRI = MF.getRegInfo();
    BitVector PinSavedRegs;
    TFI->determineCalleeSaves(MF, PinSavedRegs, RS.get());
    const bool HasCalls = MFI.hasCalls();

    for (Register Reg : FrameBound) {
      if (!Reg)
        continue;
      bool ShouldPin = HasCalls || PinSavedRegs.test(Reg.id()) ||
                       UsedCSRegs.test(Reg.id()) || MRI.isPhysRegModified(Reg);
      if (!ShouldPin) {
        for (const auto &[Node, Saves] : CSRSave) {
          if (Saves.test(Reg.id())) {
            ShouldPin = true;
            break;
          }
        }
      }
      if (!ShouldPin)
        continue;
      for (auto &KV : CSRSave)
        KV.second.reset(Reg.id());
      for (auto &KV : CSRRestore)
        KV.second.reset(Reg.id());
      CSRSave[PinNode].set(Reg.id());
      for (MachineBasicBlock *RB : ReturnBlocks) {
        if (!SFProlog || MDT->dominates(SFProlog, RB))
          CSRRestore[AuxillaryCFG.getNode(RB)].set(Reg.id());
      }
    }
  };

  PinFrameBoundCSRs();

  // Drop restores on noreturn exits (no ld/CFI after throw/abort/exit — that
  // poisons FDE at the call RA). Rehome per-CSR only onto ReturnBlocks that
  // are reachable from a Save of that CSR. Broadcasting to *all* returns
  // causes early-exit paths that never spilled to execute ld of garbage
  // (SIBsim4 / PENNANT SIGSEGV).
  auto ContainsNoReturnCall = [](const MachineBasicBlock &MBB) {
    for (const MachineInstr &MI : MBB) {
      if (!MI.isCall())
        continue;
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isGlobal()) {
          if (const Function *F = dyn_cast<Function>(MO.getGlobal()))
            if (F->doesNotReturn())
              return true;
        } else if (MO.isSymbol()) {
          StringRef N = MO.getSymbolName();
          if (N == "exit" || N == "_exit" || N == "abort" ||
              N == "__cxa_throw" || N == "__cxa_rethrow" ||
              N == "_Unwind_Resume" || N == "__assert_fail")
            return true;
        }
      }
    }
    return false;
  };

  CSRegSet ClearedRegs;
  for (auto &KV : CSRRestore) {
    MachineBasicBlock *MBB = KV.first->MatchMBB;
    if (!MBB || KV.second.empty())
      continue;

    const bool HasNoReturnCall = ContainsNoReturnCall(*MBB);
    // Pure return blocks without noreturn keep their restores.
    if (MBB->isReturnBlock() && !HasNoReturnCall)
      continue;

    bool CanReachReturn = false;
    if (!HasNoReturnCall) {
      for (df_iterator<MachineBasicBlock *> DI = df_begin(MBB),
                                            DE = df_end(MBB);
           DI != DE; ++DI) {
        if ((*DI)->isReturnBlock()) {
          CanReachReturn = true;
          break;
        }
      }
    }
    if (!CanReachReturn || HasNoReturnCall) {
      LLVM_DEBUG(dbgs() << "Clear RESTORE on noreturn/non-returning path "
                        << printMBBReference(*MBB) << "\n");
      ClearedRegs |= KV.second;
      KV.second.clear();
    }
  }

  for (unsigned Reg : ClearedRegs) {
    SmallPtrSet<MachineBasicBlock *, 8> Targets;
    bool AnySave = false;
    for (const auto &SKV : CSRSave) {
      if (!SKV.second.test(Reg) || !SKV.first->MatchMBB)
        continue;
      AnySave = true;
      // Dominates, not reachable — see path-sensitive restore comment below.
      for (MachineBasicBlock *RB : ReturnBlocks) {
        if (!ContainsNoReturnCall(*RB) &&
            MDT->dominates(SKV.first->MatchMBB, RB))
          Targets.insert(RB);
      }
    }
    // No save recorded: keep ABI safe (e.g. frame-bound race) on all returns.
    if (!AnySave) {
      for (MachineBasicBlock *RB : ReturnBlocks)
        Targets.insert(RB);
    }
    for (MachineBasicBlock *RB : Targets)
      CSRRestore[AuxillaryCFG.getNode(RB)].set(Reg);
  }

  // Path-sensitive restores on *return blocks* only:
  // Drop restore at return R when the single save does not dominate R
  // (early-exit / diamond: ray Sphere::intersect bb.10). Skip frame-bound
  // CSRs — they are re-pinned below.
  {
    SmallDenseSet<unsigned, 4> FrameBoundSet;
    SmallVector<Register, 4> FrameBound;
    MF.getSubtarget().getFrameLowering()->getFrameBoundCalleeSaves(MF,
                                                                   FrameBound);
    for (Register R : FrameBound)
      if (R)
        FrameBoundSet.insert(R.id());

    for (unsigned Reg : UsedCSRegs) {
      if (FrameBoundSet.count(Reg))
        continue;
      SmallVector<MachineBasicBlock *, 4> SaveBBs;
      for (const auto &[Node, Saves] : CSRSave) {
        if (!Saves.test(Reg) || !Node->MatchMBB)
          continue;
        SaveBBs.push_back(Node->MatchMBB);
      }
      if (SaveBBs.size() != 1)
        continue;

      MachineBasicBlock *S = SaveBBs[0];
      for (auto &[Node, Restores] : CSRRestore) {
        if (!Restores.test(Reg))
          continue;
        MachineBasicBlock *MBB = Node->MatchMBB;
        if (!MBB || !MBB->isReturnBlock())
          continue;
        if (!MDT->dominates(S, MBB))
          Restores.reset(Reg);
      }
    }
  }

  // Re-pin after path-sensitive / noreturn cleanup.
  PinFrameBoundCSRs();

  if (setupCFG()) {
    // setupCFG() splits edges and creates new blocks; rebuild dominators for
    // any subsequent queries / verification.
    MDT->recalculate(MF);
    MPDT->recalculate(MF);
  }

  // Clamp: when shrink-frame is active, all CSR save points must be in blocks
  // dominated by Prolog. A save in a no-frame block would execute sd before
  // addi sp — corrupting the stack.
  if (MachineBasicBlock *SFP = MFI.getProlog()) {
    for (const auto &[Node, Regs] : CSRSave) {
      if (Regs.empty() || !Node->MatchMBB)
        continue;
      if (!MDT->dominates(SFP, Node->MatchMBB)) {
        LLVM_DEBUG(dbgs() << "Clamp fail: CSR save in "
                          << printMBBReference(*Node->MatchMBB)
                          << " not dominated by Prolog "
                          << printMBBReference(*SFP) << " — rolling back\n");
        MFI.setProlog(nullptr);
        break;
      }
    }
  }

  verifySpillRestorePlacement();

  // Drop stack-use pollution from calculateSets(); Save/Restore lists must
  // come only from CSR placement.
  SaveBlocks.clear();
  RestoreBlocks.clear();
  setupSaveRestorePoints();

  // Final enforcement on concrete Save/Restore maps (after edge splits).
  // Guarantees ra/FP cannot remain on a delayed save point (pr51933,
  // consumer-lame scale_bitcount). Do NOT gate on hasCalls(): leaf functions
  // may still allocate ra as a GPR scratch after a conditional sd ra; the
  // shared epilogue then rets with a clobbered ra and no ld (jump to 0).
  {
    const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
    SmallVector<Register, 4> FrameBound;
    TFI->getFrameBoundCalleeSaves(MF, FrameBound);
    if (!FrameBound.empty()) {
      MachineBasicBlock *PinBlock = MFI.getProlog() ? MFI.getProlog() : Entry;
      auto &SP = SavePoints.get();
      auto &RP = RestorePoints.get();
      BitVector PinSavedRegs;
      TFI->determineCalleeSaves(MF, PinSavedRegs, RS.get());
      const MachineRegisterInfo &MRI = MF.getRegInfo();

      for (Register Reg : FrameBound) {
        if (!Reg)
          continue;

        bool InSavePoints = false;
        for (const auto &KV : SP) {
          if (any_of(KV.second, [&](const CalleeSavedInfo &C) {
                return C.getReg() == Reg.id();
              })) {
            InSavePoints = true;
            break;
          }
        }
        const bool ShouldPin =
            InSavePoints || MFI.hasCalls() || PinSavedRegs.test(Reg.id()) ||
            UsedCSRegs.test(Reg.id()) || MRI.isPhysRegModified(Reg);
        if (!ShouldPin)
          continue;

        // Strip from every block, then pin to frame prolog + returns.
        for (auto &KV : SP) {
          auto &CSIs = KV.second;
          llvm::erase_if(CSIs, [&](const CalleeSavedInfo &C) {
            return C.getReg() == Reg.id();
          });
        }
        for (auto &KV : RP) {
          auto &CSIs = KV.second;
          llvm::erase_if(CSIs, [&](const CalleeSavedInfo &C) {
            return C.getReg() == Reg.id();
          });
        }
        SavePoints.insertReg(Reg, PinBlock, &SaveBlocks);
        for (MachineBasicBlock *RB : ReturnBlocks) {
          if (!MFI.getProlog() || MDT->dominates(MFI.getProlog(), RB))
            RestorePoints.insertReg(Reg, RB, &RestoreBlocks);
        }
      }
    }
  }

  // When shrink-frame is not active, PEI keeps Prolog/Epilog at entry/returns.
  // When shrink-frame placed a Prolog, preserve it for PEI.
  MachineBasicBlock *ShrinkFrameProlog = MFI.getProlog();
  if (!ShrinkFrameProlog) {
    Prolog = nullptr;
    Epilog = nullptr;
  }

  if (SavePoints.areMultiple() || RestorePoints.areMultiple()) {
    ++NumFuncWithSplitting;
  }

  LLVM_DEBUG(dbgs() << "Final shrink wrap candidates:\n");

  LLVM_DEBUG(dbgs() << "SavePoints:\n");
  LLVM_DEBUG(SavePoints.dump(TRI));

  LLVM_DEBUG(dbgs() << "RestorePoints:\n");
  LLVM_DEBUG(RestorePoints.dump(TRI));

  if (!ShrinkFrameProlog)
    MFI.setProlog(nullptr);
  MFI.setEpilog(nullptr);
  if (!SavePoints.get().empty() && !RestorePoints.get().empty()) {
    MFI.setSavePoints(SavePoints.get());
    MFI.setRestorePoints(RestorePoints.get());
  } else {
    ++NumNotSaveOrRestore;
  }
  ++NumCandidates;

  return true;
}

MachineBasicBlock *ShrinkWrappingImpl::splitEdge(MachineBasicBlock *Pred,
                                                 MachineBasicBlock *Succ) {
  MachineBasicBlock *NewBB = MachineFunc->CreateMachineBasicBlock();

  MachineJumpTableInfo *MJTI = MachineFunc->getJumpTableInfo();
  if (MJTI)
    MJTI->ReplaceMBBInJumpTables(Succ, NewBB);

  MachineBasicBlock *FallThrough = Pred->getFallThrough(false);
  MachineFunc->insert(MachineFunc->end(), NewBB);
  bool Done = false;
  const TargetInstrInfo *TII = MachineFunc->getSubtarget().getInstrInfo();
  DebugLoc DL = Pred->findBranchDebugLoc();

  if (Pred->getFirstTerminator() == Pred->end()) {
    Done = true;
    TII->insertUnconditionalBranch(*Pred, NewBB, DebugLoc());
  } else {
    for (auto &Term : Pred->terminators()) {
      if (Term.isUnconditionalBranch() &&
          (Term.getOperand(0).getMBB() == Succ)) {
        Done = true;
        Term.getOperand(0).setMBB(NewBB);
      } else if (Term.isConditionalBranch()) {
        for (auto &MO : Term.operands()) {
          if (MO.isMBB() && (MO.getMBB() == Succ)) {
            Done = true;
            MO.setMBB(NewBB);
          }
        }
      }
    }
  }

  if (!Done && !Pred->isLayoutSuccessor(NewBB) && FallThrough == Succ) {
    TII->insertUnconditionalBranch(*Pred, NewBB, DebugLoc());
  }

  // TODO: switch

  Pred->replaceSuccessor(Succ, NewBB); // Remove old edge
  NewBB->addSuccessor(Succ);           // Add NewBB → Succ

  for (const MachineBasicBlock::RegisterMaskPair &LI : Succ->liveins())
    NewBB->addLiveIn(LI.PhysReg);

  TII->insertUnconditionalBranch(*NewBB, Succ, DebugLoc());

  return NewBB;
}

bool ShrinkWrappingImpl::setupCFG() {
  bool SplitAnyEdge = false;
  for (auto &[Node, Regs] : CSRSave) {
    if (!Regs.empty() && !Node->MatchMBB) {
      assert(Node->pred_size() == 1 &&
             "Auxillary node can have only one predecessor!");
      assert(Node->succ_size() == 1 &&
             "Auxillary node can have only one successor!");
      MachineBasicBlock *NewBB = splitEdge((*Node->pred_begin())->MatchMBB,
                                           (*Node->succ_begin())->MatchMBB);
      Node->MatchMBB = NewBB;
      SplitAnyEdge = true;
    }
  }

  for (auto &[Node, Regs] : CSRRestore) {
    if (!Regs.empty() && !Node->MatchMBB) {
      assert(Node->pred_size() == 1 &&
             "Auxillary node can have only one predecessor!");
      assert(Node->succ_size() == 1 &&
             "Auxillary node can have only one successor!");
      MachineBasicBlock *NewBB = splitEdge((*Node->pred_begin())->MatchMBB,
                                           (*Node->succ_begin())->MatchMBB);
      Node->MatchMBB = NewBB;
      SplitAnyEdge = true;
    }
  }
  return SplitAnyEdge;
}

void ShrinkWrappingImpl::setupSaveRestorePoints() {
  for (auto &[Node, Regs] : CSRSave) {
    for (Register Reg : Regs)
      SavePoints.insertReg(Reg, Node->MatchMBB, &SaveBlocks);
  }
  for (auto &[Node, Regs] : CSRRestore) {
    for (Register Reg : Regs)
      RestorePoints.insertReg(Reg, Node->MatchMBB, &RestoreBlocks);
  }
}

bool ShrinkWrappingLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || MF.empty() ||
      !ShrinkWrappingImpl::isShrinkWrappingEnabled(MF))
    return false;

  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachinePostDominatorTree *MPDT =
      &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  MachineBlockFrequencyInfo *MBFI =
      &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineOptimizationRemarkEmitter *ORE =
      &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();

  return ShrinkWrappingImpl(MDT, MPDT, MBFI, MLI, ORE).run(MF);
}

PreservedAnalyses
ShrinkWrappingPass::run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  if (MF.empty() || !ShrinkWrappingImpl::isShrinkWrappingEnabled(MF))
    return PreservedAnalyses::all();

  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachinePostDominatorTree &MPDT =
      MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF);
  MachineBlockFrequencyInfo &MBFI =
      MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);
  MachineLoopInfo &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  MachineOptimizationRemarkEmitter &ORE =
      MFAM.getResult<MachineOptimizationRemarkEmitterAnalysis>(MF);

  ShrinkWrappingImpl(&MDT, &MPDT, &MBFI, &MLI, &ORE).run(MF);
  return PreservedAnalyses::all();
}

bool ShrinkWrappingImpl::isShrinkWrappingEnabled(const MachineFunction &MF) {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();

  // Data-flow shrink-wrapping requires the target to opt in.
  if (!TFI->enableCSRSaveRestorePointsSplit())
    return false;

  switch (EnableShrinkWrappingOpt) {
  case cl::boolOrDefault::BOU_UNSET:
    return TFI->enableShrinkWrapping(MF) &&
           // Windows with CFI has some limitations that make it impossible
           // to use shrink-wrapping.
           !MF.getTarget().getMCAsmInfo().usesWindowsCFI() &&
           // Sanitizers look at the value of the stack at the location
           // of the crash. Since a crash can happen anywhere, the
           // frame must be lowered before anything else happen for the
           // sanitizers to be able to get a correct stack frame.
           !(MF.getFunction().hasFnAttribute(Attribute::SanitizeAddress) ||
             MF.getFunction().hasFnAttribute(Attribute::SanitizeThread) ||
             MF.getFunction().hasFnAttribute(Attribute::SanitizeMemory) ||
             MF.getFunction().hasFnAttribute(Attribute::SanitizeType) ||
             MF.getFunction().hasFnAttribute(Attribute::SanitizeHWAddress));
  // If EnableShrinkWrap is set, it takes precedence on whatever the
  // target sets. The rational is that we assume we want to test
  // something related to shrink-wrapping.
  case cl::boolOrDefault::BOU_TRUE:
    return true;
  case cl::boolOrDefault::BOU_FALSE:
    return false;
  }
  llvm_unreachable("Invalid shrink-wrapping state");
}
