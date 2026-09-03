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
  #include "llvm/CodeGen/LivePhysRegs.h"
  #include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
  #include "llvm/CodeGen/MachineDominators.h"
  #include "llvm/CodeGen/MachineFrameInfo.h"
  #include "llvm/CodeGen/MachineFunction.h"
  #include "llvm/CodeGen/MachineInstr.h"
  #include "llvm/CodeGen/MachineJumpTableInfo.h"
  #include "llvm/CodeGen/MachineLoopInfo.h"
  #include "llvm/CodeGen/MachinePostDominators.h"
  #include "llvm/CodeGen/MachineRegisterInfo.h"
  #include "llvm/CodeGen/RegisterClassInfo.h"
  #include "llvm/CodeGen/RegisterScavenging.h"
  #include "llvm/CodeGen/ShrinkFrameUtils.h"
  #include "llvm/CodeGen/TargetFrameLowering.h"
  #include "llvm/CodeGen/TargetInstrInfo.h"
  #include "llvm/CodeGen/TargetLowering.h"
  #include "llvm/CodeGen/TargetRegisterInfo.h"
  #include "llvm/CodeGen/TargetSubtargetInfo.h"
  #include "llvm/IR/Attributes.h"
  #include "llvm/IR/Function.h"
  #include "llvm/InitializePasses.h"
  #include "llvm/MC/MCAsmInfo.h"
  #include "llvm/MC/MCContext.h"
  #include "llvm/MC/MCSymbol.h"
  #include "llvm/Support/CommandLine.h"
  #include "llvm/Support/Compiler.h"
  #include "llvm/Support/Debug.h"
  #include "llvm/Target/TargetMachine.h"
  #include <sstream>

  using namespace llvm;

  STATISTIC(numSRReduced, "Number of CSR spills+restores reduced.");
  STATISTIC(NumFunc, "Number of functions");
  STATISTIC(NumShrinkFrame, "Number of functions with frame shrunk to non-entry");
  STATISTIC(NumShrinkFrameDup,
            "Number of return blocks duplicated for shrink-frame");
  STATISTIC(NumNotSaveOrRestore,
            "Number of cases, in which one of the sets are empty.");
  STATISTIC(NumCandidates, "Number of shrink-wrapping candidates");
  STATISTIC(
      NumFuncWithSplitting,
      "Number of functions, for which we managed to split Save/Restore points");

  static cl::opt<bool> DisableShrinkFrame(
      "disable-shrink-frame", cl::init(false), cl::Hidden,
      cl::desc("Disable moving stack frame setup from entry to a later block"));

  static cl::opt<bool> DisableShrinkFrameDup(
      "disable-riscv-shrink-frame-dup", cl::init(false), cl::Hidden,
      cl::desc("Disable BB duplication for shrink-frame on shared returns"));

  static cl::opt<unsigned> ShrinkFrameMaxDupInsns(
      "max-dup-insns", cl::init(16), cl::Hidden,
      cl::desc("Max instructions to duplicate for shrink-frame shared returns "
              "(0 disables duplication)"));

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

    /// Edge node Pred -> Succ (MatchMBB == nullptr), or nullptr.
    AuxGraphNode *getEdgeNode(MachineBasicBlock *Pred, MachineBasicBlock *Succ) {
      AuxGraphNode *PN = getNode(Pred);
      AuxGraphNode *SN = getNode(Succ);
      if (!PN || !SN)
        return nullptr;
      for (AuxGraphNode *Edge : PN->Successors) {
        if (!Edge->MatchMBB && Edge->succ_size() == 1 &&
            *Edge->succ_begin() == SN)
          return Edge;
      }
      return nullptr;
    }

    void clear() {
      for (AuxGraphNode *N : Nodes)
        delete N;
      Nodes.clear();
      Entry = nullptr;
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

    // Entry and return blocks of the current function (includes frameless
    // shared-return clones --needed for choke/exit dominance). CSR restore
    // placement must filter clones separately via allowCSRRestoreOn().
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

    /// Materialize edge save/restore points. CSRSave on the unique edge into
    /// Prolog is folded onto Prolog (no trampoline); other save/restore edges
    /// are split. Returns true if any new block was created (MDT/MPDT stale).
    bool setupCFG();

    void setupSaveRestorePoints();

    /// Determine whether a block requires a stack frame (has calls, FI uses,
    /// or defs of ra/FP/SP).
    bool blockNeedsFrame(const MachineBasicBlock &MBB) const;

    /// Move frame setup from entry to the nearest common dominator of all
    /// frame-requiring blocks. Returns true if a non-entry Prolog was chosen.
    /// \p AllowDup enables shared-return block duplication for shrink-frame.
    bool runShrinkFrame(MachineFunction &MF, bool AllowDup);

    /// Frameless shared-return clones are real exits for dominance/choke
    /// analysis but must never receive CSR restores (no frame on that path).
    bool allowCSRRestoreOn(const MachineBasicBlock *BB) const {
      return BB && !MachineFunc->getFrameInfo().isShrinkFrameClone(BB);
    }

    /// Helpers for duplicating shared returns to enable shrink-frame.
    bool isReachableFrom(MachineBasicBlock *From, MachineBasicBlock *To) const;
    bool isPureFramedPred(MachineBasicBlock *Pred, MachineBasicBlock *P0) const;
    bool isPureFramelessPred(MachineBasicBlock *Pred,
                            MachineBasicBlock *P0) const;
    bool hasNonReturnMixedJoin(MachineBasicBlock *P0) const;
    MachineBasicBlock *liftOutOfLoops(MachineBasicBlock *P) const;
    unsigned countDupInsns(const MachineBasicBlock &MBB) const;
    bool canDupSharedReturn(MachineBasicBlock &J, MachineBasicBlock *P0) const;
    bool cloneSharedReturn(MachineBasicBlock &Orig, MachineBasicBlock *P0);
    void undoAllShrinkFrameClones(MachineFunction &MF);
    /// Undo shared-return clones, clear Prolog/Epilog and MFI CSR half-state.
    /// Call instead of bare setProlog(nullptr) whenever shrink-frame must fully
    /// roll back.
    void rollbackShrinkFrame(MachineFunction &MF);
    /// Undo clones then retry shrink-frame without duplication. Prefer over
    /// rollbackShrinkFrame when Clamp rejects an off-Prolog save but shrink-frame
    /// without dup may still succeed.
    bool demoteShrinkFrameCToB(MachineFunction &MF);

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

    ReturnBlocks.clear();
    CSRUsed.clear();
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
          if (allowCSRRestoreOn(ReturnBlocks[ri]))
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
    } else if (Node->MatchMBB && Node->MatchMBB->isReturnBlock() &&
              allowCSRRestoreOn(Node->MatchMBB)) {
      // Leaf return blocks: place restores (DFA assumes exits have no uses).
      // Do NOT treat noreturn exits (e.g. call __cxa_throw + barrier) as
      // returns --restore+CFI after noreturn poisons FDE at the call RA
      // (libunwind uses pc after the call). Unwind recovers CSRs via save
      // CFI instead. Matches GCC separate shrink-wrap: epi only on paths
      // that leave the framed region toward a normal EXIT.
      // Frameless shared-return clones are skipped (no frame on that path).
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
    return llvm::blockNeedsFrame(*MachineFunc, MBB);
  }

  bool ShrinkWrappingImpl::isReachableFrom(MachineBasicBlock *From,
                                          MachineBasicBlock *To) const {
    for (auto DI = df_begin(From), DE = df_end(From); DI != DE; ++DI)
      if (*DI == To)
        return true;
    return false;
  }

  bool ShrinkWrappingImpl::isPureFramedPred(MachineBasicBlock *Pred,
                                            MachineBasicBlock *P0) const {
    return MDT->dominates(P0, Pred);
  }

  bool ShrinkWrappingImpl::isPureFramelessPred(MachineBasicBlock *Pred,
                                              MachineBasicBlock *P0) const {
    return !MDT->dominates(P0, Pred) && !isReachableFrom(P0, Pred);
  }

  bool ShrinkWrappingImpl::hasNonReturnMixedJoin(MachineBasicBlock *P0) const {
    for (MachineBasicBlock &MBB : *MachineFunc) {
      if (MBB.isReturnBlock())
        continue;
      if (MDT->dominates(P0, &MBB))
        continue;
      if (isReachableFrom(P0, &MBB))
        return true;
    }
    return false;
  }

  MachineBasicBlock *
  ShrinkWrappingImpl::liftOutOfLoops(MachineBasicBlock *P) const {
    while (MachineLoop *L = MLI->getLoopFor(P)) {
      MachineBasicBlock *PH = L->getLoopPreheader();
      if (!PH)
        return nullptr;
      P = PH;
    }
    return P;
  }

  unsigned
  ShrinkWrappingImpl::countDupInsns(const MachineBasicBlock &MBB) const {
    unsigned N = 0;
    for (const MachineInstr &MI : MBB)
      if (!MI.isMetaInstruction())
        ++N;
    return N;
  }

  bool ShrinkWrappingImpl::canDupSharedReturn(MachineBasicBlock &J,
                                              MachineBasicBlock *P0) const {
    if (!J.isReturnBlock() || blockNeedsFrame(J) || J.isEHPad())
      return false;
    if (MLI->getLoopFor(&J))
      return false;

    const TargetInstrInfo *TII = MachineFunc->getSubtarget().getInstrInfo();
    bool SawReturn = false;
    for (const MachineInstr &MI : J.terminators()) {
      if (MI.isMetaInstruction())
        continue;
      if (MI.isBundle())
        return false;
      if (TII->isTailCall(MI) || !MI.isReturn())
        return false;
      SawReturn = true;
    }
    if (!SawReturn)
      return false;

    bool HasFramed = false;
    bool HasFrameless = false;
    for (MachineBasicBlock *Pred : J.predecessors()) {
      if (isPureFramedPred(Pred, P0)) {
        HasFramed = true;
      } else if (isPureFramelessPred(Pred, P0)) {
        HasFrameless = true;
        MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
        SmallVector<MachineOperand, 4> Cond;
        if (TII->analyzeBranch(*Pred, TBB, FBB, Cond))
          return false;
      } else {
        // Mixed / ambiguous predecessor --caller gives up C.
        return false;
      }
    }
    return HasFramed && HasFrameless;
  }

  bool ShrinkWrappingImpl::cloneSharedReturn(MachineBasicBlock &Orig,
                                            MachineBasicBlock *P0) {
    MachineFunction &MF = *MachineFunc;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

    MachineBasicBlock *Clone = MF.CreateMachineBasicBlock(Orig.getBasicBlock());
    Clone->setCallFrameSize(Orig.getCallFrameSize());
    MF.insert(std::next(MachineFunction::iterator(&Orig)), Clone);

    // CloneMachineInstr copies Pre/PostInstrSymbol pointers. Emitting the same
    // temp label twice (e.g. .Lpcrel_hiN for AUIPC/%pcrel_lo) fails the
    // assembler. Remap labels defined in this block and MCSymbol operands that
    // refer to them.
    DenseMap<MCSymbol *, MCSymbol *> SymbolMap;
    for (const MachineInstr &MI : Orig) {
      auto NoteDef = [&](MCSymbol *S) {
        if (!S || SymbolMap.contains(S))
          return;
        SymbolMap[S] = MF.getContext().createNamedTempSymbol("pcrel_hi");
      };
      NoteDef(MI.getPreInstrSymbol());
      NoteDef(MI.getPostInstrSymbol());
    }

    for (MachineInstr &MI : Orig) {
      MachineInstr *NewMI = MF.CloneMachineInstr(&MI);
      NewMI->clearKillInfo();
      if (MCSymbol *S = NewMI->getPreInstrSymbol()) {
        auto It = SymbolMap.find(S);
        assert(It != SymbolMap.end());
        NewMI->setPreInstrSymbol(MF, It->second);
      }
      if (MCSymbol *S = NewMI->getPostInstrSymbol()) {
        auto It = SymbolMap.find(S);
        assert(It != SymbolMap.end());
        NewMI->setPostInstrSymbol(MF, It->second);
      }
      if (!SymbolMap.empty()) {
        for (MachineOperand &MO : NewMI->operands()) {
          if (!MO.isMCSymbol())
            continue;
          auto It = SymbolMap.find(MO.getMCSymbol());
          if (It != SymbolMap.end())
            MO.ChangeToMCSymbol(It->second, MO.getTargetFlags());
        }
      }
      Clone->push_back(NewMI);
    }

    for (auto SI = Orig.succ_begin(), SE = Orig.succ_end(); SI != SE; ++SI)
      Clone->copySuccessor(&Orig, SI);

    // Redirect frameless preds; on failure restore already-moved edges and erase
    // the unregistered clone (pair map is updated only on success).
    SmallVector<MachineBasicBlock *, 4> Redirected;
    auto FailAndCleanup = [&]() {
      for (MachineBasicBlock *Pred : Redirected) {
        MachineBasicBlock *PrevFT = Pred->getNextNode();
        const bool WasCloneFT = PrevFT == Clone;
        Pred->ReplaceUsesOfBlockWith(Clone, &Orig);
        MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
        SmallVector<MachineOperand, 4> Cond;
        if (!TII->analyzeBranch(*Pred, TBB, FBB, Cond))
          Pred->updateTerminator(WasCloneFT ? &Orig : PrevFT);
      }
      Clone->eraseFromParent();
      return false;
    };

    SmallVector<MachineBasicBlock *, 4> Preds(Orig.pred_begin(), Orig.pred_end());
    for (MachineBasicBlock *Pred : Preds) {
      if (!isPureFramelessPred(Pred, P0))
        continue;
      MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
      SmallVector<MachineOperand, 4> Cond;
      if (TII->analyzeBranch(*Pred, TBB, FBB, Cond))
        return FailAndCleanup();

      MachineBasicBlock *PrevFT = Pred->getNextNode();
      const bool WasOrigFallthrough = PrevFT == &Orig;
      Pred->ReplaceUsesOfBlockWith(&Orig, Clone);
      Redirected.push_back(Pred);
      Pred->updateTerminator(WasOrigFallthrough ? Clone : PrevFT);
    }

    if (Redirected.empty())
      return FailAndCleanup();

    // Only mutate Orig kills after redirects succeed.
    for (MachineInstr &MI : Orig)
      MI.clearKillInfo();

    if (MF.getRegInfo().tracksLiveness()) {
      MachineBasicBlock *Blocks[] = {&Orig, Clone};
      fullyRecomputeLiveIns(Blocks);
    }

    MFI.registerShrinkFrameClone(&Orig, Clone);
    ++NumShrinkFrameDup;
    LLVM_DEBUG(dbgs() << "Shrink-frame: cloned " << printMBBReference(Orig)
                      << " -> " << printMBBReference(*Clone) << "\n");
    return true;
  }

  void ShrinkWrappingImpl::undoAllShrinkFrameClones(MachineFunction &MF) {
    MachineFrameInfo &MFI = MF.getFrameInfo();
    if (!MFI.hasShrinkFrameClones())
      return;

    // Scrub CSR / return bookkeeping before blocks disappear.
    for (const auto &KV : MFI.getShrinkFrameClones()) {
      MachineBasicBlock *Clone = KV.second;
      if (AuxGraphNode *N = AuxillaryCFG.getNode(Clone)) {
        CSRSave.erase(N);
        CSRRestore.erase(N);
        N->MatchMBB = nullptr;
      }
      llvm::erase(ReturnBlocks, Clone);
    }

    undoShrinkFrameClones(MF);
  }

  void ShrinkWrappingImpl::rollbackShrinkFrame(MachineFunction &MF) {
    LLVM_DEBUG(dbgs() << "Shrink-frame rollback for " << MF.getName() << "\n");
    MachineFrameInfo &MFI = MF.getFrameInfo();
    const bool HadClones = MFI.hasShrinkFrameClones();
    // Scrub CSR maps that reference clones, then abandon Prolog + pairs.
    // Caller must recompute CSRSave/CSRRestore on the restored CFG (see Clamp
    // path in run()) — these in-memory maps are left stale on purpose here.
    undoAllShrinkFrameClones(MF);
    MFI.setProlog(nullptr);
    MFI.setEpilog(nullptr);
    MFI.clearSavePoints();
    MFI.clearRestorePoints();
    if (HadClones) {
      MDT->recalculate(MF);
      MPDT->recalculate(MF);
      MLI->calculate(*MDT);
    }
  }

  /// Undo shared-return clones, then retry shrink-frame without duplication.
  /// Returns true if a non-entry Prolog was placed. Used when Clamp rejects an
  /// off-Prolog CSR save so we retry without dup instead of abandoning
  /// shrink-frame entirely.
  bool ShrinkWrappingImpl::demoteShrinkFrameCToB(MachineFunction &MF) {
    MachineFrameInfo &MFI = MF.getFrameInfo();
    LLVM_DEBUG(dbgs() << "Shrink-frame: retry without shared-return dup for "
                      << MF.getName() << "\n");
    undoAllShrinkFrameClones(MF);
    MFI.setProlog(nullptr);
    MFI.setEpilog(nullptr);
    MFI.clearSavePoints();
    MFI.clearRestorePoints();
    MDT->recalculate(MF);
    MPDT->recalculate(MF);
    MLI->calculate(*MDT);
    return runShrinkFrame(MF, /*AllowDup=*/false);
  }

  /// True if any CFG edge illegally enters the prospective frame region (only
  /// Pred -> P0 is allowed when crossing into the framed set).
  static bool hasIllegalFrameEntryEdge(MachineFunction &MF, MachineBasicBlock *P0,
                                      MachineDominatorTree &DT) {
    for (MachineBasicBlock &Pred : MF) {
      const bool PredFramed = DT.dominates(P0, &Pred);
      for (MachineBasicBlock *Succ : Pred.successors()) {
        const bool SuccFramed = DT.dominates(P0, Succ);
        if (PredFramed == SuccFramed)
          continue;
        if (!PredFramed && SuccFramed && Succ != P0)
          return true;
        if (PredFramed && !SuccFramed)
          return true;
      }
    }
    return false;
  }

  bool ShrinkWrappingImpl::runShrinkFrame(MachineFunction &MF, bool AllowDup) {
    if (DisableShrinkFrame)
      return false;

    const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
    MachineFrameInfo &MFI = MF.getFrameInfo();

    // Bail-out conditions: these functions cannot have their frame shrunk.
    const Function &F = MF.getFunction();
    if (F.hasFnAttribute(Attribute::Naked) ||
        F.hasFnAttribute(Attribute::NoReturn) ||
        F.hasFnAttribute(Attribute::OptimizeNone) ||
        MFI.hasVarSizedObjects() ||
        MFI.hasOpaqueSPAdjustment() ||
        MF.getFunction().getCallingConv() == CallingConv::GHC ||
        MF.hasEHFunclets() ||
        F.hasFnAttribute(Attribute::SanitizeAddress) ||
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
    MachineBasicBlock *P0 = FrameBlocks[0];
    for (unsigned I = 1, E = FrameBlocks.size(); I < E; ++I) {
      P0 = MDT->findNearestCommonDominator(P0, FrameBlocks[I]);
      if (!P0)
        break;
    }

    if (!P0 || P0 == Entry)
      return false;

    P0 = liftOutOfLoops(P0);
    if (!P0 || P0 == Entry)
      return false;

    auto RebuildCFGAnalyses = [&]() {
      MDT->recalculate(MF);
      MPDT->recalculate(MF);
      MLI->calculate(*MDT);
    };

    auto CollectMixedReturns = [&](MachineBasicBlock *P,
                                  SmallVectorImpl<MachineBasicBlock *> &Out) {
      Out.clear();
      for (MachineBasicBlock &MBB : MF) {
        if (!MBB.isReturnBlock())
          continue;
        if (MDT->dominates(P, &MBB))
          continue;
        if (isReachableFrom(P, &MBB))
          Out.push_back(&MBB);
      }
    };

    auto GiveUpDup = [&]() {
      if (MFI.hasShrinkFrameClones()) {
        undoAllShrinkFrameClones(MF);
        RebuildCFGAnalyses();
      }
      return false;
    };

    // Non-return mixed join: only return duplication cannot fix this.
    if (hasNonReturnMixedJoin(P0)) {
      LLVM_DEBUG(dbgs() << "Shrink-frame bail: non-return mixed join from "
                        << printMBBReference(*P0) << "\n");
      return false;
    }

    SmallVector<MachineBasicBlock *, 4> MixedReturns;
    CollectMixedReturns(P0, MixedReturns);

    if (!MixedReturns.empty()) {
      if (!AllowDup)
        return false;

      unsigned Budget = ShrinkFrameMaxDupInsns;
      unsigned LiftAttempts = 0;
      constexpr unsigned MaxLift = 2;

      while (!MixedReturns.empty()) {
        // Prefer smaller blocks first.
        llvm::sort(MixedReturns, [&](MachineBasicBlock *A, MachineBasicBlock *B) {
          return countDupInsns(*A) < countDupInsns(*B);
        });

        for (MachineBasicBlock *J : MixedReturns) {
          // Ambiguous preds: hard give-up.
          bool Ambiguous = false;
          for (MachineBasicBlock *Pred : J->predecessors()) {
            if (!isPureFramedPred(Pred, P0) && !isPureFramelessPred(Pred, P0)) {
              Ambiguous = true;
              break;
            }
          }
          if (Ambiguous) {
            LLVM_DEBUG(dbgs() << "Shrink-frame giveUp: ambiguous pred of "
                              << printMBBReference(*J) << "\n");
            return GiveUpDup();
          }

          unsigned Cost = countDupInsns(*J);
          if (Cost > Budget || !canDupSharedReturn(*J, P0))
            continue;

          if (!cloneSharedReturn(*J, P0))
            return GiveUpDup();
          Budget -= Cost;
          RebuildCFGAnalyses();

          if (hasNonReturnMixedJoin(P0) ||
              hasIllegalFrameEntryEdge(MF, P0, *MDT))
            return GiveUpDup();
        }

        CollectMixedReturns(P0, MixedReturns);
        if (MixedReturns.empty())
          break;

        // Still mixed: undo this round's clones (if any) and lift P0 to idom,
        // even when no block was cloned (lift alone may dominate remaining
        // returns).
        if (LiftAttempts >= MaxLift) {
          LLVM_DEBUG(dbgs() << "Shrink-frame giveUp: still mixed after dup\n");
          return GiveUpDup();
        }

        undoAllShrinkFrameClones(MF);
        Budget = ShrinkFrameMaxDupInsns;
        RebuildCFGAnalyses();

        MachineDomTreeNode *PN = MDT->getNode(P0);
        if (!PN || !PN->getIDom())
          return GiveUpDup();
        MachineBasicBlock *Next = PN->getIDom()->getBlock();
        Next = liftOutOfLoops(Next);
        if (!Next || Next == Entry || Next == P0)
          return GiveUpDup();
        P0 = Next;
        ++LiftAttempts;

        if (hasNonReturnMixedJoin(P0))
          return GiveUpDup();
        CollectMixedReturns(P0, MixedReturns);
      }
    }

    // Pre-setProlog recheck: no mixed returns; every original R dominated by P0.
    CollectMixedReturns(P0, MixedReturns);
    if (!MixedReturns.empty())
      return GiveUpDup();

    for (const auto &KV : MFI.getShrinkFrameClones()) {
      MachineBasicBlock *Orig = KV.first;
      MachineBasicBlock *Clone = KV.second;
      if (!MDT->dominates(P0, Orig) || MDT->dominates(P0, Clone)) {
        LLVM_DEBUG(dbgs() << "Shrink-frame giveUp: bad domination for pair "
                          << printMBBReference(*Orig) << " / "
                          << printMBBReference(*Clone) << "\n");
        return GiveUpDup();
      }
    }

    if (hasIllegalFrameEntryEdge(MF, P0, *MDT))
      return GiveUpDup();

    // Check canUseAsPrologue (target hook).
    if (!TFI->canUseAsPrologue(*P0))
      return GiveUpDup();

    // Delayed Prolog cannot be used when entry/frameless paths touch the stack,
    // CSR saves sit before the candidate Prolog, or a shared return merges
    // framed+frameless preds. PEI has a matching hoist guard; bail here so we
    // do not leave broken shared-return clones.
    if (mustHoistShrinkFramePrologToEntry(MF, P0, *MDT)) {
      LLVM_DEBUG(dbgs() << "Shrink-frame bail: Prolog "
                        << printMBBReference(*P0)
                        << " must be at Entry for correctness\n");
      return GiveUpDup();
    }

    LLVM_DEBUG(dbgs() << "Shrink-frame: setting Prolog to "
                      << printMBBReference(*P0) << "\n");

    MFI.setProlog(P0);
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
    const bool AllowDup =
        !DisableShrinkFrameDup && ShrinkFrameMaxDupInsns != 0;
    runShrinkFrame(MF, AllowDup);

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
      // place saves in entry (no frame) --rollback.
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
          if (MFI.hasShrinkFrameClones())
            demoteShrinkFrameCToB(MF);
          else
            rollbackShrinkFrame(MF);
        }
      }
      return MFI.getProlog() != nullptr;
    }

    placeSpillsAndRestores(MF);

    // Frame-bound CSRs (ra / hard FP on RISC-V): never delay as separate
    // components — match GCC TARGET_SHRINK_WRAP_GET_SEPARATE_COMPONENTS
    // (bitmap_clear_bit RETURN_ADDR / hard FP). They always travel with the
    // main shrink-frame Prolog (addi sp + sd ra), while ordinary s* may still
    // be multi-point.
    //
    // If RA already clobbers a frame-bound reg on a path that does not go
    // through the delayed Prolog (`mv ra, a1` then early ret), hoist Prolog to
    // Entry so sd ra still precedes that use. Keep CSR maps / clones — only the
    // frame root moves (yacr2 Maze2Mech).
    //
    // MRI.isPhysRegModified(ra) can be false around calls (ABI regmask); PEI
    // still spills ra when hasCalls. Leaf functions may use ra as scratch after
    // a conditional save (scale_bitcount) — pin whenever Chow placed a save,
    // or hasCalls / determineCalleeSaves / isPhysRegModified says so.
    auto PinFrameBoundCSRs = [&]() {
      const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
      SmallVector<Register, 4> FrameBound;
      TFI->getFrameBoundCalleeSaves(MF, FrameBound);
      if (FrameBound.empty())
        return;

      MachineBasicBlock *SFProlog = MFI.getProlog();
      if (SFProlog && SFProlog != Entry &&
          mustHoistShrinkFramePrologToEntry(MF, SFProlog, *MDT) &&
          !MFI.hasShrinkFrameClones()) {
        LLVM_DEBUG(dbgs() << "ShrinkWrap: hoist Prolog to Entry (main prologue "
                            "before stack/returns)\n");
        MFI.setProlog(Entry);
        SFProlog = Entry;
      }
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
                        UsedCSRegs.test(Reg.id()) ||
                        MRI.isPhysRegModified(Reg);
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
          if (!allowCSRRestoreOn(RB))
            continue;
          if (!SFProlog || MDT->dominates(SFProlog, RB))
            CSRRestore[AuxillaryCFG.getNode(RB)].set(Reg.id());
        }
      }
    };

    PinFrameBoundCSRs();

    // Drop restores on noreturn exits (no ld/CFI after throw/abort/exit -- that
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

    auto ScrubNoreturnRestores = [&]() {
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
          // Dominates, not reachable -- see path-sensitive restore below.
          for (MachineBasicBlock *RB : ReturnBlocks) {
            if (!allowCSRRestoreOn(RB))
              continue;
            if (!ContainsNoReturnCall(*RB) &&
                MDT->dominates(SKV.first->MatchMBB, RB))
              Targets.insert(RB);
          }
        }
        // No save recorded: keep ABI safe (e.g. frame-bound race) on all returns.
        if (!AnySave) {
          for (MachineBasicBlock *RB : ReturnBlocks)
            if (allowCSRRestoreOn(RB))
              Targets.insert(RB);
        }
        for (MachineBasicBlock *RB : Targets)
          CSRRestore[AuxillaryCFG.getNode(RB)].set(Reg);
      }
    };

    // Path-sensitive restores on *return blocks* only:
    // Drop restore at return R when no save dominates R (early-exit / diamond:
    // ray Sphere::intersect bb.10). Skip frame-bound CSRs -- re-pinned below.
    //
    // When dropping from a shared return, rehome onto Pred->R edges where Pred
    // is on the save side so the hot path still restores after setupCFG splits
    // that edge (pr78622). PEI has a matching fallback if maps still miss.
    auto RehomePathSensitiveRestores = [&]() {
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
        if (SaveBBs.empty())
          continue;

        for (auto &[Node, Restores] : CSRRestore) {
          if (!Restores.test(Reg))
            continue;
          MachineBasicBlock *MBB = Node->MatchMBB;
          if (!MBB || !MBB->isReturnBlock())
            continue;
          if (MFI.isShrinkFrameClone(MBB))
            continue;
          if (any_of(SaveBBs, [&](MachineBasicBlock *S) {
                return MDT->dominates(S, MBB);
              }))
            continue;
          Restores.reset(Reg);
          for (MachineBasicBlock *Pred : MBB->predecessors()) {
            if (!any_of(SaveBBs, [&](MachineBasicBlock *S) {
                  return Pred == S || MDT->dominates(S, Pred);
                }))
              continue;
            if (AuxGraphNode *Edge = AuxillaryCFG.getEdgeNode(Pred, MBB))
              CSRRestore[Edge].set(Reg);
            else
              LLVM_DEBUG(dbgs() << "Path-sensitive rehome: no edge node "
                                << printMBBReference(*Pred) << " -> "
                                << printMBBReference(*MBB) << "\n");
          }
        }
      }
    };

    auto FinalizeCSRMaps = [&]() {
      ScrubNoreturnRestores();
      RehomePathSensitiveRestores();
      // Re-pin after path-sensitive / noreturn cleanup.
      PinFrameBoundCSRs();
    };

    FinalizeCSRMaps();

    // Clamp BEFORE setupCFG mutates the MachineFunction. If a CSR save sits
    // outside the shrink-frame region, abandon Prolog/clones and recompute CSR
    // placement on the restored CFG -- otherwise Save/Restore maps from the
    // cloned world are left applied to a shared return (save w/o restore).
    auto RebuildCSRAfterShrinkFrameAbort = [&]() {
      LLVM_DEBUG(dbgs() << "Recomputing CSR placement after shrink-frame "
                          "rollback\n");
      AuxillaryCFG.clear();
      CSRSave.clear();
      CSRRestore.clear();
      createAuxillaryCFG();
      ReversePostOrderTraversal<MachineBasicBlock *> NewRPOT(&*MF.begin());
      StackAddressUsedBlockInfo.clear();
      StackAddressUsedBlockInfo.resize(MF.getNumBlockIDs(), true);
      (void)calculateSets(MF, NewRPOT, RS.get());
      StackAddressUsedBlockInfo.clear();
      placeSpillsAndRestores(MF);
      PinFrameBoundCSRs();
      FinalizeCSRMaps();
    };

    if (MachineBasicBlock *SFP = MFI.getProlog()) {
      for (const auto &[Node, Regs] : CSRSave) {
        if (Regs.empty() || !Node->MatchMBB)
          continue;
        SmallVector<MCRegister, 4> SaveRegList;
        for (unsigned Reg : Regs)
          SaveRegList.emplace_back(Reg);
        if (!isLegalOffPrologCSRSavePoint(MF, SFP, Node->MatchMBB, SaveRegList,
                                          *MDT)) {
          LLVM_DEBUG(dbgs() << "Clamp fail: CSR save in "
                            << printMBBReference(*Node->MatchMBB)
                            << " not dominated by Prolog "
                            << printMBBReference(*SFP)
                            << " — retry without shared-return dup\n");
          if (MFI.hasShrinkFrameClones())
            demoteShrinkFrameCToB(MF);
          else
            rollbackShrinkFrame(MF);
          RebuildCSRAfterShrinkFrameAbort();
          break;
        }
      }
    }

    if (setupCFG()) {
      // setupCFG() splits edges and creates new blocks; rebuild dominators for
      // any subsequent queries / verification.
      MDT->recalculate(MF);
      MPDT->recalculate(MF);
    }

    // Strip any CSR save/restore accidentally attached to frameless shared-return
    // clones before verification and MFI materialization.
    for (const auto &KV : MFI.getShrinkFrameClones()) {
      if (AuxGraphNode *N = AuxillaryCFG.getNode(KV.second)) {
        CSRSave.erase(N);
        CSRRestore.erase(N);
      }
    }

    verifySpillRestorePlacement();

    // Drop stack-use pollution from calculateSets(); Save/Restore lists must
    // come only from CSR placement.
    SaveBlocks.clear();
    RestoreBlocks.clear();
    setupSaveRestorePoints();

    // Publish maps so mustHoist can see off-Prolog SavePoints.
    if (!SavePoints.get().empty() && !RestorePoints.get().empty()) {
      MFI.setSavePoints(SavePoints.get());
      MFI.setRestorePoints(RestorePoints.get());
    }

    // If prologue must be hoisted to entry while clones remain, undo clones in
    // ShrinkWrapping and recompute CSR maps. Never leave clones for PEI to
    // undo while RestorePoints still reflect the duplicated CFG.
    if (MachineBasicBlock *SFP = MFI.getProlog();
        SFP && SFP != Entry && MFI.hasShrinkFrameClones() &&
        mustHoistShrinkFramePrologToEntry(MF, SFP, *MDT)) {
      LLVM_DEBUG(dbgs() << "ShrinkWrap: mustHoist with clones — retry shrink-frame "
                           "without dup before PEI\n");
      demoteShrinkFrameCToB(MF);
      RebuildCSRAfterShrinkFrameAbort();
      if (setupCFG()) {
        MDT->recalculate(MF);
        MPDT->recalculate(MF);
      }
      for (const auto &KV : MFI.getShrinkFrameClones()) {
        if (AuxGraphNode *N = AuxillaryCFG.getNode(KV.second)) {
          CSRSave.erase(N);
          CSRRestore.erase(N);
        }
      }
      SaveBlocks.clear();
      RestoreBlocks.clear();
      SavePoints.clear();
      RestorePoints.clear();
      setupSaveRestorePoints();
      if (!SavePoints.get().empty() && !RestorePoints.get().empty()) {
        MFI.setSavePoints(SavePoints.get());
        MFI.setRestorePoints(RestorePoints.get());
      } else {
        MFI.clearSavePoints();
        MFI.clearRestorePoints();
      }
    }

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
        // Re-check after setupCFG (same policy as PinFrameBoundCSRs).
        // Clones incompatible with prologue hoisting were undone above.
        if (MachineBasicBlock *SFP = MFI.getProlog();
            SFP && SFP != Entry &&
            mustHoistShrinkFramePrologToEntry(MF, SFP, *MDT) &&
            !MFI.hasShrinkFrameClones()) {
          LLVM_DEBUG(dbgs() << "ShrinkWrap(final): hoist Prolog to Entry\n");
          MFI.setProlog(Entry);
        }
        MachineBasicBlock *PinBlock =
            MFI.getProlog() ? MFI.getProlog() : Entry;
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
            if (!allowCSRRestoreOn(RB))
              continue;
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

    if (!ShrinkFrameProlog) {
      // Defensive: never leave shared-return clones without a Prolog.
      // Only undo clones — do not clear Save/RestorePoints (entry-full CSR maps
      // may still be valid).
      if (MFI.hasShrinkFrameClones())
        undoAllShrinkFrameClones(MF);
      else
        MFI.setProlog(nullptr);
    }
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
    NewBB->addSuccessor(Succ);           // Add NewBB --Succ

    for (const MachineBasicBlock::RegisterMaskPair &LI : Succ->liveins())
      NewBB->addLiveIn(LI.PhysReg);

    TII->insertUnconditionalBranch(*NewBB, Succ, DebugLoc());

    return NewBB;
  }

  bool ShrinkWrappingImpl::setupCFG() {
    bool SplitAnyEdge = false;
    MachineBasicBlock *Prolog = MachineFunc->getFrameInfo().getProlog();

    for (auto &[Node, Regs] : CSRSave) {
      if (!Regs.empty() && !Node->MatchMBB) {
        assert(Node->pred_size() == 1 &&
              "Auxillary node can have only one predecessor!");
        assert(Node->succ_size() == 1 &&
              "Auxillary node can have only one successor!");
        MachineBasicBlock *Pred = (*Node->pred_begin())->MatchMBB;
        MachineBasicBlock *Succ = (*Node->succ_begin())->MatchMBB;
        assert(Pred && Succ && "Edge node endpoints must map to real MBBs");

        // When CSR save is on the unique edge into Prolog, attach it to Prolog
        // instead of splitting an empty trampoline. A trampoline before Prolog
        // is never dominated by Prolog, so mustHoist would hoist/demote and
        // undo a valid delayed frame. Only fold Succ==Prolog with that unique
        // predecessor; restores and other saves (e.g. into grow) still split.
        if (Prolog && Succ == Prolog && Prolog->pred_size() == 1 &&
            *Prolog->pred_begin() == Pred) {
          LLVM_DEBUG(dbgs() << "setupCFG: fold CSRSave edge "
                            << printMBBReference(*Pred) << " -> "
                            << printMBBReference(*Succ)
                            << " onto Prolog (no trampoline)\n");
          Node->MatchMBB = Prolog;
          continue;
        }

        MachineBasicBlock *NewBB = splitEdge(Pred, Succ);
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
        // Never fold restore edges into Prolog/Epilog: a trampoline after the
        // framed region (or before a shared return) is required for PEI order.
        MachineBasicBlock *NewBB = splitEdge((*Node->pred_begin())->MatchMBB,
                                            (*Node->succ_begin())->MatchMBB);
        Node->MatchMBB = NewBB;
        SplitAnyEdge = true;
      }
    }
    return SplitAnyEdge;
  }

  void ShrinkWrappingImpl::setupSaveRestorePoints() {
    // Belt-and-suspenders: never emit CSR save/restore on frameless shared-return
    // clones. ReturnBlocks still lists clones for choke/exit analysis; restore
    // on clone returns is inserted by PEI appendShrinkFrameCloneRestorePoints.
    MachineFrameInfo &MFI = MachineFunc->getFrameInfo();
    for (const auto &KV : MFI.getShrinkFrameClones()) {
      MachineBasicBlock *Clone = KV.second;
      if (AuxGraphNode *N = AuxillaryCFG.getNode(Clone)) {
        CSRSave.erase(N);
        CSRRestore.erase(N);
      }
    }

    for (auto &[Node, Regs] : CSRSave) {
      if (!Node->MatchMBB || MFI.isShrinkFrameClone(Node->MatchMBB))
        continue;
      for (Register Reg : Regs)
        SavePoints.insertReg(Reg, Node->MatchMBB, &SaveBlocks);
    }
    for (auto &[Node, Regs] : CSRRestore) {
      if (!Node->MatchMBB || !allowCSRRestoreOn(Node->MatchMBB))
        continue;
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

    // Data-flow shrink-wrapping requires the target to opt in (RISC-V:
    // -riscv-shrink-wrapping-dataflow).
    if (!TFI->enableCSRSaveRestorePointsSplit())
      return false;

    return TFI->enableShrinkWrapping(MF) &&
          // Windows with CFI has some limitations that make it impossible
          // to use shrink-wrapping.
          !MF.getTarget().getMCAsmInfo()->usesWindowsCFI() &&
          // Sanitizers look at the value of the stack at the location
          // of the crash. Since a crash can happen anywhere, the
          // frame must be lowered before anything else happen for the
          // sanitizers to be able to get a correct stack frame.
          !(MF.getFunction().hasFnAttribute(Attribute::SanitizeAddress) ||
            MF.getFunction().hasFnAttribute(Attribute::SanitizeThread) ||
            MF.getFunction().hasFnAttribute(Attribute::SanitizeMemory) ||
            MF.getFunction().hasFnAttribute(Attribute::SanitizeType) ||
            MF.getFunction().hasFnAttribute(Attribute::SanitizeHWAddress));
  }
