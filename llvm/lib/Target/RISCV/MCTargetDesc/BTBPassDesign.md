# BTB Fetch-Line 控制流密度优化（MC）

在 **MC** 层于控制流指令**之后**插入可收缩 NOP 池（**NBF**），经 **layout / shrink** 使每个 fetch-line 窗口内控制流条数不超过上限 **N**。不依赖 MIR BTB pass。符号与缩写见下表。

---

## 术语与符号

| 术语 / 符号 | 含义 |
| ----------- | ---- |
| **NBF** | **N**ops **B**eside branch **F**ragment，`MCFragment::FT_NopsBesideBranch`；按字节计数的可收缩 NOP 区，`setNumBytes` **只缩不增**，写出 object 时由 `writeNopData` 填充 |
| **后池** | 每条控制流在 emit 路径上**至多一片** NBF，紧挨在该控制流指令**之后**（由 `emitInstructionEnd` 插入） |
| **控制流指令** | 执行后下一条取指 PC **不保证**为当前顺序后继的指令（分支、call、jump、ret/tail 等）。本文「分支」与**控制流**同义，**不**仅指条件分支 |
| **F** | Fetch line 字节数，`BTBFetchLineSize`，选项 `-riscv-fetchline-size`，默认 32，须为 **2 的幂**（配合 `llvm::Align` / 函数对齐） |
| **N** | 每个 fetch-line 滑动窗口内允许的控制流条数上限，`BTBMaxBranchesPerFetchLine`，选项 `-riscv-btb-max-branches-per-fetchline`，默认 4 |
| **I_min** | 单条控制流指令的**最小机器码长度**：目标含 **Zca** 时为 **2**，否则 **4**（与压缩指令如 `C.BEQZ` 一致） |
| **M** | 合法 NOP 填充的**最小字节粒度**，实现中与 **I_min** 相同；`NumBytes` 为 `max(公式, M)` |
| **滑动窗口** | 半开区间 **`[w, w+F)`**，`w` 按 **F** 步进；verify 直接统计窗内条数；shrink 在此基础上用 **BCritical** 等启发式 |
| **InsertKind** | `NopsBesideBranchKind`：**`OnExecPath`**（顺序后继可能被取指）/ **`OffExecPath`**（远跳、ret 类，后继通常不执行）；决定初值公式与 shrink 优先级 |
| **NumBytes** | NBF 当前字节数；**0** 表示空池（emit 时可不创建 fragment，shrink 末可 `removeFragment`） |
| **BCritical** | shrink 时从 `BranchIdx` 起第 **(N+1)** 条控制流的 section offset，用于限制本窗可删 NOP，避免把该分支「挤进」当前窗 |
| **RestartWinBase** | 上次 shrink 修改 NBF 时的窗口起点 `w`；外层 `while (shrinkSection)` 跨调用保留，用于回退重扫 |
| **Segment** | 相邻 **`MCAlignFragment`** 之间的连续代码区；对齐**填充区内**不插入/不扫描与代码混排的 shrink 窗口 |
| **layout-preserve** | BTB object 在 lld 中 relax 删指令时用 **NOP 写回**（`hasBtbOptimized`），保持与 MC layout 一致 |
| **RIVAI_BTB_OPTIMIZED** | ELF 属性 Tag（值 32768），标记本 object 启用 BTB 优化；lld 读入后不写入输出属性流 |
| **非法 CLI** | 显式设置 **N=0**、**F=0** 或 **F 非 2 的幂** 时 `cl::callback` → `report_fatal_error` |

---

## 代码落点

| 模块                     | 文件                                                                   | 职责                                                                       |
| ---------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| CLI / MC helpers       | `llvm/include/llvm/Target/RISCV/RISCVBTBBranchRelaxation.h` | BTB 相关 `cl::opt` 与 shrink/verify API                             |
| Emit / shrink / verify | `llvm/lib/Target/RISCV/MCTargetDesc/RISCVAsmBackend.cpp`             | 分类、`emitInstructionEnd`、`refreshNBFInsertKindsAfterRelax`、`shrinkSection`、`verifyBTBFetchLineBranchLimits` |
| ELF 发射hooker           | `llvm/lib/Target/RISCV/MCTargetDesc/RISCVELFStreamer.cpp`            | `emitInstruction` 前后调 AsmBackend                                         |
| 伪指令展开                  | `llvm/lib/Target/RISCV/MCTargetDesc/RISCVMCCodeEmitter.cpp`          | `PseudoCALL`/`PseudoTAIL` → `AUIPC`+`JALR`                               |
| NBF 类型                 | `llvm/include/llvm/MC/MCSection.h`                                   | `FT_NopsBesideBranch`、`NopsBesideBranchKind`                             |
| Codegen                | `llvm/lib/CodeGen/MachineBlockPlacement.cpp`                         | BTB 开启时跳过 `alignBlocks`                                                  |
| AsmPrinter             | `llvm/lib/Target/RISCV/RISCVAsmPrinter.cpp`                          | 函数对齐 ≥ F、写 `RIVAI_BTB_OPTIMIZED`                                         |
| Layout 循环              | `llvm/lib/MC/MCAssembler.cpp`                                        | `while (shrinkSection) layoutSection`                                    |
| 链接                     | `lld/ELF/Arch/RISCV.cpp`                                             | 解析 BTB 属性、`hasBtbOptimized`、relax 时 NOP 填回                               |

---

## 设计思路

### 问题

取指按 **fetch line**（宽度 **F** 字节，如 32B）从 I-cache 读入。同一 fetch line 内若聚集过多**会改写 PC 的指令**（分支、call、jump、ret 等），微架构上往往要在 **BTB / 分支预测** 中为这些目标同时维护条目，**密度过高**会带来 BTB 压力与预测干扰。

本优化的目标不是「减少分支条数」，而是在**不改变程序语义**的前提下，通过插入可删的 **NOP 填充**，使任意滑动窗口 **`[addr, addr+F)`** 内被统计的控制流指令条数 **≤ N**（`N` 由 `-riscv-btb-max-branches-per-fetchline` 给出）。

### 策略（MC-only）

1. **Emit 阶段（乐观上界）**  
   在每条被识别的控制流指令**之后**插入一片 **NBF**（后池），初值按 §「NBF 初值公式推导」计算。初值偏大没关系：后续会收缩。

2. **Layout + relax**  
   使用 LLVM 既有 MC layout / `relaxOnce` 得到真实指令长度与 offset（含压缩、`PseudoLong*` 等）。不在 relax 内增大 NBF。

3. **Refresh InsertKind**  
   `relaxOnce` 收敛后、shrink 前，对仍标为 **On** 的 NBF：若链表中**前一 fragment** 内**最后一条**控制流（按 fixup + 字节扫描，与 shrink 同源）为 **Off-path**（如 `jal x0`、`C.J`），则将该池 **On→Off**，以便 shrink 按 Off 优先级处理；**不改 `NumBytes`**、不 layout。

4. **Shrink（收紧）**  
   `shrinkSection` 在仍满足「每窗 ≤ N」的前提下**只减小** NBF，尽量少留 NOP；每改一片池子即 `layoutSection` 更新 offset。

5. **Verify**  
   layout 结束后对每窗 `[w,w+F)` 再数一遍分支，超限则 warning/error。

6. **链接**  
   带 `RIVAI_BTB_OPTIMIZED` 的 object 在 lld 中 **layout-preserve**：relax 删掉的指令用 NOP 填回，避免链接把 MC 假设的布局挤乱。

### On / Off 两类后池

| 类型 | 语义 | shrink 倾向 |
| ---- | ---- | ----------- |
| **OnExecPath** | 条件支、call 等：**顺序后继可能被取指**（fall-through 或返回后继续） | 池内 NOP **更常被执行**，初值偏保守；shrink 时**优先**尝试缩小 On 池 |
| **OffExecPath** | 无条件跳、ret/tail 等：**顺序后继通常不执行** | 池内 NOP 多在「离轨」路径上，可用更激进的初值；shrink 在 On 之后处理 |

### 为何不用 MIR 插 NOP

MI 阶段没有最终 **section offset**，指令长度在 compress / relax 前后可变；BTB 窗口统计必须在 **MC layout 之后**与 **fixup / 真实机器码** 对齐，故由 **`emitInstructionEnd` + fragment** 完成，并关闭 `MachineBlockPlacement::alignBlocks` 以免与 NBF 重复插 NOP。

---

## NBF 初值公式推导

记号：**F** = fetch line 字节数，**N** = 每窗允许的控制流条数上限，**I_min** = 单条控制流指令在 ISA 下的**最小占用**（Zca → 2B，否则 4B，与实现中 **M** 相同）。实现里 `NumBytes = 0` 表示不插 NBF；否则 `NumBytes = max(公式结果, M)`，保证 NOP 粒度合法。

### 约束（共同）

Verify / 设计目标：对任意偏移 **w**，窗口 **`[w, w+F)`** 内控制流条数 **`Count(w) ≤ N`**。

Emit 初值是**充分条件**上的保守估计：在每条控制流后加 NOP，把**后续**控制流尽量推出当前 fetch line；Shrink 再在满足约束的前提下删 NOP。

### OnExecPath（call、条件支）

**直观**：一个 fetch line 里最多 **N** 条控制流，每条至少 **I_min** 字节（`C.BEQZ` 2B、`BEQ` 4B 等），则分支本身极端密排时最少占 **`N·I_min`**。剩余字节可作为「本条分支之后、把下一条推离本 line」的间隔上界：

```
OnBase = F - N * I_min
```

- **OnBase ≤ 0** → 不插池（`NumBytes = 0`）
- **OnBase > 0** → `NumBytes = max(OnBase, M)` → **`BTBOnCallPoolInitialBytes`**

**条件支额外项**：`relaxOnce` 后可能变为 **`[反条件支][jal][原后池]`**（`PseudoLong*`，见 §3.3）。初值过小则 relax 后 `jal` 仍可能与后续分支共线，且 **不能再增大** NBF（PC-rel 超距）。故：

```
BTBOnCondPoolInitialBytes = max(BTBOnCallPoolInitialBytes, F)
```

`emitInstructionEnd`：`isConditionalBranch` → 上式；普通 call → `BTBOnCallPoolInitialBytes`。

**例**：F=32，N=4，I_min=4 → OnBase=16；条件支初值 max(16,32)=**32**。

### OffExecPath（无条件跳、ret、tail）

顺序后继**通常不执行**，后池主要做**布局推开**，初值可更激进。

**N = 1**（每窗至多 1 条控制流）：

```
NumBytes = F
```

**N ≥ 2**：先为第一条分支留 **I_min**，剩余 **`F - I_min`** 均分为 **(N-1)** 段，每段除一条分支外还可放 NOP：

```
段长 = (F - I_min) / (N - 1)        // 整数除，向零截断
OffRaw = 段长 - I_min = (F - I_min) / (N - 1) - I_min
```

- **OffRaw ≤ 0** → 不插池  
- **OffRaw > 0** → `NumBytes = max(OffRaw, M)` → **`BTBOffPoolInitialBytes`**

**例**：F=32，N=4，I_min=4 → OffRaw = 28/3 - 4 = **9**。

### 初值与 shrink 的关系

| 阶段 | 作用 |
| ---- | ---- |
| 初值公式 | 保证「刚 emit 完」时布局**足够疏**，大概率通过 verify 或仅需少量 shrink |
| shrink | 在 **BCritical** 等约束下**减半** NBF，避免一次删掉过多导致第 (N+1) 条分支被「挤进」当前窗 |
| 最终 | 在满足 Count(w)≤N 的前提下 **NOP 尽量少** |

公式给出的是**闭式初值**；全局最优间隔分配是 NP 型的，实现采用「每分支一片后池 + 迭代 shrink」的启发式，与 verify 的滑动窗口口径一致即可。

---

## 1. 参数与 CLI

定义于 `RISCVBTBBranchRelaxation.h`（**F**、**N** 等符号见 **「术语与符号」**）。


| 选项                                          | 默认       | 作用                                                 |
| ------------------------------------------- | -------- | -------------------------------------------------- |
| `-riscv-btb-fetchline-branch-relaxation`    | false    | 总开关                                                |
| `-riscv-fetchline-size`                     | 32       | **F**；`F=0` 或非 2 幂 → `report_fatal_error`          |
| `-riscv-btb-max-branches-per-fetchline`     | 4        | **N**；`N=0` → fatal                                |
| `-riscv-btb-fetchline-violation-as-warning` | **true** | verify 超限 → warning                                |
| `-riscv-btb-shrink-section-diag-only`       | false    | `shrinkSection` 只做收集与首遍诊断后 **return false**，不缩 NBF |


**独立选项**（非 BTB 头文件）：`-riscv-branch-spacing`（`RISCVAsmBackend.cpp`）在 `emitInstructionBegin` 为条件支插入 `MCBranchSpacingFragment`，与 NBF **并存**。

**后端构造时缓存的初值**（`RISCVAsmBackend` 构造函数，flag 且 `F,N>0` 时）：

- `BTBOnCallPoolInitialBytes` = `max(F − N·I_min, M)`（若 `F−N·I_min≤0` 则为 0）
- `BTBOnCondPoolInitialBytes` = `max(BTBOnCallPoolInitialBytes, F)`（`PseudoLong*` 条件支 On 池 ≥ F，见 §4）
- `BTBOffPoolInitialBytes` = §4 Off 公式

`emitInstructionEnd` 通过 `getBTBPostPoolInitialBytes(Kind, isConditionalBranch(Inst))` 取上述缓存；**On + 条件支** 用 `BTBOnCondPoolInitialBytes`，**On + call** 用 `BTBOnCallPoolInitialBytes`。

---

## 2. 流水线（与 `MCAssembler::layout`）

```
MachineBlockPlacement（跳过 alignBlocks）
  → AsmPrinter（MF 对齐 ≥ F）
  → MC emit（emitInstructionEnd 插 NBF）
  → layout + relaxOnce
  → refreshNBFInsertKindsAfterRelax（每个 text section）
  → while (shrinkSection) layoutSection   // 每个 text section
  → performPostLayout / verify
  → 写 object（RIVAI_BTB_OPTIMIZED）
  → lld（hasBtbOptimized → layout-preserve）
```


| 阶段             | 代码位置                                                                   | 说明                                                                      |
| -------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 跳过 MI 对齐 NOP   | `MachineBlockPlacement::alignBlocks`，RISC-V + flag 时 **return**        | 避免与 MC NBF 抢布局                                                          |
| 函数入口对齐         | `RISCVAsmPrinter::runOnMachineFunction`：`MF.setAlignment(max(..., F))` | 由 `emitCodeAlignment` / `MCAlignFragment` 填充                            |
| 建池             | `RISCVELFStreamer::emitInstruction` → `emitInstructionEnd`             | **当前指令**发射**之后**插 NBF；`PseudoCALL` 等在 `expandFunctionCall` **之前**按伪指令建池 |
| Branch spacing | `emitInstructionBegin` + `needBranchSpacing`                           | 仅 `BEQ`…`BGEU`、`BEQI`/`BNEI`、`C_BEQZ`/`C_BNEZ`                          |
| Refresh        | `MCAssembler::layout` → `MCAsmBackend::refreshNBFInsertKindsAfterRelax`；RISC-V 实现在 `RISCVAsmBackend.cpp` | **relax 后、shrink 前**；仅改 `InsertKind`，不 layout |
| Shrink         | `MCAssembler::layout` 中 `while (shrinkSection)`                         | 仅 **text** section；`RestartWinBase` 跨 `while` 调用保持                      |
| Verify         | `performPostLayout` → `verifyBTBFetchLineBranchLimits`                 | **不改码**；`HasFinalLayout` 已置位后执行                                         |

### 2.1 `refreshNBFInsertKindsAfterRelax`

**动机**：条件支 emit 为 **On** 池（初值 `BTBOnCondPoolInitialBytes`）。`relaxOnce` 后 `PseudoLong*` 常在 `FT_Relaxable` 的 `VarContents` 中变为 **`[反条件支][jal x0][原 On 池]`**；池语义上跟在 **离轨 `jal`** 之后，shrink 应按 **Off** 处理（Off 池在 On 之后尝试收缩）。

**时机**：`finishLayout` / `relaxOnce` 循环结束之后、`while (shrinkSection)` 之前；`MCAssembler` 对每个 **text** `MCSection` 调用 `getBackend().refreshNBFInsertKindsAfterRelax`（默认 `MCAsmBackend` 空实现）。

**算法**（正向遍历 fragment 链表；维护 **`PrevCode`** = 最近一个含机器码的 fragment，跳过 NBF、`BranchSpacing`、纯填充等）：

1. `F` 为 NBF 且 `InsertKind==On`、`NumBytes>0`、`PrevCode!=nullptr` 时：对 **`PrevCode`** 调用 `fragmentEndsWithOffExecPathJump`；若为 true → `setInsertKind(OffExecPath)`
2. `fragmentEndsWithOffExecPathJump`：对 `PrevCode` 按 RISC-V 长度编码定末条起点 `Rel`（处理 `[32B][16B]` 尾）；`insnSize` 由 `byte[Rel]` 得 2/4，校验 `Rel+insnSize==Total`；再 `isOffExecPathInsnEncoding` 判 Off-path（**不看** `getInst()`，见 §2.2）。emit 保证 NBF 前一条必是控制流
3. 若 `F` 为 `FT_Data` / `FT_Relaxable` / 带代码的 `FT_Align`，则 `PrevCode = &F`。**只改 `InsertKind`**，不 layout

**典型命中**：`PseudoLong*` relax 后末条为 **`jal x0`**；不要求识别完整的 inv+jal 双指令模式。

### 2.2 `isOffExecPathInsnEncoding` 与 `isOffExecPathJump` 的投影关系

| 层 | 函数 | 输入 |
| --- | --- | --- |
| Emit | `isOffExecPathJump` | `MCInst` opcode / 操作数 |
| Refresh | `isOffExecPathInsnEncoding` | fragment 内机器码（`Rel` 处） |

**不是**按 MCInst 逐条 `case` 对应，而是：emit 侧所有 Off-path 伪指令/指令在发射或 relax 展开后，末条控制流只能是下面 **四种编码** 之一；refresh 只识别这四种。

| 机器码（refresh 识别） | 对应 `isOffExecPathJump`（emit） |
| -------------------- | -------------------------------- |
| 32-bit `JAL`, `rd==x0` | `JAL`（`rd==x0`）、`PseudoBR`；`**PseudoLong*` relax 尾跳**（`expandLongCondBr` / QC long 均为 `JAL x0`） |
| 32-bit `JALR`, `rd==x0` | `JALR`（`rd==x0`）、`PseudoBRIND*`、`PseudoTAILIndirect*`（末条） |
| 16-bit `C.J` | `C_J` |
| 16-bit `C.JR`（`funct4==8`，非 `C.JALR`） | `C_JR` |

**Emit 已为 Off 的池**（上表左侧 MCInst）在 refresh 中**不会**再改 `InsertKind`（只处理仍为 **On** 的 NBF）。refresh 的编码判定主要用于 **`PseudoLong*`：emit 为 On，relax 后末条 `jal x0` → On→Off**。

**故意不认为 Off**（与 `isOffExecPathJump` 一致，且不应把 On 池误改为 Off）：

- 32-bit SB 条件支（`opcode 0x63`）、`C.BEQZ`/`C.BNEZ`（未 long relax 的短分支）
- `JAL`/`JALR` 且 `rd!=x0`（call）、`C.JAL`/`C.JALR`

若将来 long relax 尾跳改为非上述四种编码，需同时改 `expandLongCondBr` 与 `isOffExecPathInsnEncoding`。

---

## 3. 控制流分类

### 3.1 emit（`isOffExecPathJump` / `isOnExecPathJump`）

二者对主表 opcode **互斥**；`emitInstructionEnd` 先判 Off，再 On。

- **Off**：`C_J`/`C_JR`、`JAL`/`JALR`（`rd==x0`）、`PseudoBR`、`PseudoBRIND*`、`PseudoTAIL`、`PseudoTAILIndirect*`
- **On**：条件支、`C_JAL`/`C_JALR`、`JAL`/`JALR`（`rd!=x0`）、`PseudoCALL`/`PseudoCALLReg`/`PseudoJump`、`PseudoCALLIndirect*`

`**isConditionalBranch`**（影响 On 池初值）：`BEQ`…`BGEU`、`BEQI`/`BNEI`、`C_BEQZ`/`C_BNEZ`。

### 3.2 shrink / verify 收集

对每个 text `MCSection` 的 fragment：

1. `**getFixups()` / `getVarFixups()**`：`fixup_riscv_jal`、`branch`、`rvc_jump`、`rvc_branch`、`call`、`call_plt`（**call** 记 `**FragOffset + fixup_offset + 4`**，即 `jalr`）
2. `**collectJalrBranchInfosFromBytes**`：扫描 `FT_Data`、`FT_Relaxable`、以及 `FT_Align` 且 `FixedSize>0` 的固定区；识别 `JALR`（`opcode 0x67`）、`C.BEQZ`/`C.BNEZ`、`C.JR`/`C.JALR`（`funct4` 8/9）
3. **NBF 列表**：`FT_NopsBesideBranch` 且 `getNumBytes()!=0`，按 `InsertKind` 分 `NopsOnExecPath` / `NopsOffExecPath`

`BranchOffsets` **排序 + unique**。

**emit 未覆盖但会进收集的**：例如仅产生上述 fixup、或机器码命中扫描的指令 → shrink/verify **计分支**但 **无后池**（应对拍 §3.3 主表）。

### 3.3 主表


| MCInst                       | 条件                   | InsertKind | 条件支 | shrink/verify                |
| ---------------------------- | -------------------- | ---------- | --- | ---------------------------- |
| `BEQ`…`BGEU`, `BEQI`/`BNEI`  | —                    | On         | 是   | `fixup_riscv_branch`         |
| `C_BEQZ`/`C_BNEZ`            | —                    | On         | 是   | `rvc_branch` + 扫描            |
| `JAL`                        | `rd==x0` / `!=x0`    | Off / On   | 否   | `fixup_riscv_jal`            |
| `JALR`                       | `rd==x0` / `!=x0`    | Off / On   | 否   | 扫描；call 另有 `call`/`call_plt` |
| `PseudoCALL`/`PseudoCALLReg` | `expandFunctionCall` | On         | 否   | `call`/`call_plt`            |
| `PseudoJump`                 | 同上                   | On         | 否   | call 类 / 扫描                  |
| `PseudoTAIL`                 | → `JALR x0`          | Off        | 否   | 同上                           |
| `PseudoCALLIndirect*`        | → `JALR x1,rs,0`     | On         | 否   | `JALR` / 扫描                  |
| `PseudoTAILIndirect*`        | → `JALR x0,rs,0`     | Off        | 否   | 同上                           |
| `C_J`                        | —                    | Off        | 否   | `rvc_jump`                   |
| `C_JR` / `C_JALR`            | funct4 8/9           | Off / On   | 否   | 扫描                           |
| `C_JAL`                      | —                    | On         | 否   | `rvc_jump`                   |
| `PseudoBR`                   | → `JAL x0`           | Off        | 否   | `jal`                        |
| `PseudoBRIND*`               | → `JALR x0`          | Off        | 否   | fixup / 扫描                   |


`**PseudoLong*`**：`relaxInstruction` → `expandLongCondBr` 写入 `VarContents`（典型 **反条件支 + `jal x0`**）；两条均有 fixup，计入窗口。emit 时条件支为 **On**，初值用 `**BTBOnCondPoolInitialBytes`（已 ≥ F）**；relax 后若末条为 Off-path，**refresh** 将邻接 NBF **On→Off**（§2.1）。

### 3.4 未纳入


| 类型                                          | emit   | shrink/verify |
| ------------------------------------------- | ------ | ------------- |
| `MRET`/`SRET`/…                             | 无      | 不收集           |
| `WFI`/`ECALL`/`EBREAK`                      | 无      | 不收集           |
| `fixup_riscv_qc_e_branch`、`nds_branch_10` 等 | 视需求补主表 | **当前收集循环未列入** |


### 3.5 emit 与 `%pcrel_hi` / `AUIPC`

`AsmPrinter` 在 `AUIPC` 前 `emitLabel(PreInstrSymbol)`。后池在 `**emitInstructionEnd`（控制流之后）** 插入，顺序为：`分支` →（可选 NBF）→ `标号` → `AUIPC`，标号与 `%pcrel_hi` fixup 同在后续指令 fragment，`getPCRelHiFixup` 可配对。若改回「下一条指令**前**」插 NBF，会破坏配对。

---

## 4. 后池初值（实现对照）

推导见上文 **「NBF 初值公式推导」**；此处为 `RISCVAsmBackend` 构造函数与 `emitInstructionEnd` 使用的汇总。**`NumBytes==0` 不创建 NBF**。

| InsertKind | 公式（与代码一致） |
| ---------- | ------------------ |
| On（call） | `F−N·I_min≤0` → 0；否则 `max(F−N·I_min, M)` → `BTBOnCallPoolInitialBytes` |
| On（条件支） | `max(BTBOnCallPoolInitialBytes, F)` → `BTBOnCondPoolInitialBytes` |
| Off, `N=1` | `F` → `BTBOffPoolInitialBytes` |
| Off, `N≥2` | `OffRaw=(F−I_min)/(N−1)−I_min`（整数除）；`≤0` → 0，否则 `max(OffRaw,M)` |

---

## 5. Shrink（`shrinkSection`）

**前置**：`EnableRISCVBTBFetchLineBranchRelaxation && F>0 && N>0`，否则立即 `return false`。

**每轮**（单次 `shrinkSection` 调用）：

1. 收集 `AlignStartOffsets` / `AlignEndOffsets`（`FT_Align` 填充起点与段尾）
2. 从 `RestartWinBase` 定位 segment / `WinBase`（若落在 align 填充区内则跳到 align 结束）
3. 全段收集 `BranchOffsets`、On/Off NBF 列表并排序
4. **诊断**（每 section 的**第一次** `shrinkSection` 调用）：对所有 `i`，若 `BranchOffsets[i+N]-BranchOffsets[i] < F`，`reportWarning` 并 dump 两侧编码（`readInsnBytesAtSectionOffset`）
5. 若 `-riscv-btb-shrink-section-diag-only` → **return false**
6. **窗口循环**：`w` 从 `WinBase` 步进 **F**，直到 `SegmentEnd`（下一个 align 填充起点）或 section 末
  - **BCritical**：从当前 `BranchIdx` 起在 `**[BranchIdx, SegmentEnd)`** 内第 **N+1** 条分支的 offset（**不是**仅数 `[w,w+F)` 窗内条数）
  - **tryShrink**：先 `**NopsOnExecPath`**，再 `**NopsOffExecPath**`；在 `[w,WinEnd)` 与 `BCritical` 约束下 `**removal = min(max_removal, …)` 再 `(removal/2)*2` 减半**
  - 若 `BCritical < WinEnd` 则跳过该池（避免把关键分支挤进窗内）
  - 成功收缩 → `RestartWinBase = w`，`Changed = true`，**break** 本轮窗口循环
  - 否则推进 `BranchIdx` / Nop 索引
7. 删除 `getNumBytes()==0` 的 NBF fragment

**外层**：`MCAssembler` 在 `Changed` 时对同一 section 调用 `**layoutSection`**，再进入 `shrinkSection`，直至返回 false。

**与 verify 的差异**：`verifyBTBFetchLineBranchLimits` 对每窗 `**[w,w+F)`** 用 `**lower_bound**` 数分支数，**>N 即报错/警告**；shrink 用 **BCritical + 后池** 启发式，二者口径相关但不相同。

**性能注意**：每轮 shrink **全段** fixup 收集 + 字节扫描；`shrinkSection` 内大量 `LLVM_DEBUG` 仅在 **Debug 构建** 或 `-debug-only=riscv-asmbackend` 时输出，Release 默认不执行。

---

## 6. Verify（`verifyBTBFetchLineBranchLimits`）

- 在 `**performPostLayout`** 调用；分支收集与 shrink **同源**（fixup + `collectJalrBranchInfosFromBytes`）
- 按 align 分段；`w` 步进 **F**；`[w,w+F)` 内分支数 **> N** → `reportWarning` 或 `reportError`（由 `-riscv-btb-fetchline-violation-as-warning` 决定）
- **不修改** fragment

---

## 7. 链接与 ELF

- **写出**：`RISCVAsmPrinter::emitAttributes` → `RIVAI_BTB_OPTIMIZED = 1`（`RISCVAttributes.h`，值 **32768**）
- **lld 解析**：`readRISCVAttributes` 置 `InputFile::hasBtbOptimized`；**不**写入输出 attributes
- **layout-preserve**：BTB object 上部分 relaxation 在删指令时 `**remove=0`**，向 `aux.writes` 写 **NOP**（如 `0x00000013` addi x0,x0,0），保持 section 布局与 MC shrink 假设一致；`R_RISCV_ALIGN` 等仍按常规定义处理

---

## 8. 限制与后续


| 项               | 说明                                                                                  |
| --------------- | ----------------------------------------------------------------------------------- |
| 大 switch / 密集分支 | shrink 迭代 × 全段扫描 × layout；见 §5                                                      |
| align           | `MCAlignFragment` 划分 segment；链接后填充大小可能变化                                            |
| **refresh**     | 已实现（§2.1）；仅改 `InsertKind`，依赖 relax 后真实机器码/fixup                         |
| 测试              | 无专用 LIT；可用 `-riscv-btb-shrink-section-diag-only` + verify                           |
| 后续              | shrink 缓存分支相对位置、减少 layout 次数；大 section 时 profile shrink 迭代耗时   |


