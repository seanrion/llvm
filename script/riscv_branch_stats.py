#!/usr/bin/env python3
"""
RISC-V 分支指令统计分析工具

对 objdump 反汇编的 RISC-V 程序进行分支指令的种类和数量统计，
并检测每个 fetch line（如 32B）地址空间内分支数是否超过阈值（如 4 条）。

用法:
  python riscv_branch_stats.py <objdump_output.txt>     # 从文件读取
  python riscv_branch_stats.py -b <riscv_binary>        # 对二进制运行 objdump
  objdump -d binary | python riscv_branch_stats.py      # 从 stdin 读取

选项:
  -f, --fetch-line-size BYTES   Fetch line 大小，默认 32
  -m, --max-branches N          每 fetch line 最大分支数，默认 4
  -s, --start-addr ADDR         限定统计起始地址（十六进制，如 0x1000）
  -e, --end-addr ADDR           限定统计结束地址（十六进制，含该地址）
"""

import re
import subprocess
import sys
import argparse
from collections import defaultdict


# RISC-V 分支/跳转指令 (标准 + 压缩扩展 C + 伪指令)
BRANCH_INSTRUCTIONS = {
    # 条件分支 (SB-type)
    "beq": "条件分支-相等",
    "bne": "条件分支-不等",
    "blt": "条件分支-小于(有符号)",
    "bge": "条件分支-大于等于(有符号)",
    "bltu": "条件分支-小于(无符号)",
    "bgeu": "条件分支-大于等于(无符号)",
    # 无条件跳转 (UJ-type, I-type)
    "jal": "无条件跳转-立即数",
    "jalr": "间接跳转-寄存器",
    # 伪指令 (objdump 常用)
    "j": "伪指令-跳转(jal x0)",
    "ret": "伪指令-返回(jalr x0,ra)",
    "beqz": "伪指令-相等零分支(beq rs,x0)",
    "bnez": "伪指令-不等零分支(bne rs,x0)",
    "blez": "伪指令-小于等于零",
    "bgez": "伪指令-大于等于零",
    "bltz": "伪指令-小于零",
    "bgtz": "伪指令-大于零",
    # 压缩扩展 (C extension)
    "c.beqz": "压缩-相等零分支",
    "c.bnez": "压缩-不等零分支",
    "c.j": "压缩-跳转",
    "c.jal": "压缩-跳转并链接",
    "c.jr": "压缩-寄存器跳转",
    "c.jalr": "压缩-寄存器跳转并链接",
}


# objdump 格式: 地址 + 冒号 + 空格 + 编码 + 若干个空格 + 指令
# 例如: "   1012c:   00a58533    add  a0,a1,a0"
_OBJDUMP_LINE_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+[0-9a-fA-F]+\s+(\S+)"
)


def parse_objdump_line(line: str) -> tuple[int, str] | None:
    """从 objdump 输出行中提取(地址, 助记符)。若为分支指令则返回，否则返回 None。"""
    line = line.strip()
    if not line or line.startswith("Disassembly") or line.startswith(":"):
        return None

    m = _OBJDUMP_LINE_RE.match(line)
    if not m:
        return None

    addr = int(m.group(1), 16)
    mnemonic = m.group(2).lower()

    if mnemonic in BRANCH_INSTRUCTIONS:
        return (addr, mnemonic)
    # 匹配 c.xxx 等；按长度降序避免 jal 误匹配 jalr
    for br in sorted(BRANCH_INSTRUCTIONS, key=len, reverse=True):
        if mnemonic == br or mnemonic.startswith(br + "."):
            return (addr, br)
    return None


def filter_by_addr_range(
    branches: list[tuple[int, str]],
    start_addr: int | None,
    end_addr: int | None,
) -> list[tuple[int, str]]:
    """按地址范围 [start_addr, end_addr] 过滤分支。None 表示不限制该边界。"""
    if start_addr is None and end_addr is None:
        return branches
    result = []
    for addr, mnemonic in branches:
        if start_addr is not None and addr < start_addr:
            continue
        if end_addr is not None and addr > end_addr:
            continue
        result.append((addr, mnemonic))
    return result


def branches_to_counts(branches: list[tuple[int, str]]) -> dict[str, int]:
    """从分支列表生成种类统计。"""
    counts: dict[str, int] = defaultdict(int)
    for _, mnemonic in branches:
        counts[mnemonic] += 1
    return dict(counts)


def parse_objdump_output(text: str) -> tuple[dict[str, int], list[tuple[int, str]]]:
    """解析 objdump 输出，返回 (分支种类统计, [(地址, 助记符), ...])。"""
    counts = defaultdict(int)
    branches: list[tuple[int, str]] = []
    for line in text.splitlines():
        parsed = parse_objdump_line(line)
        if parsed:
            addr, mnemonic = parsed
            counts[mnemonic] += 1
            branches.append((addr, mnemonic))
    return dict(counts), branches


def check_fetch_line_overflow(
    branches: list[tuple[int, str]],
    fetch_line_size: int = 32,
    max_branches: int = 4,
) -> list[tuple[int, int, list[tuple[int, str]]]]:
    """
    检测每个 fetch line（每 fetch_line_size 字节，地址空间按 fetch_line_size 对齐）内
    分支数是否超过 max_branches。例如 32B 时，边界为 0x0, 0x20, 0x40, ...
    返回 [(fetch_line_start, count, [(addr, mnemonic), ...]), ...] 仅包含超限的 fetch line。
    """
    by_fetch_line: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for addr, mnemonic in branches:
        # 32B 对齐：地址 addr 属于 [line_start, line_start+fetch_line_size) 区间
        line_start = (addr // fetch_line_size) * fetch_line_size
        by_fetch_line[line_start].append((addr, mnemonic))

    overflows: list[tuple[int, int, list[tuple[int, str]]]] = []
    for line_start in sorted(by_fetch_line.keys()):
        items = by_fetch_line[line_start]
        if len(items) > max_branches:
            overflows.append((line_start, len(items), items))
    return overflows


def print_fetch_line_report(
    overflows: list[tuple[int, int, list[tuple[int, str]]]],
    fetch_line_size: int,
    max_branches: int,
    addr_range: tuple[int | None, int | None] | None = None,
) -> None:
    """打印 fetch line 超限报告。"""
    rng_suffix = ""
    if addr_range and (addr_range[0] is not None or addr_range[1] is not None):
        s, e = addr_range
        parts = []
        if s is not None:
            parts.append(f"0x{s:x}")
        if e is not None:
            parts.append(f"0x{e:x}")
        parts_str = "-".join(parts)
        if parts_str:
            rng_suffix = f", 地址范围 {parts_str}"
    if not overflows:
        print(f"\nFetch line 检测 ({fetch_line_size}B 对齐{rng_suffix}): 无超限 (阈值 {max_branches} 条)")
        return

    print(f"\n{'=' * 70}")
    print(f"Fetch line 超限检测 ({fetch_line_size}B 对齐地址空间, 阈值 {max_branches} 条{rng_suffix})")
    print("=" * 70)
    for line_start, count, items in overflows:
        line_end = line_start + fetch_line_size - 1
        print(f"地址范围 0x{line_start:x}-0x{line_end:x}: "
              f"{count} 条分支 (超过 {max_branches})")
        print("    ",end="")
        for addr, mnemonic in sorted(items):
            print(f" 0x{addr:x}({mnemonic})",end=", ")
        print()
    print("=" * 70)


def run_objdump(binary_path: str) -> str:
    """对 RISC-V 二进制运行 objdump -d。"""
    try:
        result = subprocess.run(
            ["objdump", "-d", binary_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"错误: objdump 执行失败: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        return result.stdout
    except FileNotFoundError:
        print("错误: 未找到 objdump，请确保已安装 binutils 并在 PATH 中", file=sys.stderr)
        sys.exit(1)


def print_stats(counts: dict[str, int], addr_range: tuple[int | None, int | None] | None = None) -> None:
    """打印统计结果。addr_range 为 (start, end) 时在标题中显示限定范围。"""
    if not counts:
        print("未检测到分支指令。")
        return

    total = sum(counts.values())
    if addr_range and (addr_range[0] is not None or addr_range[1] is not None):
        start, end = addr_range
        rng = " (地址范围 "
        if start is not None:
            rng += f"0x{start:x}"
        else:
            rng += "?"
        rng += "-"
        if end is not None:
            rng += f"0x{end:x}"
        else:
            rng += "?"
        rng += ")"
    else:
        rng = ""
    print("=" * 60)
    print("RISC-V 分支指令统计" + rng)
    print("=" * 60)
    print(f"{'指令':<12} {'数量':>8} {'占比':>10} {'说明'}")
    print("-" * 60)

    # 按数量降序
    for mnemonic, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total if total else 0
        desc = BRANCH_INSTRUCTIONS.get(mnemonic, "")
        print(f"{mnemonic:<12} {cnt:>8} {pct:>9.1f}%  {desc}")

    print("-" * 60)
    print(f"{'合计':<12} {total:>8} {100.0:>9.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="统计 RISC-V objdump 反汇编中的分支指令种类和数量"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="objdump 输出文件路径（不指定则从 stdin 读取）",
    )
    parser.add_argument(
        "-b", "--binary",
        help="RISC-V 二进制文件路径，将对其运行 objdump -d",
    )
    parser.add_argument(
        "-f", "--fetch-line-size",
        type=int,
        default=32,
        metavar="BYTES",
        help="Fetch line 大小（字节），默认 32",
    )
    parser.add_argument(
        "-m", "--max-branches",
        type=int,
        default=4,
        metavar="N",
        help="每个 fetch line 内允许的最大分支数，超过则报警，默认 4",
    )
    parser.add_argument(
        "-s", "--start-addr",
        type=lambda x: int(x, 0),
        default=None,
        metavar="ADDR",
        help="限定统计起始地址（支持 0x 十六进制）",
    )
    parser.add_argument(
        "-e", "--end-addr",
        type=lambda x: int(x, 0),
        default=None,
        metavar="ADDR",
        help="限定统计结束地址（含该地址，支持 0x 十六进制）",
    )
    args = parser.parse_args()

    if args.start_addr is not None and args.end_addr is not None and args.start_addr > args.end_addr:
        print("错误: 起始地址不能大于结束地址", file=sys.stderr)
        sys.exit(1)

    if args.binary:
        text = run_objdump(args.binary)
    elif args.input:
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    _, branches = parse_objdump_output(text)
    branches = filter_by_addr_range(branches, args.start_addr, args.end_addr)
    counts = branches_to_counts(branches)
    addr_range = (args.start_addr, args.end_addr) if (args.start_addr is not None or args.end_addr is not None) else None
    print_stats(counts, addr_range)

    overflows = check_fetch_line_overflow(
        branches,
        fetch_line_size=args.fetch_line_size,
        max_branches=args.max_branches,
    )
    print_fetch_line_report(
        overflows,
        fetch_line_size=args.fetch_line_size,
        max_branches=args.max_branches,
        addr_range=addr_range,
    )


if __name__ == "__main__":
    main()
