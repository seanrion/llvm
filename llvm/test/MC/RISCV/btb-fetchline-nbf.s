## Smoke test: BTB fetch-line optimization inserts NOP pools (NBF) after branches.
## Part 1 (-diag-only): emit inserts initial pools visible in objdump.
## Part 2 (shrink): shrink NBF + verify (F=16, N=2) on dense (5 off-path c.j).

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+relax \
# RUN:   -riscv-btb-fetchline-branch-relaxation \
# RUN:   -riscv-fetchline-size=32 \
# RUN:   -riscv-btb-max-branches-per-fetchline=4 \
# RUN:   -riscv-btb-shrink-section-diag-only \
# RUN:   %s -o %t.btb.o
# RUN: llvm-objdump -d -M no-aliases %t.btb.o | FileCheck %s --check-prefix=BTB

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+relax %s -o %t.plain.o
# RUN: llvm-objdump -d -M no-aliases %t.plain.o | FileCheck %s --check-prefix=NOBTB

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+relax \
# RUN:   -riscv-btb-fetchline-branch-relaxation \
# RUN:   -riscv-fetchline-size=16 \
# RUN:   -riscv-btb-max-branches-per-fetchline=2 \
# RUN:   -riscv-btb-fetchline-violation-as-warning=false \
# RUN:   %s -o %t.shrink.o 2> %t.shrink.err
# RUN: FileCheck %s --input-file=%t.shrink.err --check-prefix=SHRINK-OK --allow-empty
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+relax \
# RUN:   -riscv-btb-fetchline-branch-relaxation \
# RUN:   -riscv-fetchline-size=16 \
# RUN:   -riscv-btb-max-branches-per-fetchline=2 \
# RUN:   -riscv-btb-fetchline-violation-as-warning=false \
# RUN:   -riscv-btb-shrink-section-diag-only \
# RUN:   %s -o %t.diag.o
# RUN: llvm-objdump -d -M no-aliases --disassemble-symbols=dense %t.shrink.o \
# RUN:   | FileCheck %s --check-prefix=SHRINK-DENSE
# RUN: llvm-objdump -d -M no-aliases --disassemble-symbols=dense %t.diag.o \
# RUN:   | FileCheck %s --check-prefix=DIAG-DENSE

	.text
	.globl test
	.type test, @function
test:
	c.beqz	a0, .Lend
.Lend:
	c.jr	ra
	.size test, .-test

	.globl dense
	.type dense, @function
dense:
	c.j	1f
	c.j	1f
	c.j	1f
	c.j	1f
	c.j	1f
1:
	c.jr	ra
	.size dense, .-dense

# BTB-LABEL: <test>:
# BTB:       c.beqz
# BTB-COUNT-8: addi    zero, zero, 0
# BTB:       c.jr
# BTB-COUNT-2: addi    zero, zero, 0

# NOBTB-NOT: addi    zero, zero, 0

# SHRINK-OK-NOT: BTB fetch-line limit violated
# SHRINK-OK-NOT: error:

# Emit-only (diag-only): full initial NOP pools between dense jumps.
# DIAG-DENSE-LABEL: <dense>:
# DIAG-DENSE-COUNT-15: addi

# After shrink: fewer NOP bytes; post-layout verify must still pass (SHRINK-OK).
# SHRINK-DENSE-LABEL: <dense>:
# SHRINK-DENSE-COUNT-6: addi
