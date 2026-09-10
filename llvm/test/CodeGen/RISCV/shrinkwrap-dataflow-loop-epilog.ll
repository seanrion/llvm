; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     < %s | FileCheck %s

; Regression for gcc.c-torture/execute/20071029-1.c style CFG:
; stack object + call in a goto loop. Under GCC-like data-flow shrink-wrapping,
; addi sp stays at entry/exit; CSR restore must NOT appear on the backedge
; between call sink and the branch that re-enters the loop.
; Noreturn exits must also NOT get CSR restore / .cfi_restore (M5b).

declare void @sink(ptr)
declare void @noreturn_exit() noreturn

define void @loop_with_stack_obj(i32 %i) {
; CHECK-LABEL: loop_with_stack_obj:
; CHECK:       addi sp, sp, -
; CHECK:       sd {{ra|s[0-9]+}},
; CHECK:       call sink
; No frame teardown / CSR reload before the loop branch.
; CHECK-NOT:   ld {{ra|s[0-9]+}},
; CHECK-NOT:   addi sp, sp, {{([1-9][0-9]*|0x[0-9a-fA-F]+)}}
; CHECK:       {{bne|beq|bnez|beqz|j|jal}}
; CHECK:       call noreturn_exit
; CHECK-NOT:   ld {{ra|s[0-9]+}},
; CHECK-NOT:   .cfi_restore
entry:
  %t = alloca [56 x i8], align 8
  br label %again

again:
  call void @sink(ptr %t)
  %inc = add i32 %i, 1
  %cond = icmp eq i32 %inc, 999
  br i1 %cond, label %die, label %again

die:
  call void @noreturn_exit()
  unreachable
}