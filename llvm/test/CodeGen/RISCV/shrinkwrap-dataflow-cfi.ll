; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     < %s | FileCheck %s --check-prefix=DF
;
; M3: with data-flow shrink-wrapping, CSR spill/restore co-locate CFI.
; Do not use nounwind - needsFrameMoves()/needsDwarfCFI would skip CFI entirely.
; Spills are emitted as a batch, then .cfi_offset directives (not strictly
; one-to-one NEXT after each sd).

declare void @sink(i64)

define i64 @early_exit_vs_cold(i64 %n, i64 %x) {
; DF-LABEL: early_exit_vs_cold:
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %ret0, label %cold

ret0:
  ret i64 0

cold:
  call void @sink(i64 %x)
  %y = add i64 %x, 1
  call void @sink(i64 %y)
  ret i64 %y
}

; DF: sd {{ra|s[0-9]+}}, {{[0-9]+}}(sp)
; DF: .cfi_offset {{ra|s[0-9]+}},
; DF: ld {{ra|s[0-9]+}}, {{[0-9]+}}(sp)
; DF: .cfi_restore {{ra|s[0-9]+}}