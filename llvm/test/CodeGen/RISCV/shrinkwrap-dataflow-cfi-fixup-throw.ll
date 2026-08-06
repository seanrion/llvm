; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     -riscv-save-csrs-early=false < %s | FileCheck %s

; Layout often places the normal return before a throw path. Epilogue CFI
; (cfi_restore / CFA=0) would poison the throw PC; ShrinkWrapCFIFixup must
; insert remember_state after the entry prologue and restore_state before
; the throw path (GCC dwarf2cfi-style).

declare ptr @__cxa_allocate_exception(i64)
declare void @__cxa_throw(ptr, ptr, ptr) noreturn

@_ZTIi = external constant ptr

define i32 @callee(i32 %i) {
; CHECK-LABEL: callee:
; CHECK:       .cfi_def_cfa_offset
; CHECK:       .cfi_offset {{ra|x1}},
; CHECK:       .cfi_remember_state
; CHECK:       .cfi_restore {{ra|x1}}
; CHECK:       .cfi_def_cfa_offset 0
; CHECK:       ret
; CHECK:       .cfi_restore_state
; CHECK:       call __cxa_throw
entry:
  %cmp3 = icmp ult i32 %i, 3
  br i1 %cmp3, label %throw_int, label %ret0

ret0:
  ret i32 0

throw_int:
  %exn = call ptr @__cxa_allocate_exception(i64 4)
  store i32 %i, ptr %exn, align 4
  call void @__cxa_throw(ptr %exn, ptr @_ZTIi, ptr null)
  unreachable
}