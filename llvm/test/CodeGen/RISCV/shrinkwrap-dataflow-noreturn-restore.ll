; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     < %s | FileCheck %s

; Noreturn exit must not get CSR restore + .cfi_restore after the call:
; that poisons FDE at the call RA (libunwind uses pc after the call).
; Cleared restores are rehomed only onto returns reachable from the CSR
; save -- not every ret (SIBsim4/PENNANT); covered by execute tests.

declare void @__cxa_throw(ptr, ptr, ptr) noreturn
declare ptr @__cxa_allocate_exception(i64)
declare void @sink(i64)

@typeinfo = external global ptr

define void @blowup_like(i64 %n) {
; CHECK-LABEL: blowup_like:
; CHECK:       sd {{s[0-9]+}},
; CHECK:       call sink
; CHECK:       call __cxa_allocate_exception
; CHECK:       call sink
; CHECK:       call __cxa_throw
; CHECK-NOT:   ld {{s[0-9]+}},
; CHECK-NOT:   .cfi_restore
entry:
  call void @sink(i64 %n)
  %exn = call ptr @__cxa_allocate_exception(i64 4)
  store i32 42, ptr %exn, align 4
  call void @sink(i64 %n)
  call void @__cxa_throw(ptr %exn, ptr @typeinfo, ptr null)
  unreachable
}