; Simple (x<<n)+x becomes SH*ADD with Zba (usually already at ISel).
; Without Zba, RV64 OptW may still strip *W to SLLI+ADD.
; Late ADD+SLLI->SH*ADD is covered by shxadd-from-slli-combiner.mir.
;
; RUN: llc -mtriple=riscv64 -mattr=+m,+zba -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ZBA
; RUN: llc -mtriple=riscv64 -mattr=+m,+zba -riscv-late-machine-combiner=true \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,ZBA
; RUN: llc -mtriple=riscv64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOZBA

define void @shl1_add_same_store(i32 signext %x, ptr %p) {
; CHECK-LABEL: shl1_add_same_store:
; ZBA:       # %bb.0:
; ZBA-NEXT:    sh1add a0, a0, a0
; ZBA-NEXT:    sw a0, 0(a1)
; ZBA-NEXT:    ret
; NOZBA:       # %bb.0:
; NOZBA-NEXT:    slli {{a[0-9]+}}, a0, 1
; NOZBA-NEXT:    add a0, {{a[0-9]+}}, {{a[0-9]+}}
; NOZBA-NEXT:    sw a0, 0(a1)
; NOZBA-NEXT:    ret
  %a = shl i32 %x, 1
  %b = add i32 %a, %x
  store i32 %b, ptr %p
  ret void
}

define void @shl2_add_same_store(i32 signext %x, ptr %p) {
; CHECK-LABEL: shl2_add_same_store:
; ZBA:       # %bb.0:
; ZBA-NEXT:    sh2add a0, a0, a0
; ZBA-NEXT:    sw a0, 0(a1)
; ZBA-NEXT:    ret
; NOZBA:       # %bb.0:
; NOZBA-NEXT:    slli {{a[0-9]+}}, a0, 2
; NOZBA-NEXT:    add a0, {{a[0-9]+}}, {{a[0-9]+}}
; NOZBA-NEXT:    sw a0, 0(a1)
; NOZBA-NEXT:    ret
  %a = shl i32 %x, 2
  %b = add i32 %a, %x
  store i32 %b, ptr %p
  ret void
}

define void @shl3_add_same_store(i32 signext %x, ptr %p) {
; CHECK-LABEL: shl3_add_same_store:
; ZBA:       # %bb.0:
; ZBA-NEXT:    sh3add a0, a0, a0
; ZBA-NEXT:    sw a0, 0(a1)
; ZBA-NEXT:    ret
; NOZBA:       # %bb.0:
; NOZBA-NEXT:    slli {{a[0-9]+}}, a0, 3
; NOZBA-NEXT:    add a0, {{a[0-9]+}}, {{a[0-9]+}}
; NOZBA-NEXT:    sw a0, 0(a1)
; NOZBA-NEXT:    ret
  %a = shl i32 %x, 3
  %b = add i32 %a, %x
  store i32 %b, ptr %p
  ret void
}
