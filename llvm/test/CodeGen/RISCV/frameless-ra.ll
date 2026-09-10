; Frameless RA follows -riscv-shrink-wrapping-dataflow (default on with
; dataflow). Opt out with -disable-riscv-frameless-ra.
;
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     < %s | FileCheck %s --check-prefixes=ON
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     -disable-riscv-frameless-ra < %s | FileCheck %s --check-prefixes=OFF
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     -stop-after=shrink-wrapping -o - < %s | FileCheck %s --check-prefixes=SW
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     -pass-remarks-analysis=frameless-ra -o /dev/null %s 2> %t.remark
; RUN: FileCheck %s --check-prefixes=REMARK < %t.remark
; End-to-end push_back_real L1–L3 coverage (RA hints + shrink-wrapping).
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     -disable-riscv-frameless-ra < %s | FileCheck %s --check-prefixes=REALOFF
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     < %s | FileCheck %s --check-prefixes=REALON
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     -stop-after=shrink-wrapping -o - < %s | FileCheck %s --check-prefixes=REALSW
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-shrink-wrapping-dataflow=true \
; RUN:     -pass-remarks-analysis=frameless-ra -o /dev/null %s 2> %t.realremark
; RUN: FileCheck %s --check-prefixes=REALREMARK < %t.realremark

declare void @grow(ptr nocapture noundef)
declare void @init()
declare void @sink(i64)

declare void @llvm.assume(i1)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull)
declare noalias noundef nonnull ptr @_Znwm(i64 noundef)
declare void @_ZdlPvm(ptr noundef nonnull, i64 noundef)

@.str = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_insert\00"

; push_back_like: L1–L3 acceptance (finish/end in entry, fast path, slow call).
define void @push_back_like(ptr nocapture noundef %v, i8 signext %x) {
; ON-LABEL: push_back_like:
; ON:         ld
; ON:         ld
; ON:         beq
; ON-NOT:     addi sp
; ON:         sb
; ON:         ret
;
; OFF-LABEL: push_back_like:
; OFF:        addi sp
entry:
  %finish_p = getelementptr i8, ptr %v, i64 8
  %end_p = getelementptr i8, ptr %v, i64 16
  %finish = load ptr, ptr %finish_p, align 8
  %end = load ptr, ptr %end_p, align 8
  %cmp = icmp eq ptr %finish, %end
  br i1 %cmp, label %slow, label %fast

fast:
  store i8 %x, ptr %finish, align 1
  %next = getelementptr i8, ptr %finish, i64 1
  store ptr %next, ptr %finish_p, align 8
  ret void

slow:
  call void @grow(ptr nocapture noundef %v)
  ret void
}

; SW-LABEL: name:            push_back_like
; SW-NOT:   prolog: ''
; SW-NOT:   prolog: {{.*}}bb.0
; SW:       prolog: {{'%bb\.[0-9]+'}}
;
; REMARK: remark: {{.*}}Frameless-path hint for %{{.*}} (cost=16384)
; REMARK: remark: {{.*}}Frameless-path hint for %{{.*}} (cost=16384)

; push_back_real: real C++ std::vector<uint8_t>::push_back IR (clang -O2).
; Layout: %0+0 begin, +8 finish, +16 end_of_storage.
define dso_local void @_Z14push_back_realRSt6vectorIhSaIhEEh(
    ptr noundef nonnull align 8 dereferenceable(24) %0, i8 noundef zeroext %1) {
entry:
  %finish_p = getelementptr inbounds nuw i8, ptr %0, i64 8
  %finish = load ptr, ptr %finish_p, align 8
  %end_p = getelementptr inbounds nuw i8, ptr %0, i64 16
  %end = load ptr, ptr %end_p, align 8
  %full = icmp eq ptr %finish, %end
  br i1 %full, label %slow, label %fast

fast:
  store i8 %1, ptr %finish, align 1
  %finish_reload = load ptr, ptr %finish_p, align 8
  %next = getelementptr inbounds nuw i8, ptr %finish_reload, i64 1
  store ptr %next, ptr %finish_p, align 8
  br label %ret

slow:
  %begin = load ptr, ptr %0, align 8
  %size = ptrtoint ptr %finish to i64
  %begin_int = ptrtoint ptr %begin to i64
  %len = sub i64 %size, %begin_int
  %len_ok = icmp sgt i64 %len, -1
  call void @llvm.assume(i1 %len_ok)
  %max = icmp eq i64 %len, 9223372036854775807
  br i1 %max, label %throw, label %grow

throw:
  call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str)
  unreachable

grow:
  %new_len = call i64 @llvm.umax.i64(i64 %len, i64 1)
  %cap = add nuw i64 %new_len, %len
  %new_cap = call i64 @llvm.umin.i64(i64 %cap, i64 9223372036854775807)
  %new_buf = call noalias noundef nonnull ptr @_Znwm(i64 noundef %new_cap)
  %insert = getelementptr inbounds nuw i8, ptr %new_buf, i64 %len
  store i8 %1, ptr %insert, align 1
  %empty = icmp eq ptr %finish, %begin
  br i1 %empty, label %update, label %copy

copy:
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %new_buf,
                                   ptr align 1 %begin, i64 %len, i1 false)
  br label %update

update:
  %begin_null = icmp eq ptr %begin, null
  br i1 %begin_null, label %done, label %free

free:
  call void @_ZdlPvm(ptr noundef nonnull %begin, i64 noundef %len)
  br label %done

done:
  %new_finish = getelementptr inbounds nuw i8, ptr %insert, i64 1
  store ptr %new_buf, ptr %0, align 8
  store ptr %new_finish, ptr %finish_p, align 8
  %new_end = getelementptr inbounds nuw i8, ptr %new_buf, i64 %new_cap
  store ptr %new_end, ptr %end_p, align 8
  br label %ret

ret:
  ret void
}

; push_back_real: L1 remark + L2 SW prolog + L3 asm (frameless RA +
; shrink-frame shared-return duplication + fold CSRSave onto Prolog).
;
; REALOFF-LABEL: _Z14push_back_realRSt6vectorIhSaIhEEh:
; REALOFF:        addi sp
;
; REALON-LABEL: _Z14push_back_realRSt6vectorIhSaIhEEh:
; REALON:         ld
; REALON:         ld
; REALON:         beq
; REALON-NOT:     addi sp
; REALON:         sb
; REALON:         ret
;
; REALSW-LABEL: name:            _Z14push_back_realRSt6vectorIhSaIhEEh
; REALSW-NOT:   prolog: ''
; REALSW-NOT:   prolog: {{.*}}bb.0
; REALSW:       prolog: {{'%bb\.[0-9]+'}}
;
; REALREMARK-DAG: remark: {{.*}}Frameless-path hint for %{{.*}} (cost=16384)
; REALREMARK-DAG: remark: {{.*}}Frameless-path hint for %{{.*}} (cost=16384)

; LiveAcrossColdCall: %x used after cold-path call -> no hint (compile smoke).
define i64 @live_across(ptr nocapture noundef %p) {
; ON-LABEL: live_across:
; ON:         ret
entry:
  %x = load i64, ptr %p, align 8
  %cold = icmp eq i64 %x, 0
  br i1 %cold, label %call, label %fast

fast:
  ret i64 %x

call:
  call void @sink(i64 %x)
  ret i64 %x
}

; Entry call before loads: skip if realign / no frameless region (smoke).
define void @entry_call_fast_exit(ptr nocapture noundef %v, i8 signext %x) {
; ON-LABEL: entry_call_fast_exit:
; ON:         ret
entry:
  call void @init()
  %finish_p = getelementptr i8, ptr %v, i64 8
  %end_p = getelementptr i8, ptr %v, i64 16
  %finish = load ptr, ptr %finish_p, align 8
  %end = load ptr, ptr %end_p, align 8
  %cmp = icmp eq ptr %finish, %end
  br i1 %cmp, label %slow, label %fast

fast:
  store i8 %x, ptr %finish, align 1
  %next = getelementptr i8, ptr %finish, i64 1
  store ptr %next, ptr %finish_p, align 8
  ret void

slow:
  call void @grow(ptr nocapture noundef %v)
  ret void
}

; Stack realign -> shouldSkipFramelessRA (compile smoke).
define void @skip_realign() {
; ON-LABEL: skip_realign:
; ON:         ret
entry:
  %buf = alloca [32 x i8], align 4096
  %p = getelementptr [32 x i8], ptr %buf, i64 0, i64 0
  store i8 0, ptr %p, align 1
  %cmp = icmp eq i8 0, 1
  br i1 %cmp, label %cold, label %exit

exit:
  ret void

cold:
  call void @sink(i64 0)
  ret void
}
