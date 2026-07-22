; ModuleID = 'builtin.module'
source_filename = "cartan_cuda"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_cos(double)
declare double @__nv_sin(double)

define void @sphere_exp_apply(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v5, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v8, 1
  br label %bb0
bb0:
  %v17 = phi { ptr, i64 } [ %v10, %entry ]
  %v18 = phi { ptr, i64 } [ %v12, %entry ]
  %v19 = phi { ptr, i64 } [ %v14, %entry ]
  %v20 = phi i32 [ %v6, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = alloca {  }, align 1
  %v23 = bitcast ptr %v22 to ptr
  %v24 = call i64 @_RINvNtNtCs8uibrORiSfe_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs8GH1ove06AH_11cartan_cuda(ptr %v23) #0
  br label %bb1
bb1:
  %v25 = zext i32 %v20 to i64
  %v26 = icmp eq i64 %v25, 0
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb2, label %bb21
bb2:
  %v28 = udiv i64 %v24, %v25
  %v29 = extractvalue { ptr, i64 } %v19, 1
  %v30 = icmp ult i64 %v28, %v29
  br i1 %v30, label %bb3, label %bb22
bb3:
  %v31 = extractvalue { ptr, i64 } %v19, 0
  %v32 = getelementptr inbounds double, ptr %v31, i64 %v28
  %v33 = load double, ptr %v32, align 8
  %v34 = fcmp olt double %v33, 0.0000001
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb5, label %bb4
bb4:
  %v36 = fmul contract double 0.5, %v33
  %v37 = fmul contract double %v36, %v33
  %v38 = fsub contract double 1.0, %v37
  br label %bb6
bb5:
  %v39 = call double @__nv_cos(double %v33) #0
  br label %bb12
bb6:
  %v40 = phi double [ %v38, %bb4 ], [ %v39, %bb13 ]
  %v41 = phi double [ 1.0, %bb4 ], [ %v58, %bb13 ]
  %v42 = icmp eq i64 %v24, 18446744073709551615
  br i1 %v42, label %bb17, label %bb14
bb7:
  %v43 = extractvalue { i8, ptr } %v67, 1
  %v44 = extractvalue { ptr, i64 } %v17, 1
  %v45 = icmp ult i64 %v24, %v44
  br i1 %v45, label %bb8, label %bb23
bb8:
  %v46 = extractvalue { ptr, i64 } %v17, 0
  %v47 = getelementptr inbounds double, ptr %v46, i64 %v24
  %v48 = load double, ptr %v47, align 8
  %v49 = fmul contract double %v40, %v48
  %v50 = extractvalue { ptr, i64 } %v18, 1
  %v51 = icmp ult i64 %v24, %v50
  br i1 %v51, label %bb9, label %bb24
bb9:
  %v52 = extractvalue { ptr, i64 } %v18, 0
  %v53 = getelementptr inbounds double, ptr %v52, i64 %v24
  %v54 = load double, ptr %v53, align 8
  %v55 = fmul contract double %v41, %v54
  %v56 = fadd contract double %v49, %v55
  store double %v56, ptr %v43, align 8
  br label %bb11
bb10:
  br label %bb11
bb11:
  ret void
bb12:
  %v57 = call double @__nv_sin(double %v33) #0
  br label %bb13
bb13:
  %v58 = fdiv contract double %v57, %v33
  br label %bb6
bb14:
  %v59 = extractvalue { ptr, i64 } %v21, 1
  %v60 = icmp ult i64 %v24, %v59
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb16, label %bb15
bb15:
  %v62 = extractvalue { ptr, i64 } %v21, 0
  %v63 = getelementptr inbounds double, ptr %v62, i64 %v24
  %v64 = insertvalue { i8, ptr } undef, i8 1, 0
  %v65 = insertvalue { i8, ptr } %v64, ptr %v63, 1
  br label %bb18
bb16:
  br label %bb17
bb17:
  %v66 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb18
bb18:
  %v67 = phi { i8, ptr } [ %v65, %bb15 ], [ %v66, %bb17 ]
  %v68 = extractvalue { i8, ptr } %v67, 0
  %v69 = zext i8 %v68 to i64
  %v70 = icmp eq i64 %v69, 1
  br i1 %v70, label %bb7, label %bb19
bb19:
  %v71 = icmp eq i64 %v69, 0
  br i1 %v71, label %bb10, label %bb20
bb20:
  unreachable
bb21:
  unreachable
bb22:
  unreachable
bb23:
  unreachable
bb24:
  unreachable
}

declare double @__nv_sqrt(double)

define void @sphere_tangent_norm(ptr %v0, i64 %v1, i32 %v2, ptr %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi i32 [ %v2, %entry ]
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = alloca {  }, align 1
  %v13 = bitcast ptr %v12 to ptr
  %v14 = call i64 @_RINvNtNtCs8uibrORiSfe_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs8GH1ove06AH_11cartan_cuda(ptr %v13) #0
  br label %bb1
bb1:
  %v15 = zext i32 %v10 to i64
  br label %bb2
bb2:
  %v16 = phi double [ 0.0, %bb1 ], [ %v30, %bb6 ]
  %v17 = phi i64 [ 0, %bb1 ], [ %v38, %bb6 ]
  %v18 = icmp ult i64 %v17, %v15
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb11, label %bb10
bb3:
  unreachable
bb4:
  %v20 = extractvalue { i64, i64 } %v37, 1
  %v21 = mul i64 %v14, %v15
  %v22 = add i64 %v21, %v20
  %v23 = extractvalue { ptr, i64 } %v9, 1
  %v24 = icmp ult i64 %v22, %v23
  br i1 %v24, label %bb6, label %bb21
bb5:
  %v25 = icmp eq i64 %v14, 18446744073709551615
  br i1 %v25, label %bb17, label %bb14
bb6:
  %v26 = extractvalue { ptr, i64 } %v9, 0
  %v27 = getelementptr inbounds double, ptr %v26, i64 %v22
  %v28 = load double, ptr %v27, align 8
  %v29 = fmul contract double %v28, %v28
  %v30 = fadd contract double %v16, %v29
  br label %bb2
bb7:
  %v31 = extractvalue { i8, ptr } %v51, 1
  %v32 = call double @__nv_sqrt(double %v16) #0
  br label %bb20
bb8:
  br label %bb9
bb9:
  ret void
bb10:
  %v33 = add i64 %v17, 1
  %v34 = insertvalue { i64, i64 } undef, i64 1, 0
  %v35 = insertvalue { i64, i64 } %v34, i64 %v17, 1
  br label %bb12
bb11:
  %v36 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb12
bb12:
  %v37 = phi { i64, i64 } [ %v35, %bb10 ], [ %v36, %bb11 ]
  %v38 = phi i64 [ %v33, %bb10 ], [ %v17, %bb11 ]
  %v39 = extractvalue { i64, i64 } %v37, 0
  %v40 = bitcast i64 %v39 to i64
  %v41 = icmp eq i64 %v40, 0
  br i1 %v41, label %bb5, label %bb13
bb13:
  %v42 = icmp eq i64 %v40, 1
  br i1 %v42, label %bb4, label %bb3
bb14:
  %v43 = extractvalue { ptr, i64 } %v11, 1
  %v44 = icmp ult i64 %v14, %v43
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb16, label %bb15
bb15:
  %v46 = extractvalue { ptr, i64 } %v11, 0
  %v47 = getelementptr inbounds double, ptr %v46, i64 %v14
  %v48 = insertvalue { i8, ptr } undef, i8 1, 0
  %v49 = insertvalue { i8, ptr } %v48, ptr %v47, 1
  br label %bb18
bb16:
  br label %bb17
bb17:
  %v50 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb18
bb18:
  %v51 = phi { i8, ptr } [ %v49, %bb15 ], [ %v50, %bb17 ]
  %v52 = extractvalue { i8, ptr } %v51, 0
  %v53 = zext i8 %v52 to i64
  %v54 = icmp eq i64 %v53, 1
  br i1 %v54, label %bb7, label %bb19
bb19:
  %v55 = icmp eq i64 %v53, 0
  br i1 %v55, label %bb8, label %bb3
bb20:
  store double %v32, ptr %v31, align 8
  br label %bb9
bb21:
  unreachable
}

define void @sphere_cos_angle(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, ptr %v5, i64 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v5, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v6, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi i32 [ %v4, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = alloca {  }, align 1
  %v18 = bitcast ptr %v17 to ptr
  %v19 = call i64 @_RINvNtNtCs8uibrORiSfe_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs8GH1ove06AH_11cartan_cuda(ptr %v18) #0
  br label %bb1
bb1:
  %v20 = zext i32 %v15 to i64
  br label %bb2
bb2:
  %v21 = phi double [ 0.0, %bb1 ], [ %v41, %bb7 ]
  %v22 = phi i64 [ 0, %bb1 ], [ %v53, %bb7 ]
  %v23 = icmp ult i64 %v22, %v20
  %v24 = xor i1 %v23, 1
  br i1 %v24, label %bb18, label %bb17
bb3:
  unreachable
bb4:
  %v25 = extractvalue { i64, i64 } %v52, 1
  %v26 = mul i64 %v19, %v20
  %v27 = add i64 %v26, %v25
  %v28 = extractvalue { ptr, i64 } %v13, 1
  %v29 = icmp ult i64 %v27, %v28
  br i1 %v29, label %bb6, label %bb27
bb5:
  %v30 = fcmp ogt double %v21, 1.0
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb9, label %bb8
bb6:
  %v32 = extractvalue { ptr, i64 } %v13, 0
  %v33 = getelementptr inbounds double, ptr %v32, i64 %v27
  %v34 = load double, ptr %v33, align 8
  %v35 = extractvalue { ptr, i64 } %v14, 1
  %v36 = icmp ult i64 %v27, %v35
  br i1 %v36, label %bb7, label %bb28
bb7:
  %v37 = extractvalue { ptr, i64 } %v14, 0
  %v38 = getelementptr inbounds double, ptr %v37, i64 %v27
  %v39 = load double, ptr %v38, align 8
  %v40 = fmul contract double %v34, %v39
  %v41 = fadd contract double %v21, %v40
  br label %bb2
bb8:
  br label %bb13
bb9:
  %v42 = fcmp olt double %v21, -1.0
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb11, label %bb10
bb10:
  br label %bb12
bb11:
  br label %bb12
bb12:
  %v44 = phi double [ -1.0, %bb10 ], [ %v21, %bb11 ]
  br label %bb13
bb13:
  %v45 = phi double [ 1.0, %bb8 ], [ %v44, %bb12 ]
  %v46 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v46, label %bb24, label %bb21
bb14:
  %v47 = extractvalue { i8, ptr } %v66, 1
  store double %v45, ptr %v47, align 8
  br label %bb16
bb15:
  br label %bb16
bb16:
  ret void
bb17:
  %v48 = add i64 %v22, 1
  %v49 = insertvalue { i64, i64 } undef, i64 1, 0
  %v50 = insertvalue { i64, i64 } %v49, i64 %v22, 1
  br label %bb19
bb18:
  %v51 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb19
bb19:
  %v52 = phi { i64, i64 } [ %v50, %bb17 ], [ %v51, %bb18 ]
  %v53 = phi i64 [ %v48, %bb17 ], [ %v22, %bb18 ]
  %v54 = extractvalue { i64, i64 } %v52, 0
  %v55 = bitcast i64 %v54 to i64
  %v56 = icmp eq i64 %v55, 0
  br i1 %v56, label %bb5, label %bb20
bb20:
  %v57 = icmp eq i64 %v55, 1
  br i1 %v57, label %bb4, label %bb3
bb21:
  %v58 = extractvalue { ptr, i64 } %v16, 1
  %v59 = icmp ult i64 %v19, %v58
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb23, label %bb22
bb22:
  %v61 = extractvalue { ptr, i64 } %v16, 0
  %v62 = getelementptr inbounds double, ptr %v61, i64 %v19
  %v63 = insertvalue { i8, ptr } undef, i8 1, 0
  %v64 = insertvalue { i8, ptr } %v63, ptr %v62, 1
  br label %bb25
bb23:
  br label %bb24
bb24:
  %v65 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb25
bb25:
  %v66 = phi { i8, ptr } [ %v64, %bb22 ], [ %v65, %bb24 ]
  %v67 = extractvalue { i8, ptr } %v66, 0
  %v68 = zext i8 %v67 to i64
  %v69 = icmp eq i64 %v68, 1
  br i1 %v69, label %bb14, label %bb26
bb26:
  %v70 = icmp eq i64 %v68, 0
  br i1 %v70, label %bb15, label %bb3
bb27:
  unreachable
bb28:
  unreachable
}

declare double @__nv_acos(double)

define void @sphere_log_apply(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v5, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v8, 1
  br label %bb0
bb0:
  %v17 = phi { ptr, i64 } [ %v10, %entry ]
  %v18 = phi { ptr, i64 } [ %v12, %entry ]
  %v19 = phi { ptr, i64 } [ %v14, %entry ]
  %v20 = phi i32 [ %v6, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = alloca {  }, align 1
  %v23 = bitcast ptr %v22 to ptr
  %v24 = call i64 @_RINvNtNtCs8uibrORiSfe_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs8GH1ove06AH_11cartan_cuda(ptr %v23) #0
  br label %bb1
bb1:
  %v25 = zext i32 %v20 to i64
  %v26 = icmp eq i64 %v25, 0
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb2, label %bb21
bb2:
  %v28 = udiv i64 %v24, %v25
  %v29 = extractvalue { ptr, i64 } %v19, 1
  %v30 = icmp ult i64 %v28, %v29
  br i1 %v30, label %bb3, label %bb22
bb3:
  %v31 = extractvalue { ptr, i64 } %v19, 0
  %v32 = getelementptr inbounds double, ptr %v31, i64 %v28
  %v33 = load double, ptr %v32, align 8
  %v34 = call double @__nv_acos(double %v33) #0
  br label %bb12
bb4:
  br label %bb6
bb5:
  %v35 = call double @__nv_sin(double %v34) #0
  br label %bb13
bb6:
  %v36 = phi double [ 1.0, %bb4 ], [ %v54, %bb13 ]
  %v37 = icmp eq i64 %v24, 18446744073709551615
  br i1 %v37, label %bb17, label %bb14
bb7:
  %v38 = extractvalue { i8, ptr } %v63, 1
  %v39 = extractvalue { ptr, i64 } %v18, 1
  %v40 = icmp ult i64 %v24, %v39
  br i1 %v40, label %bb8, label %bb23
bb8:
  %v41 = extractvalue { ptr, i64 } %v18, 0
  %v42 = getelementptr inbounds double, ptr %v41, i64 %v24
  %v43 = load double, ptr %v42, align 8
  %v44 = extractvalue { ptr, i64 } %v17, 1
  %v45 = icmp ult i64 %v24, %v44
  br i1 %v45, label %bb9, label %bb24
bb9:
  %v46 = extractvalue { ptr, i64 } %v17, 0
  %v47 = getelementptr inbounds double, ptr %v46, i64 %v24
  %v48 = load double, ptr %v47, align 8
  %v49 = fmul contract double %v33, %v48
  %v50 = fsub contract double %v43, %v49
  %v51 = fmul contract double %v50, %v36
  store double %v51, ptr %v38, align 8
  br label %bb11
bb10:
  br label %bb11
bb11:
  ret void
bb12:
  %v52 = fcmp olt double %v34, 0.0000001
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb5, label %bb4
bb13:
  %v54 = fdiv contract double %v34, %v35
  br label %bb6
bb14:
  %v55 = extractvalue { ptr, i64 } %v21, 1
  %v56 = icmp ult i64 %v24, %v55
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb16, label %bb15
bb15:
  %v58 = extractvalue { ptr, i64 } %v21, 0
  %v59 = getelementptr inbounds double, ptr %v58, i64 %v24
  %v60 = insertvalue { i8, ptr } undef, i8 1, 0
  %v61 = insertvalue { i8, ptr } %v60, ptr %v59, 1
  br label %bb18
bb16:
  br label %bb17
bb17:
  %v62 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb18
bb18:
  %v63 = phi { i8, ptr } [ %v61, %bb15 ], [ %v62, %bb17 ]
  %v64 = extractvalue { i8, ptr } %v63, 0
  %v65 = zext i8 %v64 to i64
  %v66 = icmp eq i64 %v65, 1
  br i1 %v66, label %bb7, label %bb19
bb19:
  %v67 = icmp eq i64 %v65, 0
  br i1 %v67, label %bb10, label %bb20
bb20:
  unreachable
bb21:
  unreachable
bb22:
  unreachable
bb23:
  unreachable
bb24:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define i64 @_RINvNtNtCs8uibrORiSfe_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs8GH1ove06AH_11cartan_cuda(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v5 = zext i32 %v2 to i64
  %v6 = zext i32 %v3 to i64
  %v7 = zext i32 %v4 to i64
  %v8 = icmp eq i64 %v6, 0
  br i1 %v8, label %bb10, label %bb8
bb4:
  %v9 = xor i1 %v20, 1
  br i1 %v9, label %bb6, label %bb5
bb5:
  %v10 = icmp ne i64 %v19, 18446744073709551615
  br label %bb7
bb6:
  br label %bb7
bb7:
  %v11 = phi i1 [ %v10, %bb5 ], [ 0, %bb6 ]
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb14, label %bb13
bb8:
  %v13 = sub i64 18446744073709551615, %v7
  %v14 = udiv i64 %v13, %v6
  %v15 = icmp ugt i64 %v5, %v14
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb11, label %bb9
bb9:
  br label %bb10
bb10:
  br label %bb12
bb11:
  %v17 = mul i64 %v5, %v6
  %v18 = add i64 %v17, %v7
  br label %bb12
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCs8uibrORiSfe_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs8GH1ove06AH_11cartan_cuda(ptr %v1) #0
  br label %bb4
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v21, label %bb14, label %bb15
bb14:
  br label %bb15
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCs8uibrORiSfe_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs8GH1ove06AH_11cartan_cuda(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1
  %v3 = xor i1 %v2, 1
  br i1 %v3, label %bb2, label %bb1
bb1:
  br label %bb8
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb3
bb3:
  %v5 = icmp eq i32 %v4, 1
  br i1 %v5, label %bb4, label %bb5
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  br label %bb6
bb5:
  br label %bb7
bb6:
  %v7 = icmp eq i32 %v6, 1
  br label %bb7
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1
  br i1 %v10, label %bb9, label %bb10
bb9:
  %v11 = icmp eq i8 0, 2
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb12
bb12:
  %v14 = icmp eq i32 %v13, 1
  br i1 %v14, label %bb13, label %bb14
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb15
bb14:
  br label %bb16
bb15:
  %v16 = icmp eq i32 %v15, 1
  br label %bb16
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1
  br i1 %v19, label %bb19, label %bb18
bb18:
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20
}


@llvm.used = appending global [4 x ptr] [ptr @sphere_exp_apply, ptr @sphere_tangent_norm, ptr @sphere_cos_angle, ptr @sphere_log_apply], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{ptr @sphere_exp_apply, !"kernel", i32 1}
!1 = !{ptr @sphere_tangent_norm, !"kernel", i32 1}
!2 = !{ptr @sphere_cos_angle, !"kernel", i32 1}
!3 = !{ptr @sphere_log_apply, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1, !2, !3}

!nvvmir.version = !{!4}
!4 = !{i32 2, i32 0, i32 3, i32 2}
