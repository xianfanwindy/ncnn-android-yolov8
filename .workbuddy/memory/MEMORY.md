# ncnn-android-yolov8 项目记忆

## 项目基本信息
- 腾讯 ncnn 官方 YOLOv8 Android Demo (com.tencent.yolov8ncnn)
- 构建：Gradle 8.7.3 + CMake 3.22.1 + NDK r29 + compileSdk 33 / targetSdk 35 / minSdk 24
- 推理：ncnn Vulkan GPU + OpenCV Mobile 4.13 + Mesa Turnip(可选)
- 相机：NDK Camera2 API

## 架构
- Java UI层 → JNI → yolov8ncnn.cpp → yolov8 基类 → 9种派生(det_coco/det_oiv7/det_bolt/det_weld/seg/seg_crack/pose/cls/obb)
- 模型：3种尺寸(n=320/s=480/m=640) × 6种任务 + bolt(n=640) + weld(n=640) + crack_seg(n=640) = 21组 .param+.bin
- bolt模型(taskid=6): YOLOv8n, 640×640, 4类, modeltype强制"n", param已修为raw(68列=64+4)
- weld模型(taskid=7): YOLOv8n, 640×640, 6类, modeltype强制"n", param已修为raw(70列=64+6)
- crack_seg模型(taskid=8): YOLOv8n-seg, 640×640, 1类(crack), modeltype强制"n", 继承YOLOv8_seg(实例分割)
- crack_seg param修复更复杂：3个输出blob(out0=raw检测65列, out1=掩码系数32列, out2=掩码原型), 原out0→deadout, 原out1→out2

## 已修复的构建问题
- CMake 版本：build.gradle cmake version 3.31.5 → 3.22.1（SDK 中不存在 3.31.5）
- CMakeLists.txt 变量名：set(E:\...) → set(OpenCV_DIR ...) / set(ncnn_DIR ...)（路径含冒号非法）

## 新增模型 SOP
- 项目级 Skill: `.workbuddy/skills/ncnn-add-model/` — 完整的新增模型 SOP
- 关键步骤：分析模型 → 检查输出格式(raw DFL vs decoded) → 修改param(如有) → 拷贝到assets → 修改代码文件 → 一致性审查
- 最大风险：decoded格式输出导致闪退（num_class=pred.w-64为负数），必须修改param为raw格式
- param修改原则：保留所有原始层(.bin对齐)，原out0→deadout，末尾加Concat+Reshape+Permute产生raw格式out0
- seg模型额外注意：3个输出blob(out0检测/out1掩码系数/out2掩码原型)，decoded格式需修复全部3个
- 自动化脚本：scripts/fix_ncnn_param_output.py (支持det/seg两种task-type)
