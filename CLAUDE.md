# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 构建

```bash
# Debug APK
./gradlew assembleDebug

# Release APK
./gradlew assembleRelease

# 清理构建
./gradlew clean assembleDebug
```

NDK 版本: 29.0.14206865 (定义在 `app/build.gradle`)。需要 Android SDK 33+ compileSdk。

---

## 项目概览

这是一个 **在 Android 手机上实时运行 YOLOv8 全系列视觉任务** 的 App，基于 **ncnn**（腾讯开源的神经网络推理框架）实现端侧推理加速。不只做目标检测，而是覆盖了 **检测、分割、姿态估计、分类、旋转框检测** 五大任务类型，并针对电力场景定制了专用模型。

## 9 种任务模式

App 打开相机实时预览，在画面中检测目标并绘制结果。通过下拉菜单 **即时切换任务/模型/推理后端**，无需重启：

| taskid | 任务 | 类型 | 说明 | 类数 |
|---|---|---|---|---|
| 0 | `coco` | 检测 | COCO 通用 80 类 | 80 |
| 1 | `oiv7` | 检测 | Google Open Images v7 | 601 |
| 2 | `seg` | 分割 | 检测 + 实例掩码 | 80 |
| 3 | `pose` | 姿态 | 人体 17 关键点 + 骨架 | 1(person) |
| 4 | `cls` | 分类 | 整图分类 | 1000 |
| 5 | `obb` | 旋转框 | 遥感定向检测 (DOTAv1) | 15 |
| 6 | `bolt` | 检测 | **螺栓检测**（电力） | 自定义 |
| 7 | `weld` | 检测 | **焊缝检测**（电力） | 自定义 |
| 8 | `crack_seg` | 分割 | **裂纹分割**（电力） | 自定义 |

每种任务可选 **n/s/m**（nano/small/medium）三种模型规模，以及 **320/480/640** 三种输入分辨率，组合出 9 种模型变体：

```
model_array: n-320 / s-320 / m-320 / n-480 / s-480 / m-480 / n-640 / s-640 / m-640
```

> bolt/weld/crack_seg 三个电力模型固定使用 n-640 规格。

推理后端可在 **CPU / Vulkan GPU / Mesa Turnip GPU（Adreno 专用驱动）** 三者间即时切换。

---

## 三层架构

```
┌──────────────────────────────────────────────────┐
│  Java 层 (UI + 生命周期)                          │
│  MainActivity — 相机、Spinner 切换、权限管理       │
│  YOLOv8Ncnn — 4 个 native 方法暴露给 C++          │
├──────────────────────────────────────────────────┤
│  C++ JNI 胶水层 (yolov8ncnn.cpp)                  │
│  模型加载/卸载 · 相机预览循环 · GPU 初始化 · FPS  │
├────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬───┤
│det │oiv7 │ seg │ pose│ cls │ obb │bolt │weld │crc│
│_coc│_det │     │     │     │     │_det │_det │_se│
│o   │     │     │     │     │     │     │     │g  │
└────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴───┘
```

### ① Java 层

**MainActivity.java**
- 持有 `YOLOv8Ncnn` 实例（JNI 桥接）
- `onCreate` → 设置 `SurfaceView` 预览 + 绑定三个 Spinner 的事件监听
- `onResume` → 请求相机权限，打开相机
- `onPause` → 关闭相机
- Spinner 切换时调用 `reload()` → `yolov8ncnn.loadModel(getAssets(), taskid, modelid, cpugpu)`
- 切换摄像头按钮 → `openCamera(new_facing)`

**YOLOv8Ncnn.java**
- 4 个 `native` 方法：`loadModel` / `openCamera` / `closeCamera` / `setOutputWindow`
- 静态初始化块：`System.loadLibrary("yolov8ncnn")`

### ② C++ JNI 胶水层 (`yolov8ncnn.cpp`)

**生命周期**

| 事件 | 行为 |
|---|---|
| `JNI_OnLoad` | 创建 `MyNdkCamera` + `ncnn::create_gpu_instance()` |
| `JNI_OnUnload` | 删除模型 → `ncnn::destroy_gpu_instance()` → 删除相机 |
| `loadModel` | 构造模型路径 → 销毁旧模型 → 创建新子类 → 加载参数 → 设置 target_size |
| `openCamera` | 调用 NdkCamera 打开前后摄 |
| 每帧回调 `on_image_render` | `detect()` → `draw()` → 计算 FPS |

**模型文件名构造规则**：
```cpp
// tasknames[9] = {"", "_oiv7", "_seg", "_pose", "_cls", "_obb", "_bolt", "_weld", "_crack_seg"}
// modeltype = "n" / "s" / "m"
parampath = "yolov8" + modeltype + tasknames[taskid] + ".ncnn.param"
// 例: taskid=2, modelid=0 => "yolov8n_seg.ncnn.param"
```

**GPU 初始化**：
- CPU 模式：不创建 Vulkan 实例
- GPU 模式：`ncnn::create_gpu_instance()`
- Turnip 模式：`ncnn::create_gpu_instance("libvulkan_freedreno.so")` — 绕过系统 Vulkan 驱动，直接加载 Mesa Turnip 的专用 Adreno 驱动

### ③ C++ 检测后端

**类层次**：

```
YOLOv8 (基类)
├── load() / set_det_target_size() / detect()(纯虚) / draw()(纯虚)
│
├── YOLOv8_det            → generate_proposals() + nms_sorted_bboxes()
│   ├── YOLOv8_det_coco   → draw() COCO 80 类 + 19 色
│   ├── YOLOv8_det_oiv7   → draw() OIV7 601 类
│   ├── YOLOv8_det_bolt   → draw() 螺栓专用
│   └── YOLOv8_det_weld   → draw() 焊缝专用
│
├── YOLOv8_seg            → detect() + 掩码合成 (Gemm + Sigmoid + resize_bilinear)
│   └── YOLOv8_seg_crack  → draw() 裂纹专用
│
├── YOLOv8_pose           → detect() 双输出 + 17 关键点 + draw() 骨架连线
│
├── YOLOv8_cls            → detect() 分类 + draw() 显示 Top-1
│
└── YOLOv8_obb            → detect() 双输出 + 角度解码 + NMS(RotatedRect) + draw()
```

---

## 统一的预处理/后处理流水线

所有检测类共享相同的处理模式（以 `YOLOv8_det::detect()` 为例）：

```
输入帧 RGB (640×480 或其他)
    │
    ▼
① Letterbox 缩放
   └─ 保持宽高比缩放到 target_size，短板补零到 32 的倍数
    │
    ▼
② 归一化: 像素值 × 1/255
    │
    ▼
③ ncnn 推理
   └─ 输入 blob: "in0" (归一化后的张量)
   └─ 输出 blob: "out0" (有时还有 out1, out2)
    │
    ▼
④ DFL 解码生成候选框
   └─ 每个 grid 单元格 (8/16/32 stride) 遍历
   └─ 64 维 DFL → softmax → 加权求和 → 4 个边界距离
   └─ sigmoid 计算类别分数
   └─ 保留 score > 0.25 的框
    │
    ▼
⑤ NMS 非极大值抑制
   └─ 按分数降序排序
   └─ IoU > 0.45 的框被抑制
    │
    ▼
⑥ 坐标反算 (撤销 letterbox)
   └─ (坐标 - pad偏移) / scale
   └─ 裁剪到 [0, img_w-1] / [0, img_h-1]
    │
    ▼
⑦ 按面积降序排列 → draw() 绘制
   └─ det: cv::rectangle + 标签文字
   └─ seg: 掩码混合透明色 + 矩形框
   └─ pose: 骨架连线(16对) + 关键点圆点 + 矩形框
   └─ obb: 旋转矩形 4 条边 + 标签
   └─ cls: 类名+分数
```

### 各任务输出差异

| 任务 | 输出 blobs | 后处理特点 |
|---|---|---|
| det | `out0: [144, 8400]` | 标准 DFL + NMS |
| seg | `out0: [176, 8400]` + `out1: [32, 8400]`(mask_feat) + `out2: [32, 160, 160]`(mask_protos) | `mask = Gemm(mask_feat, mask_protos)` → Sigmoid → 裁剪到检测框 → resize 到原图 |
| pose | `out0: [65, 8400]` + `out1: [51, 8400]` | 17 点 × 3 (x, y, prob)，关键点坐标解码公式 `(x + pred*2)*stride` |
| cls | `out0: [1000]` | 全图分类，无 NMS |
| obb | `out0: [79, 21504]` + `out1: [1, 21504]` | 角度解码 `sigmoid(angle) - 0.25` → `cv::RotatedRect`，NMS 使用 `rotatedRectangleIntersection` |

---

## 模型转换流程

YOLOv8 PyTorch 模型需要经过 **PNNX 工具链** 转换为 ncnn 格式：

```
yolov8n.pt
    │ yolo export
    ▼
yolov8n.torchscript  (静态形状)
    │ pnnx
    ▼
yolov8n_pnnx.py      (可编辑的 PNNX 脚本)
    │
    │ 手动编辑:
    │   view(1, 144, 6400) → view(1, 144, -1).transpose(1, 2)
    │   cat(dim=2)         → cat(dim=1)
    │   去掉后处理部分
    │
    ▼
yolov8n_pnnx.py.pt   (动态形状 TorchScript)
    │ pnnx inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
    ▼
yolov8n.ncnn.param + yolov8n.ncnn.bin  ✓
```

核心技巧：将 `view(固定尺寸)` 改为 `view(-1) + transpose`，再 `cat(dim=1)` 而不是 `dim=2`，使得模型能接受 **任意输入尺寸**，而不是只能跑 640×640。

---

## 依赖管理

所有依赖都是预编译的二进制包，不参与源码构建：

| 依赖 | 版本 | 用途 | 位置 |
|---|---|---|---|
| ncnn | 20260526 | 神经网络推理引擎 | `jni/ncnn-20260526-android-vulkan/` |
| opencv-mobile | 4.13.0 | 图像处理 (resize/pad/draw) | `jni/opencv-mobile-4.13.0-android/` |
| Mesa Turnip | 26.1.3 | Adreno GPU Vulkan 驱动 | `jniLibs/arm64-v8a/libvulkan_freedreno.so` |

CMakeLists.txt 通过 `find_package` 引用：
```cmake
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.13.0-android/sdk/native/jni)
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260526-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
```

---

## 构建方式

### 本地构建

```bash
# 确保已解压 ncnn 和 opencv-mobile 到 jni/ 目录
# 确保 libvulkan_freedreno.so 已放到 app/src/main/jniLibs/arm64-v8a/（Mesa Turnip，可选）
./gradlew assembleDebug    # Debug APK
./gradlew assembleRelease  # Release APK
```

### CI 自动构建

GitHub Actions (`.github/workflows/release-apk.yml`) 在 `workflow_dispatch` 时：
1. 自动下载 ncnn/opencv-mobile/Mesa Turnip 依赖
2. 解压到正确位置
3. 编译 Release APK
4. 用临时 keystore 签名
5. 创建 GitHub Release 并上传 APK