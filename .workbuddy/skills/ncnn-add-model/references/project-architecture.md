# ncnn-android-yolov8 项目架构参考

## 项目概览

腾讯 ncnn 官方 YOLOv8 Android Demo，在 Android 设备上实时运行 YOLOv8 推理。

- 包名：com.tencent.yolov8ncnn
- 构建：Gradle 8.7.3 + CMake 3.22.1 + NDK r29
- 编译：compileSdk 33 / targetSdk 35 / minSdk 24
- 推理：ncnn Vulkan GPU + OpenCV Mobile 4.13

## 架构层次

```
┌─────────────────────────────────────┐
│  MainActivity.java                  │  ← UI层：SurfaceView + Spinner
│    ↓ JNI                            │
│  YOLOv8Ncnn.java                    │  ← Native方法声明
├─────────────────────────────────────┤
│  yolov8ncnn.cpp                     │  ← JNI桥接：loadModel()
│    ↓                                │
│  yolov8.h/cpp                       │  ← 基类：load(), detect(), draw()
│    ↓ 多态派生                        │
│  YOLOv8_det_coco   → COCO 80类检测  │
│  YOLOv8_det_oiv7   → OIV7 601类检测 │
│  YOLOv8_det_bolt   → 螺栓4类检测    │  ← det模型参考模板
│  YOLOv8_det_weld   → 焊缝6类检测    │  ← det模型参考模板
│  YOLOv8_seg        → COCO 80类分割  │
│  YOLOv8_seg_crack  → 裂纹1类分割    │  ← seg模型参考模板
│  YOLOv8_pose       → 姿态估计        │
│  YOLOv8_cls        → 图像分类        │
│  YOLOv8_obb        → 旋转框检测      │
│    ↓                                │
│  ndkcamera.cpp                      │  ← NDK Camera2 相机管理
├─────────────────────────────────────┤
│  ncnn (Vulkan GPU)                  │  ← 推理框架
│  OpenCV Mobile 4.13                 │  ← 绘制
│  Mesa Turnip (可选)                  │  ← 高通GPU Vulkan驱动
└─────────────────────────────────────┘
```

## 类继承结构

```
YOLOv8 (基类)
  ├── load()          — 加载 .param + .bin 到 ncnn::Net
  ├── set_det_target_size() — 设置输入尺寸
  ├── detect() = 0    — 纯虚函数，各子类实现
  └── draw() = 0      — 纯虚函数，各子类实现

YOLOv8_det (检测中间类)
  ├── detect()        — 通用检测推理 + NMS 后处理
  │   输入: in0 (RGB Mat) → 输出: Object 列表 (仅rect+label+prob)
  │   推理流程: letterbox → ncnn extract("out0") → generate_proposals → NMS
  │   out0 期望: (8400, 64+num_class) raw DFL 格式
  └── draw() = 0      — 纯虚函数，各检测子类只需实现 draw()

YOLOv8_det_coco / YOLOv8_det_oiv7 / YOLOv8_det_bolt / YOLOv8_det_weld
  └── draw()          — 绘制矩形框 + 类别名 + 置信度百分比

YOLOv8_seg (分割中间类)
  ├── detect()        — 通用分割推理 + NMS + mask处理
  │   输入: in0 (RGB Mat) → 输出: Object 列表 (rect+label+prob+mask)
  │   推理流程: letterbox → ncnn extract("out0","out1","out2") → generate_proposals → NMS → mask计算
  │   out0 期望: (8400, 64+num_class) raw DFL 格式（检测头）
  │   out1 期望: (8400, 32) mask系数（掩码头）
  │   out2 期望: (32, 160, 160) mask原型（原型头）
  └── draw() = 0      — 纯虚函数，各分割子类只需实现 draw()

YOLOv8_seg_crack
  └── draw()          — 绘制矩形框 + 类别名 + mask半透明叠加 + 置信度百分比

其他中间类: YOLOv8_pose / YOLOv8_cls / YOLOv8_obb
  └── detect() + draw() — 各自完整实现
```

**关键设计**: 
- det模型: `detect()` 在 YOLOv8_det 中通用实现(1输出blob out0)，新检测模型只需实现 `draw()`
- seg模型: `detect()` 在 YOLOv8_seg 中通用实现(3输出blob out0/out1/out2)，新分割模型只需实现 `draw()`

## loadModel 函数关键逻辑

```cpp
// taskid → 任务类型映射 (当前9种)
// 0=coco, 1=oiv7, 2=seg, 3=pose, 4=cls, 5=obb, 6=bolt, 7=weld, 8=crack_seg
const char* tasknames[9] = { "", "_oiv7", "_seg", "_pose", "_cls", "_obb", "_bolt", "_weld", "_crack_seg" };

// modelid → 尺寸映射 (内置模型)
const char* modeltypes[9] = { "n", "s", "m", "n", "s", "m", "n", "s", "m" };

// 自定义模型强制variant
if (taskid == 6 || taskid == 7 || taskid == 8) modeltype = "n";
else modeltype = modeltypes[(int)modelid];

// 模型路径构建规则
parampath = "yolov8" + modeltype + tasknames[taskid] + ".ncnn.param"
modelpath = "yolov8" + modeltype + tasknames[taskid] + ".ncnn.bin"

// target_size 映射
if (taskid == 6 || taskid == 7 || taskid == 8) target_size = 640;  // 自定义模型固定640
else { modelid 0-2→320, 3-5→480, 6-8→640 }  // 内置模型按modelid映射

// 实例化 (注意det和seg的区别)
if (taskid == 0) g_yolov8 = new YOLOv8_det_coco;   // det基类
if (taskid == 1) g_yolov8 = new YOLOv8_det_oiv7;   // det基类
if (taskid == 2) g_yolov8 = new YOLOv8_seg;         // seg基类
if (taskid == 6) g_yolov8 = new YOLOv8_det_bolt;    // det派生
if (taskid == 7) g_yolov8 = new YOLOv8_det_weld;    // det派生
if (taskid == 8) g_yolov8 = new YOLOv8_seg_crack;   // seg派生 ⚠️不是det!
```

## detect() 推理流程详解

### 检测模型 (YOLOv8_det::detect)

1. **letterbox 缩放**: 输入 RGB → resize + pad 到 target_size 的倍数
2. **归一化**: 1/255.f 逐通道
3. **ncnn 推理**: `ex.input("in0", in_pad)` → `ex.extract("out0", out)`
4. **generate_proposals()**: 按 stride(8/16/32) 分段解析输出
   - 每行格式: DFL bbox(64列×4) + class scores(num_class列)
   - `num_class = pred.w - reg_max_1 * 4`（reg_max_1=16，即 `pred.w - 64`）← 这是兼容性的关键
   - DFL softmax → 线性组合 → anchor 解码
   - class scores sigmoid → 阈值筛选
5. **NMS**: 按 prob 排序 → 非极大值抑制
6. **坐标还原**: letterbox offset 逆变换

### 分割模型 (YOLOv8_seg::detect)

1-5步与检测模型完全相同（使用同一个 `generate_proposals()`）
6. **提取 mask 数据**:
   - `ex.extract("out1", mask_feat)` — mask系数 (8400×32)
   - `ex.extract("out2", mask_protos)` — mask原型 (32×160×160)
7. **计算 mask**: Gemm(mask_feat × mask_protos) → Sigmoid → resize_bilinear
8. **生成 per-object mask**: 二值化(mask_threshold=0.5) → 存入 Object.mask (CV_8UC1)

## 文件结构

```
app/src/main/
├── java/com/tencent/yolov8ncnn/
│   ├── MainActivity.java          # 主界面
│   └── YOLOv8Ncnn.java            # JNI 声明
├── jni/
│   ├── CMakeLists.txt             # 构建配置
│   ├── yolov8ncnn.cpp             # JNI 入口 + loadModel
│   ├── yolov8.h/cpp               # 基类
│   ├── yolov8_det.cpp             # 检测 detect() + coco/oiv7 draw()
│   ├── yolov8_det_bolt.cpp        # 螺栓检测 draw()          ← det模型参考模板
│   ├── yolov8_det_weld.cpp        # 焊缝检测 draw()          ← det模型参考模板
│   ├── yolov8_seg.cpp             # 分割 detect()+draw()     ← seg基类
│   ├── yolov8_seg_crack.cpp       # 裂纹分割 draw()          ← seg模型参考模板
│   ├── yolov8_pose.cpp/cls/obb
│   ├── ndkcamera.h/cpp
│   └── ncnn-*/opencv-mobile-*/    # 库文件
└── assets/                         # 模型文件 (.param + .bin)
│   ├── yolov8n_bolt.ncnn.param/bin      # bolt模型(raw格式68列)
│   ├── yolov8n_weld.ncnn.param/bin      # weld模型(raw格式70列)
│   ├── yolov8n_crack_seg.ncnn.param/bin # crack_seg模型(3blob: out0/out1/out2)
│   └── ... (其他21组模型文件)
└── res/
    ├── layout/main.xml            # UI 布局（3 个 Spinner）
    └── values/strings.xml         # Spinner 数据源
```

## Spinner 数据源

strings.xml 中定义三个 string-array：
- `task_array`: coco / oiv7 / seg / pose / cls / obb / bolt / weld / crack_seg（索引=taskid，共9项）
- `model_array`: n-320 / s-320 / m-320 / n-480 / s-480 / m-480 / n-640 / s-640 / m-640（索引=modelid）
- `cpugpu_array`: CPU / GPU / turnip（索引=cpugpu）

## 已有自定义模型清单

| taskid | 模型名 | 基类 | variant | 尺寸 | 类别数 | param格式 |
|--------|--------|------|---------|------|--------|-----------|
| 6 | bolt | YOLOv8_det | n | 640 | 4 | raw(68=64+4) |
| 7 | weld | YOLOv8_det | n | 640 | 6 | raw(70=64+6) |
| 8 | crack_seg | YOLOv8_seg | n | 640 | 1(det)+32(mask) | raw(65=64+1), mask系数32, 原型160×160 |
