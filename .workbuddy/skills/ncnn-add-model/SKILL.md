---
name: ncnn-add-model
description: 向 ncnn-android-yolov8 项目添加自定义 YOLOv8 模型(检测/分割)的 SOP 技能。当用户提到"新增模型"、"添加模型"、"集成模型"、"加个检测模型"、"加个分割模型"、"自定义模型"到 ncnn-android-yolov8 项目时触发。覆盖从模型分析、param格式适配、代码集成到构建验证的完整流程，支持检测(det)和分割(seg)两种任务类型。
agent_created: true
---

# ncnn-android-yolov8 新增模型 SOP

向 ncnn-android-yolov8 项目添加自定义 YOLOv8 模型的标准化流程。支持**检测(det)**和**分割(seg)**两种任务类型。

## 触发条件

用户提到向 ncnn-android-yolov8 项目添加新模型，关键词：新增模型、添加模型、集成模型、自定义检测模型、自定义分割模型、实例分割模型。

## 前置知识

先加载 `references/project-architecture.md` 了解项目架构，再按以下 SOP 逐步执行。

## SOP 流程

### Step 1: 分析新模型

读取新模型目录中的所有文件，获取关键信息：

1. **metadata.yaml**（如有）— 类别数、类别名称、输入尺寸
2. **model_ncnn.py**（如有）— 模型导出脚本，确认架构变体(n/s/m)
3. **model.ncnn.param** — 头部确认输入 blob 名和魔术数 `7767517`，尾部确认输出 blob 名和维度
4. **model.ncnn.bin** — 确认文件存在且完整

需要确认的 6 个关键参数：

| 参数 | 确认方式 | 重要性 | det/seg差异 |
|------|---------|--------|-------------|
| 任务类型 | 用户指定 detect/seg | CRITICAL | 决定继承哪个基类 |
| 类别数 | metadata.yaml 或 param 尾部维度 | CRITICAL | det: out0列数-64; seg: 同 |
| 类别名称列表 | metadata.yaml 或训练配置 | draw() 需要 | 相同 |
| 输入尺寸 | metadata.yaml 或 param Reshape 层 | target_size 设置 | 相同 |
| YOLOv8 变体 | model_ncnn.py 或 param 层数对比 | modeltype 命名 | 相同 |
| mask系数维度 | seg模型param中Reshape层 | CRITICAL(seg) | seg独有，YOLOv8-seg固定32 |

### Step 2: 检查输出格式兼容性（CRITICAL — 防闪退）

**这是最容易出错的一步，必须仔细检查！**

#### 检测模型期望的输出

项目代码 `YOLOv8_det::detect()` 期望模型输出 **1个blob**：
- `out0` 形状：(8400, 64+num_class)
- 每行 = DFL bbox(64列) + raw class scores(num_class列)
- 代码计算 `num_class = pred.w - 64`

#### 分割模型期望的输出

项目代码 `YOLOv8_seg::detect()` 期望模型输出 **3个blob**：
- `out0`：(8400, 64+num_class) — 检测头，raw DFL格式
- `out1`：(8400, 32) — mask系数，每检测框32个系数
- `out2`：(32, 160, 160) — mask原型矩阵

#### decoded格式（危险）

如果模型内嵌了后处理（DFL解码+anchor解码+sigmoid），输出为 decoded 格式：
- 检测模型：1个blob `out0` 形状 (8400, 4+num_class) → `num_class = 8-64 = -56` → 内存越界 → **闪退！**
- 分割模型：2个blob `out0`(decoded混合) + `out1`(mask原型) → out0列数远小于64 → **闪退！**

#### 如何判断输出格式

1. 读取 param 文件尾部，搜索最后一个产生 `out0` 的层
2. 搜索所有 `Reshape` 层，看维度参数：
   - **raw 格式**：最终 Reshape 维度 = `64 + num_class`（如 68 表示 4类）
   - **decoded 格式**：最终维度 = `4 + num_class`（如 8 表示 4类）
3. 搜索 `Sigmoid` 层：如果输出路径中有 Sigmoid → decoded 格式
4. 搜索 `Softmax` 层（bbox DFL方向）：如果输出路径中有 → decoded 格式

#### 检测模型(decoded格式)的修复方法

加载 `references/param-format-fix.md` 获取详细的修改方法。

**核心原则**：
- **保留所有原始层**（.bin 权重按层序存储，删层会破坏对齐）
- 将原 `out0` 重命名为 `deadout`（不再被读取）
- 在检测头原始 conv 输出后添加 `Concat → Reshape → Permute` 层，产生新的 raw 格式 `out0`
- 新增 **10层**

#### 分割模型(decoded格式)的修复方法（更复杂）

**⚠️ seg模型有3个输出blob，修复比det模型更复杂！**

修复步骤：
1. 将原 `out0` 重命名为 `deadout`
2. 将原 `out1` 重命名为 `out2`（掩码原型的位置正好对齐）
3. 在检测头原始 conv 输出后添加 `Concat → Reshape → Permute` 层，产生新 `out0`(raw检测, 64+num_class列) — 10层
4. 在掩码系数 conv 输出后添加 `Reshape → Permute` 层，每stride一组，最终 `Concat` 产生新 `out1`(8400×32) — 7层
- 总新增 **17层**

加载 `references/param-format-fix.md` 获取详细修改方法和实际案例。

**关键差异对比**：

| 项目 | det模型 | seg模型 |
|------|---------|---------|
| 输出blob数 | 1个(out0) | 3个(out0/out1/out2) |
| param修改 | out0→deadout + 新out0 | out0→deadout + out1→out2 + 新out0 + 新out1 |
| 新增层数 | 10层 | 17层(检测头10 + 掩码系数7) |
| 继承基类 | YOLOv8_det | YOLOv8_seg |
| detect() | 1输出blob提取 | 3输出blob提取 + mask计算 |
| draw() | 矩形+标签 | 矩形+标签+mask半透明叠加 |
| 使用自动化脚本 | `--task-type det` | `--task-type seg --mask-blobs ...` |

### Step 3: 确定模型命名

命名规则：`yolov8{variant}{task_suffix}.ncnn.param/bin`

| 场景 | variant | task_suffix | 示例 |
|------|---------|-------------|------|
| YOLOv8n + 自定义检测 | n | _xxx | yolov8n_bolt.ncnn.param |
| YOLOv8s + 自定义检测 | s | _xxx | yolov8s_bolt.ncnn.param |
| YOLOv8m + 自定义检测 | m | _xxx | yolov8m_bolt.ncnn.param |
| YOLOv8n-seg + 自定义分割 | n | _xxx_seg | yolov8n_crack_seg.ncnn.param |

variant 由实际训练架构决定（n/s/m），**不是由输入尺寸决定**。640×640 输入的 YOLOv8n 仍然是 variant="n"。

seg模型的task_suffix建议加 `_seg` 后缀以区分，如 `_crack_seg`（避免与同名的det模型混淆）。

### Step 4: 拷贝模型文件

```bash
cp <model_dir>/model.ncnn.param app/src/main/assets/yolov8{variant}{task_suffix}.ncnn.param
cp <model_dir>/model.ncnn.bin app/src/main/assets/yolov8{variant}{task_suffix}.ncnn.bin
```

如果 Step 2 修改了 param 文件，则直接将修改后的版本写入 assets 目录。

**自动化脚本方式**（推荐）：

检测模型：
```bash
python3 scripts/fix_ncnn_param_output.py <input>.param <output>.param \
    --task-type det \
    --bbox-blobs <s8> <s16> <s32> \
    --cls-blobs <s8> <s16> <s32> \
    --num-class <N> \
    --task-name <name>
```

分割模型：
```bash
python3 scripts/fix_ncnn_param_output.py <input>.param <output>.param \
    --task-type seg \
    --bbox-blobs <s8> <s16> <s32> \
    --cls-blobs <s8> <s16> <s32> \
    --mask-blobs <s8> <s16> <s32> \
    --num-class <N> \
    --task-name <name>
```

### Step 5: 修改代码文件（7个文件）

按顺序修改以下文件：

#### 5.1 `app/src/main/res/values/strings.xml`

在 `task_array` 末尾添加 `<item>{task_name}</item>`，对应新的 taskid。

#### 5.2 `app/src/main/jni/yolov8.h`

**检测模型**：在 `YOLOv8_det_oiv7` 或其他 det 派生后面添加：
```cpp
class YOLOv8_det_{task_name} : public YOLOv8_det
{
public:
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};
```

**分割模型**：在 `YOLOv8_seg` 后面添加：
```cpp
class YOLOv8_seg_{task_name} : public YOLOv8_seg
{
public:
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};
```

⚠️ seg模型继承 `YOLOv8_seg`（不是 `YOLOv8_det`），只 override `draw()`（detect() 已在基类实现，包含3输出blob提取和mask计算）。

#### 5.3 创建 draw() 实现文件

**检测模型**：创建 `app/src/main/jni/yolov8_det_{task_name}.cpp`

参照 `yolov8_det_bolt.cpp` 的模式实现 `draw()`：
- `class_names[]` 填入实际类别名称
- 颜色数组 `colors[]` 使用项目统一的 19 色方案
- 绘制逻辑（rectangle + label text）

**分割模型**：创建 `app/src/main/jni/yolov8_seg_{task_name}.cpp`

参照 `yolov8_seg_crack.cpp` 的模式实现 `draw()`：
- `class_names[]` 填入实际类别名称
- 颜色数组 `colors[]` 使用项目统一的 19 色方案
- **额外的 mask 半透明叠加绘制**（这是seg模型与det模型的核心区别）：
  - 遍历每个检测对象的 rect 区域内的 mask 行
  - 对 mask=1 的像素做半透明混色：`bgr = bgr*0.5 + color*0.5`
  - 然后画 rectangle + label text

#### 5.4 `app/src/main/jni/CMakeLists.txt`

在 `add_library` 的源文件列表中添加新源文件：
- det模型：`yolov8_det_{task_name}.cpp`
- seg模型：`yolov8_seg_{task_name}.cpp`

#### 5.5 `app/src/main/jni/yolov8ncnn.cpp`（最关键的 5 处修改）

**5.5a** taskid 范围检查：`taskid > N` → `taskid > N+1`

**5.5b** tasknames 数组：末尾添加 `"_{task_suffix}"`，数组大小 N+1

**5.5c** modeltype 处理：如果新模型只有单一 variant，添加条件逻辑：
```cpp
if (taskid == NEW_TASKID || ...) modeltype = "n";
else modeltype = modeltypes[(int)modelid];
```

**5.5d** 实例化：

检测模型：`if (taskid == NEW_TASKID) g_yolov8 = new YOLOv8_det_{task_name};`

分割模型：`if (taskid == NEW_TASKID) g_yolov8 = new YOLOv8_seg_{task_name};`

**5.5e** target_size：如果新模型固定输入尺寸：
```cpp
if (taskid == NEW_TASKID || ...) target_size = 640;
else { // 原有的 modelid → size 映射 }
```

### Step 6: 一致性审查

逐项检查：

| 检查项 | 验证方法 | det/seg差异 |
|--------|---------|-------------|
| 模型文件在 assets | `ls app/src/main/assets/yolov8{variant}{suffix}*` | 相同 |
| param 输出格式是 raw | 尾部 Reshape 维度 = 64 + num_class | seg还需检查out1/out2 |
| strings.xml 有新 task | grep task_name strings.xml | 相同 |
| yolov8.h 有新类声明 | grep class_name yolov8.h | det→YOLOv8_det_xxx / seg→YOLOv8_seg_xxx |
| draw.cpp 类名/类别数正确 | 源码检查 | seg需额外检查mask绘制逻辑 |
| CMakeLists.txt 有新源文件 | grep filename CMakeLists.txt | det→det_xxx / seg→seg_xxx |
| yolov8ncnn.cpp 5处全改 | 逐行 grep 验证 | seg实例化类名不同 |

**seg模型额外审查项**：
| 检查项 | 说明 |
|--------|------|
| param有3个输出blob | out0(检测) + out1(掩码系数) + out2(原型) |
| 原out1已重命名为out2 | 否则mask原型位置错误 |
| 原out0已重命名为deadout | 否则blob名冲突 |
| 掩码系数Reshape维度=32 | 固定值，YOLOv8-seg标准 |

### Step 7: 构建验证

提示用户在 Android Studio 中 Sync + Build + 安装测试，选择新模型运行。

## 常见陷阱

1. **输出格式不兼容**（最常见的闪退原因）— 见 Step 2
2. **variant 命名错误** — YOLOv8n 训练的模型 variant 是 "n" 不是 "m"
3. **.bin 文件对齐破坏** — 修改 param 时删层会导致 .bin 读取错位，必须保留所有原始层
4. **taskid 范围遗漏** — 只改了 tasknames 数组忘记改范围检查条件
5. **类别数不一致** — param 输出的 num_class 和 draw() 中的 class_names 数量必须一致
6. **seg模型继承错误** — 分割模型必须继承 YOLOv8_seg(不是YOLOv8_det)，否则detect()方法不对(3输出blob)
7. **seg模型param修复遗漏** — 忘记把原out1→out2(掩码原型)，只修了检测头导致out1位置错
8. **seg模型缺少mask绘制** — draw()只有矩形+标签，缺少mask半透明叠加逻辑
9. **mask系数blob定位错误** — seg模型的mask系数conv输出(32ch)和class conv输出(num_class ch)容易混淆
