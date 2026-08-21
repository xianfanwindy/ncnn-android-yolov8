# ncnn param 输出格式修复参考

## 问题背景

项目代码 `generate_proposals()` 期望模型输出 **raw DFL 格式**：

```
每行 = DFL bbox (64列) + raw class scores (num_class列)
总列数 = 64 + num_class
```

计算方式: `num_class = pred.w - reg_max_1 * 4`（reg_max_1=16，即 `pred.w - 64`）

但某些从 Ultralytics 直接导出的 ncnn 模型会内嵌完整后处理，输出 **decoded 格式**：

```
每行 = decoded bbox (4列) + sigmoid class scores (num_class列)
总列数 = 4 + num_class
```

此时 `num_class = pred.w - 64 = (4+num_class) - 64 < 0` → 内存越界 → **闪退！**

## 判断方法

读取 param 文件，搜索以下特征：

### Raw 格式特征（正确，无需修改）
- 最终输出层的 Reshape 维度 = `64 + num_class`（如 68 = 4类、144 = 80类）
- 输出路径中**没有** Sigmoid 层
- 输出路径中**没有** bbox 的 Softmax 层
- 输出路径中**没有** anchor 解码层（距离乘stride的算术运算）

### Decoded 格式特征（需修改）
- 最终输出层的 Reshape 维度 = `4 + num_class`（如 8 = 4类、84 = 80类）
- 输出路径中包含 Sigmoid 层
- 输出路径中包含 bbox 的 Softmax 层
- 输出路径中有 anchor 解码运算

---

## 检测模型 (det) 的修复方法

### 核心原则

**必须保留所有原始层** — ncnn .bin 文件按 param 文件中层的顺序存储权重数据。删除任何层都会导致后续层的权重读取偏移错位，造成推理结果错误或崩溃。

### 修复步骤

1. **保留所有原始层不变**

2. **重命名原始 out0 为 deadout**
   - 找到产生原始 `out0` 的最后一个 Concat/cat 层
   - 将其输出 blob 名从 `out0` 改为 `deadout`
   - 该层的计算仍然执行（保证 .bin 对齐），但输出不再被代码读取

3. **定位检测头原始 conv 输出 blob**
   - 在 param 文件中搜索每个 stride 的 bbox DFL conv 输出和 class score conv 输出
   - bbox conv 输出通道数 = 64（DFL reg_max=16 × 4个方向）
   - class conv 输出通道数 = num_class
   - 通常结构：
     ```
     stride 8:  conv_bbox_s8 (64ch, blob_id_A) + conv_cls_s8 (num_class_ch, blob_id_B)
     stride 16: conv_bbox_s16 (64ch, blob_id_C) + conv_cls_s16 (num_class_ch, blob_id_D)
     stride 32: conv_bbox_s32 (64ch, blob_id_E) + conv_cls_s32 (num_class_ch, blob_id_F)
     ```

4. **在 param 文件末尾添加新输出层**

   每个 stride 3 层 (Concat + Reshape + Permute) + 最终 1 层 (Concat) = 共 10 层：

   ```
   # stride 8: Concat(bbox_DFL + class) → Reshape(68) → Permute(行列互换)
   Concat      cat_xxx_s8     2 1 {bbox_blob_s8} {cls_blob_s8} {tmp1} 0=0
   Reshape     view_xxx_s8    1 1 {tmp1} {tmp2} 0=-1 1=68
   Permute     trans_xxx_s8   1 1 {tmp2} {tmp3} 0=1

   # stride 16: 同上
   Concat      cat_xxx_s16    2 1 {bbox_blob_s16} {cls_blob_s16} {tmp4} 0=0
   Reshape     view_xxx_s16   1 1 {tmp4} {tmp5} 0=-1 1=68
   Permute     trans_xxx_s16  1 1 {tmp5} {tmp6} 0=1

   # stride 32: 同上
   Concat      cat_xxx_s32    2 1 {bbox_blob_s32} {cls_blob_s32} {tmp7} 0=0
   Reshape     view_xxx_s32   1 1 {tmp7} {tmp8} 0=-1 1=68
   Permute     trans_xxx_s32  1 1 {tmp8} {tmp9} 0=1

   # 最终: Concat all strides → out0
   Concat      cat_xxx_out    3 1 {tmp3} {tmp6} {tmp9} out0 0=0
   ```

   **关键参数说明**：
   - Reshape `1=68` 中的 68 = 64(DFL) + 4(num_class)，根据实际类别数调整
   - Permute `0=1` 表示交换维度0和1（行列互换），将 (1, 68, N) 变为 (N, 68)
   - Concat `0=0` 表示沿维度0（通道维）拼接

5. **更新 param 头部计数**

   param 文件第2行格式: `{layer_count} {blob_count}`
   - layer_count = 原始层数 + 10（新增层数）
   - blob_count = 所有层涉及的唯一 blob 名称总数（需精确计算）

### 计算 blob 计数的方法

用 Python 脚本精确计算：

```python
blob_set = set()
for line in all_layer_lines:  # 包括原始层 + 新增层
    parts = line.strip().split()
    if len(parts) < 5:
        continue
    n_inputs = int(parts[2])
    n_outputs = int(parts[3])
    input_blobs = parts[4:4+n_inputs]
    output_blobs = parts[4+n_inputs:4+n_inputs+n_outputs]
    for b in input_blobs + output_blobs:
        blob_set.add(b)
blob_count = len(blob_set)
```

### .bin 文件处理

**.bin 文件不需要任何修改**。ncnn 加载 .bin 时按 param 中的层顺序逐层读取权重，保留所有原始层确保偏移对齐。新增的 Concat/Reshape/Permute 层没有权重数据，不会影响 .bin 读取。

---

## 分割模型 (seg) 的修复方法

**⚠️ seg模型比det模型更复杂！有3个输出blob，需要同时修复检测头和掩码系数头。**

### 代码期望的3个输出blob

项目代码 `YOLOv8_seg::detect()` 期望模型输出 3 个 blob：

| 输出 | 形状 | 含义 | 代码处理 |
|------|------|------|---------|
| `out0` | (8400, 64+num_class) | 检测头raw DFL格式 | generate_proposals() → NMS |
| `out1` | (8400, 32) | mask系数(每检测框32个系数) | Gemm(mask_feat × mask_protos) → mask计算 |
| `out2` | (32, 160, 160) | mask原型矩阵 | reshape → Gemm输入 |

### Ultralytics 导出的 decoded 格式 seg 模型

原始 decoded seg 模型有 2 个输出 blob：
- 原始 `out0`：把检测(decoded bbox + sigmoid class)和掩码系数混在一起经过完整后处理
- 原始 `out1`：mask原型 (32×160×160)

**注意**: decoded seg 模型的原始 out0 不仅内嵌了 DFL解码+anchor解码+sigmoid，还把掩码系数也卷在一起，所以不能简单地像det那样只修复检测头。

### 修复步骤（5步）

**Step 1: 重命名原始 out0 → deadout**

与det模型相同，找到产生原始 `out0` 的最后一个 Concat 层，将输出重命名为 `deadout`。

**Step 2: 重命名原始 out1 → out2**

原 `out1` 是 mask 原型，代码期望它在 `out2`。将产生原始 `out1` 的层的输出 blob 名从 `out1` 改为 `out2`。

⚠️ 这一步是seg模型独有的！det模型没有这一步。

**Step 3: 定位检测头 conv 输出 blob**

与det模型相同的方法，找到6个blob：
- stride 8/16/32 的 bbox DFL conv 输出 (64ch) — 每stride1个
- stride 8/16/32 的 class score conv 输出 (num_class ch) — 每stride1个

**Step 4: 定位掩码系数 conv 输出 blob**

⚠️ 这一步也是seg模型独有的！

在 param 文件中搜索每个 stride 的 mask coefficient conv 输出：
- mask coefficient conv 输出通道数 = 32
- 通常紧跟在检测头 conv 之后或穿插在一起
- 每个 stride 有独立的 mask coefficient conv，需要找到 stride 8/16/32 共 3 个 blob

定位技巧：
1. 在 decoded 输出路径中找到被 Reshape 为 `(N, 32)` 形状的 conv 层
2. 这些 conv 的 `0=32`（输出通道数32），且它们的输出 blob 在后续被 Reshape 为 `0=6400 1=32`（stride8）、`0=1600 1=32`（stride16）、`0=400 1=32`（stride32）
3. 注意：mask coefficient conv 输出是**原始conv的输出**（在decoded路径中这些输出经过了sigmoid等后处理），我们需要的是conv的原始raw输出（即sigmoid之前的blob名）

**Step 5: 在 param 文件末尾添加新输出层（17层）**

新增层数比det模型多7层（检测头10层 + 掩码系数7层 = 17层）：

```
# === 检测头: 与det模型相同的10层 ===
# stride 8: Concat(bbox + class) → Reshape(65) → Permute
Concat      cat_xxx_s8        2 1 {bbox_blob_s8} {cls_blob_s8} tmp1 0=0
Reshape     view_xxx_s8       1 1 tmp1 tmp2 0=-1 1=65
Permute     trans_xxx_s8      1 1 tmp2 tmp3 0=1

# stride 16: 同上
Concat      cat_xxx_s16       2 1 {bbox_blob_s16} {cls_blob_s16} tmp4 0=0
Reshape     view_xxx_s16      1 1 tmp4 tmp5 0=-1 1=65
Permute     trans_xxx_s16     1 1 tmp5 tmp6 0=1

# stride 32: 同上
Concat      cat_xxx_s32       2 1 {bbox_blob_s32} {cls_blob_s32} tmp7 0=0
Reshape     view_xxx_s32      1 1 tmp7 tmp8 0=-1 1=65
Permute     trans_xxx_s32     1 1 tmp8 tmp9 0=1

# 检测头最终 Concat → out0
Concat      cat_xxx_det       3 1 tmp3 tmp6 tmp9 out0 0=0

# === 掩码系数头: seg模型独有的7层 ===
# stride 8: Reshape(32) → Permute
Reshape     view_xxx_mask_s8  1 1 {mask_blob_s8} tmp10 0=-1 1=32
Permute     trans_xxx_mask_s8 1 1 tmp10 tmp11 0=1

# stride 16: 同上
Reshape     view_xxx_mask_s16 1 1 {mask_blob_s16} tmp12 0=-1 1=32
Permute     trans_xxx_mask_s16 1 1 tmp12 tmp13 0=1

# stride 32: 同上
Reshape     view_xxx_mask_s32 1 1 {mask_blob_s32} tmp14 0=-1 1=32
Permute     trans_xxx_mask_s32 1 1 tmp14 tmp15 0=1

# 掩码系数最终 Concat → out1
Concat      cat_xxx_mask      3 1 tmp11 tmp13 tmp15 out1 0=0
```

**关键参数说明**：
- 检测头 Reshape `1=65` 中的 65 = 64(DFL) + 1(num_class)，根据实际类别数调整
- 掩码系数 Reshape `1=32` 固定为32（YOLOv8-seg的mask系数维度固定为32）
- 掩码系数不需要 Concat(bbox+class)，因为mask系数只有单列数据，直接 Reshape+Permute

### 修复后param结构示意

```
原始模型结构:
  backbone → neck → [decoded det+mask head] → out0(decoded混合输出)
                       └→ [mask proto head] → out1(mask原型)

修复后结构:
  backbone → neck → [decoded det+mask head] → deadout (保留，.bin对齐)
                       └→ [mask proto head] → out2 (原型，位置不变只是改名)
  + 检测头raw输出层 → out0 (8400×65 raw DFL格式)
  + 掩码系数raw输出层 → out1 (8400×32 mask系数)
```

---

## 定位检测头 conv 输出 blob 的技巧

在 param 文件中搜索以下模式：

1. **bbox DFL conv 输出**：搜索 `Convolution` 层，其 `0=64`（输出通道64），且输出 blob 在后续被 reshape 为 `0=6400 1=64`（stride8）、`0=1600 1=64`（stride16）、`0=400 1=64`（stride32）
2. **class score conv 输出**：搜索 `0=num_class` 的 Convolution 层，如 `0=4`(bolt)、`0=6`(weld)、`0=1`(crack_seg)
3. **更直接的方法**：在原始 decoded 输出路径中逆向追踪：
   - 找到 Sigmoid 层的输入 → 这是 class score conv 的输出
   - 找到 Softmax(bbox方向) 层的输入 → 这是 bbox DFL conv 经过 Reshape 后的中间 blob
   - 逆向找到 Reshape 层的输入 → 这是 bbox DFL conv 的直接输出

## 定位掩码系数 conv 输出 blob 的技巧（仅seg模型）

1. **mask coefficient conv 输出**：搜索 `Convolution` 层，其 `0=32`（输出通道32），且输出 blob 在后续被 reshape 为 `0=6400 1=32`（stride8）、`0=1600 1=32`（stride16）、`0=400 1=32`（stride32）
2. **注意**：这些conv输出的blob名就是需要提取的raw mask系数，不需要经过任何后处理（sigmoid等是decoded路径中的层，我们跳过它们）

---

## 实际案例

### 案例1: bolt 检测模型修复 (4类)

原始 param: 202层, 242 blobs
- 输入 blob: `in0`
- 原始输出: `cat_16` → `out0` (decoded 8列格式)
- 修复: `cat_16` 输出重命名为 `deadout`
- 检测头 conv 输出:
  - stride 8 bbox: blob 184, stride 8 class: blob 203
  - stride 16 bbox: blob 190, stride 16 class: blob 209
  - stride 32 bbox: blob 196, stride 32 class: blob 215

修复后 param: 212层
新增 10 层（检测头raw输出）：
```
Concat      cat_bolt_s8     2 1 184 203 300 0=0
Reshape     view_bolt_s8    1 1 300 301 0=-1 1=68
Permute     trans_bolt_s8   1 1 301 302 0=1
Concat      cat_bolt_s16    2 1 190 209 303 0=0
Reshape     view_bolt_s16   1 1 303 304 0=-1 1=68
Permute     trans_bolt_s16  1 1 304 305 0=1
Concat      cat_bolt_s32    2 1 196 215 306 0=0
Reshape     view_bolt_s32   1 1 306 307 0=-1 1=68
Permute     trans_bolt_s32  1 1 307 308 0=1
Concat      cat_bolt_out    3 1 302 305 308 out0 0=0
```

### 案例2: weld 检测模型修复 (6类)

与bolt相同方法，区别在于：
- class conv 输出通道 = 6（不是4）
- Reshape `1=70`（64+6=70，不是68）
- 检测头 conv 输出 blob 编号不同

新增 10 层（检测头raw输出），参数 `1=70`。

### 案例3: crack_seg 分割模型修复 (1类) ⭐ 最复杂

原始 param: 229层, 283 blobs (decoded格式，2输出blob: out0混合 + out1原型)

修复后 param: 246层, 290 blobs (raw格式，3输出blob: out0检测 + out1掩码系数 + out2原型)

**Step 1**: 原 `cat_18` 输出 `out0` → `deadout`
**Step 2**: 原 `silu_147` 输出 `out1` → `out2`（mask原型改名）
**Step 3**: 检测头 conv 输出:
  - stride 8 bbox: blob 188, stride 8 class: blob 207
  - stride 16 bbox: blob 194, stride 16 class: blob 213
  - stride 32 bbox: blob 200, stride 32 class: blob 219

**Step 4**: 掩码系数 conv 输出:
  - stride 8 mask: blob 226
  - stride 16 mask: blob 232
  - stride 32 mask: blob 238

新增 17 层（10检测头 + 7掩码系数头）：
```
# 检测头raw输出 (10层)
Concat      cat_crack_s8            2 1 188 207 300 0=0
Reshape     view_crack_s8           1 1 300 301 0=-1 1=65
Permute     trans_crack_s8          1 1 301 302 0=1
Concat      cat_crack_s16           2 1 194 213 303 0=0
Reshape     view_crack_s16          1 1 303 304 0=-1 1=65
Permute     trans_crack_s16         1 1 304 305 0=1
Concat      cat_crack_s32           2 1 200 219 306 0=0
Reshape     view_crack_s32          1 1 306 307 0=-1 1=65
Permute     trans_crack_s32         1 1 307 308 0=1
Concat      cat_crack_det           3 1 302 305 308 out0 0=0

# 掩码系数raw输出 (7层)
Reshape     view_crack_mask_s8      1 1 226 309 0=-1 1=32
Permute     trans_crack_mask_s8     1 1 309 310 0=1
Reshape     view_crack_mask_s16     1 1 232 311 0=-1 1=32
Permute     trans_crack_mask_s16    1 1 311 312 0=1
Reshape     view_crack_mask_s32     1 1 238 313 0=-1 1=32
Permute     trans_crack_mask_s32    1 1 313 314 0=1
Concat      cat_crack_mask          3 1 310 312 314 out1 0=0
```

**关键点**：
- 检测头 Reshape `1=65` = 64(DFL) + 1(class)
- 掩码系数 Reshape `1=32` 固定（YOLOv8-seg mask系数维度）
- 掩码系数不需要 Concat(bbox+class)，直接 Reshape+Permute 即可
- 原out1→out2 的 rename 确保 mask 原型在正确位置
