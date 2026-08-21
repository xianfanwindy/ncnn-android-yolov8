#!/usr/bin/env python3
"""
ncnn param 输出格式修复工具

将内嵌后处理(decoded格式)的ncnn param文件修改为raw DFL输出格式,
兼容 ncnn-android-yolov8 项目的 YOLOv8_det::detect() 和 YOLOv8_seg::detect() 推理逻辑。

支持两种任务类型:
  - 检测模型 (det): 1个输出blob (out0)
  - 分割模型 (seg): 3个输出blob (out0/out1/out2)

用法:
    # 检测模型
    python3 fix_ncnn_param_output.py <input.param> <output.param> \
        --task-type det \
        --bbox-blobs <s8_bbox> <s16_bbox> <s32_bbox> \
        --cls-blobs <s8_cls> <s16_cls> <s32_cls> \
        --num-class <N>

    # 分割模型
    python3 fix_ncnn_param_output.py <input.param> <output.param> \
        --task-type seg \
        --bbox-blobs <s8_bbox> <s16_bbox> <s32_bbox> \
        --cls-blobs <s8_cls> <s16_cls> <s32_cls> \
        --mask-blobs <s8_mask> <s16_mask> <s32_mask> \
        --num-class <N> \
        --mask-dim 32

示例(bolt检测模型):
    python3 fix_ncnn_param_output.py model.ncnn.param yolov8n_bolt.ncnn.param \
        --task-type det \
        --bbox-blobs 184 190 196 \
        --cls-blobs 203 209 215 \
        --num-class 4

示例(crack_seg分割模型):
    python3 fix_ncnn_param_output.py model.ncnn.param yolov8n_crack_seg.ncnn.param \
        --task-type seg \
        --bbox-blobs 188 194 200 \
        --cls-blobs 207 213 219 \
        --mask-blobs 226 232 238 \
        --num-class 1 \
        --mask-dim 32
"""

import argparse
import sys


def parse_layer_line(line):
    """解析ncnn param层行，提取类型、名称、输入输出blob数和blob名列表"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    layer_type = parts[0]
    layer_name = parts[1]
    n_inputs = int(parts[2])
    n_outputs = int(parts[3])
    input_blobs = parts[4:4 + n_inputs]
    output_blobs = parts[4 + n_inputs:4 + n_inputs + n_outputs]
    params = parts[4 + n_inputs + n_outputs:]

    return {
        'type': layer_type,
        'name': layer_name,
        'n_inputs': n_inputs,
        'n_outputs': n_outputs,
        'input_blobs': input_blobs,
        'output_blobs': output_blobs,
        'params': params,
        'raw_line': line
    }


def count_blobs(all_layer_lines):
    """计算所有层涉及的唯一blob名称数"""
    blob_set = set()
    for line in all_layer_lines:
        parsed = parse_layer_line(line)
        if parsed is None:
            continue
        for b in parsed['input_blobs'] + parsed['output_blobs']:
            blob_set.add(b)
    return len(blob_set)


def fix_param(input_path, output_path, task_type, bbox_blobs, cls_blobs,
              mask_blobs, num_class, mask_dim, task_name):
    """
    修改param文件：
    - det模型: 保留所有原始层，重命名out0为deadout，添加raw检测输出层(10层)
    - seg模型: 保留所有原始层，重命名out0→deadout + out1→out2，添加raw检测+掩码输出层(17层)
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    magic = lines[0].strip()
    # 第2行是 layer_count blob_count

    # 处理所有层行
    modified_layers = []
    for line in lines[2:]:
        # 重命名原始 out0 → deadout
        if ' out0 ' in line or line.strip().endswith(' out0'):
            line = line.replace(' out0 ', ' deadout ')
            line = line.replace(' out0\n', ' deadout\n')
            line = line.replace(' out0 ', ' deadout ')

        # seg模型: 重命名原始 out1 → out2
        if task_type == 'seg':
            if ' out1 ' in line or line.strip().endswith(' out1'):
                line = line.replace(' out1 ', ' out2 ')
                line = line.replace(' out1\n', ' out2\n')
                line = line.replace(' out1 ', ' out2 ')

        modified_layers.append(line)

    # 计算 raw 输出每行的列数
    dfl_columns = 64  # reg_max=16 × 4个方向
    raw_columns = dfl_columns + num_class

    # 生成新增的输出层
    # tmp blob 编号从 300 开始（避免与原始 blob 冲突）
    new_layers = []
    strides = [8, 16, 32]

    # === 检测头raw输出层 (10层，det和seg都需要) ===
    stride_det_outputs = []

    for i, stride in enumerate(strides):
        bbox_blob = bbox_blobs[i]
        cls_blob = cls_blobs[i]
        tmp_concat = f"300_{stride}"
        tmp_reshape = f"301_{stride}"
        tmp_permute = f"302_{stride}"

        new_layers.append(
            f"Concat                   cat_{task_name}_s{stride}"
            f"              2 1 {bbox_blob} {cls_blob} {tmp_concat} 0=0\n"
        )
        new_layers.append(
            f"Reshape                  view_{task_name}_s{stride}"
            f"             1 1 {tmp_concat} {tmp_reshape} 0=-1 1={raw_columns}\n"
        )
        new_layers.append(
            f"Permute                  trans_{task_name}_s{stride}"
            f"            1 1 {tmp_reshape} {tmp_permute} 0=1\n"
        )
        stride_det_outputs.append(tmp_permute)

    # 检测头最终 Concat → out0
    new_layers.append(
        f"Concat                   cat_{task_name}_det"
        f"             {len(strides)} 1"
        f" {' '.join(stride_det_outputs)} out0 0=0\n"
    )

    # === 掩码系数raw输出层 (7层，仅seg模型) ===
    if task_type == 'seg' and mask_blobs:
        stride_mask_outputs = []

        for i, stride in enumerate(strides):
            mask_blob = mask_blobs[i]
            tmp_reshape_m = f"309_{stride}"
            tmp_permute_m = f"310_{stride}"

            new_layers.append(
                f"Reshape                  view_{task_name}_mask_s{stride}"
                f"      1 1 {mask_blob} {tmp_reshape_m} 0=-1 1={mask_dim}\n"
            )
            new_layers.append(
                f"Permute                  trans_{task_name}_mask_s{stride}"
                f"     1 1 {tmp_reshape_m} {tmp_permute_m} 0=1\n"
            )
            stride_mask_outputs.append(tmp_permute_m)

        # 掩码系数最终 Concat → out1
        new_layers.append(
            f"Concat                   cat_{task_name}_mask"
            f"            {len(strides)} 1"
            f" {' '.join(stride_mask_outputs)} out1 0=0\n"
        )

    all_layer_lines = modified_layers + new_layers

    # 计算新的 layer_count 和 blob_count
    layer_count = len(all_layer_lines)
    blob_count = count_blobs(all_layer_lines)

    # 写入输出文件
    with open(output_path, 'w') as f:
        f.write(magic + '\n')
        f.write(f'{layer_count} {blob_count}\n')
        for line in all_layer_lines:
            f.write(line)

    added_count = len(new_layers)
    print(f"✅ Param file fixed: {output_path}")
    print(f"   Task type: {task_type}")
    print(f"   Original layers: {len(modified_layers)}, Added: {added_count}, Total: {layer_count}")
    print(f"   Blob count: {blob_count}")
    if task_type == 'det':
        print(f"   Raw output: out0 (8400×{raw_columns} = {dfl_columns} DFL + {num_class} class)")
    else:
        print(f"   Raw output: out0 (8400×{raw_columns} = {dfl_columns} DFL + {num_class} class)")
        print(f"   Mask coeff: out1 (8400×{mask_dim} mask coefficients)")
        print(f"   Mask proto: out2 (mask prototypes, renamed from original out1)")
    print(f"   Original out0 → deadout (preserved for .bin alignment)")
    if task_type == 'seg':
        print(f"   Original out1 → out2 (mask prototype position aligned)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fix ncnn param output format: decoded → raw DFL (supports det and seg models)'
    )
    parser.add_argument('input_param', help='Input .param file path')
    parser.add_argument('output_param', help='Output .param file path')
    parser.add_argument('--task-type', choices=['det', 'seg'], required=True,
                        help='Task type: det (detection, 1 output blob) or seg (segmentation, 3 output blobs)')
    parser.add_argument('--bbox-blobs', nargs=3, required=True,
                        help='Blob IDs for stride 8/16/32 bbox DFL conv outputs')
    parser.add_argument('--cls-blobs', nargs=3, required=True,
                        help='Blob IDs for stride 8/16/32 class score conv outputs')
    parser.add_argument('--mask-blobs', nargs=3, default=None,
                        help='Blob IDs for stride 8/16/32 mask coefficient conv outputs (required for seg)')
    parser.add_argument('--num-class', type=int, required=True,
                        help='Number of detection classes')
    parser.add_argument('--mask-dim', type=int, default=32,
                        help='Mask coefficient dimension (default: 32, YOLOv8-seg standard)')
    parser.add_argument('--task-name', default='custom',
                        help='Task name prefix for new layers (default: custom)')

    args = parser.parse_args()

    # seg模型必须提供mask-blobs
    if args.task_type == 'seg' and args.mask_blobs is None:
        parser.error("--mask-blobs is required for seg task type")

    success = fix_param(
        args.input_param, args.output_param,
        args.task_type,
        args.bbox_blobs, args.cls_blobs,
        args.mask_blobs,
        args.num_class, args.mask_dim, args.task_name
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
