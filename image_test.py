import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.nn.modules.block import *

# ==========================================
# 2. 设置路径
# ==========================================
# model_path = '/disk/duanww/ultralytics/runs/detect/train-BMDStemv5T4/weights/best.pt'
model_path = "/disk/duanww/ultralytics/runs/detect/train-BMDStemv51T4-snow3/weights/best.pt"
img_name = "000017.jpg"
# img_path = f'/disk/duanww/VOC_YOLO/images/train2007/{img_name}'
img_path = f"/disk/duanww/VOC_YOLO/images/snow_train/3/train2007/{img_name}"

# ==========================================
# 3. 加载模型并挂载 Hooks
# ==========================================
print(f"Loading model from {model_path}...")
try:
    # 加载 YOLO 模型
    model = YOLO(model_path)

    # 获取底层的 nn.Module
    # 注意：Ultralytics 模型通常包装在 model.model 中
    full_model = model.model
    print(full_model.info(detailed=True))

except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 寻找 BMDStemv4 模块
target_module = None
target_name = ""

print("Searching for BMDStemv4 module...")
for name, m in full_model.named_modules():
    # 这里的类名匹配很重要，有时加载后类名可能带有前缀
    if isinstance(m, BMDStemv4) or "BMDStem" in str(type(m)):
        target_module = m
        target_name = name
        # print(f"Found BMDStemv4 at layer: {name}")
        break

if target_module is None:
    print("Error: Could not find BMDStemv4 layer in the model.")
    # 打印一下模型结构帮助调试
    # print(full_model)
    exit()

# 定义 Hook 存储字典
activations = {}


def get_activation(name):
    def hook(model, input, output):
        if isinstance(output, tuple):
            activations[name + "_0"] = output[0].detach().cpu()
            activations[name + "_1"] = output[1].detach().cpu()
        else:
            activations[name] = output.detach().cpu()

    return hook


# 注册 Hooks
# 我们直接利用 target_module 的子模块属性名
target_module.stem.retina.register_forward_hook(get_activation("retina"))
target_module.stem.wavelet.register_forward_hook(get_activation("wavelet"))
target_module.stem.snn.register_forward_hook(get_activation("low_snn"))
target_module.stem.cleaner.register_forward_hook(get_activation("high_cleaner"))
target_module.stem.act.register_forward_hook(get_activation("fusion"))
target_module.stem.shortcut.register_forward_hook(get_activation("shortcut"))
target_module.stem.register_forward_hook(get_activation("stem_output"))


# # ==========================================

# import math

# # 执行前向传播 (这会触发上面注册的 Hooks)
# model.predict(img_path, save=False, conf=0.25)

# # ==========================================
# # 5. 可视化 SNN 输出 (按通道)
# # ==========================================

# if 'snn' in activations:
#     snn_output = activations['snn'] # 预期形状: [Batch=1, Channels, Height, Width]
#     print(f"SNN Output Shape: {snn_output.shape}")

#     # 移除 Batch 维度 -> (Channels, H, W)
#     snn_map = snn_output.squeeze(0)

#     num_channels = snn_map.shape[0]
#     height = snn_map.shape[1]
#     width = snn_map.shape[2]

#     print(f"Visualizing {num_channels} channels...")

#     # 计算网格布局 (例如每行显示 8 个通道)
#     cols = 8
#     rows = math.ceil(num_channels / cols)

#     # 创建画布
#     fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
#     fig.suptitle(f'SNN Layer Output per Channel ({img_name})', fontsize=16)

#     # 展平 axes 方便遍历，处理只有一个通道的特殊情况
#     if num_channels == 1:
#         axes = [axes]
#     else:
#         axes = axes.flatten()

#     for i in range(len(axes)):
#         ax = axes[i]
#         if i < num_channels:
#             # 获取第 i 个通道的数据
#             channel_data = snn_map[i, :, :].numpy()

#             # 绘图：因为是 SNN 的 mean 输出，代表发放率，推荐使用 viridis 或 plasma
#             im = ax.imshow(channel_data, cmap='viridis', aspect='auto')

#             # 如果你想看反色的效果（类似纸张打印），可以用 cmap='Greys'

#             ax.set_title(f'Ch {i}', fontsize=8)
#             ax.axis('off') # 关闭坐标轴让画面更干净
#         else:
#             # 隐藏多余的子图
#             ax.axis('off')

#     # save_path = './test_image/change_fusion_snnlayer0_snow3.png'
#     save_path = './test_image/change_fusion_snnlayer0.png'
#     # plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"Visualization saved to {save_path}")
#     plt.tight_layout()
#     plt.show()

#     # 提示：如果只想看某一个特定通道，比如第0个
#     # plt.figure()
#     # plt.imshow(snn_map[0], cmap='viridis')
#     # plt.title("Channel 0 Detail")
#     # plt.colorbar()
#     # plt.show()

# else:
#     print("Error: Hook did not capture 'snn' output.")
######################################################################################

######################################################################################
# ================= 3. 推理 =================
# print(f"Processing image {img_path}...")
# model.predict(img_path, save=False, conf=0.25)

# # ================= 4. 处理数据 =================
# # 获取特征图: [Batch, Channel, Height, Width]
# # YOLOv8n 第一层输出通常是 [1, 16, H/2, W/2]
# feature_map = activations['stem_output'][0]

# # 【核心步骤】计算所有通道的平均值，合成一张“总图”
# # 你也可以尝试 max(dim=0) 查看最强响应，但 mean 更能代表整体信息量
# combined_map = feature_map.mean(dim=0)

# # ================= 5. 画图对比 =================
# img_raw = cv2.imread(img_path)
# img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(12, 6))
# plt.suptitle(f"Standard YOLOv8 Layer 0 Output (Aggregated)", fontsize=16)

# # 左边：原图
# plt.subplot(1, 2, 1)
# plt.imshow(img_raw)
# plt.title("Original Image")
# plt.axis('off')

# # 右边：第一层卷积的总特征图
# plt.subplot(1, 2, 2)
# plt.imshow(combined_map, cmap='viridis') # 使用 viridis 或 magma 配色
# plt.title(f"Standard Conv0 Output (Mean of {feature_map.shape[0]} Channels)\nShape: {combined_map.shape}")
# plt.axis('off')

# # 保存
# # save_path = './test_image/v4_yolo_layer0_snow3.png'
# save_path = './test_image/change_fusion_yolo_layer0.png'
# plt.tight_layout()
# plt.savefig(save_path)
# print(f"Visualization saved to {save_path}")
# plt.show()
########################################################################################

########################################################################################
# 使用 YOLO 的 predict 方法，它会自动处理预处理（resize, normalize等）
# 我们只需要运行一次推理，Hooks 就会自动抓取数据
results = model.predict(img_path, save=False, conf=0.25)

# 获取输入图像（经过预处理后的），以便对比
# YOLO 在 predict 时不直接返回 Tensor，我们需要手动读取一下原图用于展示
img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

# 开始画图
plt.figure(figsize=(24, 14))
plt.suptitle(f"Analysis of BMDStemv4 (Model: {model_path.split('/')[-1]})", fontsize=16)

# 1. 原始图片
plt.subplot(3, 4, 1)
plt.imshow(img_raw)
plt.title("Original Image")
plt.axis("off")

# # 2. Retina 输出 (Mid Channels)
# if 'retina' in activations:
#     feat = activations['retina'] # [B, C, H, W]
#     # 取前 4 个通道的平均值，查看结构
#     viz_retina = feat[0].mean(dim=0)
#     plt.subplot(3, 4, 2)
#     plt.imshow(viz_retina, cmap='viridis')
#     plt.title(f"Retina Output\nShape: {feat.shape}")
#     plt.axis('off')

# 3. Wavelet Low Frequency (LL)
if "wavelet_0" in activations:
    feat = activations["wavelet_0"]  # [B, C, H, W]
    viz_ll = feat[0].mean(dim=0)
    plt.subplot(3, 4, 3)
    plt.imshow(viz_ll, cmap="viridis")
    plt.title(f"Wavelet LL (Low Freq)\nShape: {feat.shape}")
    plt.axis("off")

# 4. Wavelet High Frequency (Before SNN)
if "wavelet_1" in activations:
    feat = activations["wavelet_1"]  # [B, 3, C, H, W]
    # 将 3 个方向 (H, V, D) 的高频能量叠加显示
    # feat[0] shape is [3, C, H, W]
    # 先对 C 维度平均，再对 3 个方向求和
    viz_high = feat[0].abs().mean(dim=1).sum(dim=0)
    plt.subplot(3, 4, 4)
    plt.imshow(viz_high, cmap="viridis")
    plt.title(f"Wavelet High Freq\nShape: {feat.shape}")
    plt.axis("off")

# 5. SNN Output (After SNN Cleaning)
# if 'snn' in activations:
if "low_snn" in activations:
    feat = activations["low_snn"]  # [B, C, H, W]
    viz_snn = feat[0].mean(dim=0)
    plt.subplot(3, 4, 5)
    plt.imshow(viz_snn, cmap="viridis")
    plt.title(f"SNN Output\nShape: {feat.shape}")
    plt.axis("off")

    # 6. fusion
if "fusion" in activations:
    feat = activations["fusion"]
    viz_fusion = feat[0].mean(dim=0)
    plt.subplot(3, 4, 6)
    # 简单的可视化对比，注意维度可能不完全对应（SNN有1x1 conv），看纹理差异
    plt.imshow(viz_fusion, cmap="viridis")
    plt.title("fusion")
    plt.axis("off")

# 7. SNN Histogram (检查稀疏性)
if "snn" in activations:
    feat = activations["snn"].numpy().flatten()
    plt.subplot(3, 4, 7)
    plt.hist(feat, bins=50, log=True, color="purple")
    plt.title("SNN Output Histogram (Log Scale)")
    plt.xlabel("Activation Value")

# 8. Shortcut Output
if "shortcut" in activations:
    feat = activations["shortcut"]
    viz_short = feat[0].mean(dim=0)
    plt.subplot(3, 4, 8)
    plt.imshow(viz_short, cmap="viridis")
    plt.title("Shortcut Branch")
    plt.axis("off")

# 9. Final Stem Output
if "stem_output" in activations:
    feat = activations["stem_output"]
    viz_out = feat[0].mean(dim=0)
    plt.subplot(3, 4, 9)
    plt.imshow(viz_out, cmap="viridis")
    plt.title(f"Final Stem Output\nShape: {feat.shape}")
    plt.axis("off")

if "high_cleaner" in activations:
    feat = activations["high_cleaner"]  # [B, C, H, W]
    viz_snn = feat[0].mean(dim=0)
    plt.subplot(3, 4, 10)
    plt.imshow(viz_snn, cmap="viridis")
    plt.title(f"high_cleaner use Dilated Convolution\nShape: {feat.shape}")
    plt.axis("off")

# 保存结果
save_dir = "./test_image/"
# save_path = save_dir +f"change_fusion_analysis_{img_name.split('.')[0]}.png"
save_path = save_dir + f"bmdstemv51_analysis_{img_name.split('.')[0]}-snow3.png"
plt.tight_layout()
plt.savefig(save_path)
print(f"Visualization saved to {save_path}")
plt.show()
#########################################################################################


###########################################################################################
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from ultralytics import YOLO

# # ================= 1. 设置路径 =================
# # 图像路径 (保持和你之前的一致)
# img_name = '000017.jpg'
# # img_path = f'/disk/duanww/VOC_YOLO/images/train2007/{img_name}'
# img_path = f'/disk/duanww/VOC_YOLO/images/snow_train/3/train2007/{img_name}'
# # 使用标准模型进行对比 (会自动下载)
# # model_name = '/disk/duanww/ultralytics/runs/detect/train-baseline/weights/best.pt'
# model_name = '/disk/duanww/ultralytics/runs/detect/train-baseline-snow-3/weights/best.pt'

# # --- FFT 转换工具函数 ---
# def plot_fft_in_axis(ax, data_2d, title_prefix):
#     """
#     辅助函数：在指定的 ax 上画出 data_2d 的频谱
#     """
#     if data_2d is None:
#         ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
#         ax.axis('off')
#         return

#     # 1. 转为 numpy
#     if hasattr(data_2d, 'cpu'):
#         img = data_2d.detach().cpu().numpy()
#     else:
#         img = data_2d.copy()

#     # 2. 如果是 RGB (H, W, 3) 转灰度
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # 3. FFT 变换
#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     # 4. 幅度谱 (Log Scale)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

#     # 5. 画图 (使用 inferno 色谱，越亮代表能量越高)
#     ax.imshow(magnitude_spectrum, cmap='magma')
#     ax.set_title(f"FFT: {title_prefix}", fontsize=10)
#     ax.axis('off')


# # ================= 2. 加载模型与 Hook =================
# print(f"Loading standard model: {model_name}...")
# model = YOLO(model_name)

# # YOLOv8 的第一层 (Stem)
# # 结构: Conv2d(3, 16, kernel_size=3, stride=2) -> BatchNorm -> SiLU
# target_layer = model.model.model[0]

# activations = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activations[name] = output.detach().cpu()
#     return hook

# target_layer.register_forward_hook(get_activation('conv0_out'))

# # ================= 3. 推理 =================
# print(f"Processing image {img_path}...")
# model.predict(img_path, save=False, conf=0.25)

# # ================= 4. 处理数据 =================
# # 获取特征图: [Batch, Channel, Height, Width]
# # YOLOv8n 第一层输出通常是 [1, 16, H/2, W/2]
# feature_map = activations['conv0_out'][0]

# # 【核心步骤】计算所有通道的平均值，合成一张“总图”
# # 你也可以尝试 max(dim=0) 查看最强响应，但 mean 更能代表整体信息量
# combined_map = feature_map.mean(dim=0)

# # ================= 5. 画图对比 =================
# img_raw = cv2.imread(img_path)
# img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(12, 6))
# plt.suptitle(f"Standard YOLOv8 Layer 0 Output (Aggregated)", fontsize=16)

# # 左边：原图
# ax1 = plt.subplot(1, 2, 1)
# plot_fft_in_axis(ax1, img_raw, "Original Image")


# ax2 = plt.subplot(1, 2, 2)
# plot_fft_in_axis(ax2, combined_map, "Standard Conv0 Output")

# # 保存
# # save_path = './test_image/standard_yolo_layer0_fft.png'
# save_path = './test_image/standard_yolo_layer0_fft_snow3.png'
# plt.tight_layout()
# plt.savefig(save_path)
# print(f"Visualization saved to {save_path}")
# plt.show()

# # ================= 4. 处理数据 =================
# # 获取特征图: [Batch, Channel, Height, Width]
# # YOLOv8n 第一层输出通常是 [1, 16, H/2, W/2]
# feature_map = activations['conv0_out'][0]

# # 【核心步骤】计算所有通道的平均值，合成一张“总图”
# # 你也可以尝试 max(dim=0) 查看最强响应，但 mean 更能代表整体信息量
# combined_map = feature_map.mean(dim=0)

# # ================= 5. 画图对比 =================
# img_raw = cv2.imread(img_path)
# img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(12, 6))
# plt.suptitle(f"Standard YOLOv8 Layer 0 Output (Aggregated)", fontsize=16)

# # 左边：原图
# plt.subplot(1, 2, 1)
# plt.imshow(img_raw)
# plt.title("Original Image")
# plt.axis('off')

# # 右边：第一层卷积的总特征图
# plt.subplot(1, 2, 2)
# plt.imshow(combined_map, cmap='viridis') # 使用 viridis 或 magma 配色
# plt.title(f"Standard Conv0 Output (Mean of {feature_map.shape[0]} Channels)\nShape: {combined_map.shape}")
# plt.axis('off')

# # 保存
# # save_path = './test_image/standard_yolo_layer0.png'
# save_path = './test_image/standard_yolo_layer0_snow3.png'
# plt.tight_layout()
# plt.savefig(save_path)
# print(f"Visualization saved to {save_path}")
# plt.show()

# #############################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# # --- FFT 转换工具函数 ---
# def plot_fft_in_axis(ax, data_2d, title_prefix):
#     """
#     辅助函数：在指定的 ax 上画出 data_2d 的频谱
#     """
#     if data_2d is None:
#         ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
#         ax.axis('off')
#         return

#     # 1. 转为 numpy
#     if hasattr(data_2d, 'cpu'):
#         img = data_2d.detach().cpu().numpy()
#     else:
#         img = data_2d.copy()

#     # 2. 如果是 RGB (H, W, 3) 转灰度
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # 3. FFT 变换
#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     # 4. 幅度谱 (Log Scale)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

#     # 5. 画图 (使用 inferno 色谱，越亮代表能量越高)
#     ax.imshow(magnitude_spectrum, cmap='magma')
#     ax.set_title(f"FFT: {title_prefix}", fontsize=10)
#     ax.axis('off')

# # --- 开始画对应的频域图 ---
# results = model.predict(img_path, save=False, conf=0.25)
# plt.figure(figsize=(24, 14))
# plt.suptitle(f"Frequency Domain Analysis (FFT Spectrum) - Corresponding to Spatial Layout", fontsize=16)

# # === 第 1 行 ===
# img_raw = cv2.imread(img_path)
# img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

# # 1. 对应 Original Image
# ax1 = plt.subplot(3, 4, 1)
# # img_raw 是 (H, W, 3)
# plot_fft_in_axis(ax1, img_raw, "Original Image")


# # 2. 对应 Retina Output

# if 'retina' in activations:
#     # 取均值变成 2D
#     data = activations['retina'][0].mean(dim=0)
#     ax2 = plt.subplot(3, 4, 2)
#     plot_fft_in_axis(ax2, data, "Retina Output")

# # 3. 对应 Wavelet LL
# ax3 = plt.subplot(3, 4, 3)
# if 'wavelet_0' in activations:
#     data = activations['wavelet_0'][0].mean(dim=0)
#     plot_fft_in_axis(ax3, data, "Wavelet LL")

# # 4. 对应 Wavelet High (Input to SNN)
# ax4 = plt.subplot(3, 4, 4)
# if 'wavelet_1' in activations:
#     # 逻辑同前：取绝对值，平均 Channel，叠加三个方向
#     data = activations['wavelet_1'][0].abs().mean(dim=1).sum(dim=0)
#     plot_fft_in_axis(ax4, data, "Wavelet High Freq")

# # === 第 2 行 ===

# # 5. 对应 SNN Output
# ax5 = plt.subplot(3, 4, 5)
# if 'snn' in activations:
#     data = activations['snn'][0].mean(dim=0)
#     plot_fft_in_axis(ax5, data, "SNN Output")

# # 6. 对应 Fusion
# ax6 = plt.subplot(3, 4, 6)
# if 'fusion' in activations:
#     data = activations['fusion'][0].mean(dim=0)
#     plot_fft_in_axis(ax6, data, "Fusion Output")

# # 7. 对应 Histogram (直方图无法做 2D FFT)
# ax7 = plt.subplot(3, 4, 7)
# ax7.axis('off')

# # 8. 对应 Shortcut
# ax8 = plt.subplot(3, 4, 8)
# if 'shortcut' in activations:
#     data = activations['shortcut'][0].mean(dim=0)
#     plot_fft_in_axis(ax8, data, "Shortcut Branch")

# # === 第 3 行 ===

# # 9. 对应 Final Stem Output
# ax9 = plt.subplot(3, 4, 9)
# if 'stem_output' in activations:
#     data = activations['stem_output'][0].mean(dim=0)
#     plot_fft_in_axis(ax9, data, "Final Stem Output")

# # 10. 留空 或 显示 SNN 差异 (可选)
# ax10 = plt.subplot(3, 4, 10)
# ax10.axis('off')

# # 11. 留空
# ax11 = plt.subplot(3, 4, 11)
# ax11.axis('off')

# # 12. 留空
# ax12 = plt.subplot(3, 4, 12)
# ax12.axis('off')

# # 保存
# save_dir = './test_image/'
# # save_path_fft = save_dir + f"bmdstemv4_analysis_{img_name.split('.')[0]}-ONLY_FFT.png"
# save_path_fft = save_dir + f"bmdstemv4_analysis_{img_name.split('.')[0]}-snow3_ONLY_FFT.png"
# plt.tight_layout()
# plt.savefig(save_path_fft)
# print(f"FFT Analysis saved to {save_path_fft}")
# plt.show()
