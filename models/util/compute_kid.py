from pathlib import Path
import numpy as np
import torch


from sklearn.metrics.pairwise import polynomial_kernel
from torchvision.models.inception import inception_v3
import torch.nn as nn
from PIL import Image
import os

# 加载Inception V3模型并设置为评估模式
model = inception_v3(pretrained=False)
model.load_state_dict(torch.load('./models/inception_v3_google-1a9a5a14.pth'))
model.eval()

# 获取Inception V3模型的倒数第二层（特征提取层）
inception_features = nn.Sequential(*list(model.children())[:-1])


# 图片
task = 'super_resolution'
seed = 'seed_0'

label_root = Path(f'./saved_results/{task}/{seed}/truth')
delta_recon_root = Path(f'./saved_results/{task}/{seed}/recon')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = img.convert("RGB")
            images.append(img)
    return images

# 加载真实图像和生成图像
real_images = load_images_from_folder(label_root)
generated_images = load_images_from_folder(delta_recon_root)

# 将图像转换为 NumPy 数组
real_images = np.array([np.array(img) for img in real_images])
generated_images = np.array([np.array(img) for img in generated_images])



# from sklearn.metrics.pairwise import polynomial_kernel
# from scipy.linalg import sqrtm

# def calculate_kid(real_images, generated_images):
#     # 加载Inception V3模型
#     inception = inception_v3(include_top=False, pooling='avg')
#     # 预处理图像
#     real_images = preprocess_input(real_images)
#     generated_images = preprocess_input(generated_images)
#     # 提取真实图像和生成图像的特征向量
#     real_features = inception.predict(real_images)
#     generated_features = inception.predict(generated_images)
#     # 计算特征向量之间的核矩阵
#     kernel_matrix = polynomial_kernel(real_features, generated_features)
#     # 计算核矩阵的Fréchet距离
#     kid = np.trace(kernel_matrix) + np.trace(real_features) + np.trace(generated_features) - 3 * np.trace(
#         sqrtm(kernel_matrix))
#     return kid

# # 示例用法
# kid = calculate_kid(real_images, generated_images)
# print("KID:", kid)


# def preprocess_input(images):
#     # 图像预处理
#     images = torch.tensor(images).float()
#     images = images.permute(0, 3, 1, 2)  # 调整通道顺序
#     images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
#     images = (images / 255.0 - 0.5) * 2.0  # 归一化到[-1, 1]范围

#     return images



























import numpy as np
from scipy import linalg

def calculate_features(images, model):
    # 图像预处理
    images = torch.tensor(images).float()
    images = images.permute(0, 3, 1, 2)  # 调整通道顺序
    images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    images = (images / 255.0 - 0.5) * 2.0  # 归一化到[-1, 1]范围

    # 提取特征
    with torch.no_grad():
        features = model(images)

    # 将特征拉平为2D向量
    features = torch.flatten(features, start_dim=1)

    # 转换为numpy数组
    features = features.numpy()

    return features

# 假设 real_images 和 generated_images 分别是真实图像和生成图像的数组
real_features = calculate_features(real_images, inception_features)
generated_features = calculate_features(generated_images, inception_features)


def calculate_kid(real_features, generated_features):
    # 计算真实图像和生成图像的协方差矩阵
    cov_real = np.cov(real_features, rowvar=False)
    cov_generated = np.cov(generated_features, rowvar=False)

    # 计算特征均值的差异
    mean_diff = np.mean(real_features, axis=0) - np.mean(generated_features, axis=0)

    # 计算KID指标
    kid_score = np.sum(mean_diff * mean_diff) + np.trace(cov_real + cov_generated - 2 * linalg.sqrtm(np.dot(cov_real, cov_generated)))

    return kid_score

# 计算KID指标
kid_score = calculate_kid(real_features, generated_features)

print(kid_score)















# device = 'cuda:0'
# loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)


# task = 'super_resolution'
# seed = 'seed_0'

# label_root = Path(f'./saved_results/{task}/{seed}/truth')
# delta_recon_root = Path(f'./saved_results/{task}/{seed}/recon')

# psnr_delta_list = []
# # psnr_normal_list = []

# lpips_delta_list = []
# # lpips_normal_list = []

# ssim_delta_list = []

# for idx in tqdm(range(1)):
#     fname = str(idx).zfill(5)

#     label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
#     delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
#     # normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]

#     psnr_delta = peak_signal_noise_ratio(label, delta_recon)
#     # psnr_normal = peak_signal_noise_ratio(label, normal_recon)

#     psnr_delta_list.append(psnr_delta)
#     # psnr_normal_list.append(psnr_normal)

#     delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
#     # normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
#     label = torch.from_numpy(label).permute(2, 0, 1).to(device)

#     gray_label = np.array(rgb_to_grayscale(label.cpu())).squeeze()
#     gray_delta_recon = np.array(rgb_to_grayscale(delta_recon.cpu())).squeeze()
#     # print(gray_label.shape)
#     ssim_d = ssim(gray_delta_recon, gray_label, data_range=255)
#     # print(ssim_d)
#     ssim_delta_list.append(ssim_d)

#     delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
#     # normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
#     label = label.view(1, 3, 256, 256) * 2. - 1.

#     delta_d = loss_fn_vgg(delta_recon, label)
#     # normal_d = loss_fn_vgg(normal_recon, label)

#     lpips_delta_list.append(delta_d)
#     # lpips_normal_list.append(normal_d)

# psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
# lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)
# ssim_delta_avg = sum(ssim_delta_list) / len(ssim_delta_list)

# # psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
# # lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)

# print(f'Delta PSNR: {psnr_delta_avg}')
# print(f'Delta LPIPS: {lpips_delta_avg}')
# print(f'Delta SSIM: {ssim_delta_avg}')

# # print(f'Normal PSNR: {psnr_normal_avg}')
# # print(f'Normal LPIPS: {lpips_normal_avg}')