import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import editdistance
import numpy as np
import torch
import numpy as np
from PIL import Image
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from tqdm import tqdm

    # dibco_256.add_argument('--num_block', default=[1, 2, 4, 8, 4, 2, 6, 4, 1], type=int, nargs='+', help='')

# def validation(model, criterion, evaluation_loader, is_save_img = False):
#     """ validation or evaluation """

#     valid_loss = 0.0
#     count = 0
#     total_pixels = 0
#     correct_pixels = 0
#     tp = 0  # True Positive
#     fp = 0  # False Positive
#     fn = 0  # False Negative
#     tn = 0  # True Negative

#     model.eval()  # 设置模型为评估模式
#     with torch.no_grad():  # 禁用梯度计算，减少内存占用
#         for i, (image_tensors, gt_image_tensors) in enumerate(evaluation_loader):
#             batch_size = image_tensors.size(0)
#             image = image_tensors.cuda()
#             gt_image = gt_image_tensors.cuda()

#             preds = model(image)
#             # preds = preds.float()

#             # 计算损失
#             cost = criterion(preds, gt_image).mean()
#             valid_loss += cost.item()
#             count += 1

#             # 将预测结果转换为二值图像
#             preds_binary = (preds > 0.5).float()  # 假设是二分类任务，使用 Sigmoid 激活并阈值化

#             # 将预测和真实标签从 GPU 移到 CPU 并转换为 NumPy 数组
#             preds_binary_np = preds_binary.cpu().numpy().flatten()
#             gt_image_np = gt_image.cpu().numpy().flatten()

#             # 更新总像素数和正确分类的像素数
#             total_pixels += len(gt_image_np)
#             correct_pixels += (preds_binary_np == gt_image_np).sum()

#             # 更新混淆矩阵
#             tp += ((preds_binary_np == 1) & (gt_image_np == 1)).sum()
#             fp += ((preds_binary_np == 1) & (gt_image_np == 0)).sum()
#             fn += ((preds_binary_np == 0) & (gt_image_np == 1)).sum()
#             tn += ((preds_binary_np == 0) & (gt_image_np == 0)).sum()
#             if is_save_img:
#                 pred_image = preds_binary_np[0, 0]  # 形状为 [H, W]
#                 # 将二值图像的值从 [0, 1] 转换为 [0, 255]
#                 pred_image = (pred_image * 255).astype(np.uint8)
#                 # 创建 PIL 图像对象并保存
#                 img = Image.fromarray(pred_image)
#                 img.save(f"output_{i}.png")


#     # 计算平均验证损失
#     val_loss = valid_loss / count

#     # 计算准确率（Accuracy）
#     accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0

#     # 计算精确率（Precision）
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0

#     # 计算召回率（Recall）
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#     # 计算像素级分类误差（Pixel-wise Classification Error）
#     pixel_error_rate = (total_pixels - correct_pixels) / total_pixels if total_pixels > 0 else 0

#     return val_loss, accuracy, precision, recall, pixel_error_rate


def validation(model, criterion, evaluation_loader):
    """ 包含FM/p-FM/PSNR/DRD指标的验证函数 """
    
    valid_loss = 0.0
    count = 0
    
    # 初始化指标累加器
    total_fm, total_pfm, total_psnr, total_drd = 0.0, 0.0, 0.0, 0.0

    
    model.eval()
    with torch.no_grad():
        for image_tensors, gt_image_tensors in evaluation_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.cuda()
            gt_image = gt_image_tensors.cuda()
            
            # 前向传播
            preds = model(image)
            # preds = torch.sigmoid(preds)
            
            # 计算损失
            loss = criterion(preds, gt_image)
            loss = loss.mean()
            valid_loss += loss.item()

            preds_binary = (preds > 0.5).float()

            preds_binary = 1.0 - preds_binary
            gt_image = 1.0 - gt_image
            
            # 转换为CPU numpy数组处理
            preds_np = preds_binary.cpu().numpy()[:, 0]  # 去除通道维度
            gt_np = gt_image.cpu().numpy()[:, 0]
            
            # 逐样本计算指标
            for i in range(batch_size):
                pred = preds_np[i]
                gt = gt_np[i]
 
                # === 计算FM ===
                # tp = np.sum((pred == 0) & (gt == 0))
                # fp = np.sum((pred == 0) & (gt == 1))
                # fn = np.sum((pred == 1) & (gt == 0))

                tp = np.sum((pred == 1) & (gt == 1))
                fp = np.sum((pred == 1) & (gt == 0))
                fn = np.sum((pred == 0) & (gt == 1))
                
                tp_plus_fn = tp + fn
                if tp_plus_fn == 0:
                    recall = 1.0  # 真实无正样本，Recall=1
                else:
                    recall = tp / tp_plus_fn
                # 计算Precision
                tp_plus_fp = tp + fp
                if tp_plus_fp == 0:
                    if tp_plus_fn == 0:
                        precision = 1.0  # 真实和预测均无正样本，Precision=1
                    else:
                        precision = 0.0  # 真实有正样本但预测全负，Precision=0
                else:
                    precision = tp / tp_plus_fp
                fm = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                total_fm += fm
                
                # === 计算PSNR ===
                pred_psnr = 1.0 - pred
                gt_psnr = 1.0 - gt

                mse = np.mean((pred_psnr - gt_psnr) ** 2)
                psnr = 10 * np.log10(1.0 / mse) if mse != 0 else 100  # 像素范围[0,1]
                total_psnr += psnr
                # psnr = psnr_metric(gt, pred, data_range=1.0) 
                
                # === 计算p-FM（结构相似性改进）===
                # 标记真实图像的连通区域
                gt_label = label(gt.astype(int), connectivity=1)
                num_gt_regions = np.max(gt_label)
                p_recall = 0.0
                
                if num_gt_regions > 0:
                    # 计算每个真实区域是否被预测覆盖
                    for region_id in range(1, num_gt_regions+1):
                        region_mask = (gt_label == region_id)
                        if np.any(pred[region_mask] == 1):
                            p_recall += 1
                    p_recall /= num_gt_regions
                else:
                    p_recall = 1.0  # 无真实区域时视为完美召回
                
                pfm_val = 2 * p_recall * precision / (p_recall + precision) if (p_recall + precision) > 0 else 0
                total_pfm += pfm_val
                
                # === 计算DRD ===
                def compute_drd(gt, pred):
                    """ 距离倒数失真度量计算 """
                    diff = (gt != pred).astype(np.uint8)
                    if np.sum(diff) == 0:
                        return 0.0
                    
                    # 计算正确像素的距离变换
                    correct_pixels = (gt == pred).astype(np.uint8)
                    distance_map = distance_transform_edt(1 - correct_pixels)
                    
                    # 计算失真区域加权值
                    drd_sum = np.sum(1.0 / (distance_map[diff == 1] + 1.0))  # +1避免除零
                    
                    # 计算归一化因子（8x8非重叠块中包含真实前景的块数）
                    h, w = gt.shape
                    block_size = 8
                    nubn = 0
                    for y in range(0, h, block_size):
                        for x in range(0, w, block_size):
                            block = gt[y:y+block_size, x:x+block_size]
                            if np.any(block == 1):
                                nubn += 1
                    nubn = max(nubn, 1)  # 至少1个块防止除零
                    
                    return drd_sum / nubn
                
                drd = compute_drd(gt, pred)
                total_drd += drd
 
                count += 1

    
    # 计算平均指标
    avg_loss = valid_loss / len(evaluation_loader)
    avg_fm = total_fm / count
    avg_pfm = total_pfm / count 
    avg_psnr = total_psnr / count
    avg_drd = total_drd / count
    
    return avg_loss, avg_fm, avg_pfm, avg_psnr, avg_drd