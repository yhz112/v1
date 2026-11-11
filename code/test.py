import torch

import os
import re
import json
import valid
from utils import utils
from utils import option
from data import dataset
from collections import OrderedDict
from model import transformer_imgbin
import numpy as np
from PIL import Image
from loss import CombinedLoss, AdvancedBinaryLoss,FocalLoss, dice_bce_loss

def module_test(model, save_dir, criterion, evaluation_loader) :

    valid_loss = 0.0
    count = 0
    
    # 初始化指标累加器
    total_fm, total_pfm, total_psnr, total_drd = 0.0, 0.0, 0.0, 0.0

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算，减少内存占用
        for i, image_tensors in enumerate(evaluation_loader):
             
            batch_size = image_tensors.size(0)
            image = image_tensors.cuda()
            # print(f"image:{image.shape} {image.dtype}")

            preds = model(image)
            # preds = torch.sigmoid(preds)
            preds = preds.float()
            # thresh_msp = thresh_msp.float()
            # print(f"thrsh_msp:{thresh_msp}")
            # 将预测结果转换为二值图像
            # preds_binary = ((preds) < 0.1).float()  # 假设是二分类任务，使用 Sigmoid 激活并阈值化
            preds_binary = ((preds > 0.5)).float()  # 假设是二分类任务，使用 Sigmoid 激活并阈值化

            # 将预测和真实标签从 GPU 移到 CPU 并转换为 NumPy 数组
            preds_binary_np = preds_binary.cpu().numpy()

            pred_image = preds_binary_np[0, 0]  # 形状为 [H, W]
            # 将二值图像的值从 [0, 1] 转换为 [0, 255]
            pred_image = (pred_image * 255).astype(np.uint8)
            # 创建 PIL 图像对象并保存
            img = Image.fromarray(pred_image)
            img.save(f"{save_dir}/output_{i}.png")




def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    # logger = utils.get_logger(args.save_dir)
    # logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # model = transformer_imgbin.create_model(num_class= args.num_class, num_block= args.num_block)
    model = transformer_imgbin.create_model(num_class= args.num_class, 
                                            num_block= args.num_block, 
                                            img_size= args.img_size,
                                            layer_conv_num = args.layer_conv_num)


    pth_path = args.save_dir + '/best_psnr.pth'
    # pth_path = args.save_dir + '/best_val.pth'
    # pth_path = args.save_dir + '/best_drd.pth'
    print('loading transformer img binary checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model.eval()

    # scripted_model = torch.jit.script(model)
    # output_path = args.save_dir + "/best_val.pt"
    # torch.jit.save(scripted_model, output_path)  # 如果你使用了 script

# # Move the model to the device (CPU or GPU) before tracing
#     model = model.to("cuda")

#     # Define an example input tensor with the same shape as your model's input.
#     # Make sure the batch size is 1 and the dimensions match what your model expects.
#     # For instance, if your model takes RGB images of size 256x256, you would do:
#     example_input = torch.randn(1, 3, 512, 512).to(device)  # Adjust dimensions as needed
# # Use torch.jit.trace to convert the model into a TorchScript model
#     traced_model = torch.jit.trace(model, example_input)

#     # Save the traced model to a file
#     output_path = args.save_dir + "/traced_model.pt"
#     traced_model.save(output_path)

    # logger.info(f'Traced model saved to {output_path}')

    model = model.cuda()

    # logger.info('Loading test loader...')
    # train_dataset = dataset.myLoadDS(args.test_data_path)

    test_dataset = dataset.myLoadDS(args.test_data_path, is_test= True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    criterion = dice_bce_loss()

    model.eval()
    with torch.no_grad():
        # val_loss, accuracy, precision, recall, pixel_error_rate = valid.validation(model,
        #                                                              criterion,
        #                                                              test_loader)

        module_test(model, args.save_dir,criterion, test_loader)
        
    # logger.info(
    #     f'Test. loss : {val_loss:0.3f} \t acc : {accuracy:0.4f} \t pre : {precision:0.4f} \t recall: {recall:0.4f}')

if __name__ == '__main__':
    args = option.get_args_parser()
    main()



