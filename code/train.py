import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from data import dataset
from tqdm import tqdm

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import transformer_imgbin
from loss import CombinedLoss, dice_bce_loss, FocalLoss, TverskyBoundaryLoss
from torch.cuda.amp import autocast, GradScaler

from torchinfo import summary

def compute_loss(args, model, image, gt_image, batch_size, criterion):
    # preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    preds  = model(image)
    # preds = torch.sigmoid(preds)
    # preds = preds.float()

    torch.backends.cudnn.enabled = False
    total_loss = criterion(preds, gt_image)
    torch.backends.cudnn.enabled = True
    return total_loss



def main():

    args = option.get_args_parser()
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    # 如果使用 CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True   # 启用cudnn自动优化
        torch.backends.cudnn.deterministic = False  # 关闭确定性模式

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir, flush_secs = 60, comment = args.exp_name)

    model = transformer_imgbin.create_model(num_class= args.num_class, 
                                            num_block= args.num_block, 
                                            img_size= args.img_size,
                                            layer_conv_num = args.layer_conv_num)

    # summary(model, input_data=torch.ones(32, 3, 224, 224), col_names=("num_params", "mult_adds"), verbose=1)
    # import sys
    # sys.exit(0)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))
    logger.info(f"To view training metrics, run:\ntensorboard --logdir={args.save_dir} --bind_all")


    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()


    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-5, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    # Load pretrained model if it exists
    pretrained_model_path = os.path.join(args.save_dir, 'last_model.pth')  # or 'best_WER.pth'
    if os.path.exists(pretrained_model_path):
        logger.info(f"Loading pretrained model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['model'])
        model_ema.ema.load_state_dict(checkpoint['state_dict_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Pretrained model loaded successfully.")

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(args.train_data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               prefetch_factor=2,
                                               num_workers=args.num_workers)
                                            #    collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(args.val_data_path)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)


    # criterion = torch.nn.BCELoss() 
    criterion = dice_bce_loss()
    # criterion = TverskyBoundaryLoss()
    # criterion = CombinedLoss()
    # criterion = BinaryEqualizedFocalLoss()
    # criterion = AdvancedBinaryLoss()
    # criterion = FocalLoss()
    # criterion = MseFocalLoss()
    # criterion = FocalSSIMLoss()


    train_loss, best_loss = 0.0, float('inf')
    best_fm, best_psnr, best_drd = 0.0, 0.0, float('inf')


    #### ---- train & eval ---- ####
    with tqdm(total=args.total_iter + 1, desc="Training Progress") as pbar:
        for nb_iter in range(1, args.total_iter + 1):

            optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

            # optimizer.zero_grad()

            batch = next(train_iter)
            image = batch[0].cuda()
            image1 = batch[1].cuda()
            batch_size = image.size(0)

            pbar.set_description(f"Iter {nb_iter}/{args.total_iter}")

            loss = compute_loss(args, model, image, image1, batch_size, criterion)
            loss.backward()
            optimizer.first_step(zero_grad=True)


            loss_perturbed = compute_loss(args, model, image, image1, batch_size, criterion)
            loss_perturbed.backward()
            optimizer.second_step(zero_grad=True)

            # model.zero_grad()
            model_ema.update(model, num_updates=nb_iter / 2)
            # model_ema.update(model)
            train_loss += loss.item()

            pbar.update(1)
            pbar.set_postfix({
                'lr': f"{current_lr:.2e}",
                'loss': f"{loss.item():.4f}"
            })

            # writer.add_scalars('Train/Losses', {
            #     'total': loss.item(),
            #     'focal': focal_loss.item(),
            #     'dice': dice_loss.item()
            # }, nb_iter)
            writer.add_scalar('0_Train/Loss', loss.item(), nb_iter)
            writer.add_scalar('1_Learning_Rate', current_lr, nb_iter)



            if nb_iter % args.print_iter == 0:
                train_loss_avg = train_loss / args.print_iter
                logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

                writer.add_scalar('Train/lr', current_lr, nb_iter)
                writer.add_scalar('Train/train_loss', train_loss_avg, nb_iter)
                train_loss = 0.0


            if nb_iter % args.eval_iter == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, fm, pfm, psnr, drd = valid.validation(model_ema.ema,
                                                                criterion,
                                                                val_loader)

                    writer.add_scalar('3_Validation/Loss', val_loss, nb_iter)
                    writer.add_scalar('4_Metrics/F-Measure', fm, nb_iter)
                    writer.add_scalar('4_Metrics/Pseudo-FM', pfm, nb_iter)
                    writer.add_scalar('4_Metrics/PSNR', psnr, nb_iter)
                    writer.add_scalar('4_Metrics/DRD', drd, nb_iter)

                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'last_model.pth'))

                    if val_loss < best_loss:
                        logger.info(f'val improved from {best_loss:.5f} to {val_loss:.5f}!!!')
                        best_loss = val_loss
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_val.pth'))

                    if fm > best_fm:
                        logger.info(f'fm improved from {best_fm:.5f} to {fm:.5f}!!!')
                        best_fm = fm
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_fm.pth'))

                    if psnr > best_psnr:
                        logger.info(f'psnr improved from {best_psnr:.5f} to {psnr:.5f}!!!')
                        best_psnr = psnr
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_psnr.pth'))

                    if drd < best_drd:
                        logger.info(f'drd improved from {best_drd:.5f} to {drd:.5f}!!!')
                        best_drd = drd
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_drd.pth'))

                    logger.info(
                        f'Val. loss : {val_loss:0.5f} \t fm : {fm:0.5f} \t pfm : {pfm:0.5f} \t psnr : {psnr:0.5f} \t drd : {drd:0.5f} \t ')

                    model.train()

    writer.close()
    logger.info(f"Training complete! Final model saved to {args.save_dir}")


if __name__ == '__main__':
    main()