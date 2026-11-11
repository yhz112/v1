import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Origami for HTR',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out-dir', type=str, default='./output', help='output directory')
    parser.add_argument('--train-bs', default=80, type=int, help='train batch size')
    parser.add_argument('--val-bs', default=100, type=int, help='validation batch size')
    parser.add_argument('--num-workers', default=12, type=int, help='nb of workers')
    parser.add_argument('--eval-iter', default=1000, type=int, help='nb of iterations to run evaluation')
    parser.add_argument('--total-iter', default=150000, type=int, help='nb of total iterations for training')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='nb of iterations for warm-up')
    parser.add_argument('--print-iter', default=50, type=int, help='nb of total iterations to print information')
    parser.add_argument('--max-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=5e-1, type=float, help='weight decay')
    parser.add_argument('--use-wandb', action='store_true', default=False, help = 'wheteher use wandb, otherwise use tensorboard')
    parser.add_argument('--exp-name',type=str, default='IAM_HTR_ORIGAMI_NET', help='experimental name (save dir will be out_dir + exp_name)')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')

    parser.add_argument('--img-size', default=[1024, 64], type=int, nargs='+', help='image size')
    parser.add_argument('--num_class', default = 1, type=int, help='number of class')
    parser.add_argument('--num_block', default=[3, 4, 8, 2, 2, 2], type=int, nargs='+', help='')
    parser.add_argument('--layer_conv_num', default=[16, 32, 64, 128], type=int, nargs='+', help='')


    
    parser.add_argument('--patch-size', default=[64, 8], type=int, nargs='+', help='patch size')
    parser.add_argument('--mask-ratio', default=0.3, type=float, help='mask ratio')
    parser.add_argument('--cos-temp', default=8, type=int, help='cosine similarity classifier temperature')
    parser.add_argument('--max-span-length', default=8, type=int, help='max mask length')
    parser.add_argument('--spacing', default=0, type=int, help='the spacing between two span masks')
    parser.add_argument('--proj', default=8, type=float, help='projection value')

    parser.add_argument('--dpi-min-factor', default=0.5,type=float)
    parser.add_argument('--dpi-max-factor', default=1.5, type=float)
    parser.add_argument('--perspective-low', default=0., type=float)
    parser.add_argument('--perspective-high', default=0.4, type=float)
    parser.add_argument('--elastic-distortion-min-kernel-size', default=3, type=int)
    parser.add_argument('--elastic-distortion-max-kernel-size', default=3, type=int)
    parser.add_argument('--elastic_distortion-max-magnitude', default=20, type=int)
    parser.add_argument('--elastic-distortion-min-alpha', default=0.5, type=float)
    parser.add_argument('--elastic-distortion-max-alpha', default=1, type=float)
    parser.add_argument('--elastic-distortion-min-sigma', default=1, type=int)
    parser.add_argument('--elastic-distortion-max-sigma', default=10, type=int)
    parser.add_argument('--dila-ero-max-kernel', default=2, type=int )
    parser.add_argument('--jitter-contrast', default=0.4, type=float)
    parser.add_argument('--jitter-brightness', default=0.4, type=float )
    parser.add_argument('--jitter-saturation', default=0.4, type=float )
    parser.add_argument('--jitter-hue', default=0.2, type=float)

    parser.add_argument('--dila-ero-iter', default=1, type=int, help='nb of iterations for dilation and erosion kernel')
    parser.add_argument('--blur-min-kernel', default=3, type=int)
    parser.add_argument('--blur-max-kernel', default=5, type=int)
    parser.add_argument('--blur-min-sigma', default=3, type=int)
    parser.add_argument('--blur-max-sigma', default=5, type=int)
    parser.add_argument('--sharpen-min-alpha', default=0, type=int)
    parser.add_argument('--sharpen-max-alpha', default=1, type=int)
    parser.add_argument('--sharpen-min-strength', default=0, type=int)
    parser.add_argument('--sharpen-max-strength', default=1, type=int)
    parser.add_argument('--zoom-min-h', default=0.8, type=float)
    parser.add_argument('--zoom-max-h', default=1, type=float)
    parser.add_argument('--zoom-min-w', default=0.99, type=float)
    parser.add_argument('--zoom-max-w', default=1, type=float)
    parser.add_argument('--proba', default=0.5, type=float)

    parser.add_argument('--ema-decay', default=0.9999, type=float, help='Exponential Moving Average (EMA) decay')
    parser.add_argument('--alpha', default=1, type=float, help='kld loss ratio')

    subparsers = parser.add_subparsers(title="dataset setting", dest="subcommand")

    img_12_9_bin = subparsers.add_parser("img_12_9_bin",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")
    img_12_9_bin.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/img_bin_dataset_12_9/train/',
                    help='train data list')
    img_12_9_bin.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/img_bin_dataset_12_9/val/',
                    help='val data list')
    img_12_9_bin.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/img_bin_dataset_12_9/test/',
                    help='test data list')
    img_12_9_bin.add_argument('--train-bs', default=100, type=int, help='train batch size')
    img_12_9_bin.add_argument('--val-bs', default=200, type=int, help='validation batch size')
    img_12_9_bin.add_argument('--total-iter', default=100000, type=int, help='nb of total iterations for training1')
    img_12_9_bin.add_argument('--num_block', default=[3, 4, 4, 4, 4, 3], type=int, nargs='+', help='')

    img_256_bin = subparsers.add_parser("img_256_bin",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")
    img_256_bin.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/img_bin_dataset_256_256/train/',
                    help='train data list')
    img_256_bin.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/img_bin_dataset_256_256/val/',
                    help='val data list')
    img_256_bin.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/img_bin_dataset_256_256/test/',
                    help='test data list')

    img_256_bin.add_argument('--train-bs', default=24, type=int, help='train batch size')
    img_256_bin.add_argument('--val-bs', default=64, type=int, help='validation batch size')
    img_256_bin.add_argument('--max-lr', default=1.5e-3, type=float, help='learning rate')

    img_256_bin.add_argument('--eval-iter', default=400, type=int, help='nb of iterations to run evaluation')
    img_256_bin.add_argument('--warm-up-iter', default=100, type=int, help='nb of iterations for warm-up')
    img_256_bin.add_argument('--print-iter', default=50, type=int, help='nb of total iterations to print information')
    img_256_bin.add_argument('--total-iter', default=8000, type=int, help='nb of total iterations for training1')

    # img_256_bin.add_argument('--num_block', default=[3, 4, 4, 4, 4, 3], type=int, nargs='+', help='')
    # img_256_bin.add_argument('--layer_conv_num', default=[32, 64, 96, 128], type=int, nargs='+', help='')
    img_256_bin.add_argument('--num_block', default=[1, 2, 2, 2, 4, 2, 4, 2, 1], type=int, nargs='+', help='')
    img_256_bin.add_argument('--layer_conv_num', default=[32, 64, 128, 256], type=int, nargs='+', help='')


    img_total_bin = subparsers.add_parser("img_total_bin",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")

    img_total_bin.add_argument('--train-bs', default=1, type=int, help='train batch size')
    img_total_bin.add_argument('--val-bs', default=4, type=int, help='validation batch size')

    img_total_bin.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/img_bin_total_dataset_resize_2048_1024/train/',
                    help='train data list')
    img_total_bin.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/img_bin_total_dataset_resize_2048_1024/val/',
                    help='val data list')
    img_total_bin.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/img_bin_total_dataset_resize_2048_1024/test/',
                    help='test data list')

    img_512_bin = subparsers.add_parser("img_512_bin",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")

    img_512_bin.add_argument('--train-bs', default=100, type=int, help='train batch size')
    img_512_bin.add_argument('--val-bs', default=160, type=int, help='validation batch size')
    img_512_bin.add_argument('--num_block', default=[3, 4, 8, 2, 2, 2], type=int, nargs='+', help='')

    img_512_bin.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/img_bin_total_dataset_resize_512_512/train/',
                    help='train data list')
    img_512_bin.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/img_bin_total_dataset_resize_512_512/val/',
                    help='val data list')
    img_512_bin.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/img_bin_total_dataset_resize_512_512/test/',
                    help='test data list')
    img_512_bin.add_argument('--img-size', default=[1024, 64], type=int, nargs='+', help='image size')


    img_2048_288 = subparsers.add_parser("img_2048_288",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")
    img_2048_288.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/img_bin_2048_288/train/',
                    help='train data list')
    img_2048_288.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/img_bin_2048_288/val/',
                    help='val data list')
    img_2048_288.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/img_bin_2048_288/test/',
                    help='test data list')

    img_2048_288.add_argument('--train-bs', default=4, type=int, help='train batch size')
    img_2048_288.add_argument('--val-bs', default=8, type=int, help='validation batch size')
    img_2048_288.add_argument('--max-lr', default=1.5e-3, type=float, help='learning rate')

    img_2048_288.add_argument('--eval-iter', default=400, type=int, help='nb of iterations to run evaluation')
    img_2048_288.add_argument('--warm-up-iter', default=100, type=int, help='nb of iterations for warm-up')
    img_2048_288.add_argument('--print-iter', default=50, type=int, help='nb of total iterations to print information')
    img_2048_288.add_argument('--total-iter', default=8000, type=int, help='nb of total iterations for training1')

    # img_2048_288.add_argument('--num_block', default=[1, 2, 2, 3, 3, 2, 3, 2, 1], type=int, nargs='+', help='')
    # img_2048_288.add_argument('--layer_conv_num', default=[32, 64, 96, 128], type=int, nargs='+', help='')
    img_2048_288.add_argument('--num_block', default=[1, 2, 2, 2, 4, 2, 4, 2, 1], type=int, nargs='+', help='')
    img_2048_288.add_argument('--layer_conv_num', default=[32, 64, 128, 256], type=int, nargs='+', help='')



    dibco2017 = subparsers.add_parser("dibco2017",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")
    dibco2017.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/DIBCO/DIBCO2017/train/',
                    help='train data list')
    dibco2017.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/DIBCO/DIBCO2017/val/',
                    help='val data list')
    dibco2017.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/DIBCO/DIBCO2017/test/',
                    help='test data list')

    dibco2017.add_argument('--train-bs', default=32, type=int, help='train batch size')
    dibco2017.add_argument('--val-bs', default=96, type=int, help='validation batch size')
    dibco2017.add_argument('--max-lr', default=1.5e-3, type=float, help='learning rate')


    dibco2017.add_argument('--eval-iter', default=400, type=int, help='nb of iterations to run evaluation')
    dibco2017.add_argument('--warm-up-iter', default=100, type=int, help='nb of iterations for warm-up')
    dibco2017.add_argument('--print-iter', default=50, type=int, help='nb of total iterations to print information')
    dibco2017.add_argument('--total-iter', default=20000, type=int, help='nb of total iterations for training1')

    dibco2017.add_argument('--num_block', default=[1, 2, 2, 2, 3, 2, 4, 2, 1], type=int, nargs='+', help='')
    dibco2017.add_argument('--layer_conv_num', default=[32, 64, 96, 128], type=int, nargs='+', help='')


    dibco2018 = subparsers.add_parser("dibco2018",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")
    dibco2018.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/DIBCO/DIBCO2018/train/',
                    help='train data list')
    dibco2018.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/DIBCO/DIBCO2018/val/',
                    help='val data list')
    dibco2018.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/dibco_256_256/test/',
                help='test data list')

    dibco2018.add_argument('--train-bs', default=16, type=int, help='train batch size')
    dibco2018.add_argument('--val-bs', default=48, type=int, help='validation batch size')
    dibco2018.add_argument('--max-lr', default=1.5e-3, type=float, help='learning rate')


    dibco2018.add_argument('--eval-iter', default=400, type=int, help='nb of iterations to run evaluation')
    dibco2018.add_argument('--warm-up-iter', default=100, type=int, help='nb of iterations for warm-up')
    dibco2018.add_argument('--print-iter', default=50, type=int, help='nb of total iterations to print information')
    dibco2018.add_argument('--total-iter', default=8000, type=int, help='nb of total iterations for training1')

    dibco2018.add_argument('--num_block', default=[1, 2, 2, 2, 4, 2, 4, 2, 1], type=int, nargs='+', help='')
    dibco2018.add_argument('--layer_conv_num', default=[32, 64, 128, 256], type=int, nargs='+', help='')


    dibco_256 = subparsers.add_parser("dibco_256",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")
    dibco_256.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/dibco_256_256/train/',
                help='train data list')
    dibco_256.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/dibco_256_256/val/',
                help='val data list')
    dibco_256.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/dibco_256_256/test/',
                help='test data list')

    dibco_256.add_argument('--train-bs', default=32, type=int, help='train batch size')
    dibco_256.add_argument('--val-bs', default=64, type=int, help='validation batch size')
    dibco_256.add_argument('--max-lr', default=1.5e-3, type=float, help='learning rate')


    dibco_256.add_argument('--eval-iter', default=400, type=int, help='nb of iterations to run evaluation')
    dibco_256.add_argument('--warm-up-iter', default=100, type=int, help='nb of iterations for warm-up')
    dibco_256.add_argument('--print-iter', default=50, type=int, help='nb of total iterations to print information')
    dibco_256.add_argument('--total-iter', default=10000, type=int, help='nb of total iterations for training1')

    # dibco_256.add_argument('--num_block', default=[3, 4, 4, 4, 4, 3], type=int, nargs='+', help='')
    # dibco_256.add_argument('--num_block', default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], type=int, nargs='+', help='')
    # dibco_256.add_argument('--layer_conv_num', default=[32, 64, 96, 128, 160], type=int, nargs='+', help='')
    #                                              0   1  2 3  4  5  6  7
    # dibco_256.add_argument('--num_block', default=[1, 2, 2, 4, 4, 2, 4, 3, 1], type=int, nargs='+', help='')
    # dibco_256.add_argument('--num_block', default=[1, 2, 2, 6, 4, 2, 4, 2, 1], type=int, nargs='+', help='')
    # dibco_256.add_argument('--num_block', default=[1, 2, 3, 6, 4, 2, 4, 1, 1], type=int, nargs='+', help='')
    dibco_256.add_argument('--num_block', default=[1, 2, 2, 2, 4, 2, 4, 2, 1], type=int, nargs='+', help='')
    # dibco_256.add_argument('--layer_conv_num', default=[32, 64, 96, 128], type=int, nargs='+', help='')
    dibco_256.add_argument('--layer_conv_num', default=[32, 64, 128, 256], type=int, nargs='+', help='')
    # dibco_256.add_argument('--layer_conv_num', default=[64, 96, 128, 192], type=int, nargs='+', help='')
    # dibco_256.add_argument('--layer_conv_num', default=[32, 48, 96, 128], type=int, nargs='+', help='')
    # dibco_256.add_argument('--layer_conv_num', default=[16, 32, 64, 96], type=int, nargs='+', help='')


    dibco_512 = subparsers.add_parser("dibco_512",
                                description='Dataset parser for training on img bin',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on img bin")
    dibco_512.add_argument('--train_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/dibco_512_512/train/',
                help='train data list')
    dibco_512.add_argument('--val_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/dibco_512_512/val/',
                help='val data list')
    dibco_512.add_argument('--test_data-path', type=str, default='/home/yzy/dataset/cable_img_bin_dataset/DIBCO/DIBCO2017/test/',
                help='test data list')

    dibco_512.add_argument('--train-bs', default=16, type=int, help='train batch size')
    dibco_512.add_argument('--val-bs', default=32, type=int, help='validation batch size')
    dibco_512.add_argument('--max-lr', default=1e-3, type=float, help='learning rate')


    dibco_512.add_argument('--eval-iter', default=400, type=int, help='nb of iterations to run evaluation')
    dibco_512.add_argument('--warm-up-iter', default=100, type=int, help='nb of iterations for warm-up')
    dibco_512.add_argument('--print-iter', default=50, type=int, help='nb of total iterations to print information')
    dibco_512.add_argument('--total-iter', default=80000, type=int, help='nb of total iterations for training1')

    dibco_512.add_argument('--num_block', default=[3, 4, 4, 4, 4, 3], type=int, nargs='+', help='')
    dibco_512.add_argument('--layer_conv_num', default=[32, 64, 128, 192], type=int, nargs='+', help='')

    return parser.parse_args()
    