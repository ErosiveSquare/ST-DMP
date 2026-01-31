import argparse
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch

from agent import METHODS
from experiment.dataset import DATASETS
from multi_runs import multiple_run
from multi_runs_joint import multiple_run_joint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')


class Logger:
    """同时输出到控制台和日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_logging(args):
    """设置日志系统，创建日志文件夹和日志文件"""
    # 创建日志文件夹
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 使用日期时间命名日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{args.dataset}_{args.method}_bs{args.buffer_size}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 设置日志记录器
    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"日志保存至: {log_path}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 100)
    
    return logger, log_path


def get_params():
    parser = argparse.ArgumentParser()
    # experiment related
    parser.add_argument('--dataset',            default='cifar10',  type=str, choices=DATASETS.keys())
    parser.add_argument('--n_tasks',            default='10',       type=int)
    parser.add_argument('--n_classes',          default=None,       type=int, help='总类别数，默认按数据集自动设置')
    parser.add_argument('--buffer_size',        default=200,        type=int)
    parser.add_argument('--method',             default='mose',     type=str, choices=METHODS.keys())

    parser.add_argument('--seed',               default=0,          type=int)
    parser.add_argument('--run_nums',           default=10,         type=int)
    parser.add_argument('--epoch',              default=1,          type=int)
    parser.add_argument('--lr',                 default=1e-3,       type=float)
    parser.add_argument('--wd',                 default=1e-4,       type=float)
    parser.add_argument('--batch_size',         default=10,         type=int)
    parser.add_argument('--buffer_batch_size',  default=64,         type=int)

    parser.add_argument('--continual',          default='on',       type=str, choices=['off', 'on'])

    # mose control
    parser.add_argument('--ins_t',              default=0.07,       type=float)
    parser.add_argument('--expert',             default='3',        type=str, choices=['0','1','2','3'])
    parser.add_argument('--expert1',            default='2',       type=str, choices=['0','1','2','3'])
    parser.add_argument('--expert2',            default='3',       type=str, choices=['0','1','2','3'])
    parser.add_argument('--n_experts',          default=4,          type=int)
    parser.add_argument('--classifier',         default='ncm',      type=str, choices=['linear', 'ncm'])
    parser.add_argument('--augmentation',       default='ocm',      type=str, choices=['ocm', 'scr', 'none', 'simclr', 'randaug', 'trivial'])
    # mose hyper-params (new)
    parser.add_argument('--lambda_uncert',      default=5.0,        type=float, help='view consistency loss weight')
    parser.add_argument('--distill_temperature', default=2.0,       type=float, help='知识蒸馏温度')
    parser.add_argument('--distill_weight',      default=1.0,       type=float, help='知识蒸馏损失权重')
    parser.add_argument('--u_buffer_capacity',   default=64,        type=int,   help='U-buffer 容量')
    parser.add_argument('--align_weight',        default=1.0,       type=float, help='特征对齐损失权重')
    parser.add_argument('--teacher_student_l2_weight', default=5.0, type=float, help='教师-学生特征蒸馏权重')
    parser.add_argument('--T_min',               default=0.1,       type=float, help='hard-only 采样温度下界')
    parser.add_argument('--topk_ratio',          default=0.2,       type=float, help='相似类 Top-K 采样比例')
    parser.add_argument('--topk_min',            default=1,         type=int,   help='相似类 Top-K 最小值')
    parser.add_argument('--topk_max',            default=64,        type=int,   help='相似类 Top-K 最大值')

    parser.add_argument('--gpu_id',             default=0,          type=int)
    parser.add_argument('--n_workers',          default=8,          type=int)

    # teacher confusion / 类引导重放
    parser.add_argument('--confusion_temperature', default=None,    type=float, help='teacher confusion softmax温度；默认沿用distill_temperature')
    parser.add_argument('--teacher_confusion_top_k', default=3,     type=int,   help='对旧类取top-k概率累加')
    parser.add_argument('--teacher_confusion_top_n', default=5,     type=int,   help='按混淆得分选出的旧类数用于主缓冲区采样')

    # checkpoint output control (optional)
    parser.add_argument('--save_path',          type=str, default=argparse.SUPPRESS)
    parser.add_argument('--task_ckpt_root',     type=str, default=argparse.SUPPRESS)
    parser.add_argument('--final_ckpt_root',    type=str, default=argparse.SUPPRESS)
    parser.add_argument('--final_ckpt_name',    type=str, default=argparse.SUPPRESS)

    # diagnostics
    parser.add_argument('--diag_attn',          action='store_true', help='enable attention diagnostics logs')

    # logging 
    parser.add_argument('--exp_name',           default='tmp',      type=str)
    parser.add_argument('--wandb_project',      default='ocl',      type=str)
    parser.add_argument('--wandb_entity',                           type=str)
    parser.add_argument('--wandb_log',          default='off',      type=str, choices=['off', 'online'])
    args = parser.parse_args()

    # fallback for expert1/expert2 when not provided
    if getattr(args, 'expert1', None) is None:
        args.expert1 = args.expert
    if getattr(args, 'expert2', None) is None:
        args.expert2 = args.expert

    return args


def main(args):
    # 设置日志
    logger, log_path = setup_logging(args)
    
    try:
        torch.cuda.set_device(args.gpu_id)
        args.cuda = torch.cuda.is_available()

        print('Arguments =')
        for arg in vars(args):
            print('\t' + arg + ':', getattr(args, arg))

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print('[CUDA is unavailable]')

        if args.continual == 'on':
            multiple_run(args)
        else:
            multiple_run_joint(args)
            
    finally:
        # 记录结束时间并关闭日志
        print('=' * 100)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"日志已保存至: {log_path}")
        logger.close()
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal


if __name__ == '__main__':
    args = get_params()
    main(args)
