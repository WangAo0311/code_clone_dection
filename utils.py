import os
import random
import numpy as np
import torch
import torch.distributed as dist
import argparse
import logging
logger = logging.getLogger()


def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # ✅ 修正拼写
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   
    torch.use_deterministic_algorithms(True, warn_only=False)
def setup_logger(output_dir):
    logger = logging.getLogger()
    log_file_path = os.path.join(output_dir, "log_epoch_train.log")

    if any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        return

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    ))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)



def parse_args():
    parser = argparse.ArgumentParser()

    # 必要参数
    #parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--load_model_path", type=str)

    # 数据文件
    parser.add_argument("--train_filename", type=str)
    parser.add_argument("--dev_filename", type=str)
    parser.add_argument("--test_filename", type=str)

    # 模型相关
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=32)

    # 训练策略
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--train_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--beam_size", type=int, default=10)

    # 环境
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--slide_window_size", type=int, default=0,
                    help=">0 时启用 sliding window")
    parser.add_argument("--slide_stride", type=int, default=0,
                        help="窗口步长；默认=window_size//2")
    parser.add_argument("--pooling", type=str, default="cls",
                        choices=["cls", "mean", "attention"])
    
    parser.add_argument("--openai_model", type=str, default="",
                    help="填 text-embedding-ada-002 时跳过训练")
    parser.add_argument("--openai_batch", type=int, default=16,
                    help="一次请求多少条文本，=1 表示单条单条发")
    
    parser.add_argument(
        "--concat_desc_code",      # ← 新 flag
        action="store_true",
        help="如果存在 Code*_desc 字段，则拼接成 'desc <sep> code' 后再送模型"
    )
    
    return parser.parse_args()

def init_device(args):
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device
    if args.local_rank in [-1, 0]:
        logger.warning("Device: %s, n_gpu: %d, distributed: %s",
                       device, args.n_gpu, args.local_rank != -1)
    return device