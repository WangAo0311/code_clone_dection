from losses import nt_xent_loss
import logging
from modeling import AttnPooling, sliding_windows,get_sentence_embedding,load_model
from utils import set_seed, setup_logger,parse_args,init_device
from data import Example,read_examples,convert_examples_to_features,prepare_dataset,make_collate_fn
from test import test
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
    get_linear_schedule_with_warmup
)
import os
from evaluate import *
from train import *
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
#best_threshold = 0.32
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def main():
    dev_dataset = {}
    args = parse_args()
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"[proc {os.getpid()}] local_rank={args.local_rank}, cuda:{torch.cuda.current_device()}")

    if args.openai_model:          
        args.do_train = False
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.local_rank in [-1, 0]:
        setup_logger(args.output_dir)
        logger.info(f"Arguments: {args}")
    init_device(args)
    config_class, model_class, tokenizer_class = AutoConfig, AutoModel, AutoTokenizer
    model_name = args.model_name_or_path.lower()
    trust_remote = "codet5p" in model_name
    
    if not args.openai_model:
        config = config_class.from_pretrained(
            args.config_name or args.model_name_or_path,
            trust_remote_code=trust_remote
        )
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name or args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            trust_remote_code=trust_remote
        )
        model = load_model(args, model_class, config, tokenizer)
        if hasattr(model, "module"):
            model.module.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
    else:
        model = None
        tokenizer = None
    collate_fn = make_collate_fn(tokenizer)
    best_threshold = 0.32
    if args.do_eval and (not args.do_train) and args.openai_model:
        best_threshold = eval_gpt(args, model, tokenizer, collate_fn, logger)
    elif args.do_train:
        best_threshold = train(args,model,tokenizer,collate_fn,logger,dev_dataset)
    elif args.do_eval and  not(args.do_train):
        best_threshold = evaluate(args, model, tokenizer, collate_fn, logger, dev_dataset, 100)
    if args.do_test:

        test(args, model, tokenizer, collate_fn, logger, best_threshold)


if __name__ == "__main__":
    main()