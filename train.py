import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm
from modeling import get_sentence_embedding
from data import read_examples, prepare_dataset
from losses import nt_xent_loss
from torch.optim import AdamW
import os
from datetime import datetime
import torch.nn as nn
from evaluate import evaluate
from transformers import (
    get_linear_schedule_with_warmup
)
def train(args, model,tokenizer,collate_fn,logger,dev_dataset):
    # if torch.cuda.device_count() > 1:
    #     logger.info(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    # else:
    #     logger.info("Using single GPU")
    """ Train the model """
    model_name = args.model_name_or_path.lower()
    train_examples = read_examples(args.train_filename)
    #train_dataset = prepare_dataset(train_examples, tokenizer, args.max_source_length, model_name)
    train_dataset= prepare_dataset(train_examples, tokenizer, args.max_source_length,args)
    if args.local_rank == -1:
        
        g = torch.Generator()
        g.manual_seed(args.seed)            
        train_sampler = RandomSampler(train_dataset, generator=g)
    else:
        
        train_sampler = DistributedSampler(
            train_dataset,
            seed=args.seed,                
            shuffle=True
        )

    #train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size // args.gradient_accumulation_steps,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Define optimizer_grouped_parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight","attnpool.q.weight"])],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight","attnpool.q.weight"])],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * 0.1),
        num_training_steps=t_total
    )

    global_step = 0
    if args.local_rank in [-1, 0]:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

    nb_tr_examples, nb_tr_steps,global_step = 0,0,0
    for epoch in range(args.num_train_epochs):
        model.train()

        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        bar = tqdm(train_dataloader, total=len(train_dataloader)) if args.local_rank in [-1, 0] else train_dataloader
        for step, batch in enumerate(bar):
            try:
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask, task_ids,_ = batch
                unique_tasks = set(t.item() for t in task_ids)
                if len(unique_tasks) == 1:
                    logger.info(f"⚠️ Skipping batch {step}, all task_ids = {list(unique_tasks)}")
                    continue
                #print("Forward start", flush=True)
                
                
                sen_vec1 = get_sentence_embedding(model, source_ids, source_mask, model_name, tokenizer,args)
                sen_vec2 = get_sentence_embedding(model, target_ids, target_mask, model_name, tokenizer,args)
                loss = nt_xent_loss(sen_vec1, sen_vec2, task_ids)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                if args.local_rank in (-1, 0):     
                    bar.set_description(f"epoch {epoch} step {step} loss {loss:.4f}")
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # source_mask是attention mask，1的地方是有效token
                    if source_mask is not None:
                        real_lengths = source_mask.sum(dim=1)  # 每个样本的真实token数
                        max_real_length = real_lengths.max().item()
                    else:
                        # 如果没mask，退而求其次，直接用 source_ids.size(1) 当作seq_len
                        max_real_length = source_ids.size(1)

                    total_tokens = source_ids.size(0) * source_ids.size(1)

                    print(f"⚠️ Warning: Skipped batch {step}, batch_size={source_ids.size(0)}, seq_len={source_ids.size(1)}, max_real_length={max_real_length}, total_tokens={total_tokens} due to OOM")
                    logger.info(f"⚠️ Warning: Skipped batch {step}, , task_ids={task_ids.tolist()},batch_size={source_ids.size(0)}, seq_len={source_ids.size(1)}, max_real_length={max_real_length}, total_tokens={total_tokens} due to OOM")
                    
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        if args.do_eval:
            best_threshold = evaluate(args, model, tokenizer, collate_fn, logger, dev_dataset, epoch)
        
    return best_threshold





