
from data import read_examples, prepare_dataset
from torch.utils.data import DataLoader, RandomSampler
from modeling import get_sentence_embedding
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from datetime import datetime
def fixed_sampler(dataset, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return RandomSampler(dataset, generator=g)
def evaluate(args, model, tokenizer, collate_fn, logger, dev_dataset, epoch):
    if args.local_rank not in [-1, 0]:
        return
    model_name = args.model_name_or_path.lower()
    model.eval()
    #dev_dataset = {}
    current_epoch = epoch if 'epoch' in locals() else 0
    threshold_log_path = os.path.join(args.output_dir, f"thresholds_epoch{current_epoch}.tsv")
#tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    if 'dev_loss' in dev_dataset:
        eval_examples, eval_data = dev_dataset['dev_loss']
    else:
        eval_examples = read_examples(args.dev_filename)
        eval_data = prepare_dataset(eval_examples, tokenizer, args.max_source_length,args)
        dev_dataset['dev_loss'] = (eval_examples, eval_data)

    eval_sampler = fixed_sampler(eval_data, args.seed) 
    # eval_sampler = RandomSampler(eval_data) 
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,  collate_fn=collate_fn,drop_last=False)
    if args.local_rank in [-1, 0]:
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
    

    
    cos_right, cos_wrong = [], []

    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, target_ids, target_mask,task_ids ,_= batch
        with torch.no_grad():
            sen_vec1 = get_sentence_embedding(model, source_ids, source_mask, model_name, tokenizer,args)
            sen_vec2 = get_sentence_embedding(model, target_ids, target_mask, model_name, tokenizer,args)

        cos = nn.CosineSimilarity(dim=1)(sen_vec1, sen_vec2)
        
        cos_right += cos.tolist()
        K = 3

        for i in range(len(sen_vec1)):
            neg_count = 0
            for j in range(len(sen_vec1)):
                if i==j or torch.equal(task_ids[i], task_ids[j]):
                    continue
                cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec1[j]).item())
                cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec2[j]).item())
                neg_count += 1
                if neg_count > K:
                    break
    #threshold_log_path = os.path.join(args.output_dir, f"thresholds_epoch{epoch}.tsv")
    best_f1, best_precision, best_recall, best_threshold = 0, 0, 0, 0

    

    with open(threshold_log_path, "w") as tf:
        tf.write("threshold\trecall\tprecision\tF1\n")
        for i in range(1, 100):
            threshold = i / 100
            tp = sum([1 for s in cos_right if s >= threshold])
            fp = sum([1 for s in cos_wrong if s >= threshold])
            fn = len(cos_right) - tp

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            tf.write(f"{threshold:.2f}\t{recall:.4f}\t{precision:.4f}\t{f1:.4f}\n")

            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = threshold

    if args.local_rank in [-1, 0]:
        logger.info(f"[Epoch {epoch}] eval: best F1 = {best_f1:.4f}, precision = {best_precision:.4f}, recall = {best_recall:.4f}, threshold = {best_threshold:.2f}")
    
        result_str = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Epoch {epoch}]: recall={best_recall:.3f}, precision={best_precision:.3f}, F1={best_f1:.3f}, threshold={best_threshold:.2f}\n"
        with open(os.path.join(args.output_dir, 'result'), 'a+') as f:
            f.write(result_str)
# 保存当前 checkpoint
    if not args.openai_model:
        checkpoint_dir = os.path.join(args.output_dir, "checkpoint-last")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_to_save = model.module if hasattr(model, 'module') else model
        if args.do_train:
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, f"{epoch}_pytorch_model.bin"))
    if args.local_rank in [-1, 0]:
                logger.info("  Best F1:%s", best_f1)
                logger.info("  "+"*"*20)
                logger.info("  Recall:%s", best_recall)
                logger.info("  "+"*"*20)
                logger.info("  Precision:%s", best_precision)
                logger.info("  "+"*"*20)
                logger.info("  Best threshold:%s", best_threshold)
                logger.info("  "+"*"*20)
    return best_threshold


def eval_gpt(args, model, tokenizer, collate_fn, logger):
    if args.local_rank not in [-1, 0]:
        return
    model_name = args.model_name_or_path.lower() 
    model_eval = model
    eval_examples = read_examples(args.dev_filename)
    eval_data = prepare_dataset(eval_examples, tokenizer,
                                args.max_source_length, args)
    eval_sampler = fixed_sampler(eval_data, args.seed) 
    eval_loader = DataLoader(eval_data,
                            sampler=eval_sampler,
                            batch_size=args.eval_batch_size,
                            collate_fn=collate_fn,
                            drop_last=False)

    cos_right, cos_wrong, processed = [], [], 0
    bar = tqdm(eval_loader, total=len(eval_loader),
            desc="Eval  (OpenAI)" if args.openai_model else "Eval  (HF)",
            ncols=100)

    for batch in bar:
        if args.openai_model:
            codes1, _, codes2, _, task_ids,idx = batch
            sen_vec1 = get_sentence_embedding(None, codes1, None, model_name, tokenizer, args)
            sen_vec2 = get_sentence_embedding(None, codes2, None, model_name, tokenizer, args)
        # else:
        #     batch = tuple(t.to(args.device) for t in batch)
        #     s_ids, s_mask, t_ids, t_mask, task_ids = batch
        #     with torch.no_grad():
        #         sen_vec1 = get_sentence_embedding(model_eval, s_ids, s_mask, model_name, tokenizer, args)
        #         sen_vec2 = get_sentence_embedding(model_eval, t_ids, t_mask, model_name, tokenizer, args)
        cos = nn.CosineSimilarity(dim=1)(sen_vec1, sen_vec2)
        cos_right += cos.tolist()
        K = 3
        for i in range(len(sen_vec1)):
            neg_count = 0
            for j in range(len(sen_vec1)):
                # 跳过正例
                if i == j:
                    continue
                # 跳过同 task 的样本（可选，严格对齐训练逻辑）
                if task_ids[i] == task_ids[j]:
                    continue
                cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec1[j]).item())
                cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec2[j]).item())
                neg_count += 1
                if neg_count > K:
                    break

    best_f1 = best_precision = best_recall = best_threshold = 0
    threshold_log_path = os.path.join(args.output_dir, f"thresholds_eval_GPT.tsv")
    with open(threshold_log_path, "w") as tf:
        tf.write("threshold\trecall\tprecision\tF1\n")
        for i in range(1, 100):
            t = i / 100
            tp = sum(s >= t for s in cos_right)
            fp = sum(s >= t for s in cos_wrong)
            fn = len(cos_right) - tp
            p  = tp / (tp + fp) if tp + fp else 0
            r  = tp / (tp + fn) if tp + fn else 0
            f1 = 2*p*r/(p+r) if p+r else 0
            
            tf.write(f"{t:.2f}\t{r:.4f}\t{p:.4f}\t{f1:.4f}\n")
            if f1 > best_f1:
                best_f1, best_precision, best_recall, best_threshold = f1, p, r, t

    if args.local_rank in [-1, 0]:
        logger.info(f"[GPT] eval: best F1 = {best_f1:.4f}, precision = {best_precision:.4f}, recall = {best_recall:.4f}, threshold = {best_threshold:.2f}")
    
        result_str = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [GPT]: recall={best_recall:.3f}, precision={best_precision:.3f}, F1={best_f1:.3f}, threshold={best_threshold:.2f}\n"
        with open(os.path.join(args.output_dir, 'result'), 'a+') as f:
            f.write(result_str)
        # 保存当前 checkpoint
    if args.local_rank in [-1, 0]:
                logger.info("  Best F1:%s", best_f1)
                logger.info("  "+"*"*20)
                logger.info("  Recall:%s", best_recall)
                logger.info("  "+"*"*20)
                logger.info("  Precision:%s", best_precision)
                logger.info("  "+"*"*20)
                logger.info("  Best threshold:%s", best_threshold)
                logger.info("  "+"*"*20)
    return best_f1
