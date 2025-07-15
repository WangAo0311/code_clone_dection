
from modeling import get_sentence_embedding
import os
import torch
from data import read_examples, prepare_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
#import torch.nn.functional as F
def fixed_sampler(dataset, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return RandomSampler(dataset, generator=g)
def fetch_code(example, side):                 # side 1=code1, 2=code2
    return example.code1 if side == 1 else example.code2

def save_case(fail_dir, case_type,
              idx_a, side_a, idx_b, side_b, sim,
              eg_a, eg_b):
    """
    把 (源码A, 源码B) 保存成 txt，文件名包含行号+侧标记+余弦分数
    """
    os.makedirs(fail_dir, exist_ok=True)
    fname = f"{case_type}_{idx_a:05d}{side_a}_{idx_b:05d}{side_b}_sim{sim:.2f}.txt"
    path = os.path.join(fail_dir, fname)

    with open(path, "w", encoding="utf-8") as fout:
        fout.write(f"# {case_type}   cos={sim:.4f}\n")
        fout.write(f"# task_a={eg_a.task} task_b={eg_b.task}\n\n")
        fout.write("######## Source A ########\n")
        fout.write(fetch_code(eg_a, side_a) + "\n\n")
        fout.write("######## Source B ########\n")
        fout.write(fetch_code(eg_b, side_b) + "\n")

def test(args, model, tokenizer, collate_fn, logger, best_threshold=0.32):
    """
    评测 + 保存失败样例：
        FP / FN → output_dir/failed_cases/*.txt
    """
    if args.local_rank not in [-1, 0]:
        return

    # ---------------- ① 准备数据 ---------------- #
    model_name = args.model_name_or_path.lower()
    if model:
        model.eval()

    eval_examples = read_examples(args.test_filename)
    eval_data     = prepare_dataset(eval_examples, tokenizer,
                                    args.max_source_length, args)

    eval_loader = DataLoader(
        eval_data,
        sampler=fixed_sampler(eval_data, args.seed),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        drop_last=False
    )

    if args.local_rank in [-1, 0]:
        logger.info("\n***** Running test *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size   = %d", args.eval_batch_size)

    # ---------------- ② 主循环 ---------------- #
    cos_right, cos_wrong = [], []
    false_pos, false_neg = [], []         # (idx_a, side_a, idx_b, side_b, cos)

    bar = tqdm(eval_loader, total=len(eval_loader),
               desc="Test (OpenAI)" if args.openai_model else "Test (HF)",
               ncols=100)
    processed = 0
    K = 3          # 每个正样本随机采 K 个负例

    for batch in bar:
        # ---- 取句向量 ---- #
        if args.openai_model:                           # GPT 路径
            codes1, _, codes2, _, task_ids, idx_batch = batch
            idx_batch = idx_batch.tolist()              # [B]
            sen_vec1 = get_sentence_embedding(None, list(codes1), None,
                                              model_name, tokenizer, args)
            sen_vec2 = get_sentence_embedding(None, list(codes2), None,
                                              model_name, tokenizer, args)
        else:                                           # HF 路径
            # batch: ids1, mask1, ids2, mask2, task, idx
            s_ids, s_mask, t_ids, t_mask, task_ids, idx_batch = batch
            idx_batch = idx_batch.tolist()              # 保留 cpu list

            s_ids, s_mask = s_ids.to(args.device), s_mask.to(args.device)
            t_ids, t_mask = t_ids.to(args.device), t_mask.to(args.device)

            with torch.no_grad():
                sen_vec1 = get_sentence_embedding(model, s_ids, s_mask,
                                                  model_name, tokenizer, args)
                sen_vec2 = get_sentence_embedding(model, t_ids, t_mask,
                                                  model_name, tokenizer, args)

        # ---- 正例余弦 ---- #
        cos_pos = F.cosine_similarity(sen_vec1, sen_vec2)  # [B]
        cos_right.extend(cos_pos.cpu().tolist())

        # ---- FN 记录 ---- #
        for k, sim in enumerate(cos_pos.cpu().tolist()):
            if sim < best_threshold:          # 真克隆没过阈
                false_neg.append((idx_batch[k], 1,  # code1
                                  idx_batch[k], 2,  # code2
                                  sim))

        # ---- 负例采样 (FP) ---- #
        B = len(sen_vec1)
        for i in range(B):
            neg_cnt = 0
            for j in range(B):
                if i == j or task_ids[i] == task_ids[j]:
                    continue

                # code1_i vs code1_j
                sim1 = F.cosine_similarity(sen_vec1[i], sen_vec1[j], dim=0).item()
                if sim1 >= best_threshold:
                    false_pos.append((idx_batch[i], 1, idx_batch[j], 1, sim1))
                # code1_i vs code2_j
                sim2 = F.cosine_similarity(sen_vec1[i], sen_vec2[j], dim=0).item()
                if sim2 >= best_threshold:
                    false_pos.append((idx_batch[i], 1, idx_batch[j], 2, sim2))
                cos_wrong.extend([sim1, sim2])  # 用于指标
                #cos_wrong.append(sim1)
                neg_cnt += 1
                if neg_cnt >= K:
                    break

        processed += B
        bar.set_postfix(done=processed)

    # ---------------- ③ 计算指标 ---------------- #
    tp = sum(s >= best_threshold for s in cos_right)
    fp = sum(s >= best_threshold for s in cos_wrong)
    fn = len(cos_right) - tp
    
    precision = tp / (tp + fp) if tp + fp else 0
    recall    = tp / (tp + fn) if tp + fn else 0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0         
    
    
    if args.local_rank in [-1, 0]:
        result_str = (f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
                      f"Test: recall={recall:.3f}, precision={precision:.3f}, "
                      f"F1={f1:.3f}, threshold={best_threshold:.2f}\n")
        with open(os.path.join(args.output_dir, "result"), "a+", encoding="utf-8") as f:
            f.write(result_str)
        with open("result.txt", "a+", encoding="utf-8") as f:
            f.write(result_str)
            pos_cnt = len(cos_right)         
            neg_cnt = len(cos_wrong) 
            f.write("Pos / Neg = %d / %d  (%.2f : 1)\n" % (pos_cnt, neg_cnt, pos_cnt/neg_cnt))

    
    if args.local_rank in [-1, 0]:
        logger.info("Test metrics using best threshold")
        for k, v in dict(recall=recall, precision=precision,
                         F1=f1, threshold=best_threshold).items():
            logger.info("  %s = %.4f", k, v)

    # ---------------- ④ 保存失败样例 ---------------- #
    fail_dir = os.path.join(args.output_dir, "failed_cases")
    for idx_a, side_a, idx_b, side_b, sim in false_neg:
        save_case(fail_dir, "FN",
                  idx_a, side_a, idx_b, side_b, sim,
                  eval_examples[idx_a], eval_examples[idx_b])
    for idx_a, side_a, idx_b, side_b, sim in false_pos:
        save_case(fail_dir, "FP",
                  idx_a, side_a, idx_b, side_b, sim,
                  eval_examples[idx_a], eval_examples[idx_b])

    logger.info("Mis-classified pairs saved to %s  (FN=%d, FP=%d)",
                fail_dir, len(false_neg), len(false_pos))

    # ---------------- ⑤ 阈值扫描曲线 ---------------- #
    if args.local_rank in [-1, 0]:
        thresh_log = os.path.join(args.output_dir, "thresholds_test.tsv")
        with open(thresh_log, "w") as tf:
            tf.write("threshold\trecall\tprecision\tF1\n")
            for i in range(1, 100):
                th = i / 100
                tp_ = sum(s >= th for s in cos_right)
                fp_ = sum(s >= th for s in cos_wrong)
                fn_ = len(cos_right) - tp_
                prec_ = tp_ / (tp_ + fp_) if tp_ + fp_ else 0
                rec_  = tp_ / (tp_ + fn_) if tp_ + fn_ else 0
                f1_   = 2*prec_*rec_/(prec_+rec_) if prec_+rec_ else 0
                tf.write(f"{th:.2f}\t{rec_:.4f}\t{prec_:.4f}\t{f1_:.4f}\n")


        logger.info("✔ Threshold curve written to %s", thresh_log)




# def test(args, model, tokenizer, collate_fn, logger, best_threshold=0.32):
#     if args.local_rank not in [-1, 0]:
#         return
#     model_name = args.model_name_or_path.lower()
#     if model:                       # OpenAI 路径下 model=None
#             model.eval()

#         # 1) 数据集
#     eval_examples = read_examples(args.test_filename)
#     eval_data     = prepare_dataset(
#         eval_examples, tokenizer, args.max_source_length, args
#     )
#     #eval_sampler = RandomSampler(eval_data) 
#     eval_sampler = fixed_sampler(eval_data,args.seed) 
#     #if args.local_rank == -1 else DistributedSampler(eval_data)

#     eval_loader  = DataLoader(
#         eval_data,
#         sampler   = eval_sampler,
#         batch_size= args.eval_batch_size,
#         collate_fn= collate_fn,
#         drop_last = False
#     )

#     if args.local_rank in [-1, 0]:
#         logger.info("\n***** Running test *****")
#         logger.info("  Num examples = %d", len(eval_examples))
#         logger.info("  Batch size   = %d", args.eval_batch_size)

#     # 2) 主循环
#     cos_right, cos_wrong = [], []
#     bar     = tqdm(eval_loader, total=len(eval_loader),
#                     desc="Test (OpenAI)" if args.openai_model else "Test (HF)",
#                     ncols=100)
#     processed = 0

#     for batch in bar:
#         # ------- 2.1 取句向量 -------
#         if args.openai_model:                      # GPT：batch 里的 ids 是 str
#             codes1, _, codes2, _, task_ids = batch
            
#             sen_vec1 = get_sentence_embedding(
#                 None, list(codes1), None, model_name, tokenizer, args
#             )
#             sen_vec2 = get_sentence_embedding(
#                 None, list(codes2), None, model_name, tokenizer, args
#             )
#         else:                                      # 本地 HF 模型
#             batch = tuple(t.to(args.device) for t in batch)
#             s_ids, s_mask, t_ids, t_mask, task_ids = batch
#             with torch.no_grad():
#                 sen_vec1 = get_sentence_embedding(
#                     model, s_ids, s_mask, model_name, tokenizer, args
#                 )
#                 sen_vec2 = get_sentence_embedding(
#                     model, t_ids, t_mask, model_name, tokenizer, args
#                 )

#         # ------- 2.2 正、负例余弦 -------
#         cos_right.extend(
#             torch.nn.functional.cosine_similarity(sen_vec1, sen_vec2).tolist()
#         )
#         K = 3

#         for i in range(len(sen_vec1)):
#             neg_count = 0
#             for j in range(len(sen_vec1)):
#                 if i==j or torch.equal(task_ids[i], task_ids[j]):
#                     continue
#                 cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec1[j]).item())
#                 cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec2[j]).item())
#                 neg_count += 1
#                 if neg_count > K:
#                     break

#         # ------- 2.3 进度条 -------
#         processed += len(sen_vec1)
#         bar.set_postfix(done=processed)

#     if not args.do_eval:
#         best_threshold = 0.32
#         logger.info("using default eval_threshold: %s", best_threshold)
#     if args.local_rank in [-1, 0]:
#         logger.info("using eval_threshold: %s", best_threshold)

#     # 主评估
#     tp = sum([1 for s in cos_right if s >= best_threshold])
#     fp = sum([1 for s in cos_wrong if s >= best_threshold])
#     fn = len(cos_right) - tp

#     precision = tp / (tp + fp) if tp + fp > 0 else 0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

#     logger.info("Test metrics using best threshold")
#     result = {'recall': recall, 'precision': precision, 'F1': f1, 'threshold': best_threshold}
#     for key in sorted(result.keys()):
#         if args.local_rank in [-1, 0]:
#             logger.info("  %s = %s", key, str(result[key]))

#     # 保存 threshold 曲线
#     if args.local_rank in [-1, 0]:
#         result_str = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test: recall={recall:.3f}, precision={precision:.3f}, F1={f1:.3f}, threshold={best_threshold:.2f}\n"
#         with open(os.path.join(args.output_dir, 'result'), 'a+') as f:
#             f.write(result_str)
#         with open('result.txt', 'a+') as f:
#             f.write(result_str)
#     threshold_log_path = os.path.join(args.output_dir, "thresholds_test.tsv")
#     with open(threshold_log_path, "w") as tf:
#         tf.write("threshold\trecall\tprecision\tF1\n")
#         for i in range(1, 100):
#             threshold = i / 100
#             tp = sum([1 for s in cos_right if s >= threshold])
#             fp = sum([1 for s in cos_wrong if s >= threshold])
#             fn = len(cos_right) - tp

#             precision = tp / (tp + fp) if tp + fp > 0 else 0
#             recall = tp / (tp + fn) if tp + fn > 0 else 0
#             f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
#             tf.write(f"{threshold:.2f}\t{recall:.4f}\t{precision:.4f}\t{f1:.4f}\n")