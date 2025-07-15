import json
import torch
import torch.nn.functional as F
class Example:
    #def __init__(self, code1: str, code2: str, task: int, lang1: str, lang2: str):
    def __init__(self, idx,code1: str, code2: str, task: int , desc1: str = "", desc2: str = ""):
        self.idx   = idx
        self.code1 = code1
        self.code2 = code2
        self.task = task
        self.desc1 = desc1
        self.desc2 = desc2


def read_examples(filename):
    """读取 jsonl 格式样本，返回 Example 列表"""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for  i, line in enumerate(f):
            # if i > 200:
            #     break

            js = json.loads(line)
            code1 = ' '.join(js['Code1'].strip().split())
            code2 = ' '.join(js['Code2'].strip().split())
            desc1 = ' '.join(js.get("Code1_desc", "").split())
            desc2 = ' '.join(js.get("Code2_desc", "").split())
            task = int(js['Task'])
            examples.append(Example(i,code1, code2, task, desc1, desc2))
    return examples

def convert_examples_to_features(examples, tokenizer, max_length,args = None):
    features = []
    concat = bool(args and getattr(args, "concat_desc_code", False))
    model_name = args.model_name_or_path.lower()
    use_gpt = bool(args and args.openai_model)
    if not use_gpt:
        if "unixcoder" in model_name:
            sep_tok = tokenizer.sep_token or "<sep>"        # UniXcoder 自带 <sep>
            prefix, suffix = "", ""                         # 无前后缀
        elif "codet5p" in model_name:
            # CodeT5+ 专用 special tokens
            sep_tok = "<sep>"
            prefix, suffix = "<encoder-only> ", ""
        else:                                               
            sep_tok = tokenizer.sep_token or "[SEP]"
            prefix, suffix = "", ""
        sep_id   = tokenizer.convert_tokens_to_ids(sep_tok)
        pref_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        suf_ids  = tokenizer(suffix, add_special_tokens=False)["input_ids"] if suffix else []
        #trunc = args.slide_window_size == 0
        trunc = True
    for example in examples:
        txt1, txt2 = example.code1, example.code2
        if use_gpt:                             # ===== GPT 分支 =====
            # 直接存原始字符串；attention_mask 用 None 占位
            features.append((txt1, None,
                             txt2, None,
                             example.task,
                             example.idx
                             ))
            continue
        if concat:
            #txt1 = f"{prefix}{example.desc1} {sep_tok} {example.code1}{suffix}".strip()
            #txt2 = f"{prefix}{example.desc2} {sep_tok} {example.code2}{suffix}".strip()
            desc1_ids = tokenizer(
                example.desc1, add_special_tokens=False, truncation=False
            )["input_ids"]
            desc2_ids = tokenizer(
                example.desc2, add_special_tokens=False, truncation=False
            )["input_ids"]
            code1_ids = tokenizer(
                example.code1, add_special_tokens=False,
                truncation=True, max_length=max_length
            )["input_ids"]
            code2_ids = tokenizer(
                example.code2, add_special_tokens=False,
                truncation=True, max_length=max_length
            )["input_ids"]
            ids1 = pref_ids + desc1_ids + [sep_id] + code1_ids + suf_ids
            ids2 = pref_ids + desc2_ids + [sep_id] + code2_ids + suf_ids
            mask1 = [1] * len(ids1)
            mask2 = [1] * len(ids2)
        else:
            #txt1, txt2 = example.code1, example.code2
            enc1 = tokenizer(txt1, truncation=trunc, padding=False,
                        max_length=max_length, return_tensors="pt")
            enc2 = tokenizer(txt2, truncation=trunc, padding=False,
                            max_length=max_length, return_tensors="pt")
            
            ids1, mask1 = enc1["input_ids"], enc1["attention_mask"]
            ids2, mask2 = enc2["input_ids"], enc2["attention_mask"]
           
            
        
        features.append((
            torch.tensor(ids1), torch.tensor(mask1),
            torch.tensor(ids2), torch.tensor(mask2),
            example.task,
            example.idx
            
        ))
        
        
    return features

def prepare_dataset(examples, tokenizer, max_length, args):
    feats = convert_examples_to_features(examples, tokenizer, max_length, args)
    ids1, m1, ids2, m2, tasks,idxs = [], [], [], [], [], []
    for f in feats:
        if isinstance(f[0], str):          # GPT 保存字符串
            ids1.append(f[0]); m1.append(None)
            ids2.append(f[2]); m2.append(None)
        else:                              # 本地模型保存 Tensor
            ids1.append(f[0].squeeze(0));  m1.append(f[1].squeeze(0))
            ids2.append(f[2].squeeze(0));  m2.append(f[3].squeeze(0))
        tasks.append(torch.tensor(f[4]))
        idxs.append(torch.tensor(f[5]))
    return list(zip(ids1, m1, ids2, m2, tasks,idxs))



def make_collate_fn(tokenizer):
        pad_id = tokenizer.pad_token_id if tokenizer else 0

        def collate_fn(batch):
            
            ids1, m1, ids2, m2, task , idx = zip(*batch)
            if isinstance(ids1[0], str):
                return list(ids1), None, list(ids2), None, torch.tensor(task).long(),torch.tensor(idx).long()
            # —— 统一到当前 batch 的最长长度 ——
            def pad_to_max(seq_list, pad_val):
                L = max(x.size(0) for x in seq_list)
                out = []
                for x in seq_list:
                    pad_len = L - x.size(0)
                    out.append(F.pad(x, (0, pad_len), value=pad_val))
                return torch.stack(out)
            ids1 = pad_to_max(ids1, pad_id)
            m1   = pad_to_max(m1,   0)
            ids2 = pad_to_max(ids2, pad_id)
            m2   = pad_to_max(m2,   0)
            task = torch.tensor(task).long()
            idx  = torch.tensor(idx).long()
            return ids1, m1, ids2, m2, task,idx

        return collate_fn 