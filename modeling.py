import torch
import torch.nn as nn
import openai 
import tiktoken
import logging
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.nn.parallel import DistributedDataParallel as DDP
logger = logging.getLogger()
class AttnPooling(nn.Module):
    """
    单层自注意力池化：先把窗口向量映射到同维 Query，再做 Softmax 权重求和。
    """
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):                 # x: [B, N_win, hidden]
        w = (self.q(x) * x).sum(-1) / x.size(-1)**0.5   # [B, N_win]
        w = torch.softmax(w, dim=-1).unsqueeze(-1)      # [B, N_win, 1]
        
        
        return (x * w).sum(1)                           # [B, hidden]
# def sliding_windows(input_ids, window, stride):
#     """把 [B, L] 拆成 List[[B, win_len]]，不足 window 的尾部也保留"""
#     bsz, seqlen = input_ids.shape
#     idx = list(range(0, seqlen, stride))
#     splits = [input_ids[:, i:i+window] for i in idx]
#     return splits    

def sliding_windows(t, window, stride, pad_token_id):
    """
    把张量 [B, L] 拆成等长窗口 List[Tensor[B, window]]。
    尾块不足 window 时右侧补 pad_token_id。
    """
    bsz, seqlen = t.size()
    windows = []
    for start in range(0, seqlen, stride):
        end = start + window
        chunk = t[:, start:end]                       # [B, <=window]
        if chunk.size(1) < window:                    # 末尾补 pad
            pad_len = window - chunk.size(1)
            pad = t.new_full((bsz, pad_len), pad_token_id)
            chunk = torch.cat([chunk, pad], dim=1)
        windows.append(chunk)
        if end >= seqlen:
            break
    return windows


def get_sentence_embedding(model, input_ids, attention_mask, model_name, tokenizer=None, args = None):
    """
    自动适配不同模型结构和封装方式，返回句向量表示（如 [CLS] embedding）

    参数说明：
    - model: 当前模型（可能被 DataParallel 或 DDP 包装）
    - input_ids: 输入 token ids [batch_size, seq_len]
    - attention_mask: attention mask [batch_size, seq_len]
    - model_name: 模型名称（如 "codet5p", "longformer", "unixcoder"...）
    返回：
    - Tensor: 句向量 [batch_size, hidden_size]
    """
    if args.openai_model:
        enc = tiktoken.encoding_for_model(args.openai_model)  
        texts = input_ids 
        MAX_TOK = 8191
        def truncate(txt: str) -> str:
            ids = enc.encode(txt)
            if len(ids) <= MAX_TOK:
                return txt
            # 超长就截断再 decode 回字符串
            return enc.decode(ids[:MAX_TOK])
        texts = [truncate(t) for t in texts]    
        # 分批请求 OpenAI（batch 默认 1）
        vecs = []
        for i in range(0, len(texts), args.openai_batch):
            chunk = texts[i:i + args.openai_batch]
            rsp   = openai.embeddings.create(model=args.openai_model,
                                             input=chunk,
                                             encoding_format="float")
            vecs.extend(d.embedding for d in rsp.data)

        return torch.tensor(vecs, dtype=torch.float32,
                            device=args.device if hasattr(args, "device") else "cpu")
    base_model = model.module if hasattr(model, "module") else model
    
    if args.slide_window_size == 0 or input_ids.size(1) <= args.slide_window_size:
        if "codet5p" in model_name:
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]
        elif "unixcoder" in model_name:
            # outputs = base_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # return outputs[0][:, 0, :]  
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]

        else:

            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]
    
    try:

        win    = args.slide_window_size
        stride = args.slide_stride or win // 2
        pad_id = tokenizer.pad_token_id if tokenizer else 0

        id_pieces   = sliding_windows(input_ids,     win, stride, pad_id)
        mask_pieces = sliding_windows(attention_mask, win, stride, 0)
        CHUNK = getattr(args, "window_batch", 4)
        emb_list = []
        B = input_ids.size(0)
        
        
        for i in range(0, len(id_pieces), CHUNK):
            # --------- 2.1 组一个“微批” ----------
            slice_ids   = id_pieces[i : i + CHUNK]   # list, 每个 [B, win]
            slice_masks = mask_pieces[i : i + CHUNK]

            # ❷ 组合成微批张量  →  [CHUNK*B, win]
            ids_batch  = torch.stack(slice_ids , 0).flatten(0, 1)
            mask_batch = torch.stack(slice_masks, 0).flatten(0, 1)
            # --------- 2.2 前向 ----------
            if model.training:
                out = base_model(input_ids=ids_batch, attention_mask=mask_batch)
            else:
                with torch.no_grad():
                    out = base_model(input_ids=ids_batch, attention_mask=mask_batch)

            # --------- 2.3 拆回每个窗口 ----------
            # 先 [CHUNK*B, hidden] → [CHUNK, B, hidden]
            out_windows = out.last_hidden_state[:, 0, :].view(-1, B, out.last_hidden_state.size(-1))
            emb_list.extend([w for w in out_windows])   # 追加 CHUNK 个 [B, h]

        if not emb_list:
            raise RuntimeError("⚠️ emb_list is empty! Check slide_window_size/stride.")

        # --------- ③ Pool over windows ----------
        emb_stack = torch.stack(emb_list, dim=1)        # [B, N_win, hidden]

        if args.pooling == "mean":
            return emb_stack.mean(dim=1)
        elif args.pooling == "attention":
            return base_model.attnpool(emb_stack)
        else:                                           # 'cls'
            logger.warning(f"⚠️ Unknown pooling type '{args.pooling}', fallback to CLS pooling.")
            return emb_stack[:, 0, :]
     
    except RuntimeError as e:
        if "out of memory" in str(e):
            max_len = 512
            input_ids_trunc     = input_ids[:, :max_len]
            attention_mask_trunc = attention_mask[:, :max_len]
            logger.info(f"⚠️ OOM in get_sentence_embedding: seq_len={input_ids.size(1)}, error={repr(e)}")
            outputs = base_model(
                input_ids=input_ids_trunc,
                attention_mask=attention_mask_trunc
            )
            return outputs.last_hidden_state[:, 0, :]
        else:
            logger.error(f"❌ Unexpected RuntimeError in get_sentence_embedding: {e}")
            raise




def load_model(args, model_class, config, tokenizer):
    model_name = args.model_name_or_path.lower()
    
    if "codet5p" in model_name:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.add_special_tokens({
        "additional_special_tokens": ["<encoder-only>", "<sep>"]})
        model.resize_token_embeddings(len(tokenizer))

    elif "longformer" in model_name:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.pooling == "attention":
        hidden = config.hidden_size
        model.add_module("attnpool", AttnPooling(hidden))

    if args.load_model_path is not None:
        if args.local_rank in [-1, 0]:
            logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path, map_location="cpu"))
    model.to(args.device)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    return model