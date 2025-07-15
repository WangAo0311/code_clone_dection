import torch
import torch.nn.functional as F
def nt_xent_loss(z1, z2, task_ids, tau=0.1):
    """
    SimCLR / NT-Xent，保留同 task 的正例，过滤其余同 task。
    """
    device, B = z1.device, z1.size(0)
    z   = torch.cat([z1, z2], dim=0)                 # [2B, H]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / tau

    # 正例位置：i ↔ (i±B)
    pos_index = torch.arange(2*B, device=device)
    pos_index = (pos_index + B) % (2*B)              # [2B]

    # —— 构造 mask ——  
    eye_mask   = torch.eye(2*B, dtype=torch.bool, device=device)
    task_all   = torch.cat([task_ids, task_ids], dim=0)  # [2B]
    same_task  = task_all.unsqueeze(0) == task_all.unsqueeze(1)

    # 仅屏蔽 (a) 自身，(b) 同 task & 不是正例
    invalid_mask = eye_mask | (same_task & ~(torch.nn.functional.one_hot(pos_index, 2*B).bool()))

    sim = sim.masked_fill(invalid_mask, -1e4)

    # 交叉熵
    labels = pos_index
    return F.cross_entropy(sim, labels)