import torch
import torch.distributed as dist


class AverageMeter:

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    def update_sum(self, total, n):
        self.total += float(total)
        self.count += n

    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0

    def sync(self):
        #把所有进程的 total 和 count 加起来，外面不需要知道自己跑在几张卡上
        if not (dist.is_available() and dist.is_initialized()):
            return self

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        packed = torch.tensor([self.total, float(self.count)], dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        self.total = float(packed[0])
        self.count = int(packed[1])
        return self
