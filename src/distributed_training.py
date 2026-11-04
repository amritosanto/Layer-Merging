"""Reference-style synchronous replicas: weighted local losses, summed scaled gradients."""
import copy
import torch
from torch import nn
from .objectives import training_loss

class PreservationWorker(nn.Module):
    def __init__(self,teacher,source,merger,cfg,device):
        super().__init__(); self.teacher=teacher; self.source=source; self.merger=merger; self.cfg=cfg; self.device=device
    def forward(self,ids,pairs):
        with torch.cuda.device(self.device),torch.autocast('cuda',dtype=torch.float16):
            return training_loss(self.teacher,self.source,self.merger,ids,pairs,self.cfg)

class DualWorkers:
    def __init__(self,teacher,source,merger,cfg):
        self.workers=[PreservationWorker(teacher,source,merger,cfg,torch.device('cuda:0'))]
        if not cfg.single_gpu_debug:
            self.workers.append(PreservationWorker(copy.deepcopy(teacher).to('cuda:1'),copy.deepcopy(source).to('cuda:1'),
                                                  copy.deepcopy(merger).to('cuda:1'),cfg,torch.device('cuda:1')))

    @torch.no_grad()
    def sync(self):
        primary=self.workers[0].merger
        for worker in self.workers[1:]:
            flat=torch.cat([p.detach().reshape(-1) for p in primary.parameters()]).to(worker.device)
            offset=0
            for target in worker.merger.parameters():
                target.copy_(flat[offset:offset+target.numel()].view_as(target)); offset+=target.numel()
            for target,original in zip(worker.merger.buffers(),primary.buffers()):target.copy_(original)
            worker.merger.train(primary.training)

    def clear(self):
        for worker in self.workers:worker.merger.zero_grad(set_to_none=True)

    def loss(self,ids,pairs):
        shards=torch.tensor_split(ids,len(self.workers)); active=[i for i,s in enumerate(shards) if len(s)]
        outputs=torch.nn.parallel.parallel_apply([self.workers[i] for i in active],
                  [(shards[i].to(self.workers[i].device),pairs) for i in active],devices=active)
        weights=[len(shards[i])/len(ids) for i in active]
        loss=sum(w*o[0].to('cuda:0') for w,o in zip(weights,outputs))
        logs={k:sum(w*o[1][k].to('cuda:0') for w,o in zip(weights,outputs)) for k in outputs[0][1]}
        return loss,logs,[len(s) for s in shards]

    @torch.no_grad()
    def reduce(self):
        # Each local loss was weighted by shard / global batch; SUM, not another average.
        for worker in self.workers[1:]:
            flat=torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).reshape(-1)
                            for q in worker.merger.parameters()]).to(self.workers[0].device)
            offset=0
            for p in self.workers[0].merger.parameters():
                value=flat[offset:offset+p.numel()].view_as(p); offset+=p.numel()
                if p.grad is None:p.grad=value.clone()
                else:p.grad.add_(value)

    def synchronize(self):
        for worker in self.workers:torch.cuda.synchronize(worker.device)
