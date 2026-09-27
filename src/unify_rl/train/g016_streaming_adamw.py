"""Exact AdamW with moments durably on CPU and one-parameter GPU staging."""
from __future__ import annotations
import math
import torch

def normalize_g016_streaming_state(optimizer):
    """Restore the FP32-CPU state contract after Optimizer.load_state_dict."""
    owner=getattr(optimizer,'optimizer',optimizer);tensor_count=0
    for state in owner.state.values():
        for key,value in list(state.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                state[key]=value.detach().to(device='cpu',dtype=torch.float32);tensor_count+=1
    return {'version':'clean29529_g016_streaming_state_normalization_v1','state_count':len(owner.state),'tensor_count':tensor_count,'device':'cpu','dtype':'torch.float32'}

def materialize_g016_streaming_state(optimizer):
    owner=getattr(optimizer,'optimizer',optimizer);count=0
    for group in owner.param_groups:
        for parameter in group['params']:
            state=owner.state[parameter]
            if not state:
                state['step']=torch.tensor(0.0,dtype=torch.float32,device='cpu')
                state['master_param']=parameter.detach().float().cpu().clone()
                state['exp_avg']=torch.zeros(parameter.shape,dtype=torch.float32,device='cpu')
                state['exp_avg_sq']=torch.zeros(parameter.shape,dtype=torch.float32,device='cpu')
                if group.get('amsgrad',False):state['max_exp_avg_sq']=torch.zeros(parameter.shape,dtype=torch.float32,device='cpu')
            count+=1
    return {'version':'clean29529_g016_bf16_policy_fp32_cpu_master_state_v1','parameter_count':count,'state_device':'cpu','master_dtype':'torch.float32','moment_dtype':'torch.float32'}

class G016StreamingCPUAdamW(torch.optim.AdamW):
    """AdamW-equivalent update without materializing all moments on GPU."""
    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            beta1,beta2=group['betas'];lr=float(group['lr']);eps=float(group['eps']);wd=float(group['weight_decay']);maximize=bool(group.get('maximize',False));amsgrad=bool(group.get('amsgrad',False))
            for parameter in group['params']:
                grad=parameter.grad
                if grad is None:continue
                if grad.is_sparse:raise RuntimeError('G016 streaming AdamW forbids sparse gradients')
                state=self.state[parameter]
                if not state:
                    state['step']=torch.tensor(0.0,dtype=torch.float32,device='cpu')
                    state['master_param']=parameter.detach().float().cpu().clone()
                    state['exp_avg']=torch.zeros(parameter.shape,dtype=torch.float32,device='cpu')
                    state['exp_avg_sq']=torch.zeros(parameter.shape,dtype=torch.float32,device='cpu')
                    if amsgrad:state['max_exp_avg_sq']=torch.zeros(parameter.shape,dtype=torch.float32,device='cpu')
                state['step'].add_(1);step=float(state['step'].item());g=grad.detach().float();g=-g if maximize else g
                exp_avg=state['exp_avg'].to(device=parameter.device,dtype=torch.float32,non_blocking=False);exp_avg_sq=state['exp_avg_sq'].to(device=parameter.device,dtype=torch.float32,non_blocking=False)
                exp_avg.mul_(beta1).add_(g,alpha=1-beta1);exp_avg_sq.mul_(beta2).addcmul_(g,g,value=1-beta2)
                if amsgrad:
                    maximum=state['max_exp_avg_sq'].to(device=parameter.device,dtype=torch.float32,non_blocking=False);torch.maximum(maximum,exp_avg_sq,out=maximum);denominator=maximum.sqrt().div_(math.sqrt(1-beta2**step)).add_(eps);state['max_exp_avg_sq'].copy_(maximum.cpu());del maximum
                else:denominator=exp_avg_sq.sqrt().div_(math.sqrt(1-beta2**step)).add_(eps)
                master=state['master_param'].to(device=parameter.device,dtype=torch.float32,non_blocking=False)
                if wd:master.mul_(1-lr*wd)
                master.addcdiv_(exp_avg,denominator,value=-lr/(1-beta1**step))
                parameter.copy_(master.to(dtype=parameter.dtype))
                state['master_param'].copy_(master.cpu());state['exp_avg'].copy_(exp_avg.cpu());state['exp_avg_sq'].copy_(exp_avg_sq.cpu());del master,exp_avg,exp_avg_sq,denominator
        return loss
