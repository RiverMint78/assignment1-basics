import math
from collections.abc import Callable

import torch


class SGD(torch.optim.Optimizer):
    """Example from `cs336_assignment1_basics.pdf`"""

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]  # Get state associated with p.
            t = state.get("t", 0)  # Get iteration number from the state, or 0.
            grad = p.grad.data  # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
            state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= (beta1 := betas[0]) < 1.0:
            raise ValueError(f"Invalid beta1 (betas[0]) parameter: {betas[0]}")
        if not 0.0 <= (beta2 := betas[1]) < 1.0:
            raise ValueError(f"Invalid beta2 (betas[1]) parameter: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1: float = group["beta1"]
            beta2: float = group["beta2"]
            eps: float = group["eps"]
            lr: float = group["lr"]
            weight_decay: float = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad: torch.Tensor = p.grad
                state: dict = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                state["step"] += 1
                t: int = state["step"]
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1: float = 1.0 - beta1**t
                bias_correction2: float = 1.0 - beta2**t
                step_size: float = lr / bias_correction1
                denom: torch.Tensor = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1.0)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
