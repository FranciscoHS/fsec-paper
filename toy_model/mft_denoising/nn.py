import torch
import torch.nn as nn
import torch.nn.functional as F


class JumpReLUFunction(torch.autograd.Function):
    """
    JumpReLU: y = x * 1[x > theta].

    Backward:
      - dL/dx uses the gate as a multiplicative mask (gated identity STE).
      - dL/dtheta uses a rectangular-kernel pseudo-derivative around theta
        with bandwidth eps. This mirrors the JumpReLU-SAE recipe
        (Rajamanoharan et al.) and lets a learnable per-neuron threshold move.
    """

    @staticmethod
    def forward(ctx, x, theta, eps):
        gate = (x > theta).to(x.dtype)
        ctx.save_for_backward(x, theta, gate)
        ctx.eps = float(eps)
        return x * gate

    @staticmethod
    def backward(ctx, grad_out):
        x, theta, gate = ctx.saved_tensors
        eps = ctx.eps
        grad_x = grad_out * gate
        grad_theta = None
        if ctx.needs_input_grad[1]:
            kernel = ((x - theta).abs() < eps).to(x.dtype) / (2.0 * eps)
            grad_per_sample = -grad_out * x * kernel  # (..., H)
            # Reduce all leading (batch) dims down to theta's shape.
            while grad_per_sample.dim() > theta.dim():
                grad_per_sample = grad_per_sample.sum(dim=0)
            grad_theta = grad_per_sample
        return grad_x, grad_theta, None


class JumpReLU(nn.Module):
    """JumpReLU activation with optional learnable per-neuron threshold."""

    def __init__(self, hidden_size: int, theta_init: float = 0.1,
                 learn_threshold: bool = False, eps: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.learn_threshold = learn_threshold
        init = torch.full((hidden_size,), float(theta_init))
        if learn_threshold:
            self.theta = nn.Parameter(init)
        else:
            self.register_buffer("theta", init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return JumpReLUFunction.apply(x, self.theta, self.eps)

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, theta_init~={float(self.theta.mean()):.3f}, "
                f"learn_threshold={self.learn_threshold}, eps={self.eps}")


class Tanh3(nn.Module):
    """f(x) = tanh(x)^3 — original activation in this project."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) ** 3


def make_activation(kind: str, hidden_size: int, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind == "tanh3":
        return Tanh3()
    if kind == "jumprelu":
        return JumpReLU(
            hidden_size=hidden_size,
            theta_init=kwargs.get("theta_init", 0.1),
            learn_threshold=kwargs.get("learn_threshold", False),
            eps=kwargs.get("eps", 0.1),
        )
    raise ValueError(f"Unknown activation: {kind}")


class TwoLayerNet(nn.Module):
    """Encoder/decoder two-layer net with switchable hidden activation."""

    def __init__(
        self,
        input_size: int = 1000,
        hidden_size: int = 256,
        encoder_initialization_scale: float = 1.0,
        decoder_initialization_scale: float = 1.0,
        activation: str = "tanh3",
        activation_kwargs: dict | None = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

        with torch.no_grad():
            nn.init.xavier_normal_(self.fc1.weight)
            self.fc1.weight.mul_(encoder_initialization_scale)
            if self.fc1.bias is not None:
                self.fc1.bias.zero_()
            nn.init.xavier_normal_(self.fc2.weight)
            self.fc2.weight.mul_(decoder_initialization_scale)
            if self.fc2.bias is not None:
                self.fc2.bias.zero_()

        self.activation_kind = activation
        self.act = make_activation(activation, hidden_size, **(activation_kwargs or {}))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        return self.act(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out.squeeze(0) if squeeze else out


if __name__ == "__main__":
    for kind in ("tanh3", "jumprelu"):
        m = TwoLayerNet(input_size=1000, hidden_size=256, activation=kind)
        y = m(torch.randn(4, 1000))
        print(kind, y.shape)
