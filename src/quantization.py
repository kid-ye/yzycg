"""
Model Quantization — Section V-A
Equations 7 & 8: convert FP weights to fixed-point integers, then
replace multiplications with bit-shift + addition operations.
"""
import torch
import torch.nn as nn
import copy


# ---------------------------------------------------------------------------
# Equation 7:  Y = round(2^n * X)   →   X ≈ Y / 2^n
# ---------------------------------------------------------------------------

def quantize_tensor(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Quantize a floating-point tensor to fixed-point with n fractional bits.
    Returns the integer tensor Y such that X ≈ Y / 2^n.
    """
    scale = 2 ** n
    return torch.round(x * scale).to(torch.int32)


def dequantize_tensor(y: torch.Tensor, n: int) -> torch.Tensor:
    """Recover approximate float from quantized integer: X ≈ Y / 2^n."""
    return y.to(torch.float32) / (2 ** n)


# ---------------------------------------------------------------------------
# Equation 8:  N × X  →  bit-shift + add
# Decompose Y (integer) into powers-of-two, then apply shifts to N.
# ---------------------------------------------------------------------------

def bitshift_multiply(N: torch.Tensor, Y: torch.Tensor, n: int) -> torch.Tensor:
    """
    Compute N * (Y / 2^n) using only bit-shifts and additions (Equation 8).

    N : input activation  (float tensor)
    Y : quantized weight  (int32 tensor, same shape as N after broadcasting)
    n : quantization bits

    For each non-zero bit position a in Y:
        contribution = N * 2^(a - n)
                     = N << (a-n)  if a-n > 0
                     = N >> (n-a)  if a-n < 0
    """
    result = torch.zeros_like(N, dtype=torch.float32)
    Y_int = Y.to(torch.int32)

    # Handle sign: decompose |Y|, then apply sign at the end
    sign = torch.sign(Y_int).float()
    Y_abs = Y_int.abs()

    max_bits = int(Y_abs.max().item()).bit_length() if Y_abs.numel() > 0 else 0

    for a in range(max_bits):
        bit_mask = (Y_abs >> a) & 1          # 1 where bit a is set
        shift = a - n
        if shift >= 0:
            contribution = N * (2 ** shift)  # left shift
        else:
            contribution = N / (2 ** (-shift))  # right shift (arithmetic)
        result += contribution * bit_mask.float() * sign

    return result


# ---------------------------------------------------------------------------
# Quantized Conv1d layer — drops in as a replacement for nn.Conv1d
# ---------------------------------------------------------------------------

class QuantizedConv1d(nn.Module):
    """
    Conv1d whose weights are stored as fixed-point integers (Equation 7).
    Forward pass uses standard float arithmetic (dequantize on-the-fly).
    For true hardware deployment, replace with bitshift_multiply.
    """
    def __init__(self, conv: nn.Conv1d, n: int):
        super().__init__()
        self.n = n
        self.stride   = conv.stride
        self.padding  = conv.padding
        self.dilation = conv.dilation
        self.groups   = conv.groups

        # Store quantized integer weights (Equation 7)
        self.register_buffer('weight_q', quantize_tensor(conv.weight.data, n))
        if conv.bias is not None:
            self.register_buffer('bias_q', quantize_tensor(conv.bias.data, n))
        else:
            self.bias_q = None

    def forward(self, x):
        # Dequantize for float conv (Equation 7 inverse)
        w = dequantize_tensor(self.weight_q, self.n)
        b = dequantize_tensor(self.bias_q,   self.n) if self.bias_q is not None else None
        return nn.functional.conv1d(x, w, b,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    groups=self.groups)


class QuantizedLinear(nn.Module):
    """Linear layer with fixed-point weights."""
    def __init__(self, linear: nn.Linear, n: int):
        super().__init__()
        self.n = n
        self.register_buffer('weight_q', quantize_tensor(linear.weight.data, n))
        if linear.bias is not None:
            self.register_buffer('bias_q', quantize_tensor(linear.bias.data, n))
        else:
            self.bias_q = None

    def forward(self, x):
        w = dequantize_tensor(self.weight_q, self.n)
        b = dequantize_tensor(self.bias_q,   self.n) if self.bias_q is not None else None
        return nn.functional.linear(x, w, b)


# ---------------------------------------------------------------------------
# Apply quantization to a full model
# ---------------------------------------------------------------------------

def quantize_model(model: nn.Module, n: int) -> nn.Module:
    """
    Return a deep copy of model with all Conv1d and Linear layers replaced
    by their quantized equivalents at n fractional bits (Equation 7).
    """
    q_model = copy.deepcopy(model)

    def _replace(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv1d):
                setattr(module, name, QuantizedConv1d(child, n))
            elif isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child, n))
            else:
                _replace(child)

    _replace(q_model)
    q_model.eval()
    return q_model


# ---------------------------------------------------------------------------
# Utility: compare model size before/after quantization
# ---------------------------------------------------------------------------

def model_size_bytes(model: nn.Module) -> int:
    return sum(
        p.numel() * p.element_size()
        for p in list(model.parameters()) + list(model.buffers())
    )


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.model import TripletECGModel

    model = TripletECGModel().eval()

    print(f"Original size : {model_size_bytes(model) / 1024:.1f} KB")

    for n in [4, 6, 8]:
        q = quantize_model(model, n=n)
        # Verify forward pass shape
        x = torch.randn(1, 1, 1000)
        with torch.no_grad():
            out = q.encoder(x)
        print(f"n={n:2d} bits | quantized size: {model_size_bytes(q)/1024:.1f} KB | "
              f"output shape: {tuple(out.shape)}")
