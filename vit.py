"""Our tiny streaming Conformer ("ViT" below): waveform -> one logit vector per 64 ms buffer.

    waveform [B, N], 16 kHz, int16 scale
      -> front end: STFT restarted at every 1024-sample buffer (256 points, hop 128: 7 frames x 129
         bins), bins below 150 Hz zeroed, learned per-bin PCEN
      -> stem: ReLU(Conv2d(1 -> 8)) over (frequency, time), one token per `time_stride` frames
      -> Linear to embed_dim, then per layer: convolution module (depthwise k 7 over tokens +
         pointwise, residual, LayerNorm), then nn.TransformerEncoderLayer (post-LN) whose
         attention carries an ALiBi bias
      -> per-token logits, averaged over the 7 frames of each buffer

causal=True (streamable, what the AudioMoth runs): the stem and convolutions look only at the past,
the attention at the last `context` tokens. causal=False (float reference): centred convolutions,
attention over the whole input with a symmetric ALiBi bias. The firmware C (firmware/) computes the
causal model token by token; export.py writes its weights.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

BUFFER = 1024            # samples per buffer (64 ms at 16 kHz), the AudioMoth's DMA transfer
N_FFT, HOP = 256, 128    # 7 frames per buffer
HIGHPASS_HZ = 150.0
SAMPLE_RATE = 16000


class PCEN(nn.Module):
    """Per-bin learned PCEN (Wang et al. 2017) on the STFT power E:
    (E / (M + eps)^alpha + delta)^r - delta^r, M[t] = (1 - s) M[t - 1] + s E[t]; alpha and s through a
    sigmoid, delta and r through a softplus, initialised at alpha 0.98, delta 2, r 0.5, s for a ~3 s
    time constant. M starts from m_init (e.g. the mean training power, as the firmware) or E[0]."""

    EPS = 1e-6

    def __init__(self, bins: int):
        super().__init__()
        inv_sigmoid = lambda y: math.log(y / (1 - y))  # noqa: E731
        inv_softplus = lambda y: math.log(math.expm1(y))  # noqa: E731
        self.raw_alpha = nn.Parameter(torch.full((bins,), inv_sigmoid(0.98)))
        self.raw_smooth = nn.Parameter(torch.full((bins,), inv_sigmoid(2 / 376)))  # 375 frames of 8 ms
        self.raw_delta = nn.Parameter(torch.full((bins,), inv_softplus(2.0)))
        self.raw_r = nn.Parameter(torch.full((bins,), inv_softplus(0.5)))

    def values(self) -> dict[str, torch.Tensor]:
        return {"alpha": torch.sigmoid(self.raw_alpha), "smooth": torch.sigmoid(self.raw_smooth),
                "delta": F.softplus(self.raw_delta), "r": F.softplus(self.raw_r)}

    def forward(self, power: torch.Tensor, m_init: torch.Tensor | None = None) -> torch.Tensor:
        """power [B, bins, T] -> [B, bins, T]."""
        v = {k: x.unsqueeze(-1) for k, x in self.values().items()}
        # M by a parallel prefix scan of the affine recurrence M[t] = a M[t - 1] + b[t]
        a = (1 - v["smooth"]).expand_as(power).clone()
        b = v["smooth"] * power
        if m_init is None:
            a[..., 0], b[..., 0] = 0, power[..., 0]
        d = 1
        while d < power.shape[-1]:
            a, b = torch.cat([a[..., :d], a[..., d:] * a[..., :-d]], -1), torch.cat([b[..., :d], a[..., d:] * b[..., :-d] + b[..., d:]], -1)
            d *= 2
        m = b if m_init is None else a * m_init.unsqueeze(-1) + b
        return (power / (m + self.EPS).pow(v["alpha"]) + v["delta"]).pow(v["r"]) - v["delta"].pow(v["r"])


class FrontEnd(nn.Module):
    """Waveform [B, N] -> PCEN spectrogram [B, 129, 7 * (N // BUFFER)], the STFT restarted at every
    buffer (no frame spans two buffers, as on the device)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT, periodic=False), persistent=False)
        mask = torch.ones(N_FFT // 2 + 1, 1)
        mask[:math.ceil(HIGHPASS_HZ / (SAMPLE_RATE / N_FFT))] = 0
        self.register_buffer("mask", mask, persistent=False)
        self.pcen = PCEN(N_FFT // 2 + 1)

    def power(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[-1] // BUFFER
        x = waveform[:, :n * BUFFER].reshape(-1, BUFFER)
        mag = torch.stft(x, N_FFT, HOP, window=self.window, center=False, return_complex=True).abs()
        mag = mag.reshape(waveform.shape[0], n, *mag.shape[1:]).movedim(1, 2).flatten(2)  # [B, bins, n * 7]
        return (mag * self.mask) ** 2

    def forward(self, waveform: torch.Tensor, m_init: torch.Tensor | None = None) -> torch.Tensor:
        return self.pcen(self.power(waveform), m_init)


def time_padding(kernel: int, causal: bool) -> tuple[int, int]:
    """Left / right zero padding keeping the length: the past only (causal) or centred."""
    return (kernel - 1, 0) if causal else ((kernel - 1) // 2, kernel - 1 - (kernel - 1) // 2)


class Stem(nn.Module):
    """ReLU(Conv2d(1 -> channels)) over (frequency, time), one token per time_stride frames,
    flattened channel-major to channels * out_bins features. aligned: windows on the 7-frame buffers
    (time_kernel = time_stride = 7, one token per buffer, no padding)."""

    def __init__(self, channels: int, freq_kernel: int, freq_stride: int, time_kernel: int, time_stride: int,
                 aligned: bool = False, causal: bool = True, bins: int = N_FFT // 2 + 1):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, (freq_kernel, time_kernel), (freq_stride, time_stride), padding=(freq_kernel // 2, 0))
        self.time_stride, self.aligned, self.causal = time_stride, aligned, causal
        self.out_bins = (bins - 1) // freq_stride + 1
        self.out_features = channels * self.out_bins

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = spec.unsqueeze(1)
        if not self.aligned:
            x = F.pad(x, time_padding(self.conv.kernel_size[1], self.causal))
        return torch.relu(self.conv(x)).flatten(1, 2)  # [B, features, tokens]


class ConvModule(nn.Module):
    """The Conformer's convolution module (Gulati et al. 2020), reduced to what an int8 streaming
    firmware runs cheaply: LayerNorm(x + Dropout(Linear(ReLU(DepthwiseConv1d(x))))), over tokens."""

    def __init__(self, dim: int, kernel: int, dropout: float, causal: bool):
        super().__init__()
        self.kernel_size, self.causal = kernel, causal
        self.depthwise = nn.Conv1d(dim, dim, kernel, groups=dim)
        self.pointwise = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(F.pad(x.transpose(1, 2), time_padding(self.kernel_size, self.causal))).transpose(1, 2)
        return self.norm(x + self.dropout(self.pointwise(torch.relu(y))))


class ViT(nn.Module):
    """waveform [B, N] -> buffer logits [B, N // 1024, n_classes].

    Args:
        stem: Stem's arguments.
        layers: encoder layers (1 to 3 in the paper).
        causal: streamable (True) or the float reference (False).
        dim, heads, ffn, context, conv_kernel, dropout: token width, attention heads, FFN width,
            tokens each query sees (causal), depthwise kernel in tokens, dropout rate.
    """

    def __init__(self, n_classes: int, stem: dict, layers: int, causal: bool = True, dim: int = 32, heads: int = 4,
                 ffn: int = 128, context: int = 32, conv_kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.causal, self.context, self.n_classes = causal, context, n_classes
        self.spectrogram = FrontEnd()
        self.stem = Stem(**stem, causal=causal)
        self.patch_embed = nn.Linear(self.stem.out_features, dim)
        self.input_dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList([ConvModule(dim, conv_kernel, dropout, causal) for _ in range(layers)])
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, ffn, dropout, batch_first=True)
                                     for _ in range(layers)])
        self.embed_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, n_classes)
        # ALiBi slopes 2^(-8h / heads): 0.25, 0.0625, 0.0156, 0.0039 for 4 heads
        self.register_buffer("alibi_slopes", 2.0 ** (-8.0 / heads * torch.arange(1, heads + 1)), persistent=False)
        self.frames_per_buffer = (BUFFER - N_FFT) // HOP + 1

    def attention_mask(self, n: int) -> torch.Tensor:
        """[heads, n, n] additive mask: -slope * age within the causal window, -inf elsewhere; or
        -slope * |i - j| everywhere (non-causal)."""
        pos = torch.arange(n, device=self.alibi_slopes.device)
        age = (pos[:, None] - pos[None, :]).float()
        slopes = self.alibi_slopes.view(-1, 1, 1)
        if not self.causal:
            return -slopes * age.abs()
        return (-slopes * age).masked_fill(~((age >= 0) & (age < self.context)), float("-inf"))

    def token_logits(self, spec: torch.Tensor) -> torch.Tensor:
        """PCEN spectrogram [B, 129, T] -> [B, tokens, n_classes]."""
        x = self.input_dropout(self.patch_embed(self.stem(spec).transpose(1, 2)))
        mask = self.attention_mask(x.shape[1]).repeat(x.shape[0], 1, 1)
        # PyTorch's fused inference path mishandles 3-D masks (pytorch/pytorch#102333): use the regular one
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for conv, layer in zip(self.convs, self.layers):
                x = layer(conv(x), src_mask=mask)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        return self.classifier(self.embed_dropout(x))

    def forward(self, waveform: torch.Tensor, m_init: torch.Tensor | None = None) -> torch.Tensor:
        spec = self.spectrogram(waveform, m_init)
        frames = spec.shape[-1]
        z = self.token_logits(spec).repeat_interleave(self.stem.time_stride, dim=1)[:, :frames]
        return z.reshape(z.shape[0], frames // self.frames_per_buffer, self.frames_per_buffer, -1).mean(2)

    def receptive_field(self) -> int:
        """Buffers of past that reach a causal buffer logit: the stem's kernel, then per layer the
        convolution (kernel - 1 tokens) and the attention window (context - 1 tokens)."""
        tokens = len(self.layers) * (self.convs[0].kernel_size - 1 + self.context - 1)
        frames = tokens * self.stem.time_stride + self.stem.conv.kernel_size[1] - 1
        return math.ceil(frames / self.frames_per_buffer)

    def macs_per_buffer(self, tokens_seen: int) -> float:
        """Multiply-accumulates per 64 ms buffer as the firmware computes them (stem, embedding,
        convolution modules, q/k/v, attention products over min(context, tokens_seen) keys, output
        projection, FFN, classifier; LayerNorm, softmax and the front end excluded)."""
        d, ffn = self.patch_embed.out_features, self.layers[0].linear1.out_features
        keys = min(self.context, tokens_seen) if self.causal else tokens_seen
        conv = self.stem.conv
        per_token = (conv.out_channels * self.stem.out_bins * conv.kernel_size[0] * conv.kernel_size[1]
                     + self.patch_embed.in_features * d + d * self.n_classes
                     + len(self.layers) * (d * self.convs[0].kernel_size + d * d + 4 * d * d + 2 * keys * d + 2 * d * ffn))
        return per_token * self.frames_per_buffer / self.stem.time_stride


# --- readouts: from buffer logits to the decisions of each task -----------------------------------

def ema_weights(n: int, tau_buffers: float) -> torch.Tensor:
    """[n, n] lower-triangular W with ema = W @ logits along time: ema_0 = z_0, ema_t = (1 - a) ema_{t-1} + a z_t."""
    a = 1.0 - math.exp(-1.0 / tau_buffers)
    t, s = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    w = torch.where(s <= t, a * (1 - a) ** (t - s).clamp(min=0), torch.zeros(()))
    w[:, 0] = (1 - a) ** torch.arange(n)
    return w


def clip_logits(z: torch.Tensor, tau_s: float = 0.5, skip: int = 5, temperature: float | None = 5.0) -> torch.Tensor:
    """Species logits of a clip as the device decides: the exponential moving average (time constant
    tau_s) of the buffer logits z [B, buffers, k], max over buffers >= skip (the first buffers have
    less than the attention context). temperature: a smooth max (log-mean-exp) for training; None:
    the exact max (scoring)."""
    ema = torch.einsum("ts,bsk->btk", ema_weights(z.shape[1], tau_s * SAMPLE_RATE / BUFFER).to(z), z)[:, skip:]
    if temperature is None:
        return ema.amax(dim=1)
    return (torch.logsumexp(temperature * ema, dim=1) - math.log(ema.shape[1])) / temperature


def window_logit(z: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
    """One decision per 3 s window (detection, same regime as the clip models): a smooth max of all
    its buffer logits z [B, buffers, 1] -> [B]."""
    return (torch.logsumexp(temperature * z[..., 0], dim=1) - math.log(z.shape[1])) / temperature
