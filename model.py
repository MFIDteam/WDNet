# net/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# =========================
# LayerNorm 与工具
# =========================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu    = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# =========================
# GDFN
# =========================
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# =========================
# 注意力
# =========================
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads  = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv        = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) H W -> b head c (H W)', head=self.num_heads)
        k = rearrange(k, 'b (head c) H W -> b head c (H W)', head=self.num_heads)
        v = rearrange(v, 'b (head c) H W -> b head c (H W)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = attn @ v

        out = rearrange(out, 'b head c (H W) -> b (head c) H W', H=h, W=w)
        return self.project_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# =========================
# Patch Embedding & Resizing
# =========================
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

# =========================
# Prompt（暗光专用）
# =========================
class LowLightPromptBlock(nn.Module):
    """面向暗光增强的可学习提示原语"""
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3      = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb     = x.mean(dim=(-2, -1))                     # B, C
        weights = F.softmax(self.linear_layer(emb), dim=1) # B, L

        prompt = self.prompt_param.expand(B, -1, -1, -1, -1)   # B,L,C,S,S
        prompt = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt).sum(dim=1)
        prompt = F.interpolate(prompt, (H, W), mode='bilinear', align_corners=False)
        prompt = self.conv3x3(prompt)
        return prompt

# =========================================================
# 小波 & MoE：暗光增强命名
# =========================================================
def _gaussian_kernel1d(sigma: float, radius: int = None, device=None, dtype=torch.float32):
    if radius is None:
        radius = int(3.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-12)
    return g

def _gaussian_blur(img, sigma=1.0):
    B, C, H, W = img.shape
    g = _gaussian_kernel1d(sigma, device=img.device, dtype=img.dtype)
    g_col = g.view(1, 1, -1, 1)
    g_row = g.view(1, 1, 1, -1)
    out = F.conv2d(img, g_row.repeat(C, 1, 1, 1),
                   padding=(0, g_row.shape[3] // 2), groups=C)
    out = F.conv2d(out, g_col.repeat(C, 1, 1, 1),
                   padding=(g_col.shape[2] // 2, 0), groups=C)
    return out

def _laplacian(img):
    k = torch.tensor([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    C = img.shape[1]
    return F.conv2d(img, k.repeat(C, 1, 1, 1), padding=1, groups=C)

def _minmax(x, eps=1e-12):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _wave_energy(HL, LH, HH):
    return HL.abs() + LH.abs() + HH.abs()

class HaarDWT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5],
                           [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5],
                           [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])
        k = torch.stack([ll, hl, lh, hh], dim=0).unsqueeze(1)      # (4,1,2,2)
        self.register_buffer('w', k.repeat(channels, 1, 1, 1))     # (4C,1,2,2)
        self.C = channels

    def forward(self, x):
        B, C, H, W = x.shape
        y  = F.conv2d(x, self.w, stride=2, groups=C)               # (B,4C,H/2,W/2)
        y  = y.view(B, C, 4, H // 2, W // 2)
        LL = y[:, :, 0]
        HL = y[:, :, 1]
        LH = y[:, :, 2]
        HH = y[:, :, 3]
        return LL, HL, LH, HH

class HaarIWT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5],
                           [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5],
                           [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5,  0.5]])
        k = torch.stack([ll, hl, lh, hh], dim=0).unsqueeze(1)      # (4,1,2,2)
        self.register_buffer('w', k.repeat(channels, 1, 1, 1))     # (4C,1,2,2)
        self.C = channels

    def forward(self, LL, HL, LH, HH):
        B, C, h, w = LL.shape
        y   = torch.stack([LL, HL, LH, HH], dim=2).view(B, C * 4, h, w)  # (B,4C,h,w)
        out = F.conv_transpose2d(y, self.w, stride=2, groups=C)
        return out

class IllumWTConv(nn.Module):
    """面向光照表征的 DW3x3+PW1x1"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

# === Drop-in replacement for LowLightWaveletMoEGate: Soft-MoE with two stability tricks ===
# 软性MoE（softmax + 自层正偏置 + 温度），并实现两点小技巧：
# 1) 对“其它层”的 H* 计算使用 no_grad，并对其特征停止梯度（detach），减少重入反传/显存开销；
# 2) 跨层只做“读”：其它层的特征不回传梯度，但 q_proj / v_proj / gate 等路由参数仍可学习。
class LowLightWaveletMoEGate(nn.Module):
    """
    Soft-MoE 版本（推荐）：
      - 不再使用 min-max 和 tau 截断；
      - 把“当前层”也纳入 softmax 权重，并给当前层加正偏置 bias0，保证自层优先；
      - 使用温度 T 控制权重分布的尖锐度；
      - 默认对 enc_others 的 H* 计算使用 no_grad，并对 Hs/enc_others 停止梯度，降低不稳定与显存开销。
    """

    def __init__(self, in_ch_curr, prompt_ch,
                 sim_dim=32, sigma=1.0, beta=1.0,
                 # Soft-MoE 关键超参
                 temperature=0.7, bias0=1.0,
                 # 两点小技巧开关
                 detach_others: bool = True,
                 stop_grad_others: bool = True,
                 # 向后兼容：允许老代码传 tau 或其它没用到的关键字
                 tau=None, **unused):
        super().__init__()
        # 小波分析/重建
        self.dwt   = HaarDWT(in_ch_curr)
        self.iwt   = HaarIWT(in_ch_curr)
        self.beta  = beta
        self.sigma = sigma

        # 相似度投影
        self.q_proj = nn.Conv2d(prompt_ch, sim_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(1,         sim_dim, 1, bias=False)

        # 显著性融合（Hw + Hlog -> H*）
        self.fuse   = nn.Conv2d(2, 1, 1, bias=True)

        # 提示投影到 prompt_ch
        self.to_prompt = IllumWTConv(in_ch_curr, prompt_ch)

        # 门控（跳连/提示）
        self.gate      = nn.Conv2d(1, 2, 1, bias=True)

        # Soft-MoE 参数
        self.temperature = float(temperature)
        self.bias0       = float(bias0)

        # 技巧：控制跨层梯度
        self.detach_others    = bool(detach_others)
        self.stop_grad_others = bool(stop_grad_others)

    @staticmethod
    def _Hstar_from_feat(x, sigma, fuse):
        B, C, H, W = x.shape
        dwt = HaarDWT(C).to(x.device)
        LL, HL, LH, HH = dwt(x)
        Hw   = _wave_energy(HL, LH, HH)
        Hw   = _minmax(F.interpolate(Hw, size=(H, W),
                                     mode='bilinear', align_corners=False)).mean(dim=1, keepdim=True)
        LLup = F.interpolate(LL, size=(H, W), mode='bilinear', align_corners=False)
        Hlog = _minmax(_laplacian(_gaussian_blur(LLup, sigma)).abs()).mean(dim=1, keepdim=True)
        Hs   = torch.sigmoid(fuse.to(x.device)(torch.cat([Hw, Hlog], dim=1)))
        return Hs

    @staticmethod
    def _sim_l2norm(Q, H, v_proj):
        V  = v_proj(H)                         # (B,sim,H,W)
        Qn = F.normalize(Q, dim=1)
        Vn = F.normalize(V, dim=1)
        sim = (Qn * Vn).mean(dim=(1, 2, 3), keepdim=True)
        return sim

    def forward(self, enc_curr, enc_others, prompt_feat):
        eps = 1e-6
        B, C, H, W = enc_curr.shape

        # 当前层 H*
        LL, HL, LH, HH = self.dwt(enc_curr)
        h2, w2 = LL.shape[-2:]
        Hw   = _wave_energy(HL, LH, HH)
        Hw   = _minmax(F.interpolate(Hw, size=(H, W),
                                     mode='bilinear', align_corners=False)).mean(dim=1, keepdim=True)
        LLup = F.interpolate(LL, size=(H, W), mode='bilinear', align_corners=False)
        Hlog = _minmax(_laplacian(_gaussian_blur(LLup, self.sigma)).abs()).mean(dim=1, keepdim=True)
        Hcurr = torch.sigmoid(self.fuse(torch.cat([Hw, Hlog], dim=1)))  # (B,1,H,W)

        # 其它层 H*
        H_others = []
        if len(enc_others) > 0:
            if self.detach_others:
                with torch.no_grad():
                    for enc in enc_others:
                        Hs = self._Hstar_from_feat(enc, self.sigma, self.fuse)
                        Hs = F.interpolate(Hs, size=(H, W), mode='bilinear', align_corners=False)
                        H_others.append(Hs)
            else:
                for enc in enc_others:
                    Hs = self._Hstar_from_feat(enc, self.sigma, self.fuse)
                    Hs = F.interpolate(Hs, size=(H, W), mode='bilinear', align_corners=False)
                    H_others.append(Hs)

            if self.stop_grad_others:
                H_others = [h.detach() for h in H_others]

        # Soft-MoE 权重
        Q = self.q_proj(prompt_feat)  # (B,sim,H,W)
        sims = [self._sim_l2norm(Q, Hcurr, self.v_proj)]
        for Hs in H_others:
            sims.append(self._sim_l2norm(Q, Hs, self.v_proj))
        logits = torch.cat(sims, dim=1)  # (B, 1+N, 1, 1)

        T  = max(self.temperature, eps)
        b0 = self.bias0
        logits_curr   = logits[:, 0:1] / T + b0
        logits_others = logits[:, 1:]  / T if logits.size(1) > 1 else logits[:, 1:]
        logits = torch.cat([logits_curr, logits_others], dim=1)
        w = torch.softmax(logits, dim=1)  # (B,1+N,1,1)

        Hs_all = [Hcurr] + H_others
        Hmix = 0.0
        for i, Hs in enumerate(Hs_all):
            Hmix = Hmix + w[:, i:i+1] * Hs

        # 小波域调制 & IWT
        Hlow  = F.interpolate(Hmix, size=(h2, w2), mode='bilinear', align_corners=False)
        scale = 1.0 + self.beta * Hlow
        HLm   = HL * scale
        LHm   = LH * scale
        HHm   = HH * scale
        recon = self.iwt(LL, HLm, LHm, HHm)

        # 提示 + 门控
        wave_prompt = self.to_prompt(recon)
        gates       = torch.sigmoid(self.gate(Hmix))
        g_skip, g_prompt = gates[:, 0:1], gates[:, 1:2]

        return wave_prompt, g_skip, g_prompt


# =========================================================
# 多尺度树突 DNM（用于 L1-L3 融合）—— 3 路版
# =========================================================
# —— 替换你文件里的 MultiScaleDNM —— #
class MultiScaleDNM(nn.Module):
    """
    多尺度树突神经元融合（3 路）
    归一化策略：
      - dendrite_norm: 在树突聚合 z (N, out_ch, M) 上，沿 M 维做 RMSNorm（无均值项），减少分支数/尺度的影响；
      - soma_norm: 在胞体输出 v (B,C,H,W) 上，做 GroupNorm(1,C) ≈ LayerNorm2d。
    """
    def __init__(self, in_dims, out_ch=96, M=4, activation='relu',
                 use_dendrite_norm=True, use_soma_norm=True, eps=1e-6):
        super().__init__()
        assert isinstance(in_dims, (list, tuple)) and len(in_dims) > 0
        self.in_dims  = list(in_dims)
        self.num_syn  = len(in_dims)
        self.out_ch   = out_ch
        self.M        = M
        self.activation = activation
        self.eps = eps

        # 可学习的线性权重（树突层）
        self.W = nn.ParameterList()
        self.q = nn.ParameterList()
        for d in self.in_dims:
            Wk = nn.Parameter(torch.randn(out_ch, M, d) * (1.0 / (d ** 0.5)))
            qk = nn.Parameter(torch.full((out_ch, M), 0.1))
            self.W.append(Wk)
            self.q.append(qk)

        # 胞体线性
        self.soma_scale = nn.Parameter(torch.ones(out_ch))
        self.soma_bias  = nn.Parameter(torch.zeros(out_ch))

        # 归一化模块
        self.use_dendrite_norm = use_dendrite_norm
        self.use_soma_norm     = use_soma_norm
        if self.use_soma_norm:
            # GroupNorm(1,C) 等价于“按通道的 LayerNorm2d”
            self.soma_gn = nn.GroupNorm(1, out_ch)

        # 树突归一（按 M 维的 RMSNorm，逐通道有可学习缩放）
        if self.use_dendrite_norm:
            self.dendrite_gamma = nn.Parameter(torch.ones(out_ch, 1))  # (C,1) 作用在 M 维归一后

    def _act(self, x):
        if self.activation is None:
            return x
        if self.activation == 'relu':
            return F.relu(x, inplace=False)
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        return x

    def forward(self, feats):
        assert len(feats) == self.num_syn, f"Expect {self.num_syn} feature maps, got {len(feats)}"
        B, _, H, W = feats[0].shape
        N = B * H * W

        syn_accum = None  # 将各路 z 按树突分支维累加
        for k, x in enumerate(feats):
            Bk, Ck, Hk, Wk = x.shape
            assert Hk == H and Wk == W, "All feats must share spatial size"
            assert Ck == self.in_dims[k], f"in_dims[{k}]={self.in_dims[k]}, but got C={Ck}"

            # (B,H,W,Ck) -> (N, Ck)
            x_flat = x.permute(0, 2, 3, 1).reshape(N, Ck)          # (N, Ck)
            Wk = self.W[k]                                          # (C_out, M, Ck)
            qk = self.q[k]                                          # (C_out, M)

            # z: (N, C_out, M)
            z = torch.einsum('nc,omc->nom', x_flat, Wk)             # 先线性
            z = z - qk.unsqueeze(0)                                 # 偏置/阈值
            z = self._act(z)

            # —— 树突归一：沿 M 维 RMSNorm，逐通道 gamma 缩放 —— #
            if self.use_dendrite_norm:
                # rms over M: (N, C, 1)
                rms = torch.sqrt(torch.mean(z * z, dim=2, keepdim=True) + self.eps)
                z = z / rms                                          # 归一到单位 RMS
                z = z * self.dendrite_gamma.unsqueeze(0)             # (1,C,1) 缩放

            syn_accum = z if syn_accum is None else (syn_accum + z)  # 按路求和（仍是 N×C×M）

        # 胞体整合：沿 M 维求和 -> (N, C)
        v = syn_accum.sum(dim=2)

        # 线性 + 激活
        v = v * self.soma_scale.unsqueeze(0) + self.soma_bias.unsqueeze(0)
        v = self._act(v)

        # 回到 (B,C,H,W)
        v = v.view(B, H, W, self.out_ch).permute(0, 3, 1, 2).contiguous()

        # —— 胞体输出再做一次通道归一 —— #
        if self.use_soma_norm:
            v = self.soma_gn(v)

        return v

# ========================
# 主干网络（在 L3/L2/L1 注入小波 Prompt + 末端 3 路 DNM 融合）
# =========================
class PromptIR(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4, 6, 6, 8],
        num_refinement_blocks = 4,
        heads = [1, 2, 4, 8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        decoder = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ---- Prompt（3 个：H/2, H/4, H/8）----
        if self.decoder:
            self.prompt1 = LowLightPromptBlock(prompt_dim=64,  prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = LowLightPromptBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = LowLightPromptBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)

            self.p3_align = nn.Conv2d(320, 128, kernel_size=1, bias=False)  # for L3
            self.p2_align = nn.Conv2d(128,  64, kernel_size=1, bias=False)  # for L2/L1
            self.p1_align = nn.Conv2d( 64,  64, kernel_size=1, bias=False)  # identity

        # 接口兼容的 1×1（保留）
        self.chnl_reduce1 = nn.Conv2d( 64,  64, 1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, 1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, 1, bias=bias)

        # ========= Encoder =========
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, 1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128, int(dim*2**1), 1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim*2**1))

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256, int(dim*2**2), 1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**3), num_heads=heads[3],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        # ========= Decoder 顶层 =========
        self.up4_3 = Upsample(int(dim*2**3))                         # 384 -> 192
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**2)*2,
                                            int(dim*2**2), 1, bias=bias)  # 384->192

        # latent 注入（704 = 384 + 320）
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512,  # 192 + 512 = 704
                                             num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2) + 512,
                                             int(dim*2**3), 1, bias=bias)  # 704 -> 384

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        # H/4 -> H/2
        self.up3_2 = Upsample(int(dim*2**2))                          # 192 -> 96
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2),
                                            int(dim*2**1), 1, bias=bias)  # 192->96

        self.noise_level2 = TransformerBlock(dim=int(dim*2**2) + 256,  # 192 + 256 = 448
                                             num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**2) + 256,
                                             int(dim*2**2), 1, bias=bias)  # 448 -> 192

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        # H/2 -> H
        self.up2_1 = Upsample(int(dim*2**1))                           # 96 -> 48

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1) + 128,   # 96 + 128 = 224
                                             num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1) + 128,
                                             int(dim*2**1), 1, bias=bias)  # 224 -> 96

        self.noise_level0 = TransformerBlock(dim=int(dim*2**1) + 128,   # 96 + 128 = 224
                                             num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level0 = nn.Conv2d(int(dim*2**1) + 128,
                                             int(dim*2**1), 1, bias=bias)  # 224 -> 96

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # 原始输出头（in_ch = 2*dim，例如 96）
        self.output = nn.Conv2d(int(dim*2**1), out_channels, 3, 1, 1, bias=bias)

        # ---------- 暗光：小波提示 ----------
        if self.decoder:
            self.wave_prompt_l3 = LowLightWaveletMoEGate(in_ch_curr=int(dim*2**2),
                                                         prompt_ch=128,
                                                         sim_dim=32, sigma=1.0,
                                                         beta=1.0, tau=0.2)
            self.wave_prompt_l2 = LowLightWaveletMoEGate(in_ch_curr=int(dim*2**1),
                                                         prompt_ch=64,
                                                         sim_dim=32, sigma=1.0,
                                                         beta=1.0, tau=0.2)
            self.wave_prompt_l1 = LowLightWaveletMoEGate(in_ch_curr=int(dim*2**0),
                                                         prompt_ch=64,
                                                         sim_dim=32, sigma=1.0,
                                                         beta=1.0, tau=0.2)

        # ---------- 末端多尺度 DNM：仅融合 decoder 三层（dec1/dec2/dec3） ----------
        # dec1: 96 (H), dec2: 96 (H/2), dec3: 192 (H/4)
        self.dnm_fusion = MultiScaleDNM(
            in_dims=[int(dim*2**1),  # dec1
                     int(dim*2**1),  # dec2
                     int(dim*2**2)], # dec3
            out_ch=int(dim*2**1),   # 96
            M=4,
            activation='relu'
        )

        # 逐尺度轻量归一（GroupNorm，3 路）
        self.dnm_norm_dec1 = nn.GroupNorm(1, int(dim * 2 ** 1))  # 96
        self.dnm_norm_dec2 = nn.GroupNorm(1, int(dim * 2 ** 1))  # 96
        self.dnm_norm_dec3 = nn.GroupNorm(1, int(dim * 2 ** 2))  # 192

        # 尺度 SE（3 个尺度 -> 3 个系数），拼接后全局池化 + 两层 1x1
        se_in_ch = int(dim * 2 ** 1) + int(dim * 2 ** 1) + int(dim * 2 ** 2)  # 96+96+192=384
        self.scale_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(se_in_ch, 64, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, 3, 1, bias=True),  # 输出 3 个权重
            nn.Sigmoid()
        )

        # DNM 残差门控（σ(gate)∈(0,1)，初始化为很小值 ≈ 0.12）
        self.dnm_gate = nn.Parameter(torch.tensor(-2.0))  # alpha = sigmoid(-2) ~ 0.119

    def forward(self, inp_img, noise_emb=None):
        # ========== Encoder ==========
        x1 = self.patch_embed(inp_img)                 # 3 -> 48
        e1 = self.encoder_level1(x1)                   # H,   C=48

        x2 = self.down1_2(e1)                          # H/2, C=96
        e2 = self.encoder_level2(x2)                   # H/2, C=96

        x3 = self.down2_3(e2)                          # H/4, C=192
        e3 = self.encoder_level3(x3)                   # H/4, C=192

        x4  = self.down3_4(e3)                         # H/8, C=384
        lat = self.latent(x4)                          # H/8, C=384

        # ---------- latent（H/8）生成 prompt3，并注入 ----------
        if self.decoder:
            p3  = self.prompt3(lat)                    # B,320,H/8,W/8
            lat = torch.cat([lat, p3], dim=1)          # 384+320=704
            lat = self.noise_level3(lat)               # 704
            lat = self.reduce_noise_level3(lat)        # 704 -> 384 (解码 L4)
        # dec4 = lat                                     # decoder L4 特征: B,384,H/8（不进 DNM，仅供上层使用）

        # ========== Decoder L3（H/4）==========
        d3 = self.up4_3(lat)                           # H/8 -> H/4, C: 384->192
        d3 = torch.cat([d3, e3], dim=1)                # B,384,H/4
        d3 = self.reduce_chan_level3(d3)               # -> B,192,H/4
        d3 = self.decoder_level3(d3)                   # B,192,H/4

        if self.decoder:
            p3_up = F.interpolate(p3, size=d3.shape[-2:],
                                  mode='bilinear', align_corners=False)  # B,320,H/4
            p3_up = self.p3_align(p3_up)               # 320 -> 128

            wp3, g_skip3, g_p3 = self.wave_prompt_l3(enc_curr=e3,
                                                     enc_others=[e1, e2],
                                                     prompt_feat=p3_up)
            d3_w = d3    * (0.5 + g_skip3)
            p3_w = p3_up * (0.5 + g_p3)

            d3  = torch.cat([d3_w, p3_w, wp3], dim=1)  # 192 + 128 + 128 = 448
            d3  = self.noise_level2(d3)                # B,448,H/4
            d3  = self.reduce_noise_level2(d3)         # -> B,192,H/4
        dec3 = d3                                      # decoder L3: B,192,H/4

        p2 = self.prompt2(d3)                          # B,128,H/4

        # ========== Decoder L2（H/2）==========
        d2 = self.up3_2(d3)                            # H/4 -> H/2, C: 192->96
        d2 = torch.cat([d2, e2], dim=1)                # B,192,H/2
        d2 = self.reduce_chan_level2(d2)               # -> B,96,H/2
        d2 = self.decoder_level2(d2)                   # B,96,H/2

        if self.decoder:
            p2_up = F.interpolate(p2, size=d2.shape[-2:],
                                  mode='bilinear', align_corners=False)  # B,128,H/2
            p2_up = self.p2_align(p2_up)               # 128 -> 64

            wp2, g_skip2, g_p2 = self.wave_prompt_l2(enc_curr=e2,
                                                     enc_others=[e1, e3],
                                                     prompt_feat=p2_up)
            d2_w = d2    * (0.5 + g_skip2)
            p2_w = p2_up * (0.5 + g_p2)

            d2  = torch.cat([d2_w, p2_w, wp2], dim=1)  # 96 + 64 + 64 = 224
            d2  = self.noise_level1(d2)                # B,224,H/2
            d2  = self.reduce_noise_level1(d2)         # -> B,96,H/2
        dec2 = d2                                      # decoder L2: B,96,H/2

        p1 = self.prompt1(d2)                          # B,64,H/2

        # ========== Decoder L1（H）==========
        d1 = self.up2_1(d2)                            # H/2 -> H, C: 96->48
        d1 = torch.cat([d1, e1], dim=1)                # B,96,H

        if self.decoder:
            p1_up = F.interpolate(p1, size=d1.shape[-2:],
                                  mode='bilinear', align_corners=False)  # B,64,H
            p1_up = self.p1_align(p1_up)               # 64 -> 64

            wp1, g_skip1, g_p1 = self.wave_prompt_l1(enc_curr=e1,
                                                     enc_others=[e2, e3],
                                                     prompt_feat=p1_up)
            d1_w = d1    * (0.5 + g_skip1)
            p1_w = p1_up * (0.5 + g_p1)

            d1  = torch.cat([d1_w, p1_w, wp1], dim=1)  # 96 + 64 + 64 = 224
            d1  = self.noise_level0(d1)                # B,224,H
            d1  = self.reduce_noise_level0(d1)         # -> B,96,H

        # 可按需启用/关闭这两行：
        d1 = self.decoder_level1(d1)
        # d1 = self.refinement(d1)
        dec1 = d1  # decoder L1: B,96,H

        # ========== 多尺度 DNM 融合（仅 dec1/dec2/dec3） + 归一/SE 重标定 + 门控 ==========
        B, C1, H, W = dec1.shape
        dec2_up = F.interpolate(dec2, size=(H, W), mode='bilinear', align_corners=False)
        dec3_up = F.interpolate(dec3, size=(H, W), mode='bilinear', align_corners=False)

        # 逐尺度归一
        dec1_n = self.dnm_norm_dec1(dec1)
        dec2_n = self.dnm_norm_dec2(dec2_up)
        dec3_n = self.dnm_norm_dec3(dec3_up)

        # 尺度 SE：得到 B×3×1×1 的权重
        cat_all = torch.cat([dec1_n, dec2_n, dec3_n], dim=1)
        w = self.scale_se(cat_all)  # B,3,1,1
        w1, w2, w3 = w[:, 0:1], w[:, 1:2], w[:, 2:3]

        dec1_w = dec1_n * w1
        dec2_w = dec2_n * w2
        dec3_w = dec3_n * w3

        # DNM 融合（稳态输入）
        fused_raw = self.dnm_fusion([dec1_w, dec2_w, dec3_w])  # B,96,H,W

        # 残差门控融合：y = dec1 + α * fused_raw
        alpha = torch.sigmoid(self.dnm_gate)
        fused = dec1 + alpha * fused_raw

        # 抛光
        fused = self.refinement(fused)

        # 输出头 + 残差到图像域
        out = self.output(fused) + inp_img
        return out
