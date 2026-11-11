
import torch
import torch.nn as nn
from einops import rearrange

class LightMambaS4(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, d_model=64, d_state=32):
        super().__init__()
        # 分块投影 + 下采样
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),  # 下采样
        )
        
        # 共享参数的 Mamba 层（减少内存）
        self.mamba_layer = MambaBlock(d_model, d_state)
        
    def forward(self, x):
        # 输入: [B, C, H, W]
        x = self.patch_embed(x)  # [B, d_model, H/16, W/16]
        x = rearrange(x, "b c h w -> b (h w) c")  # 序列长度: (H/16)*(W/16)
        x = self.mamba_layer(x)  # [B, seq, d_model]
        return x

class MambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 状态空间参数 (A, B, C, D)
        self.A = nn.Parameter(torch.randn(d_state, d_state) / d_state)
        self.B = nn.Linear(d_model, d_state, bias=False)  # 输入投影到状态空间
        self.C = nn.Linear(d_model, d_state, bias=False)  # 输入投影到状态空间
        self.D = nn.Parameter(torch.randn(d_model))
        
        # 动态时间步长生成
        self.proj_dt = nn.Linear(d_model, 1)

    def _discretize(self, A, dt):
        """离散化参数 (兼容 PyTorch 1.8.1)"""
        # dt: [B, seq, 1] → [B, seq, 1, 1]
        dt = dt.unsqueeze(-1).unsqueeze(-1)
        # A: [d_state, d_state] → [1, 1, d_state, d_state]
        A = A.unsqueeze(0).unsqueeze(0)
        dA = torch.matrix_exp(A * dt)  # [B, seq, d_state, d_state]
        return dA

    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # 动态生成参数
        dt = torch.sigmoid(self.proj_dt(x))  # [B, seq, 1]
        B_t = self.B(x)  # [B, seq, d_state]
        C_t = self.C(x)  # [B, seq, d_state]
        
        # 离散化 A
        dA = self._discretize(self.A, dt)  # [B, seq, d_state, d_state]
        
        # 初始化隐藏状态
        h = torch.zeros(batch, self.d_state).to(x.device)
        outputs = []
        
        for t in range(seq_len):
            # 状态更新: h = dA * h + B_t * x
            h = torch.einsum("bij,bj->bi", dA[:, t], h) + B_t[:, t, :] * x[:, t, :]
            # 输出: y = C_t * h + D * x
            y = C_t[:, t, :] * h + self.D * x[:, t, :]
            outputs.append(y.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

class LiteConvMamba(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 深度可分离卷积（减少计算量）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels, in_channels, 1),  # 逐点卷积
        )
        
        # 极简 Mamba（单层 + 小维度）
        self.mamba = MambaBlock(d_model=in_channels, d_state=16)
        
    def forward(self, x):
        # 输入: [B, C, H, W]
        x_conv = self.conv(x)
        x_mamba = rearrange(x, "b c h w -> b (h w) c")
        x_mamba = self.mamba(x_mamba)
        x_mamba = rearrange(x_mamba, "b (h w) c -> b c h w", h=x.shape[2])
        return x_conv + x_mamba  # 残差连接