import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块：Conv2d + GN + ReLU + Conv2d + GN + ReLU"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(
            8, out_channels), num_channels=out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(
            8, out_channels), num_channels=out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    """下采样模块：MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    """上采样模块：转置卷积 + 跳跃连接拼接 + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块
    用于自适应地重新校准通道权重
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_UNet(nn.Module):
    """
    双分支UNet + 分支内SE注意力 + 融合层SE注意力
    - 分支A: 6通道原始影像
    - 分支B: 13通道指数
    - A/B分支每一层都有SE注意力
    - 融合层也有SE注意力
    """

    def __init__(self, in_channels_a=6, in_channels_b=13, num_classes=3,
                 features=[64, 128, 256, 512], se_reduction=16):
        super(SE_UNet, self).__init__()

        # ========== 分支A：6通道原始影像 ==========
        self.enc1_a = DoubleConv(in_channels_a, features[0])
        self.se1_a = SEBlock(features[0], reduction=se_reduction)

        self.down1_a = Down(features[0], features[1])
        self.se2_a = SEBlock(features[1], reduction=se_reduction)

        self.down2_a = Down(features[1], features[2])
        self.se3_a = SEBlock(features[2], reduction=se_reduction)

        self.down3_a = Down(features[2], features[3])
        self.se4_a = SEBlock(features[3], reduction=se_reduction)

        self.down4_a = Down(features[3], features[3] * 2)
        self.se5_a = SEBlock(features[3] * 2, reduction=se_reduction)

        # ========== 分支B：13通道指数 ==========
        self.enc1_b = DoubleConv(in_channels_b, features[0])
        self.se1_b = SEBlock(features[0], reduction=se_reduction)

        self.down1_b = Down(features[0], features[1])
        self.se2_b = SEBlock(features[1], reduction=se_reduction)

        self.down2_b = Down(features[1], features[2])
        self.se3_b = SEBlock(features[2], reduction=se_reduction)

        self.down3_b = Down(features[2], features[3])
        self.se4_b = SEBlock(features[3], reduction=se_reduction)

        self.down4_b = Down(features[3], features[3] * 2)
        self.se5_b = SEBlock(features[3] * 2, reduction=se_reduction)

        # ========== 融合层：通道拼接 + SE注意力 ==========
        self.fusion_conv = nn.Conv2d(
            features[3] * 4, features[3] * 2, kernel_size=1)
        self.fusion_gn = nn.GroupNorm(num_groups=min(8, features[3] * 2),
                                      num_channels=features[3] * 2)
        self.fusion_relu = nn.ReLU(inplace=True)

        self.se_attention = SEBlock(features[3] * 2, reduction=se_reduction)

        # ========== 解码器（标准UNet） ==========
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, img, indices):
        """
        Args:
            img: 6通道原始影像 [B, 6, 128, 128]
            indices: 13通道指数 [B, 13, 128, 128]
        Returns:
            logits: [B, 3, 128, 128]
        """
        # ===== 分支A编码 =====
        e1_a = self.enc1_a(img)
        e1_a = self.se1_a(e1_a)

        e2_a = self.down1_a(e1_a)
        e2_a = self.se2_a(e2_a)

        e3_a = self.down2_a(e2_a)
        e3_a = self.se3_a(e3_a)

        e4_a = self.down3_a(e3_a)
        e4_a = self.se4_a(e4_a)

        e5_a = self.down4_a(e4_a)
        e5_a = self.se5_a(e5_a)

        # ===== 分支B编码 =====
        e1_b = self.enc1_b(indices)
        e1_b = self.se1_b(e1_b)

        e2_b = self.down1_b(e1_b)
        e2_b = self.se2_b(e2_b)

        e3_b = self.down2_b(e2_b)
        e3_b = self.se3_b(e3_b)

        e4_b = self.down3_b(e3_b)
        e4_b = self.se4_b(e4_b)

        e5_b = self.down4_b(e4_b)
        e5_b = self.se5_b(e5_b)

        # ===== 融合：通道拼接 -> 降维 -> SE注意力 =====
        e5 = torch.cat([e5_a, e5_b], dim=1)
        e5 = self.fusion_conv(e5)
        e5 = self.fusion_gn(e5)
        e5 = self.fusion_relu(e5)
        e5 = self.se_attention(e5)

        # ===== 解码（跳跃连接：对应层的特征相加） =====
        d1 = self.up1(e5, e4_a + e4_b)
        d2 = self.up2(d1, e3_a + e3_b)
        d3 = self.up3(d2, e2_a + e2_b)
        d4 = self.up4(d3, e1_a + e1_b)

        logits = self.out_conv(d4)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ========== 快速测试 ==========
if __name__ == "__main__":
    model = SE_UNet(in_channels_a=6, in_channels_b=13, num_classes=3)

    img = torch.randn(4, 6, 128, 128)
    indices = torch.randn(4, 13, 128, 128)

    output = model(img, indices)
    print(f"Input - img: {img.shape}, indices: {indices.shape}")
    print(f"Output logits: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
