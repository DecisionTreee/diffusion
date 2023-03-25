import math
import torch
import torch.nn as nn


class PositionEmbeddings(nn.Module):
    def __init__(self, channel, scale=1.0):
        super().__init__()
        self.channel = channel
        self.scale = scale

    def forward(self, time):
        device = time.device
        half_dim = self.channel // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(time * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Attention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.norm = nn.GroupNorm(32, channel)
        self.to_qkv = nn.Conv2d(channel, channel * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(channel, channel, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.channel, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


class DownSamples(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Conv2d(channel, channel, 3, 2, 1)

    def forward(self, x, time_emb):
        return self.layer(x)


class UpSamples(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.layer = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x, time_emb):
        x = self.up(x)
        x = self.layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, use_attention=False):
        super().__init__()
        self.activation = nn.SiLU()

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_channel)
        self.conv2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

        self.time_mlp = nn.Linear(time_emb_dim, out_channel)

        self.residual_connection = nn.Conv2d(in_channel, out_channel, 1, 1, 0) if in_channel != out_channel else nn.Identity()
        self.attention = Attention(out_channel) if use_attention else nn.Identity()

    def forward(self, x, time_emb):
        out = self.activation(self.norm1(x))
        out = self.conv1(out)
        out += self.time_mlp(self.activation(time_emb))[:, :, None, None]

        out = self.activation(self.norm2(out))
        out = self.conv2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out


class Unet(nn.Module):
    def __init__(self, image_channel, base_channel, channel_multi=(1, 2, 4, 8), num_res_block=2, time_emb_dim=128 * 4, time_emb_scale=1.0, attention_resolution=(1, )):
        super().__init__()
        self.activation = nn.SiLU()

        self.time_mlp = nn.Sequential(
            PositionEmbeddings(base_channel, time_emb_scale),
            nn.Linear(base_channel, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.input = nn.Conv2d(image_channel, base_channel, 3, 1, 1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channel]

        now_channel = base_channel
        for i, multi in enumerate(channel_multi):
            out_channel = base_channel * multi
            for _ in range(num_res_block):
                self.downs.append(ResidualBlock(now_channel, out_channel, time_emb_dim, i in attention_resolution))  # i in attention_resolution
                now_channel = out_channel
                channels.append(now_channel)
            if i != len(channel_multi) - 1:
                self.downs.append(DownSamples(now_channel))
                channels.append(now_channel)

        self.middle = nn.ModuleList([
            ResidualBlock(now_channel, now_channel, time_emb_dim, True),
            ResidualBlock(now_channel, now_channel, time_emb_dim, False),
        ])

        for i, multi in reversed(list(enumerate(channel_multi))):
            out_channel = base_channel * multi
            for _ in range(num_res_block + 1):
                self.ups.append(ResidualBlock(channels.pop() + now_channel, out_channel, time_emb_dim, i in attention_resolution))
                now_channel = out_channel
            if i != 0:
                self.ups.append(UpSamples(now_channel))

        self.out_norm = nn.GroupNorm(32, base_channel)
        self.out_conv = nn.Conv2d(base_channel, image_channel, 3, 1, 1)

    def forward(self, x, time):
        time_emb = self.time_mlp(time)

        x = self.input(x)
        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb)
            skips.append(x)
        for layer in self.middle:
            x = layer(x, time_emb)
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        return x
