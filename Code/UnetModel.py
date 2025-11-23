import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """
    Attention mechanism
    """
    def __init__(self, d_in_channels, e_in__channels, out_channels, kernel_size=(1,1), stride=(1,1)):
        super().__init__()
        if out_channels == 0:
            out_channels = 1
            
        self.We = nn.Conv2d(
                in_channels  = e_in__channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride = stride)
        
        self.Wd = nn.Conv2d(
                in_channels  = d_in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride = stride)

        self.sigmoid = nn.Sigmoid()

        self.Watt = nn.Conv2d(
                in_channels  = out_channels,
                out_channels = 1,
                kernel_size  = kernel_size,
                stride = stride)

    def forward(self, d,e):

        WeE = self.We(e)
        WdD = self.Wd(d)

        B = self.sigmoid(WeE + WdD)    # B = sigmoid(We*e+Wd*d)
        A = self.sigmoid(self.Watt(B)) # A = sigmoid(Watt*B)

        new_skip_e = torch.mul(A, e)

        return new_skip_e


class CausalConvBlock(nn.Module):
    """
    Each block in the encoder consists of a Conv2d layer followed by batch normalization, 
    dropout, and a `LeakyRelu' activation function. In addition, an attention block within the skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class CausalTransConvBlock(nn.Module):
    """
    Each block in the decoder consists of a ConvTranspose2d layer followed by batch normalization, 
    dropout, and a `LeakyRelu' activation function. In addition, an attention block within the skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

        self.att_block = AttentionBlock(in_channels, out_channels, out_channels//2)

    def forward(self, x, skip, EnableSkipAtt):
        if EnableSkipAtt:
            skip = self.att_block(x,skip)
        
        return self.conv(torch.cat((x, skip), 1)),skip


class UNET(nn.Module):
    """
    UNET architecture

    Args:
        in_channel (int): Number of input channels.
        activation (int): Activation fnction at the end of the UNET.
        EnableSkipAttention (True/False): Flag to enable attention in the skip connections.
    """
    def __init__(self, in_channel, activation = 'sigmoid', EnableSkipAttention = 0, use_rtf=True):
        super(UNET, self).__init__()
        self.use_rtf = use_rtf # Flag
        self.EnableSkipAttention = EnableSkipAttention

        # Encoder blocks
        self.conv_block_1 = CausalConvBlock(in_channel, 32, (6, 3), (2, 2))
        self.conv_block_2 = CausalConvBlock(32,  32,  (7, 4), (2, 2))
        self.conv_block_3 = CausalConvBlock(32,  64,  (7, 5), (2, 2))
        self.conv_block_4 = CausalConvBlock(64,  64,  (6, 6), (2, 2))
        self.conv_block_5 = CausalConvBlock(64,  96,  (6, 6), (2, 2))
        self.conv_block_6 = CausalConvBlock(96,  96,  (6, 6), (2, 2))
        self.conv_block_7 = CausalConvBlock(96,  128, (2, 2), (2, 2))
        self.conv_block_8 = CausalConvBlock(128, 256, (2, 2), (1, 1))

        # Only register rtf_encoder if we want to use it
        if False:
            self.rtf_encoder = nn.Sequential(
                # Input: [B, 8, 514, 497]
                nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)),   # [B, 64, 129, 125]
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.5),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(5, 5), padding=(2, 2)),  # [B, 128, 26, 25]
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.5),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1)),  # [B, 256, 9, 9]
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.LeakyReLU(inplace=True),

                nn.AdaptiveAvgPool2d((1, 1))  # Final: [B, 256, 1, 1]
            )
            #self.bottleneck_fusion = nn.Conv2d(512, 256, kernel_size=1)
            # Attention-based fusion: learn how to combine e8 and RTF-encoded features
            self.attn_layer = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.rtf_encoder = None  # Important!
            self.attn_layer  = None

        # Decoder blocks
        self.tran_conv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2,2),
                stride=(1,1)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True))
        self.tran_conv_block_2 = CausalTransConvBlock(256, 128, (2, 2), (2, 2))
        self.tran_conv_block_3 = CausalTransConvBlock(128, 96,  (6, 6), (2, 2))
        self.tran_conv_block_4 = CausalTransConvBlock(96,  96,  (6, 6), (2, 2))
        self.tran_conv_block_5 = CausalTransConvBlock(96,  64,  (6, 6), (2, 2))
        self.tran_conv_block_6 = CausalTransConvBlock(64,  64,  (7, 5), (2, 2))
        self.tran_conv_block_7 = CausalTransConvBlock(64,  32,  (7, 4), (2, 2))
        self.tran_conv_block_8 = CausalTransConvBlock(32,  32,  (6, 3), (2, 2))

        if EnableSkipAttention == 0: # 'Standard' mode
            self.last_conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=in_channel,
                    kernel_size=1,
                    stride=1),
                nn.BatchNorm2d(num_features=in_channel),
                nn.Dropout2d(0.5),
                nn.LeakyReLU(inplace=True))
        else:
            self.last_conv_block = CausalTransConvBlock(32, in_channel, (1, 1), (1, 1))

        # Dense layer    
        if activation == 'tanh':
            self.dense = nn.Sequential(nn.Linear(514, 514),nn.Tanh())
        else:
            self.dense = nn.Sequential(nn.Linear(514, 514),nn.Sigmoid())

        
    def forward(self, x, rtf=None):
        # Ilai Z 
        # Changing from reciving x to reciving rtf
        #skip = torch.zeros_like(x)
        # x shape is torch.Size([16, 8, 514, 497])
        # Encoder blocks
        rtf_rand =   torch.randn_like(rtf)
        rtf = x # Only for 24.09.25 model
        # e1 = self.conv_block_1(x)  #torch.Size([16, 32, 255, 248])
        e1 = self.conv_block_1(rtf)  #torch.Size([16, 32, 255, 248])
        e2 = self.conv_block_2(e1) #torch.Size([16, 32, 125, 123])
        e3 = self.conv_block_3(e2) #torch.Size([16, 64, 60, 60])
        e4 = self.conv_block_4(e3) #torch.Size([16, 64, 28, 28])
        e5 = self.conv_block_5(e4) #torch.Size([16, 96, 12, 12])
        e6 = self.conv_block_6(e5) #torch.Size([16, 96, 4, 4])
        e7 = self.conv_block_7(e6) #torch.Size([16, 128, 2, 2])
        e8 = self.conv_block_8(e7) #torch.Size([16, 256, 1, 1])

        # if self.rtf_encoder is not None and rtf is not None:
        #     # rtf shape is torch.Size([8, 8, 514, 497])
        #     rtf_encoded = self.rtf_encoder(rtf)  # [B, 256, 1, 1]
        #     fused = torch.cat([e8, rtf_encoded], dim=1)  # [B, 512, 1, 1]
        #     attn = self.attn_layer(fused)  # [B, 256, 1, 1], between 0-1
        #     e8 = e8 * attn  # Apply attention weights

        if self.EnableSkipAttention == 0: # Standard mode
            EnableSkipAtt = 0
        else: # Attention mode
            EnableSkipAtt = 1

        # Decoder blocks
        d = self.tran_conv_block_1(e8)
        d,_ = self.tran_conv_block_2(d, e7, EnableSkipAtt)
        d,_ = self.tran_conv_block_3(d, e6, EnableSkipAtt)
        d,_ = self.tran_conv_block_4(d, e5, EnableSkipAtt)
        d,_ = self.tran_conv_block_5(d, e4, EnableSkipAtt)
        d,_ = self.tran_conv_block_6(d, e3, EnableSkipAtt)
        d,_ = self.tran_conv_block_7(d, e2, EnableSkipAtt)
        d,_ = self.tran_conv_block_8(d, e1, EnableSkipAtt)

        if self.EnableSkipAttention == 0: # Standard mode
            d = self.last_conv_block(d)
        else: # Attention mode
           # d,skip = self.last_conv_block(d, x, EnableSkipAtt) # 07 . 08 Ilai Z
            d,skip = self.last_conv_block(d, rtf, EnableSkipAtt)

        d = d.permute(0,1,3,2)
        d = self.dense(d).permute(0,1,3,2)

        return d,skip       
    
# For attention between RTF and 

def split_complex(x):
    # x: [B, M, 2F, T] -> (xr, xi) each [B,M,F,T]
    F2 = x.shape[2]
    assert F2 % 2 == 0, "freq dim must be 2*F"
    F = F2 // 2
    xr = x[:, :, :F, :]
    xi = x[:, :, F:, :]
    return xr, xi

def complex_inner_conj(h_r, h_i, y_r, y_i):
    # sum_m conj(h_m) * y_m  -> complex scalar per (F,T)
    # real:  sum_m (h_r*y_r + h_i*y_i)
    # imag:  sum_m (-h_r*y_i + h_i*y_r)
    re = (h_r * y_r + h_i * y_i).sum(dim=1, keepdim=False)
    im = (-h_r * y_i + h_i * y_r).sum(dim=1, keepdim=False)
    return re, im  # [B, F, T] each

import torch
import torch.nn as nn

class PhysCorrFeat(nn.Module):
    """
    Build phase-aware correlation features per TF-bin from raw complex inputs:
      Inputs: rtf_raw, mix_raw: [B, M, 2F, T]  (real/imag stacked)
      Output: corr_feats:       [B, 3,   F, T]  (Re{p}, Im{p}, |p|)
    """
    def __init__(self):
        super().__init__()

    def forward(self, rtf_raw, mix_raw):
        eps = 1e-8
        B, M, F2, T = mix_raw.shape
        assert F2 % 2 == 0, f"F must be even (real/imag stacked). Got F={F2}."
        Fpos = F2 // 2

        # split complex (real/imag halves)
        hr, hi = rtf_raw[:, :, :Fpos, :], rtf_raw[:, :, Fpos:, :]
        yr, yi = mix_raw[:, :, :Fpos, :], mix_raw[:, :, Fpos:, :]

        # mic-wise l2 normalization (scale-invariant)
        h_norm = (hr**2 + hi**2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
        y_norm = (yr**2 + yi**2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
        hr, hi = hr / h_norm, hi / h_norm
        yr, yi = yr / y_norm, yi / y_norm

        # p = h^H y (complex), per (Fpos,T)
        pre = (hr * yr + hi * yi).sum(dim=1, keepdim=True)      # [B,1,Fpos,T]
        pim = (-hr * yi + hi * yr).sum(dim=1, keepdim=True)     # [B,1,Fpos,T]
        mag = torch.sqrt(pre**2 + pim**2)                       # [B,1,Fpos,T]

        # stack to 2*Fpos with phase-aware placement
        corr_re = torch.cat([pre, torch.zeros_like(pre)], dim=2)        # Re in real half
        corr_im = torch.cat([torch.zeros_like(pim), pim], dim=2)        # Im in imag half
        corr_ma = torch.cat([0.5 * mag, 0.5 * mag], dim=2)              # |p| duplicated, scaled

        corr_feats = torch.cat([corr_re, corr_im, corr_ma], dim=1)      # [B,3,2*Fpos,T]
        return corr_feats
    


class AttentionFusionBlock(nn.Module):
    """
    Local temporal cross-attention between mixture and RTF.
    Acts as a learned subspace-tracking filter per frequency bin.

    mix, rtf_estimate: [B, C, F, T]
    returns: fused [B, stem_ch, F, T]
    """

    def __init__(self, rtf_in_ch, mix_in_ch,
                 stem_ch=8, num_heads=1,
                 attn_win=48, stride=32):
        super().__init__()
        self.attn_win = attn_win
        self.stride = stride

        # --- light feature stems ---
        self.rtf_stem = nn.Sequential(
            nn.Conv2d(rtf_in_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.mix_stem = nn.Sequential(
            nn.Conv2d(mix_in_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )

        # --- local cross-attention ---
        self.attn = nn.MultiheadAttention(embed_dim=stem_ch,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(stem_ch)

        self.post = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, mix, rtf_estimate):
        B, _, F, T = mix.shape

        # --- stems ---
        mix_s = self.mix_stem(mix)          # [B,S,F,T]
        rtf_s = self.rtf_stem(rtf_estimate) # [B,S,F,T]

        # flatten frequency dimension
        mix_flat = mix_s.permute(0, 2, 3, 1).reshape(B * F, T, -1)
        rtf_flat = rtf_s.permute(0, 2, 3, 1).reshape(B * F, T, -1)

        fused = torch.zeros_like(mix_flat, device=mix.device)

        # --- sliding local attention ---
        for t_start in range(0, T, self.stride):
            t_end = min(t_start + self.attn_win, T)
            q = mix_flat[:, t_start:t_end, :]
            k = rtf_flat[:, t_start:t_end, :]
            v = rtf_flat[:, t_start:t_end, :]
            attn_out, _ = self.attn(q, k, v, need_weights=False)
            fused[:, t_start:t_end, :] = self.norm(q + attn_out)

        # reshape back
        fused = fused.reshape(B, F, T, -1).permute(0, 3, 1, 2)
        return self.post(fused)


class UNETDualInput(nn.Module):
    """
    Dual-input UNet:
      - Inputs: RTF [B, C_rtf, F, T], MIX [B, C_mix, F, T]
      - Stage 0: light stems on each, then concat -> encoder as usual
      - Decoder + (optional) skip attention preserved
      - Output channels = out_channels (typically equals C_mix)

    Args:
        rtf_in_ch (int):  channels of RTF input (e.g., 8)
        mix_in_ch (int):  channels of MIX input (e.g., 8)
        out_channels (int): number of output channels (usually mix_in_ch)
        activation (str): 'sigmoid' or 'tanh' for the dense tail
        EnableSkipAttention (int): 0 = off, 1 = on
        stem_each (int): channels produced by each stem before concat
    """
    def __init__(
        self,
        rtf_in_ch: int,
        mix_in_ch: int,
        out_channels: int,
        activation: str = 'tanh',
        EnableSkipAttention: int = 0,
        stem_each: int = 8,
    ):
        super().__init__()
        self.EnableSkipAttention = EnableSkipAttention
        self.out_channels = out_channels


        self.attn_fusion = AttentionFusionBlock(rtf_in_ch, mix_in_ch, stem_ch=8, num_heads=2)
        in_channel = 8 + mix_in_ch            # rtf_s + mix_s + corr(3)
        # ---- Encoder ----
        self.conv_block_1 = CausalConvBlock(in_channel, 32,  (6, 3), (2, 2))
        self.conv_block_2 = CausalConvBlock(32,        32,  (7, 4), (2, 2))
        self.conv_block_3 = CausalConvBlock(32,        64,  (7, 5), (2, 2))
        self.conv_block_4 = CausalConvBlock(64,        64,  (6, 6), (2, 2))
        self.conv_block_5 = CausalConvBlock(64,        96,  (6, 6), (2, 2))
        self.conv_block_6 = CausalConvBlock(96,        96,  (6, 6), (2, 2))
        self.conv_block_7 = CausalConvBlock(96,        128, (2, 2), (2, 2))
        self.conv_block_8 = CausalConvBlock(128,       256, (2, 2), (1, 1))

        # ---- Decoder ----
        self.tran_conv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True),
        )
        self.tran_conv_block_2 = CausalTransConvBlock(256, 128, (2, 2), (2, 2))
        self.tran_conv_block_3 = CausalTransConvBlock(128, 96,  (6, 6), (2, 2))
        self.tran_conv_block_4 = CausalTransConvBlock(96,  96,  (6, 6), (2, 2))
        self.tran_conv_block_5 = CausalTransConvBlock(96,  64,  (6, 6), (2, 2))
        self.tran_conv_block_6 = CausalTransConvBlock(64,  64,  (7, 5), (2, 2))
        self.tran_conv_block_7 = CausalTransConvBlock(64,  32,  (7, 4), (2, 2))
        self.tran_conv_block_8 = CausalTransConvBlock(32,  32,  (6, 3), (2, 2))

        # ---- Last block ----
        if EnableSkipAttention == 0:
            self.last_conv_block = nn.Sequential(
                nn.Conv2d(32, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(0.5),
                nn.LeakyReLU(inplace=True),
            )
        else:
            # This block expects a skip tensor with channels == out_channels.
            self.last_conv_block = CausalTransConvBlock(32, out_channels, (1, 1), (1, 1))

        # ---- Dense tail over F=514 ----
        if activation == 'tanh':
            self.dense = nn.Sequential(nn.Linear(514, 514), nn.Tanh())
        else:
            self.dense = nn.Sequential(nn.Linear(514, 514), nn.Sigmoid())

    def forward(self, mix, rtf_estimate, DUAL_MODEL):
        """
        rtf: [B, C_rtf, 514, T]
        mix: [B, C_mix, 514, T]
        returns:
            d:    [B, out_channels, 514, T]
            skip: last skip used (only meaningful when attention is enabled at the last block)
        """

        ### ONLY FOR MIX MODEL:

        x = self.attn_fusion(mix, rtf_estimate)
        x = torch.cat([x, mix], dim=1)
        # ---- encoder ----
        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = self.conv_block_6(e5)
        e7 = self.conv_block_7(e6)
        e8 = self.conv_block_8(e7)

        EnableSkipAtt = 1 if self.EnableSkipAttention else 0

        # ---- decoder ----
        d = self.tran_conv_block_1(e8)
        d, _ = self.tran_conv_block_2(d, e7, EnableSkipAtt)
        d, _ = self.tran_conv_block_3(d, e6, EnableSkipAtt)
        d, _ = self.tran_conv_block_4(d, e5, EnableSkipAtt)
        d, _ = self.tran_conv_block_5(d, e4, EnableSkipAtt)
        d, _ = self.tran_conv_block_6(d, e3, EnableSkipAtt)
        d, _ = self.tran_conv_block_7(d, e2, EnableSkipAtt)
        d, _ = self.tran_conv_block_8(d, e1, EnableSkipAtt)

        # ---- last step ----
        if EnableSkipAtt == 0:
            d = self.last_conv_block(d)
            last_skip = None
        else:
            # Provide MIX as the final skip so its channels match out_channels.
            # If you prefer RTF here, ensure its channels == out_channels.
            d, last_skip = self.last_conv_block(d, mix, EnableSkipAtt)

        # ---- dense over frequency dim (assumed 514) ----
        d = d.permute(0, 1, 3, 2)        # [B, C, T, F]
        d = self.dense(d)                # linear over F
        d = d.permute(0, 1, 3, 2)        # [B, C, F, T]

        return d, last_skip
