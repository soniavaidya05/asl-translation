import torch
import torch.nn as nn


class BottleneckLayer(nn.Module):
    """Bottleneck fusion layer (Nagrani et al. 2021). channels only share info through the averaged tokens"""

    def __init__(self, d_model=128, nhead=4, num_tokens=4, d_ff=512, dropout=0.1, learn_tokens=True):
        super().__init__()
        self.num_tokens = num_tokens

        # only the first layer has its own tokens, later layers get them from the layer before
        self.tokens = None
        if learn_tokens:
            self.tokens = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

        self.encoders = nn.ModuleList()
        for _ in range(3):
            self.encoders.append(nn.TransformerEncoderLayer(
                d_model, nhead, d_ff, dropout, activation="gelu", batch_first=True))

    def forward(self, hands, face, body, tokens=None, masks=None):
        # hands, face, body: [B, T, d_model] (T can be different per channel)
        # masks: list of 3 bool tensors [B, T], True = padded frame
        batch_size = hands.shape[0]
        if tokens is None:
            tokens = self.tokens.expand(batch_size, -1, -1)
        if masks is None:
            masks = [None, None, None]

        token_mask = torch.zeros(batch_size, self.num_tokens, dtype=torch.bool, device=hands.device)

        new_streams = []
        new_tokens = []
        for x, mask, encoder in zip([hands, face, body], masks, self.encoders):
            T = x.shape[1]
            if mask is None:
                mask = torch.zeros(batch_size, T, dtype=torch.bool, device=x.device)

            # zero out padded frames first so NaNs from missing keypoints can't leak in
            x = x.masked_fill(mask.unsqueeze(-1), 0)

            combined = torch.cat([x, tokens], dim=1)
            full_mask = torch.cat([mask, token_mask], dim=1)
            out = encoder(combined, src_key_padding_mask=full_mask)

            new_streams.append(out[:, :T].masked_fill(mask.unsqueeze(-1), 0))
            new_tokens.append(out[:, T:])

        avg_tokens = torch.stack(new_tokens).mean(dim=0)
        return new_streams, avg_tokens


class BottleneckFusion(nn.Module):
    """Stacked bottleneck layers. Layer 1 gathers info into the tokens, layer 2+ spreads it back to each channel"""

    def __init__(self, d_model=128, nhead=4, num_tokens=4, num_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        assert num_layers >= 1, "need at least one layer"
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(BottleneckLayer(d_model, nhead, num_tokens, d_ff, dropout, learn_tokens=(i == 0)))

    def forward(self, hands, face, body, masks=None):
        batch_size = hands.shape[0]
        streams = [hands, face, body]
        if masks is None:
            masks = [torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device) for x in streams]

        tokens = None
        for layer in self.layers:
            streams, tokens = layer(*streams, tokens=tokens, masks=masks)

        # decoder memory is hands, face, body, tokens all in one sequence
        memory = torch.cat(streams + [tokens], dim=1)
        token_mask = torch.zeros(batch_size, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        memory_mask = torch.cat(list(masks) + [token_mask], dim=1)
        return memory, memory_mask, streams, tokens
