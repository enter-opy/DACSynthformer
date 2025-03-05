import torch
import torch.nn as nn

# Rotary Positional Embedding
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, self.dim, 2.0) / self.dim))
        t = torch.arange(max_len).unsqueeze(1)
        freqs = t * inv_freq.unsqueeze(0)
        self.register_buffer('sinusoid', torch.cat([freqs.sin(), freqs.cos()], dim=-1))

    def forward(self, x, offset=0):
        n, seq_len, d = x.shape
        sinusoid = self.sinusoid[offset:offset + seq_len, :].to(x.device)
        sinusoid = sinusoid.repeat_interleave(2, dim=1)  # Ensure sinusoid covers all dimensions
        sin_part, cos_part = sinusoid[:, :d//2], sinusoid[:, d//2:]

        x_sin = x[:, :, :d//2] * sin_part - x[:, :, d//2:] * cos_part
        x_cos = x[:, :, :d//2] * cos_part + x[:, :, d//2:] * sin_part

        return torch.cat((x_sin, x_cos), dim=-1)

# Multiembedding Layer
class MultiEmbedding(nn.Module):
    def __init__(self, vocab_size, per_token_embed_size, num_tokens):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, per_token_embed_size) for _ in range(num_tokens)])

    def forward(self, x):
        # x shape: (batch, seq_len, num_tokens)
        embeddings = [self.embeddings[i](x[:, :, i]) for i in range(len(self.embeddings))]
        return torch.cat(embeddings, dim=-1)  # Concatenate embeddings along the last dimension
    
#--------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, forward_expansion, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, src, mask):
        attn_output, _ = self.attention(src, src, src, attn_mask=mask)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src
    
#-------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, max_len, num_codebooks, vocab_size):
        super().__init__()
        
        self.embed_size=embed_size
        self.num_layers=num_layers
        self.heads=heads
        self.forward_expansion=forward_expansion
        self.dropout=dropout
        self.max_len=max_len  # used for rotary encoder
        self.num_codebooks=num_codebooks
        self.vocab_size=vocab_size
        
        self.embed = MultiEmbedding(vocab_size, embed_size//num_codebooks, num_codebooks)  #2nd arg is embedding size per token
        self.pos_encoder = RotaryPositionalEmbedding(embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, forward_expansion, heads, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, num_codebooks * vocab_size)

    def forward(self, src, src_mask):
        src = self.embed(src)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        logits = self.output_layer(src)
        return logits.view(src.shape[0], src.shape[1], -1, 1024)  # Reshape for individual token vocab
