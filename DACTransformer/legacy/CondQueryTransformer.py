"""
In this work, we introduce a novel approach to conditional transformer architectures that leverages class-specific query projections in the attention mechanism. Unlike traditional multi-head attention, where each head uses a single set of learned query, key, and value projections, our method employs multiple class-specific query projections while maintaining shared key and value projections across all classes. This architecture allows for class-dependent attention patterns while still benefiting from the representational power of multi-head attention.

The core innovation lies in how class information is utilized during both training and inference. During training with one-hot class vectors, only the relevant class-specific query projection is effectively used for each input. This encourages each class projection to specialize in attending to features most relevant to its corresponding class. During inference, the model can accept non-one-hot class vectors, enabling a weighted combination of class-specific queries. This mechanism allows for smooth interpolation between class-specific attention patterns, potentially generating novel outputs that blend characteristics of multiple classes.

This approach differs from standard conditional transformers in two key aspects. First, it provides a more granular method of incorporating class information directly into the attention mechanism, rather than relying solely on concatenated class embeddings or global conditioning signals. Second, it offers a natural way to perform "soft" class conditioning during inference, opening up possibilities for controlled generation and smooth transitions between different class-specific behaviors. This architecture aims to enhance the model's ability to generate diverse and controllable outputs, particularly in tasks requiring fine-grained control over generated content or the ability to smoothly transition between different styles or classes.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #print(f"------- In Rotary forward, x.shape is ={x.shape}")
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
        #print(f"------- In MultiEmbedding will return a vector of shape  ={torch.cat(embeddings, dim=-1).shape}")
        return torch.cat(embeddings, dim=-1)  # Concatenate embeddings along the last dimension
    

#########################################################################################
#    new
# ########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassDependentQueryAttention(nn.Module):
    def __init__(self, num_classes, d_model, num_heads, dropout=0.1, verbose=0):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.verbose = verbose
        
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_classes)])
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, conditioning, mask=None):
        batch_size, seq_len, _ = x.size()
        
        if self.verbose > 1:
            print(f"ClassDependentQueryAttention input shape: {x.shape}")
            print(f"Conditioning shape: {conditioning.shape}")
        
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        class_queries = [W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
                         for W_q in self.W_q]
        
        if conditioning is None:
            q = sum(query for query in class_queries) / self.num_classes
        else:
            class_weights = conditioning[:, :, :self.num_classes]
            q = sum(class_weights[:, :, i].unsqueeze(1).unsqueeze(-1) * query 
                    for i, query in enumerate(class_queries))
        
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, num_classes, embed_size, cond_size, num_heads, forward_expansion, dropout=0.1, verbose=0):
        super().__init__()
        self.embed_size = embed_size
        self.cond_size = cond_size
        self.d_model = embed_size + cond_size
        self.verbose = verbose
        
        self.attention = ClassDependentQueryAttention(num_classes, self.d_model, num_heads, dropout, verbose)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, conditioning, mask=None):
        if self.verbose > 1:
            print(f"TransformerBlock input shape: {x.shape}")
        
        # Apply normalization and dropout before concatenating conditioning
        x = self.norm1(x)
        x = self.dropout(x)
        
        # Concatenate conditioning
        x_with_cond = torch.cat([x, conditioning], dim=-1)
        
        # Apply attention
        attn_output = self.attention(x_with_cond, conditioning, mask)
        
        # Remove conditioning from attention output
        attn_output = attn_output[:, :, :self.embed_size]
        
        # Apply residual connection
        x = x + attn_output
        
        # Apply feed-forward layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        if self.verbose > 1:
            print(f"TransformerBlock output shape: {x.shape}")
        
        return x

class ClassConditionedTransformer(nn.Module):
    def __init__(self,  embed_size, num_layers, num_heads, forward_expansion, dropout, max_len, num_classes, num_codebooks, vocab_size, cond_size, verbose=0) :

        super().__init__()
        self.num_classes = num_classes
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.cond_size = cond_size
        self.verbose = verbose  
        self.forward_expansion= forward_expansion
        self.max_len=max_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.embedding = MultiEmbedding(vocab_size, embed_size // num_codebooks, num_codebooks)  
        self.positional_encoding = RotaryPositionalEmbedding(embed_size, max_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_classes, embed_size, cond_size, num_heads, forward_expansion, dropout, verbose)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(embed_size, num_codebooks * vocab_size)
        
    def forward(self, x, conditioning=None, mask=None):
        if self.verbose > 0:
            print(f"Input shape: {x.shape}")
            if conditioning is not None:
                print(f"Conditioning shape: {conditioning.shape}")
        
        x = self.embedding(x)
        x = self.positional_encoding(x)  # Apply positional encoding to embeddings
        
        for block in self.transformer_blocks:
            x = block(x, conditioning, mask)
        
        logits = self.final_layer(x)
        logits = logits.view(logits.size(0), logits.size(1), self.num_codebooks, self.vocab_size)
        
        if self.verbose > 0:
            print(f"Output shape: {logits.shape}")
        
        return logits

# # Usage example:
# model = ClassConditionedTransformer(
#     num_classes=7,
#     num_codebooks=8,
#     vocab_size=1024,
#     embed_size=512,
#     cond_size=8,  # 7 classes + 1 additional parameter
#     num_heads=8,
#     num_layers=6,
#     forward_expansion=4,
#     dropout=0.1,
#     verbose=1
# )

# # Example input
# batch_size = 32
# seq_length = 100
# x = torch.randint(0, 1024, (batch_size, seq_length, 8))  # [batch_size, seq_length, num_codebooks]
# conditioning = torch.rand(batch_size, seq_length, 8)  # [batch_size, seq_length, cond_size]
# mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

# output = model(x, conditioning, mask)