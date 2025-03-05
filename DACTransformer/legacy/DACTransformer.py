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
    
#--------------------------------------------------------------
#input_size is embed_size+conditioning vector size.

class TransformerBlock(nn.Module):
    def __init__(self, input_size, embed_size, forward_expansion, num_heads, dropout, verbose=0):
        super().__init__()
        
        self.verbose=verbose
        if verbose >0 : 
            print(f'TBLOCK 0 (init) , input_size is {input_size}, embed_size is {embed_size}')
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_size)
        
        if input_size != embed_size :  # the difference is the size of the conditional vector
            self.linear1 = nn.Linear(input_size, embed_size)  # Linear layer to reduce dimension to embed_size
        else : 
            self.linear1 = None
            
        self.feed_forward = nn.Sequential(
            ###--------- nn.Linear(input_size, forward_expansion * embed_size),
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, src, mask):
        if self.verbose >0 : 
            print(f'TBLOCK 1, src.shape is {src.shape}')
        attn_output, _ = self.attention(src, src, src, attn_mask=mask)
        if self.verbose >0 : 
            print(f'TBLOCK 2, attn_output.shape is {attn_output.shape}')
        src = self.norm1(src + self.dropout(attn_output)) # residual connection from src
        
        if self.linear1 != None : 
            src = self.linear1(src)  # Adjust dimension back to embed_size
        
        if self.verbose >0 : 
            print(f'TBLOCK 3, src.shape is {src.shape}')
        ff_output = self.feed_forward(src)
        if self.verbose >0 : 
            print(f'TBLOCK 4, ff_output.shape is {ff_output.shape}')
        src = self.norm2(src + self.dropout(ff_output))   # residual connection from src
        if self.verbose >0 : 
            print(f'TBLOCK 5, src.shape is {src.shape}')
        return src
    
    
#-------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, forward_expansion, dropout, max_len, num_classes, num_codebooks, vocab_size, cond_size, verbose=0):
        super().__init__()
        
        self.verbose=verbose
        self.embed_size = embed_size
        self.cond_size = cond_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.max_len = max_len  # used for rotary encoder
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        self.embed = MultiEmbedding(vocab_size, embed_size // num_codebooks, num_codebooks)  # 2nd arg is embedding size per token
        print(f'Get a coder with embed_size={embed_size}. cond_size={cond_size}, max_len={max_len}')
        self.pos_encoder = RotaryPositionalEmbedding(embed_size+cond_size, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size + cond_size, embed_size, forward_expansion, num_heads, dropout, verbose) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, num_codebooks * vocab_size)

        
    def forward(self, src, cond_expanded, src_mask):
        src = self.embed(src)
        
        if (self.verbose > 5) :
            print(f'In TransformerDecoder, before concating cond, source.shape is {src.shape}')
        
        if cond_expanded != None : 
            if (self.verbose > 5) :
                print(f'In TransformerDecoder, cond_expanded has shape {cond_expanded.shape}')
        else :
            if (self.verbose > 5) :
                print(f'no conditional expansion of embedding')

        
        if cond_expanded != None :  # else unconditional
            src = torch.cat((src, cond_expanded), dim=-1)
        
        if (self.verbose > 5) :
            print(f'In TransformerDecoder, after concatenating src and cond_expanded, source.shape is {src.shape}')
        
        src = self.pos_encoder(src)
        
        if (self.verbose > 5) :
            print(f'In TransformerDecoder, after positional encodeing, source.shape is {src.shape}')


        # Pass through each transformer layer, re-concatenating conditioning vector
        # --- for layer in self.layers:
        for i, layer in enumerate(self.layers):
            if (self.verbose > 0) :
                print(f'For feeding layer {i}, source.shape is {src.shape}')
            src = layer(src, src_mask)
            if i != len(self.layers) - 1: # don't concat cond data on the last iteration
                if cond_expanded != None :  # else unconditional
                    src = torch.cat((src, cond_expanded), dim=-1)
        
#         print(f'NOW get the logits with source.shape = {src.shape}')
        logits = self.output_layer(src)
#         print(f'shape of logits from output_layer is {logits.shape}')
#         print(f'shape of logits returned from transfermer is {logits.view(src.shape[0], src.shape[1], -1, 1024).shape}')
        return logits.view(src.shape[0], src.shape[1], -1, 1024)  # Adjust reshape as needed
    
    
