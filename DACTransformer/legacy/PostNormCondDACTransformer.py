import torch
import torch.nn as nn

"""
In this version of the Transformer, the positional encoding is added to the embedding vector at the input. No conditioning data is concatenated at the input. In each transformer block, the input in first normalized, then the conditioning vector is concatenated with normalized embedding, then that vector is linearly projected back down to the model size for processing by the attention block. That means, the conditioning info never gets positional information explicitly, however, the conditioning information is added for each element in the sequence and pertains only to that sequence element (its history is unimportant). 

"""

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
    def __init__(self, input_size, embed_size, num_heads, dropout, forward_expansion, verbose=0):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.input_size = input_size
        self.verbose = verbose
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.linear_reduce = nn.Linear(input_size, embed_size)  # Adjust size after concatenation

    def forward(self, src, cond, mask=None):
        """
        Forward pass for the TransformerBlock with masking.

        Args:
        - src (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size).
        - cond (torch.Tensor): Conditioning tensor of shape (batch_size, seq_len, cond_size).
        - mask (torch.Tensor): Attention mask of shape (seq_len, seq_len).

        Returns:
        - torch.Tensor: Processed tensor of shape (batch_size, seq_len, embed_size).
        """
        
        # Normalize embeddings first
        normalized_src = self.norm1(src)
        if self.verbose >0  :
            print(f"Normalized src shape: {normalized_src.shape}")

        if self.verbose >0  :
            print(f"cond shape before expanding is : {cond.shape}")
            
        ### # Expand conditional vector to match sequence length
        ### cond_expanded = cond.unsqueeze(1).expand(-1, normalized_src.size(1), -1)
        ### if self.verbose >0  :
        ###    print(f"Condition vector expanded to: {cond_expanded.shape}")
        cond_expanded=cond

        # Concatenate conditional vector after normalization
        combined = torch.cat((normalized_src, cond_expanded), dim=-1)
        if self.verbose >0  :
            print(f"Combined shape (src + cond): {combined.shape}")

        # Reduce dimensionality back to embed_size
        reduced = self.linear_reduce(combined)
        if self.verbose >0  :
            print(f"Reduced shape: {reduced.shape}")

        # Apply self-attention with masking
        attention_output, _ = self.attention(reduced, reduced, reduced, attn_mask=mask)
        if self.verbose >0 :
            print(f"Attention output shape: {attention_output.shape}")
        x = self.dropout_layer(attention_output) + reduced

        # Apply feed-forward network
        forward = self.feed_forward(self.norm2(x))
        if self.verbose >0 :
            print(f"Feed-forward output shape: {forward.shape}")
        out = self.dropout_layer(forward) + x
        return out

    
    
#-------------------------------------------------------------------

class PostNormCondDACTransformerDecoder(nn.Module):

    def __init__(self, embed_size, num_layers, num_heads, forward_expansion, dropout, max_len, num_classes, num_codebooks, vocab_size, cond_size,  verbose=False):
    # num_classes isn´t used here, but it is in other decoders
    #def __init__(self, vocab_size, num_codebooks, embed_size, cond_size, num_layers, num_heads, dropout, forward_expansion, max_len, verbose=False):
        super(PostNormCondDACTransformerDecoder, self).__init__()
        self.embed_size = embed_size
        self.input_size = embed_size + cond_size  # Combined input size
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.verbose = verbose
        
        self.num_heads=num_heads
        self.forward_expansion=forward_expansion
        self.dropout = dropout
        self.max_len=max_len
        self.cond_size=cond_size
        self.num_classes=num_classes
        self.num_layers=num_layers
        
        
        print(f"Setting up MultiEmbedding with vocab_size= {vocab_size}, embed_size= {embed_size}, num_codebooks= {num_codebooks}")
        self.multi_embedding = MultiEmbedding(vocab_size, embed_size // num_codebooks, num_codebooks)
        print(f"Setting up RotaryPositionalEmbedding with embed_size= {embed_size}, max_len= {max_len}")        
        self.positional_embedding = RotaryPositionalEmbedding(embed_size, max_len)

        # Create transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_size=self.input_size,
                    embed_size=embed_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    verbose=verbose  # Pass verbose flag to each block
                )
                for _ in range(num_layers)
            ]
        )
        

        
        self.dropout_layer = nn.Dropout(dropout)
        self.final_layer = nn.Linear(embed_size, num_codebooks * vocab_size)

    def forward(self, src, cond, mask=None):
        """
        Forward pass for the TransformerDecoder.

        Args:
        - src (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_codebooks).
        - cond (torch.Tensor): Conditioning tensor of shape (batch_size, seq_len, cond_size).
        - mask (torch.Tensor): Attention mask of shape (seq_len, seq_len).

        Returns:
        - logits (torch.Tensor): Output logits reshaped to match target shape.
        """
        if (self.verbose > 5) :
            print(f"Source shape: {src.shape}")
            print(f"Condition shape: {cond.shape}")
            if mask is not None:
                print(f"Mask shape: {mask.shape}")

        # Embed the input tokens
        src = self.multi_embedding(src)
        if (self.verbose > 5) :
            print(f"Embedding output shape: {src.shape}")

        # Apply positional encoding and dropout
        src = self.positional_embedding(src)
        src = self.dropout_layer(src)

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if (self.verbose > 6) :
                print(f"Passing through layer {i}")
            src = layer(src, cond, mask)

            
        logits = self.final_layer(src)
        logits = logits.view(logits.size(0), logits.size(1), self.num_codebooks, self.vocab_size)
        
        if self.verbose > 0:
            print(f"Output shape: {logits.shape}")
            print(f"================================================================")
        
        return logits
    
        ### # Reshape logits as per original implementation
        ### logits = src
        ### if (self.verbose > 5) :
        ###     print(f"Final logits shape before reshaping: {logits.shape}")
        ### #return logits.view(src.shape[0], src.shape[1], -1, 1024)
        ### logits = logits.view(logits.size(0), logits.size(1), self.num_codebooks, self.vocab_size)
        ### return logits


    
