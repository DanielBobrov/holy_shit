import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from torch.cuda.amp import autocast


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class InputEncoder(nn.Module):
    """
    InputEncoder transforms input tokens into an internal knowledge-space representation.
    It combines token embeddings with positional encodings.
    """
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tokens into an internal representation.
        
        Args:
            tokens: Tensor of token indices, shape [batch_size, seq_len]
            
        Returns:
            Tensor: Embedded representation, shape [batch_size, seq_len, d_model]
        """
        embeddings = self.token_embedding(tokens)  # [batch_size, seq_len, d_model]
        embeddings = self.position_encoding(embeddings)
        return self.dropout(embeddings)


class ModifiedMultiHeadAttention(nn.Module):
    """
    Modified multi-head attention with adapter layers for fine-tuning attention.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Adapter layers (down-projection followed by up-projection)
        adapter_dim = d_model // 8  # Reduced dimensionality for the adapter
        self.adapter_down = nn.Linear(d_model, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, d_model)
        self.adapter_act = nn.GELU()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None, output_attentions=False):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Adapter layers (residual connection)
        adapter_output = self.adapter_down(output)
        adapter_output = self.adapter_act(adapter_output)
        adapter_output = self.adapter_up(adapter_output)
        
        final_output = output + adapter_output
        
        if output_attentions:
            return final_output, attn_weights
        return final_output


class ThinkingBlock(nn.Module):
    """
    ThinkingBlock implements a single block of thinking with self-attention and feed-forward layers.
    It accumulates knowledge using linear layers while self-attention implements reasoning.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = ModifiedMultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dynamic scaling factor for residual connections
        self.res_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, mask=None, output_attentions=False):
        # Self-attention with residual connection and layer norm
        if output_attentions:
            attn_output, attn_weights = self.self_attn(x, x, x, mask, output_attentions=True)
            x = self.norm1(x + self.res_scale * attn_output)
            
            # Feed-forward with residual connection and layer norm
            ff_output = self.ff(x)
            x = self.norm2(x + self.res_scale * ff_output)
            
            return x, attn_weights
        else:
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.res_scale * attn_output)
            
            # Feed-forward with residual connection and layer norm
            ff_output = self.ff(x)
            x = self.norm2(x + self.res_scale * ff_output)
            
            return x


class ThinkingLayer(nn.Module):
    """
    ThinkingLayer implements multiple thinking blocks for iterative reasoning.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ThinkingBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None, output_attentions=False):
        all_attentions = [] if output_attentions else None
        
        for layer in self.layers:
            if output_attentions:
                x, attn_weights = layer(x, mask, output_attentions=True)
                all_attentions.append(attn_weights)
            else:
                x = layer(x, mask)
                
        if output_attentions:
            return x, all_attentions
        return x


class RecurrentFeedbackMechanism(nn.Module):
    """
    Implements a recurrent feedback loop that feeds thinking layer outputs
    back for further iterations to refine the reasoning process.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Use a combination of linear projection and non-linearity
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        
        # Adaptive gating mechanism to control information flow
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, processed_data: torch.Tensor) -> torch.Tensor:
        """
        Adjusts the processed data for further iterations.
        
        Args:
            processed_data: Output tensor from the thinking layer [batch_size, seq_len, d_model]
            
        Returns:
            Modified tensor for the next iteration [batch_size, seq_len, d_model]
        """
        # Project the data through the feedback mechanism
        projection = self.projection(processed_data)
        
        # Apply gating mechanism to control information flow
        gate_values = self.gate(processed_data)
        
        # Combine original and projected data based on gate values
        feedback = gate_values * projection + (1 - gate_values) * processed_data
        
        # Apply layer normalization
        return self.norm(feedback)


class OutputDecoder(nn.Module):
    """
    OutputDecoder transforms the final internal state into output tokens.
    It maps the refined representation from the iterative reasoning process to the vocabulary.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the final representation into output logits.
        
        Args:
            x: Final representation tensor [batch_size, seq_len, d_model]
            
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        return self.output_projection(x)


class IterativeReasoningModel(nn.Module):
    """
    IterativeReasoningModel integrates all components and implements the iterative reasoning process.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_iterations: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = InputEncoder(vocab_size, d_model, dropout)
        self.thinking_layer = ThinkingLayer(d_model, num_heads, d_ff, num_layers, dropout)
        self.feedback_mechanism = RecurrentFeedbackMechanism(d_model, dropout)
        self.decoder = OutputDecoder(d_model, vocab_size)
        
        self.d_model = d_model
        self.max_iterations = max_iterations
        
        # Initialize weights with improved method
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters using the Kaiming method with fan-in mode."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
        
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_iterations: int = None,
        output_attentions: bool = False,
        output_intermediate: bool = False,
        use_mixed_precision: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List, List]]:
        """
        Process input tokens through the model with iterative reasoning.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len, seq_len]
            num_iterations: Number of reasoning iterations (default: max_iterations)
            output_attentions: Whether to return attention weights
            output_intermediate: Whether to return intermediate representations
            use_mixed_precision: Whether to use mixed precision training
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
            Optional: attention weights and intermediate representations
        """
        if num_iterations is None:
            num_iterations = self.max_iterations
            
        # Track outputs for visualization if requested
        intermediate_representations = [] if output_intermediate else None
        all_attention_weights = [] if output_attentions else None
        
        # Use mixed precision if requested
        context_manager = autocast() if use_mixed_precision else torch.no_grad()
        
        with context_manager:
            # Encode inputs
            embeddings = self.encoder(tokens)
            
            # Iterative reasoning process
            representations = embeddings
            if output_intermediate:
                intermediate_representations.append(representations.detach().clone())
            
            for _ in range(num_iterations):
                # Process through thinking layer
                if output_attentions:
                    representations, attn_weights = self.thinking_layer(
                        representations, mask, output_attentions=True
                    )
                    all_attention_weights.append(attn_weights)
                else:
                    representations = self.thinking_layer(representations, mask)
                
                # Apply feedback mechanism to refine representations
                representations = self.feedback_mechanism(representations)
                
                if output_intermediate:
                    intermediate_representations.append(representations.detach().clone())
            
            # Decode final representations
            logits = self.decoder(representations)
        
        if output_attentions or output_intermediate:
            return logits, all_attention_weights, intermediate_representations
        return logits
    
    def generate(
        self,
        tokens: torch.Tensor,
        mask_token_id: int,
        max_len: int = 100,
        num_iterations: int = None
    ) -> torch.Tensor:
        """
        Generate output by replacing masked tokens with predicted values.
        
        Args:
            tokens: Input token indices with masks [batch_size, seq_len]
            mask_token_id: ID of the mask token
            max_len: Maximum sequence length
            num_iterations: Number of reasoning iterations
            
        Returns:
            Generated sequence [batch_size, seq_len]
        """
        batch_size, seq_len = tokens.shape
        
        # Copy input tokens
        output_tokens = tokens.clone()
        
        # Find positions of mask tokens
        mask_positions = (output_tokens == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) == 0:
            return output_tokens  # No masks to fill
            
        # Forward pass with the current tokens
        logits = self.forward(output_tokens, num_iterations=num_iterations)
        
        # For each mask position, select the token with highest probability
        for batch_idx, seq_idx in zip(mask_positions[0], mask_positions[1]):
            output_tokens[batch_idx, seq_idx] = logits[batch_idx, seq_idx].argmax(dim=-1)
            
        return output_tokens
    
    def visualize_attention(
        self, 
        tokens: torch.Tensor,
        token_ids_to_words: Dict[int, str],
        mask_token_id: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention patterns for a given input.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            token_ids_to_words: Mapping from token ids to readable words
            mask_token_id: ID of the mask token
            save_path: Path to save the visualization (if None, display instead)
        """
        # Generate predictions with attention outputs
        with torch.no_grad():
            _, attention_weights, _ = self.forward(
                tokens, output_attentions=True, output_intermediate=True
            )
        
        # Get token labels for the sequence
        seq_tokens = []
        for t in tokens[0].cpu().numpy():
            if t in token_ids_to_words:
                seq_tokens.append(token_ids_to_words[t])
            else:
                seq_tokens.append(f"<{t}>")
        
        # Create a figure for the attention visualization
        fig, axes = plt.subplots(
            len(attention_weights),
            len(attention_weights[0]),
            figsize=(20, 15),
            squeeze=False
        )
        
        # Find the mask position
        mask_pos = (tokens[0] == mask_token_id).nonzero(as_tuple=True)[0].item()
        
        for iter_idx, iter_attentions in enumerate(attention_weights):
            for layer_idx, layer_attention in enumerate(iter_attentions):
                # Get attention weights from the mask token to all other tokens
                # Take the first head's attention for simplicity
                attn = layer_attention[0, 0, mask_pos].cpu()
                
                ax = axes[iter_idx, layer_idx]
                sns.heatmap(
                    attn.unsqueeze(0),
                    annot=False,
                    fmt="",
                    cmap="viridis",
                    xticklabels=seq_tokens,
                    yticklabels=["MASK"],
                    ax=ax
                )
                ax.set_title(f"Iteration {iter_idx+1}, Layer {layer_idx+1}")
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
        
    def visualize_iterative_reasoning(
        self,
        tokens: torch.Tensor,
        mask_token_id: int,
        token_ids_to_words: Dict[int, str],
        save_path: Optional[str] = None
    ):
        """
        Visualize how the representation at the mask position evolves during iterations.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            mask_token_id: ID of the mask token
            token_ids_to_words: Mapping from token ids to readable words
            save_path: Path to save the visualization (if None, display instead)
        """
        # Generate predictions with intermediate outputs
        with torch.no_grad():
            logits, _, intermediate = self.forward(
                tokens, output_attentions=True, output_intermediate=True
            )
        
        # Find the mask position
        mask_pos = (tokens[0] == mask_token_id).nonzero(as_tuple=True)[0].item()
        
        # Get the evolution of the hidden state at the mask position
        mask_states = [hidden[0, mask_pos].cpu().numpy() for hidden in intermediate]
        
        # Use PCA to reduce the dimensionality for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_states = pca.fit_transform(mask_states)
        
        # Create a figure for the visualization
        plt.figure(figsize=(10, 8))
        
        # Plot the trajectory of the representation
        plt.plot(
            reduced_states[:, 0],
            reduced_states[:, 1],
            marker='o',
            linestyle='-'
        )
        
        # Add annotations for each iteration
        for i, (x, y) in enumerate(reduced_states):
            plt.annotate(
                f"{i}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        
        plt.title("Evolution of Mask Token Representation")
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()


class ReasoningDataset(torch.utils.data.Dataset):
    """Dataset for the reasoning task."""
    
    def __init__(self, data_path: str, vocab_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            
        # Create token to index mapping
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        
        # Special tokens
        self.mask_token_id = self.token2idx.get("<mask>")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize input and target text
        input_tokens = self._tokenize(item["input_text"])
        target_tokens = self._tokenize(item["target_text"])
        
        return {
            "input_ids": input_tokens,
            "target_ids": target_tokens,
            "type": item["type"]
        }
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Convert text to token indices."""
        tokens = []
        current_token = ""
        
        for char in text:
            current_token += char
            if current_token in self.token2idx:
                tokens.append(self.token2idx[current_token])
                current_token = ""
                
        if current_token:  # Handle any remaining characters
            for char in current_token:
                if char in self.token2idx:
                    tokens.append(self.token2idx[char])
        
        return torch.tensor(tokens, dtype=torch.long)


def train_model(
    model: IterativeReasoningModel,
    train_dataset: ReasoningDataset,
    val_dataset: ReasoningDataset,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    weight_decay: float = 0.01,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[str] = None,
    use_mixed_precision: bool = False,
    label_smoothing: float = 0.1,
    use_consistency_loss: bool = True,
    consistency_weight: float = 0.1,
    gradient_accumulation_steps: int = 1
):
    """Train the model on the given dataset with enhanced training techniques."""
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=label_smoothing
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    
    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Forward pass with mixed precision if enabled
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    if use_consistency_loss:
                        # Forward pass with intermediate outputs for consistency loss
                        logits, _, intermediate = model(
                            input_ids, 
                            output_intermediate=True,
                            use_mixed_precision=True
                        )
                        
                        # Main cross-entropy loss
                        ce_loss = criterion(
                            logits.view(-1, logits.size(-1)),
                            target_ids.view(-1)
                        )
                        
                        # Consistency loss between consecutive iterations
                        cons_loss = 0.0
                        for i in range(1, len(intermediate)):
                            cons_loss += F.mse_loss(
                                intermediate[i],
                                intermediate[i-1].detach()
                            )
                        
                        loss = ce_loss + consistency_weight * cons_loss
                    else:
                        logits = model(input_ids, use_mixed_precision=True)
                        loss = criterion(
                            logits.view(-1, logits.size(-1)),
                            target_ids.view(-1)
                        )
                
                # Backward pass with gradient scaling
                scaler.scale(loss / gradient_accumulation_steps).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                if use_consistency_loss:
                    # Forward pass with intermediate outputs for consistency loss
                    logits, _, intermediate = model(
                        input_ids, 
                        output_intermediate=True
                    )
                    
                    # Main cross-entropy loss
                    ce_loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                    
                    # Consistency loss between consecutive iterations
                    cons_loss = 0.0
                    for i in range(1, len(intermediate)):
                        cons_loss += F.mse_loss(
                            intermediate[i],
                            intermediate[i-1].detach()
                        )
                    
                    loss = ce_loss + consistency_weight * cons_loss
                else:
                    logits = model(input_ids)
                    loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                
                # Backward pass
                (loss / gradient_accumulation_steps).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                
                logits = model(input_ids)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
    return model


def evaluate_model(
    model: IterativeReasoningModel,
    test_dataset: ReasoningDataset,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate the model on the test dataset."""
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_types = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            types = batch["type"]
            
            # Generate predictions
            predictions = model.generate(
                input_ids,
                mask_token_id=test_dataset.mask_token_id
            )
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_ids.cpu().numpy())
            all_types.extend(types)
    
    # Calculate metrics
    correct = 0
    total = 0
    
    # Group metrics by type
    metrics_by_type = {}
    
    for pred, target, type_name in zip(all_predictions, all_targets, all_types):
        # Check if predictions match targets
        match = (pred == target).all()
        
        if type_name not in metrics_by_type:
            metrics_by_type[type_name] = {"correct": 0, "total": 0}
            
        metrics_by_type[type_name]["total"] += 1
        if match:
            metrics_by_type[type_name]["correct"] += 1
            correct += 1
        total += 1
    
    # Print overall accuracy
    print(f"Overall Accuracy: {correct/total:.4f}")
    
    # Print accuracy by type
    print("\nAccuracy by Type:")
    for type_name, metrics in metrics_by_type.items():
        accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        print(f"{type_name}: {accuracy:.4f} ({metrics['correct']}/{metrics['total']})")
    
    return metrics_by_type


def main():
    # Paths to data files
    vocab_path = "/home/user/Pycharm/shit/reasoning/vocab.json"
    train_path = "/home/user/Pycharm/shit/reasoning/train.json"
    valid_path = "/home/user/Pycharm/shit/reasoning/valid.json"
    test_path = "/home/user/Pycharm/shit/reasoning/test.json"
    
    # Load datasets
    train_dataset = ReasoningDataset(train_path, vocab_path)
    val_dataset = ReasoningDataset(valid_path, vocab_path)
    test_dataset = ReasoningDataset(test_path, vocab_path)
    
    # Create model
    model = IterativeReasoningModel(
        vocab_size=len(train_dataset.vocab),
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        max_iterations=20,
        dropout=0.1
    )
    
    # Train model
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        num_epochs=100,  # Can be adjusted based on when grokking occurs
        learning_rate=1e-4
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=32
    )
    
    # Save model
    torch.save(model.state_dict(), "reasoning_model.pt")


if __name__ == "__main__":
    main()

