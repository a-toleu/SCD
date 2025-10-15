#!/usr/bin/env python
"""
Complete End-to-End Pipeline for Speaker Change Detection
Includes: Audio Loading → Wav2Vec2 Embedding → Graph Construction → Training → Evaluation
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import webrtcvad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score
)


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_csv_data(csv_path):
    """Load CSV file with audio paths and change points"""
    print(f"\n Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Parse JSON format change_points
    df['change_points'] = df['change_points_json'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    
    print(f" Loaded {len(df)} audio files")
    return df


# ============================================================================
# 2. AUDIO SEGMENTATION
# ============================================================================

def segment_filtering(segments, vad_mode=1, sample_rate=16000):
    """Filter silent segments using Voice Activity Detection"""
    vad = webrtcvad.Vad(vad_mode)
    filtered_indices = []

    for i, segment in enumerate(segments):
        segment_pcm = (segment * 32767).astype(np.int16).tobytes()
        frame_duration_ms = 30
        frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2

        has_voice = False
        for j in range(0, len(segment_pcm), frame_bytes):
            frame = segment_pcm[j:j + frame_bytes]
            if len(frame) < frame_bytes:
                break
            if vad.is_speech(frame, sample_rate=sample_rate):
                has_voice = True
                break

        if has_voice:
            filtered_indices.append(i)
 
    return filtered_indices


def segment_and_label_audio(audio_path, change_points, segment_length=1.0, sample_rate=16000):
    """Segment audio and generate binary labels"""
    
    # Extract time points
    if len(change_points) > 0 and isinstance(change_points[0], (list, tuple)):
        change_points = [x[0] for x in change_points]
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    samples_per_segment = int(segment_length * sr + 0.5)
    total_segments = int(np.ceil(len(y) / samples_per_segment))
    
    segments = []
    labels_dict = {}
    change_index = 0
    
    for i in range(total_segments):
        start_sample = i * samples_per_segment
        end_sample = min(start_sample + samples_per_segment, len(y))
        segment = y[start_sample:end_sample]
        segments.append(segment)
        
        # Check if contains change point
        segment_start = i * segment_length
        segment_end = segment_start + segment_length
        
        while change_index < len(change_points) and change_points[change_index] < segment_end:
            if segment_start <= change_points[change_index] < segment_end:
                labels_dict[i] = 1
                change_index += 1
                break
            else:
                change_index += 1
        
        if i not in labels_dict:
            labels_dict[i] = 0
    
    true_labels = np.array([x[1] for x in sorted(labels_dict.items())])
    
    # VAD filtering
    valid_indices = segment_filtering(segments)
    
    filtered_segments = [segments[i] for i in valid_indices]
    filtered_labels = [true_labels[i] for i in valid_indices]
    
    return filtered_segments, np.array(filtered_labels)


def process_dataset(df, segment_length=1.0):
    """Process entire dataset"""
    all_labels = []
    all_segments = []
    
    print(f"\n Processing {len(df)} audio files...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        audio_path = row['audio_path']
        change_points = row['change_points']
        
        if not os.path.exists(audio_path):
            print(f" File not found: {audio_path}")
            continue
        
        try:
            segments, labels = segment_and_label_audio(
                audio_path, change_points, segment_length=segment_length
            )
            
            all_labels.append(labels)
            all_segments.append(segments)
            
        except Exception as e:
            print(f" Error processing {audio_path}: {e}")
            continue
    
    print(f" Processed {len(all_labels)} files")
    return all_labels, all_segments


# ============================================================================
# 3. WAV2VEC2 FEATURE EXTRACTION
# ============================================================================

def pad_or_truncate(segment, target_length=16000):
    """Pad or truncate audio segment to target length"""
    current_length = len(segment)
    if current_length < target_length:
        pad_width = target_length - current_length
        segment = np.pad(segment, (0, pad_width), mode='constant')
    else:
        segment = segment[:target_length]
    return segment


def extract_wav2vec2_features(segments_list, labels_list, 
                               target_sr=16000, batch_size=512, device=None):
    """Extract features using Wav2Vec2"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n Extracting Wav2Vec2 features (device: {device})")
    
    # Load model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    
    for i, segments in enumerate(tqdm(segments_list, desc="Extracting features")):
        # Standardize length
        segments_fixed = [pad_or_truncate(seg, 16000) for seg in segments]
        num_segments = len(segments_fixed)
        
        file_embeddings = []
        file_labels = torch.tensor(labels_list[i], dtype=torch.float32).view(-1, 1).to(device)
        
        # Process in batches
        for start in range(0, num_segments, batch_size):
            end = min(start + batch_size, num_segments)
            batch = np.array(segments_fixed[start:end])
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                inputs = processor(
                    batch_tensor.cpu(), 
                    sampling_rate=target_sr,
                    return_tensors="pt", 
                    padding=True
                ).input_values.to(device)
                
                if inputs.dim() == 3:
                    inputs = inputs.squeeze(0)
                
                outputs = model(inputs).last_hidden_state
                embeddings = outputs.mean(dim=1)
                file_embeddings.append(embeddings)
            
            del batch_tensor, inputs, outputs, embeddings
            torch.cuda.empty_cache()
        
        if file_embeddings:
            all_features.append(torch.cat(file_embeddings, dim=0))
        else:
            all_features.append(torch.zeros((0, model.config.hidden_size)).to(device))
        
        all_labels.append(file_labels)
    
    print(f" Feature extraction complete!")
    return all_features, all_labels


# ============================================================================
# 4. GRAPH CONSTRUCTION WITH POSITIONAL ENCODING
# ============================================================================

def construct_graph_with_positions(features_tensor, labels_tensor, 
                                   k_neighbors=15, time_window=3, 
                                   similarity_threshold=0.3):
    """
    Construct graph from embeddings with positional information
    - KNN-based similarity edges
    - Temporal edges between adjacent segments
    - Position indices for positional encoding
    """
    num_nodes = features_tensor.shape[0]
    
    if num_nodes == 0:
        return Data(
            x=features_tensor,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros(0, dtype=torch.float),
            y=labels_tensor.squeeze(),
            position=torch.arange(0, dtype=torch.long)
        )
    
    # Normalize features
    features_norm = (features_tensor - features_tensor.mean(dim=0)) / (features_tensor.std(dim=0) + 1e-8)
    node_features = F.normalize(features_norm, p=2, dim=1).cpu().numpy()
    
    edge_index = []
    edge_attr = []
    
    # 1. KNN-based similarity edges
    if num_nodes > k_neighbors:
        knn = NearestNeighbors(n_neighbors=min(k_neighbors+1, num_nodes), metric='cosine')
        knn.fit(node_features)
        distances, indices = knn.kneighbors(node_features)
        
        for i in range(num_nodes):
            for j_idx in range(1, min(k_neighbors+1, len(indices[i]))):
                j = indices[i, j_idx]
                similarity = 1 - distances[i, j_idx]
                
                if similarity > similarity_threshold:
                    # Temporal decay
                    time_decay = np.exp(-abs(i - j) / 50.0)
                    weight = similarity * time_decay
                    
                    edge_index.append([i, j])
                    edge_attr.append(weight)
    
    # 2. Temporal edges (adjacent segments)
    for i in range(num_nodes):
        for offset in range(1, time_window + 1):
            if i + offset < num_nodes:
                temporal_weight = 1.0 / offset
                edge_index.append([i, i + offset])
                edge_index.append([i + offset, i])
                edge_attr.append(temporal_weight)
                edge_attr.append(temporal_weight)
    
    # Convert to tensors
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
    
    # IMPORTANT: Position indices for positional encoding
    position_indices = torch.arange(num_nodes, dtype=torch.long)
    
    return Data(
        x=features_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels_tensor.squeeze(),
        position=position_indices  # For positional encoding
    )


# ============================================================================
# 5. IMPROVED GCN MODEL WITH POSITIONAL ENCODING
# ============================================================================

class ImprovedGCNWithPositionalEncoding(nn.Module):
    """Improved GCN with Sinusoidal Positional Encoding"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, max_nodes, 
                 positional_dim=64, num_layers=3, dropout=0.3, 
                 use_sinusoidal=True, use_residual=True):
        super(ImprovedGCNWithPositionalEncoding, self).__init__()
        
        self.use_residual = use_residual
        self.num_layers = num_layers
        
        # Positional encoding
        if use_sinusoidal:
            self.register_buffer('pos_encoding', 
                               self._create_sinusoidal_encoding(max_nodes, positional_dim))
        else:
            self.pos_encoding = nn.Parameter(torch.randn(max_nodes, positional_dim))
        
        # Input transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim + positional_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output head
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def _create_sinusoidal_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding (like Transformers)"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x, edge_index, edge_weight, positions):
        # Add positional encoding
        pos_emb = self.pos_encoding[positions]
        x = torch.cat([x, pos_emb], dim=1)
        
        # Transform
        x = self.feature_transform(x)
        
        # GCN layers with residual
        for i in range(self.num_layers):
            identity = x if self.use_residual else None
            
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            if self.use_residual and identity is not None:
                x = x + identity
            
            x = self.dropouts[i](x)
        
        # Output (logits)
        x = self.output_layer(x)
        return x.squeeze()


# ============================================================================
# 6. TRAINING FUNCTION
# ============================================================================

def train_improved_gnn(train_graphs, input_dim, hidden_dim=128, output_dim=1, 
                      positional_dim=64, num_layers=3, epochs=300, lr=0.001, 
                      batch_size=8, device="cuda", weight_decay=1e-4,
                      class_weight_multiplier=5.0, patience=50, 
                      use_sinusoidal=True, dropout=0.3):
    """Train improved GCN"""
    
    print("\n" + "="*70)
    print(" Training Improved GCN with Positional Encoding")
    print("="*70)
    print(f"Hidden dim:      {hidden_dim}")
    print(f"Positional dim:  {positional_dim}")
    print(f"Num layers:      {num_layers}")
    print(f"Epochs:          {epochs}")
    print(f"Class weight:    {class_weight_multiplier}x")
    print("="*70)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    max_nodes = max(graph.x.shape[0] for graph in train_graphs)
    
    model = ImprovedGCNWithPositionalEncoding(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        max_nodes=max_nodes,
        positional_dim=positional_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_sinusoidal=use_sinusoidal,
        use_residual=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # Class weights
    all_labels = torch.cat([g.y for g in train_graphs])
    pos_count = all_labels.sum().item()
    neg_count = (all_labels == 0).sum().item()
    pos_weight = (neg_count / pos_count) * class_weight_multiplier
    
    print(f"\n Positive weight: {pos_weight:.2f}x")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            if batch.edge_index.numel() == 0:
                continue
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.position)
            loss = criterion(logits, batch.y.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches == 0:
            continue
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f" Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(torch.load('best_model.pt'))
    print(f" Training complete! Best loss: {best_loss:.4f}")
    
    return model


# ============================================================================
# 7. EVALUATION FUNCTION
# ============================================================================

def evaluate_improved_gnn(model, test_graphs, device="cuda"):
    """Evaluate model per-graph, then report macro (mean) and micro (pooled) results."""
    model.eval()

    per_f1, per_prec, per_rec, per_far, per_mdr = [], [], [], [], []
    total_tn = total_fp = total_fn = total_tp = 0
    num_evaluated = 0

    with torch.no_grad():
        for i, graph in enumerate(test_graphs, start=1):
            graph = graph.to(device)
            if graph.edge_index.numel() == 0:
                print(f"[{i}/{len(test_graphs)}] skipped: empty edge_index")
                continue

            # Forward
            logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.position)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).detach().cpu().numpy().astype(int).ravel()
            y_true = graph.y.detach().cpu().numpy().astype(int).ravel()

            # Per-graph metrics
            f1   = f1_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            rec  = recall_score(y_true, preds, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            mdr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            # Collect
            per_f1.append(f1)
            per_prec.append(prec)
            per_rec.append(rec)
            per_far.append(far)
            per_mdr.append(mdr)

            total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp
            num_evaluated += 1

            # Print per-graph
            print(f"[{i}/{len(test_graphs)}] nodes={graph.x.size(0)} "
                  f"| F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} "
                  f"| FAR: {far:.4f} | MDR: {mdr:.4f}")

    if num_evaluated == 0:
        print("No graphs evaluated.")
        return {'f1_score': 0.0, 'mdr': 0.0, 'far': 0.0, 'precision': 0.0, 'recall': 0.0}

    # Macro (mean of per-graph metrics)
    macro_f1   = float(np.mean(per_f1))
    macro_prec = float(np.mean(per_prec))
    macro_rec  = float(np.mean(per_rec))
    macro_far  = float(np.mean(per_far))
    macro_mdr  = float(np.mean(per_mdr))

    # Micro (pooled across all graphs)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0.0)
    micro_far  = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
    micro_mdr  = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0

    # Summary print
    print("\n" + "="*70)
    print("Per-graph Macro Averages (mean over graphs)")
    print("="*70)
    print(f"F1: {macro_f1:.4f} | Precision: {macro_prec:.4f} | Recall: {macro_rec:.4f}")
    print(f"FAR: {macro_far:.4f} | MDR: {macro_mdr:.4f}")
    print("-"*70)
    print("Micro (Pooled) Results")
    print("-"*70)
    print(f"F1: {micro_f1:.4f} | Precision: {micro_prec:.4f} | Recall: {micro_rec:.4f}")
    print(f"FAR: {micro_far:.4f} | MDR: {micro_mdr:.4f}")
    print("="*70)

    # Keep backward-compatible keys as MICRO (pooled), and also return macro block
    return {
        'f1_score': micro_f1,
        'mdr': micro_mdr,
        'far': micro_far,
        'precision': micro_prec,
        'recall': micro_rec,
        'macro': {
            'f1_score': macro_f1,
            'mdr': macro_mdr,
            'far': macro_far,
            'precision': macro_prec,
            'recall': macro_rec,
        }
    }

# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Complete Speaker Change Detection Pipeline')
    parser.add_argument('csv_file', type=str, help='CSV file with audio paths')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--segment-length', type=float, default=1.0, help='Segment length (seconds)')
    parser.add_argument('--k-neighbors', type=int, default=15, help='KNN neighbors')
    parser.add_argument('--time-window', type=int, default=3, help='Temporal window')
    parser.add_argument('--similarity-threshold', type=float, default=0.3, help='Similarity threshold')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--positional-dim', type=int, default=64, help='Positional encoding dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--class-weight', type=float, default=5.0, help='Class weight multiplier')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pipeline
    print("\n" + "="*70)
    print(" Complete Speaker Change Detection Pipeline")
    print("="*70)
    
    # 1. Load CSV
    df = load_csv_data(args.csv_file)
    
    # 2. Process audio
    labels, segments = process_dataset(df, segment_length=args.segment_length)
    
    # 3. Extract Wav2Vec2 features
    features, labels = extract_wav2vec2_features(segments, labels, device=device)
    
    # 4. Construct graphs
    print(f"\n Constructing graphs...")
    graphs = []
    for feat, lab in tqdm(zip(features, labels), total=len(features), desc="Building graphs"):
        graph = construct_graph_with_positions(
            feat, lab,
            k_neighbors=args.k_neighbors,
            time_window=args.time_window,
            similarity_threshold=args.similarity_threshold
        )
        graphs.append(graph)
    
    print(f" Constructed {len(graphs)} graphs")
    
    # 5. Train or Test
    if not args.test:
        # Training
        model = train_improved_gnn(
            graphs,
            input_dim=768,
            hidden_dim=args.hidden_dim,
            positional_dim=args.positional_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            class_weight_multiplier=args.class_weight
        )
        
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
        
        max_nodes = max(graph.x.shape[0] for graph in graphs)
        config = {
            'hidden_dim': args.hidden_dim,
            'positional_dim': args.positional_dim,
            'num_layers': args.num_layers,
            'segment_length': args.segment_length,
            'k_neighbors': args.k_neighbors,
            'time_window': args.time_window,
            'similarity_threshold': args.similarity_threshold,
            'max_nodes':max_nodes
        }

        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f" Configuration saved to {args.output_dir}/config.json")

        # Evaluate on training set
        results = evaluate_improved_gnn(model, graphs, device=device)
    else:
        # Testing
        import json as jsonlib

        print("\n Testing mode")
        if not os.path.exists(os.path.join(args.output_dir, 'final_model.pt')):
            print(" No trained model found!")
            return
        
        cfg_path  = os.path.join(args.output_dir, 'config.json')

        # 1) Load training config
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = jsonlib.load(f)
            
            test_max_nodes = max(g.x.shape[0] for g in graphs) if len(graphs) > 0 else 1
            
            train_hidden_dim     = cfg.get("hidden_dim", args.hidden_dim)
            train_positional_dim = cfg.get("positional_dim", args.positional_dim)
            train_num_layers     = cfg.get("num_layers", args.num_layers)
            train_max_nodes      = int(cfg.get("max_nodes", test_max_nodes))  # fallback to test if missing
            train_use_sinusoidal = True  # you trained with sinusoidal per your code
            train_dropout        = 0.3   # keep consistent with training if you changed it


        max_nodes_for_test = max(train_max_nodes, test_max_nodes)


        
        model = ImprovedGCNWithPositionalEncoding(
            input_dim=768,
            hidden_dim=train_hidden_dim,
            output_dim=1,
            max_nodes=max_nodes_for_test,
            positional_dim=train_positional_dim,
            num_layers=args.num_layers
        ).to(device)
        
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'final_model.pt')))
        results = evaluate_improved_gnn(model, graphs, device=device)
    
    print("\n Complete!")


if __name__ == "__main__":
    main()