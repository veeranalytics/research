# Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv
import igraph as ig, leidenalg
import random

import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# For reproducibility
random.seed(12)
np.random.seed(12)
torch.manual_seed(12)

# Get Data
url = 'https://media.githubusercontent.com/media/veeranalytics/research/refs/heads/main/Movies_Dataset_Full.csv'
df = pd.read_csv(url)

####################################################################################################################
###################### Method 04: Dynamic, Hierarchical, and Semantically-rich Community-enhanced ##################
###################### Knowledge Graph Recommender via Hybrid Path- and Graph-based Reasoning    ###################
####################################################################################################################
# Data Preparation
user_col = 'UserID'
item_col = 'MovieID'
label_col = 'label'

users = df[user_col].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=12)
train_df = df[df[user_col].isin(train_users)].reset_index(drop=True)
test_df = df[df[user_col].isin(test_users)].reset_index(drop=True)

# Build KG with Features
def build_kg_with_features(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        user = str(row[user_col])
        movie = str(row[item_col])
        if user not in G:
            G.add_node(user, node_type='user',
                       age=int(row['Age']),
                       gender=0 if row['Gender'] == 'M' else 1,
                       occupation=int(row['Occupation']),
                       zipcode=str(row['Zip-code']))
        if movie not in G:
            G.add_node(movie, node_type='movie',
                       title=row['Title'],
                       genres=row['Genres'],
                       year=int(row['Year']))
        G.add_edge(user, movie,
                   rating=float(row['Rating']),
                   timestamp=float(row['Timestamp']),
                   label=int(row[label_col]))
    return G

def enrich_with_communities(G):
    import igraph as ig, leidenalg
    G_ig = ig.Graph.TupleList(G.edges(), directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    node_community = {}
    for comm_id, comm in enumerate(partition):
        for node in comm:
            node_community[G_ig.vs[node]['name']] = comm_id
    nx.set_node_attributes(G, node_community, 'community')
    for node, comm in node_community.items():
        comm_node = f'comm_{comm}'
        if comm_node not in G:
            G.add_node(comm_node, node_type='community')
        G.add_edge(node, comm_node, relation='belongs_to')
    return G

# Node Feature Matrix
def get_node_idx_and_features(G):
    all_nodes = list(G.nodes)
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    features = []
    for n in all_nodes:
        ntype = G.nodes[n].get('node_type', '')
        community = G.nodes[n].get('community', 0)
        degree = G.degree[n]
        if ntype == 'user':
            age = G.nodes[n].get('age', 0)
            gender = G.nodes[n].get('gender', 0)
            occupation = G.nodes[n].get('occupation', 0)
            features.append([1, 0, degree, community, age, gender, occupation])
        elif ntype == 'movie':
            year = G.nodes[n].get('year', 0)
            features.append([0, 1, degree, community, year, 0, 0])
        else:
            features.append([0, 0, degree, community, 0, 0, 0])
    node_features = torch.tensor(features, dtype=torch.float32)
    return all_nodes, node_to_idx, node_features

def get_edge_index(G, node_to_idx):
    edges = list(G.edges)
    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges] +
                              [[node_to_idx[v], node_to_idx[u]] for u, v in edges], dtype=torch.long).T
    return edge_index

# Model Components
class SAGEGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = F.relu(self.sage1(x, edge_index))
        h = self.sage2(h, edge_index)
        return h

class PathEncoder(nn.Module):
    def __init__(self, node_feat_dim, emb_dim=16, hidden_dim=16):
        super().__init__()
        self.fc = nn.Linear(node_feat_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
    def forward(self, path_feats):
        x = self.fc(path_feats)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

class PathAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, path_embeds):
        scores = self.attn(path_embeds)
        weights = torch.softmax(scores, dim=0)
        return (weights * path_embeds).sum(dim=0)

class FusionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, gnn_emb, path_emb):
        fused = torch.cat([gnn_emb, path_emb], dim=-1)
        return self.mlp(fused)

# Path Sampling with Features
def sample_paths_with_features(G, src, tgt, node_to_idx, node_features, num_paths=3, max_len=6):
    paths = []
    try:
        sp = nx.shortest_path(G, src, tgt)
        if len(sp) <= max_len:
            feats = node_features[[node_to_idx[n] for n in sp]]
            paths.append(feats)
        for _ in range(num_paths-1):
            try:
                path = nx.shortest_path(G, src, tgt)
                if len(path) > 2 and random.random() > 0.5:
                    path.insert(1, f"comm_{G.nodes[path[1]].get('community',0)}")
                path = path[:max_len]
                feats = node_features[[node_to_idx[n] for n in path if n in node_to_idx]]
                paths.append(feats)
            except:
                continue
    except:
        pass
    if not paths:
        feats = node_features[[node_to_idx.get(src,0), node_to_idx.get(tgt,0)]]
        paths = [feats]
    return paths

# Dynamic Training Loop (KG/community is re-enriched every interval)
def train_dynamic_hybrid(train_df, G, gnn_model, path_encoder, path_attn, fusion,
                        user_col, item_col, label_col, epochs=3, batch_size=64, device='cuda', update_interval=10000):
    optimizer = torch.optim.Adam(list(gnn_model.parameters()) +
                                 list(path_encoder.parameters()) +
                                 list(path_attn.parameters()) +
                                 list(fusion.parameters()), lr=0.01)
    all_items = train_df[item_col].astype(str).unique()
    train_pos = train_df[train_df[label_col]==1][[user_col, item_col]]
    step = 0
    gnn_model.train(); path_encoder.train(); path_attn.train(); fusion.train()
    for epoch in range(epochs):
        losses = []
        for idx in range(0, len(train_pos), batch_size):
            batch = train_pos.iloc[idx:idx+batch_size]
            # --- DYNAMIC: Incrementally update KG/community every interval ---
            if step % update_interval == 0 and step > 0:
                G = enrich_with_communities(G)
            all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
            node_features = node_features.to(device)
            edge_index = get_edge_index(G, node_to_idx).to(device)
            node_embeds = gnn_model(node_features, edge_index)
            batch_loss = []
            for _, row in batch.iterrows():
                u, i = str(row[user_col]), str(row[item_col])
                if u not in node_to_idx or i not in node_to_idx: continue
                neg_i = random.choice([item for item in all_items if item != i])
                user_emb = node_embeds[node_to_idx[u]]
                pos_item_emb = node_embeds[node_to_idx[i]]
                neg_item_emb = node_embeds[node_to_idx[neg_i]]
                pos_paths = sample_paths_with_features(G, u, i, node_to_idx, node_features)
                neg_paths = sample_paths_with_features(G, u, neg_i, node_to_idx, node_features)
                pos_paths_t = [p.unsqueeze(0).to(device) for p in pos_paths]
                neg_paths_t = [p.unsqueeze(0).to(device) for p in neg_paths]
                pos_embeds = torch.stack([path_encoder(p) for p in pos_paths_t])
                neg_embeds = torch.stack([path_encoder(p) for p in neg_paths_t])
                pos_vec = path_attn(pos_embeds)
                neg_vec = path_attn(neg_embeds)
                def fix_shape(x):
                    if x.dim() == 1: return x.unsqueeze(0)
                    elif x.dim() == 3: return x.squeeze(0)
                    elif x.dim() == 2 and x.shape[0] != 1: return x.mean(0, keepdim=True)
                    return x
                pos_gnn_vec = fix_shape(user_emb + pos_item_emb)
                pos_path_vec = fix_shape(pos_vec)
                neg_gnn_vec = fix_shape(user_emb + neg_item_emb)
                neg_path_vec = fix_shape(neg_vec)
                pos_score = fusion(pos_gnn_vec, pos_path_vec)
                neg_score = fusion(neg_gnn_vec, neg_path_vec)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
                batch_loss.append(loss)
            if batch_loss:
                loss = torch.stack(batch_loss).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            step += len(batch)
        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}")

# Negative Sampling Evaluation
def evaluate_hybrid_neg_sampling(
    test_df, G, node_to_idx, node_features, 
    gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col, Ks=[1,5,10,20], num_negs=50, max_users=100, device='cuda'
):
    all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
    node_features = node_features.to(device)
    edge_index = get_edge_index(G, node_to_idx).to(device)
    user_positive = defaultdict(list)
    user_scores = defaultdict(dict)
    users = test_df[user_col].astype(str).unique()[:max_users]
    all_items = test_df[item_col].astype(str).unique()
    gnn_model.eval(); path_encoder.eval(); path_attn.eval(); fusion.eval()
    with torch.no_grad():
        node_embeds = gnn_model(node_features, edge_index)
        for u in tqdm(users):
            if u not in node_to_idx: continue
            pos_items = test_df[(test_df[user_col]==int(u)) & (test_df[label_col]==1)][item_col].astype(str).tolist()
            if not pos_items: continue
            user_positive[u].extend(pos_items)
            neg_candidates = list(set(all_items) - set(pos_items))
            if len(neg_candidates) < num_negs*len(pos_items): continue
            neg_items = np.random.choice(neg_candidates, min(len(neg_candidates), num_negs*len(pos_items)), replace=False)
            candidate_items = pos_items + list(neg_items)
            for item in candidate_items:
                if item not in node_to_idx: continue
                user_emb = node_embeds[node_to_idx[u]]
                item_emb = node_embeds[node_to_idx[item]]
                paths = sample_paths_with_features(G, u, item, node_to_idx, node_features)
                paths_t = [p.unsqueeze(0).to(device) for p in paths]
                path_embeds = torch.stack([path_encoder(p) for p in paths_t])
                path_vec = path_attn(path_embeds)
                def fix_shape(x):
                    if x.dim() == 1: return x.unsqueeze(0)
                    elif x.dim() == 3: return x.squeeze(0)
                    elif x.dim() == 2 and x.shape[0] != 1: return x.mean(0, keepdim=True)
                    return x
                gnn_vec = fix_shape(user_emb + item_emb)
                path_vec = fix_shape(path_vec)
                score = fusion(gnn_vec, path_vec).item()
                user_scores[u][item] = score

    def hit_rate_at_k(ranked, true, k): return int(len(set(ranked[:k]) & set(true)) > 0)
    def mrr_at_k(ranked, true, k):
        for idx, item in enumerate(ranked[:k]):
            if item in true:
                return 1.0 / (idx + 1)
        return 0.0
    def recall_at_k(ranked, true, k):
        pred_top_k = ranked[:k]
        hits = len(set(pred_top_k) & set(true))
        return hits / min(len(true), k) if true else 0
    def ndcg_at_k(ranked, true, k):
        dcg = 0.0
        for i, item in enumerate(ranked[:k]):
            if item in true:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true), k)))
        return dcg / idcg if idcg > 0 else 0.0

    print("\n--- Ranking Metrics (neg-sampling, hybrid) ---")
    for k in Ks:
        hit_rates, mrrs, recalls, ndcgs = [], [], [], []
        for u in user_positive:
            if u not in user_scores: continue
            scores = user_scores[u]
            ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
            true = user_positive[u]
            hit_rates.append(hit_rate_at_k(ranked, true, k))
            mrrs.append(mrr_at_k(ranked, true, k))
            recalls.append(recall_at_k(ranked, true, k))
            ndcgs.append(ndcg_at_k(ranked, true, k))
        print(f"HitRate@{k}: {np.mean(hit_rates):.4f}")
        print(f"MRR@{k}:     {np.mean(mrrs):.4f}")
        print(f"Recall@{k}:  {np.mean(recalls):.4f}")
        print(f"NDCG@{k}:    {np.mean(ndcgs):.4f}")
        
# Main Pipeline
# Record the start time before the step
start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

G = build_kg_with_features(train_df)
G = enrich_with_communities(G)
all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
node_features = node_features.to(device)

gnn_model = SAGEGNN(node_features.shape[1], 32, 16).to(device)
path_encoder = PathEncoder(node_features.shape[1], emb_dim=16, hidden_dim=16).to(device)
path_attn = PathAttention(hidden_dim=16).to(device)
fusion = FusionMLP(in_dim=32, hidden_dim=16).to(device)

train_dynamic_hybrid(
    train_df, G, gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col, epochs=5, batch_size=64, device=device, update_interval=10000
)

G_test = build_kg_with_features(test_df)
G_test = enrich_with_communities(G_test)
_, node_to_idx_test, node_features_test = get_node_idx_and_features(G_test)
node_features_test = node_features_test.to(device)

evaluate_hybrid_neg_sampling(
    test_df, G_test, node_to_idx_test, node_features_test,
    gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col, Ks=[1,5,10,20], num_negs=50, max_users=100, device=device
)

# Record the end time after the step
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The Method 04 took: {elapsed_time:.4f} seconds")


