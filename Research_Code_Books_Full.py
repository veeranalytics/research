# Import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.nn import SAGEConv
import igraph as ig, leidenalg
import time
from datetime import datetime

warnings.filterwarnings("ignore")
#%matplotlib inline

# For reproducibility
random.seed(12)
np.random.seed(12)
torch.manual_seed(12)

# Get Data
url = 'https://media.githubusercontent.com/media/veeranalytics/research/refs/heads/main/Books_Dataset_Full.csv'
df = pd.read_csv(url)

####################################################################################################################
#################### Method 01: Community Enhanced Knowledge Graph for Recommendation ##############################
####################################################################################################################
#Get the main columns
user_col = 'User-ID'
item_col = 'ISBN'
label_col = 'label'

# Split by user
users = df[user_col].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=12)
train_df = df[df[user_col].isin(train_users)].reset_index(drop=True)
test_df = df[df[user_col].isin(test_users)].reset_index(drop=True)

# Build Knowledge Graph with Communities
def build_kg_with_features(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        user = str(row['User-ID'])
        book = str(row['ISBN'])
        if user not in G:
            G.add_node(user, node_type='user', age=float(row['Age']) if not pd.isna(row['Age']) else 0)
        if book not in G:
            G.add_node(book, node_type='book',
                       title=row['Book-Title'],
                       author=row['Book-Author'],
                       year=float(row['Year-Of-Publication']) if not pd.isna(row['Year-Of-Publication']) else 0,
                       publisher=row['Publisher'],
                       country=row['Country'])
        G.add_edge(user, book, label=int(row['label']), rating=float(row['Book-Rating']) if not pd.isna(row['Book-Rating']) else 0)
    return G

def enrich_with_communities(G):
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

# Feature Extraction for GNN
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
            features.append([1, 0, 0, degree, community, age, 0, 0, 0, 0])
        elif ntype == 'book':
            year = G.nodes[n].get('year', 0)
            features.append([0, 1, 0, degree, community, 0, year, 0, 0, 0])
        else:
            features.append([0, 0, 1, degree, community, 0, 0, 0, 0, 0])
    node_features = torch.tensor(features, dtype=torch.float32)
    return all_nodes, node_to_idx, node_features

# Path-based Encoder and Attention
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

# Simple Path Sampler
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

# Model and Training Loop (Pairwise BPR Loss)
class ScoreMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(x)

def train_cekgr(train_df, G, node_to_idx, node_features, path_encoder, path_attn, scorer,
                user_col, item_col, epochs=3, batch_size=64, device='cuda'):
    optimizer = torch.optim.Adam(list(path_encoder.parameters()) +
                                 list(path_attn.parameters()) +
                                 list(scorer.parameters()), lr=0.01)
    all_items = train_df[item_col].astype(str).unique()
    train_pos = train_df[train_df['label']==1][[user_col, item_col]]
    path_encoder.train(); path_attn.train(); scorer.train()
    for epoch in range(epochs):
        losses = []
        for idx in range(0, len(train_pos), batch_size):
            batch = train_pos.iloc[idx:idx+batch_size]
            batch_loss = []
            for _, row in batch.iterrows():
                u, i = str(row[user_col]), str(row[item_col])
                if u not in node_to_idx or i not in node_to_idx:
                    continue
                neg_i = random.choice([item for item in all_items if item != i])
                # Sample paths
                pos_paths = sample_paths_with_features(G, u, i, node_to_idx, node_features)
                neg_paths = sample_paths_with_features(G, u, neg_i, node_to_idx, node_features)
                pos_paths_t = [p.unsqueeze(0).to(device) for p in pos_paths]
                neg_paths_t = [p.unsqueeze(0).to(device) for p in neg_paths]
                pos_embeds = torch.stack([path_encoder(p) for p in pos_paths_t])
                neg_embeds = torch.stack([path_encoder(p) for p in neg_paths_t])
                pos_vec = path_attn(pos_embeds)
                neg_vec = path_attn(neg_embeds)
                pos_score = scorer(pos_vec)
                neg_score = scorer(neg_vec)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
                batch_loss.append(loss)
            if batch_loss:
                loss = torch.stack(batch_loss).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}")

# Negative Sampling Evaluation
def evaluate_cekgr_neg_sampling(
    test_df, G, node_to_idx, node_features, path_encoder, path_attn, scorer,
    user_col, item_col, Ks=[1,5,10,20], num_negs=50, max_users=100, device='cuda'):
    user_positive = defaultdict(list)
    user_scores = defaultdict(dict)
    users = test_df[user_col].astype(str).unique()[:max_users]
    all_items = test_df[item_col].astype(str).unique()
    path_encoder.eval(); path_attn.eval(); scorer.eval()
    print("Evaluating (neg-sampling, path)...")
    with torch.no_grad():
        for u in tqdm(users):
            if u not in node_to_idx: continue
            pos_items = test_df[(test_df[user_col]==int(u)) & (test_df['label']==1)][item_col].astype(str).tolist()
            if not pos_items: continue
            user_positive[u].extend(pos_items)
            neg_candidates = list(set(all_items) - set(pos_items))
            if len(neg_candidates) < num_negs*len(pos_items):
                continue
            neg_items = np.random.choice(neg_candidates, min(len(neg_candidates), num_negs*len(pos_items)), replace=False)
            candidate_items = pos_items + list(neg_items)
            for item in candidate_items:
                if item not in node_to_idx: continue
                paths = sample_paths_with_features(G, u, item, node_to_idx, node_features)
                paths_t = [p.unsqueeze(0).to(device) for p in paths]
                path_embeds = torch.stack([path_encoder(p) for p in paths_t])
                path_vec = path_attn(path_embeds)
                score = scorer(path_vec).item()
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

    print("\n--- Ranking Metrics (neg-sampling, path) ---")
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
        
# Running the Pipeline
# Record the start time before the step
start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Build KG ---
G = build_kg_with_features(train_df)
G = enrich_with_communities(G)
all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
node_features = node_features.to(device)

# --- Models ---
path_encoder = PathEncoder(node_features.shape[1], emb_dim=16, hidden_dim=16).to(device)
path_attn = PathAttention(hidden_dim=16).to(device)
scorer = ScoreMLP(hidden_dim=16).to(device)

# --- Training ---
train_cekgr(
    train_df, G, node_to_idx, node_features, path_encoder, path_attn, scorer,
    user_col, item_col, epochs=5, batch_size=64, device=device
)

# --- Build KG/test features ---
G_test = build_kg_with_features(test_df)
G_test = enrich_with_communities(G_test)
_, node_to_idx_test, node_features_test = get_node_idx_and_features(G_test)
node_features_test = node_features_test.to(device)

# --- Evaluation ---
evaluate_cekgr_neg_sampling(
    test_df, G_test, node_to_idx_test, node_features_test, path_encoder, path_attn, scorer,
    user_col, item_col, Ks=[1,5,10,20], num_negs=50, max_users=100, device=device
)

# Record the end time after the step
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The Method 01 took: {elapsed_time:.4f} seconds")

####################################################################################################################
################################# Method 02: Collaborative filtering (Product)  ####################################
####################################################################################################################
# Prepare the Dataset
user_col = 'User-ID'
item_col = 'ISBN'
rating_col = 'Book-Rating'
label_col = 'label'

# Index users and items for embedding lookup
user_ids = df[user_col].astype(str).unique()
item_ids = df[item_col].astype(str).unique()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {i: j for j, i in enumerate(item_ids)}

# Train/test split by user
users = df[user_col].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=12)
train_df = df[df[user_col].isin(train_users)].reset_index(drop=True)
test_df = df[df[user_col].isin(test_users)].reset_index(drop=True)

# Matrix Factorization Model
class MFModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.out = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        x = u * v  # elementwise product
        return self.out(x).squeeze(-1)

# Training Loop (Pairwise BPR Loss)
def train_cf_model(train_df, user_to_idx, item_to_idx, model, epochs=5, batch_size=1024, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    all_items = list(item_to_idx.keys())
    train_pos = train_df[train_df[label_col]==1][[user_col, item_col]]
    model.train()
    for epoch in range(epochs):
        losses = []
        for idx in range(0, len(train_pos), batch_size):
            batch = train_pos.iloc[idx:idx+batch_size]
            user_idx, pos_item_idx, neg_item_idx = [], [], []
            for _, row in batch.iterrows():
                u = user_to_idx.get(str(row[user_col]), None)
                i = item_to_idx.get(str(row[item_col]), None)
                if u is None or i is None: continue
                # Negative sample
                neg_i = np.random.choice([item for item in all_items if item != row[item_col]])
                neg_i_idx = item_to_idx.get(str(neg_i), None)
                if neg_i_idx is None: continue
                user_idx.append(u)
                pos_item_idx.append(i)
                neg_item_idx.append(neg_i_idx)
            if not user_idx: continue
            user_idx = torch.tensor(user_idx, dtype=torch.long, device=device)
            pos_item_idx = torch.tensor(pos_item_idx, dtype=torch.long, device=device)
            neg_item_idx = torch.tensor(neg_item_idx, dtype=torch.long, device=device)
            pos_scores = model(user_idx, pos_item_idx)
            neg_scores = model(user_idx, neg_item_idx)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}")

# Efficient Ranking Metrics with Negative Sampling
def evaluate_cf_ranking(
    test_df, user_to_idx, item_to_idx, model,
    user_col, item_col, label_col,
    Ks=[1,5,10,20], num_negs=50, max_users=200, device='cuda'
):
    from collections import defaultdict
    user_positive = defaultdict(list)
    user_scores = defaultdict(dict)
    users = test_df[user_col].astype(str).unique()[:max_users]
    all_items = test_df[item_col].astype(str).unique()
    model.eval()
    with torch.no_grad():
        for u in tqdm(users):
            u_idx = user_to_idx.get(u, None)
            if u_idx is None: continue
            pos_items = test_df[(test_df[user_col]==int(u)) & (test_df[label_col]==1)][item_col].astype(str).tolist()
            if not pos_items: continue
            user_positive[u].extend(pos_items)
            neg_candidates = list(set(all_items) - set(pos_items))
            if len(neg_candidates) < num_negs*len(pos_items): continue
            neg_items = np.random.choice(neg_candidates, min(len(neg_candidates), num_negs*len(pos_items)), replace=False)
            candidate_items = pos_items + list(neg_items)
            user_idx_tensor = torch.tensor([u_idx]*len(candidate_items), dtype=torch.long, device=device)
            item_idx_tensor = torch.tensor([item_to_idx[item] for item in candidate_items], dtype=torch.long, device=device)
            scores = model(user_idx_tensor, item_idx_tensor).cpu().numpy().tolist()
            for item, score in zip(candidate_items, scores):
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

    print("\n--- Ranking Metrics (neg-sampling, MF) ---")
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
        
# Run the Pipeline
# Record the start time before the step
start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_users = len(user_ids)
num_items = len(item_ids)
mf_model = MFModel(num_users, num_items, emb_dim=32).to(device)

# Training
train_cf_model(train_df, user_to_idx, item_to_idx, mf_model, epochs=5, batch_size=1024, device=device)

# Evaluation (with negative sampling, recommended)
evaluate_cf_ranking(
    test_df, user_to_idx, item_to_idx, mf_model,
    user_col, item_col, label_col, Ks=[1,5,10,20], num_negs=50, max_users=200, device=device
)

# Record the end time after the step
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The Method 02 took: {elapsed_time:.4f} seconds")

###################################################################################################################
###################### Method 03: Static, Hierarchical, and Semantically-rich Community-enhanced ###################
###################### Knowledge Graph Recommender via Hybrid Path- and Graph-based Reasoning    ###################
####################################################################################################################
# Prepare Data
user_col = 'User-ID'
item_col = 'ISBN'
label_col = 'label'

# Index users and items for embedding lookup
user_ids = df[user_col].astype(str).unique()
item_ids = df[item_col].astype(str).unique()

# Train/test split by user
users = df[user_col].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=12)
train_df = df[df[user_col].isin(train_users)].reset_index(drop=True)
test_df = df[df[user_col].isin(test_users)].reset_index(drop=True)

# KG Construction & Community Detection
def build_kg_with_features(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        user = str(row[user_col])
        book = str(row[item_col])
        if user not in G:
            G.add_node(user, node_type='user', age=float(row['Age']) if not pd.isna(row['Age']) else 0)
        if book not in G:
            G.add_node(book, node_type='book',
                       title=row['Book-Title'],
                       author=row['Book-Author'],
                       year=float(row['Year-Of-Publication']) if not pd.isna(row['Year-Of-Publication']) else 0,
                       publisher=row['Publisher'],
                       country=row['Country'])
        G.add_edge(user, book, label=int(row['label']), rating=float(row['Book-Rating']) if not pd.isna(row['Book-Rating']) else 0)
    return G

def enrich_with_communities(G):
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

# Node Features
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
            features.append([1, 0, 0, degree, community, age, 0, 0, 0, 0])
        elif ntype == 'book':
            year = G.nodes[n].get('year', 0)
            features.append([0, 1, 0, degree, community, 0, year, 0, 0, 0])
        else:
            features.append([0, 0, 1, degree, community, 0, 0, 0, 0, 0])
    node_features = torch.tensor(features, dtype=torch.float32)
    return all_nodes, node_to_idx, node_features

def get_edge_index(G, node_to_idx):
    edges = list(G.edges)
    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges] +
                              [[node_to_idx[v], node_to_idx[u]] for u, v in edges], dtype=torch.long).T
    return edge_index

# Model Components
class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index).relu()
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

class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index).relu()
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

# Sample Paths for PathEncoder
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

# Training Loop (Pairwise BPR Loss, Hybrid Reasoning)
def train_hybrid(
    train_df, G, node_to_idx, node_features, edge_index,
    gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, epochs=3, batch_size=64, device='cuda'):
    optimizer = torch.optim.Adam(list(gnn_model.parameters()) +
                                 list(path_encoder.parameters()) +
                                 list(path_attn.parameters()) +
                                 list(fusion.parameters()), lr=0.01)
    all_items = train_df[item_col].astype(str).unique()
    train_pos = train_df[train_df[label_col]==1][[user_col, item_col]]
    gnn_model.train(); path_encoder.train(); path_attn.train(); fusion.train()
    for epoch in range(epochs):
        losses = []
        for idx in range(0, len(train_pos), batch_size):
            batch = train_pos.iloc[idx:idx+batch_size]
            batch_loss = []
            node_embeds = gnn_model(node_features.to(device), edge_index.to(device))
            for _, row in batch.iterrows():
                u, i = str(row[user_col]), str(row[item_col])
                if u not in node_to_idx or i not in node_to_idx: continue
                neg_i = random.choice([item for item in all_items if item != i])
                if neg_i not in node_to_idx: continue
                # Path
                pos_paths = sample_paths_with_features(G, u, i, node_to_idx, node_features)
                neg_paths = sample_paths_with_features(G, u, neg_i, node_to_idx, node_features)
                pos_paths_t = [p.unsqueeze(0).to(device) for p in pos_paths]
                neg_paths_t = [p.unsqueeze(0).to(device) for p in neg_paths]
                pos_embeds = torch.stack([path_encoder(p) for p in pos_paths_t])
                neg_embeds = torch.stack([path_encoder(p) for p in neg_paths_t])
                pos_vec = path_attn(pos_embeds)
                neg_vec = path_attn(neg_embeds)
                user_emb = node_embeds[node_to_idx[u]]
                pos_item_emb = node_embeds[node_to_idx[i]]
                neg_item_emb = node_embeds[node_to_idx[neg_i]]
                gnn_vec_pos = user_emb + pos_item_emb
                gnn_vec_neg = user_emb + neg_item_emb
                if gnn_vec_pos.dim() == 1: gnn_vec_pos = gnn_vec_pos.unsqueeze(0)
                if gnn_vec_neg.dim() == 1: gnn_vec_neg = gnn_vec_neg.unsqueeze(0)
                if pos_vec.dim() == 1: pos_vec = pos_vec.unsqueeze(0)
                if neg_vec.dim() == 1: neg_vec = neg_vec.unsqueeze(0)
                pos_score = fusion(gnn_vec_pos, pos_vec)
                neg_score = fusion(gnn_vec_neg, neg_vec)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
                batch_loss.append(loss)
            if batch_loss:
                loss = torch.stack(batch_loss).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}")

# Efficient Evaluation (Negative Sampling)
def evaluate_hybrid_neg_sampling(
    test_df, G, node_to_idx, node_features, edge_index,
    gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col,
    Ks=[1,5,10,20], num_negs=50, max_users=100, device='cuda'):
    user_positive = defaultdict(list)
    user_scores = defaultdict(dict)
    users = test_df[user_col].astype(str).unique()[:max_users]
    all_items = test_df[item_col].astype(str).unique()
    gnn_model.eval(); path_encoder.eval(); path_attn.eval(); fusion.eval()
    print("Evaluating (neg-sampling, hybrid)...")
    with torch.no_grad():
        node_embeds = gnn_model(node_features.to(device), edge_index.to(device))
        for u in tqdm(users):
            if u not in node_to_idx: continue
            pos_items = test_df[(test_df[user_col]==int(u)) & (test_df[label_col]==1)][item_col].astype(str).tolist()
            if not pos_items: continue
            user_positive[u].extend(pos_items)
            neg_candidates = list(set(all_items) - set(pos_items))
            if len(neg_candidates) < num_negs*len(pos_items):
                continue
            neg_items = np.random.choice(neg_candidates, min(len(neg_candidates), num_negs*len(pos_items)), replace=False)
            candidate_items = pos_items + list(neg_items)
            for item in candidate_items:
                if item not in node_to_idx: continue
                paths = sample_paths_with_features(G, u, item, node_to_idx, node_features)
                paths_t = [p.unsqueeze(0).to(device) for p in paths]
                path_embeds = torch.stack([path_encoder(p) for p in paths_t])
                path_vec = path_attn(path_embeds)
                user_emb = node_embeds[node_to_idx[u]]
                item_emb = node_embeds[node_to_idx[item]]
                gnn_vec = user_emb + item_emb
                if gnn_vec.dim() == 1: gnn_vec = gnn_vec.unsqueeze(0)
                if path_vec.dim() == 1: path_vec = path_vec.unsqueeze(0)
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
        
# Run the Pipeline
# Record the start time before the step
start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Build KG ---
G = build_kg_with_features(train_df)
G = enrich_with_communities(G)
all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
node_features = node_features.to(device)
edge_index = get_edge_index(G, node_to_idx).to(device)

# --- Models ---
gnn_model = GNNModel(node_features.shape[1], hidden_dim=32, out_dim=16).to(device)
path_encoder = PathEncoder(node_features.shape[1], emb_dim=16, hidden_dim=16).to(device)
path_attn = PathAttention(hidden_dim=16).to(device)
fusion = FusionMLP(in_dim=32, hidden_dim=16).to(device)

# --- Training ---
train_hybrid(
    train_df, G, node_to_idx, node_features, edge_index,
    gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, epochs=5, batch_size=64, device=device
)

# --- Test graph/features ---
G_test = build_kg_with_features(test_df)
G_test = enrich_with_communities(G_test)
_, node_to_idx_test, node_features_test = get_node_idx_and_features(G_test)
node_features_test = node_features_test.to(device)
edge_index_test = get_edge_index(G_test, node_to_idx_test).to(device)

# --- Evaluation ---
evaluate_hybrid_neg_sampling(
    test_df, G_test, node_to_idx_test, node_features_test, edge_index_test,
    gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col, Ks=[1,5,10,20], num_negs=50, max_users=100, device=device
)

# Record the end time after the step
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The Method 03 took: {elapsed_time:.4f} seconds")

####################################################################################################################
###################### Method 04: Dynamic, Hierarchical, and Semantically-rich Community-enhanced ##################
###################### Knowledge Graph Recommender via Hybrid Path- and Graph-based Reasoning    ###################
####################################################################################################################
# Data Split and Fields
user_col = 'User-ID'
item_col = 'ISBN'
label_col = 'label'

# Train/test split by user (for fairness)
users = df[user_col].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=12)
train_df = df[df[user_col].isin(train_users)].reset_index(drop=True)
test_df = df[df[user_col].isin(test_users)].reset_index(drop=True)

# KG Build and Dynamic Community Update
def build_kg_with_features(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        user = str(row[user_col])
        book = str(row[item_col])
        if user not in G:
            G.add_node(user, node_type='user', age=float(row['Age']) if not pd.isna(row['Age']) else 0)
        if book not in G:
            G.add_node(book, node_type='book',
                       title=row['Book-Title'],
                       author=row['Book-Author'],
                       year=float(row['Year-Of-Publication']) if not pd.isna(row['Year-Of-Publication']) else 0,
                       publisher=row['Publisher'],
                       country=row['Country'])
        G.add_edge(user, book, label=int(row['label']), rating=float(row['Book-Rating']) if not pd.isna(row['Book-Rating']) else 0)
    return G

def dynamic_enrich_with_communities(G):
    import igraph as ig, leidenalg
    G_ig = ig.Graph.TupleList(G.edges(), directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    node_community = {}
    for comm_id, comm in enumerate(partition):
        for node in comm:
            node_community[G_ig.vs[node]['name']] = comm_id
    nx.set_node_attributes(G, node_community, 'community')
    # Add community nodes and edges dynamically
    for node, comm in node_community.items():
        comm_node = f'comm_{comm}'
        if comm_node not in G:
            G.add_node(comm_node, node_type='community')
        G.add_edge(node, comm_node, relation='belongs_to')
    return G, node_community

# Node Features
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
            features.append([1, 0, 0, degree, community, age, 0, 0, 0, 0])
        elif ntype == 'book':
            year = G.nodes[n].get('year', 0)
            features.append([0, 1, 0, degree, community, 0, year, 0, 0, 0])
        else:
            features.append([0, 0, 1, degree, community, 0, 0, 0, 0, 0])
    node_features = torch.tensor(features, dtype=torch.float32)
    return all_nodes, node_to_idx, node_features

def get_edge_index(G, node_to_idx):
    edges = list(G.edges)
    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges] +
                              [[node_to_idx[v], node_to_idx[u]] for u, v in edges], dtype=torch.long).T
    return edge_index

# Model Components
class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index).relu()
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

# Path Sampling (with Node Features)
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

# Dynamic Hybrid Training Loop (with Dynamic Communities)
def train_dynamic_hybrid(
    train_df, G, gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col,
    epochs=3, batch_size=64, device='cuda',
    update_interval=1000
):
    optimizer = torch.optim.Adam(list(gnn_model.parameters()) +
                                 list(path_encoder.parameters()) +
                                 list(path_attn.parameters()) +
                                 list(fusion.parameters()), lr=0.01)
    all_items = train_df[item_col].astype(str).unique()
    train_pos = train_df[train_df[label_col]==1][[user_col, item_col]]
    step = 0
    for epoch in range(epochs):
        losses = []
        idx = 0
        while idx < len(train_pos):
            batch = train_pos.iloc[idx:idx+batch_size]
            # --- Dynamic community update every interval ---
            if (step % update_interval == 0) and (step > 0):
                G, _ = dynamic_enrich_with_communities(G)
            all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
            edge_index = get_edge_index(G, node_to_idx)
            node_embeds = gnn_model(node_features.to(device), edge_index.to(device))
            batch_loss = []
            for _, row in batch.iterrows():
                u, i = str(row[user_col]), str(row[item_col])
                if u not in node_to_idx or i not in node_to_idx: continue
                neg_i = random.choice([item for item in all_items if item != i])
                if neg_i not in node_to_idx: continue
                # Path
                pos_paths = sample_paths_with_features(G, u, i, node_to_idx, node_features)
                neg_paths = sample_paths_with_features(G, u, neg_i, node_to_idx, node_features)
                pos_paths_t = [p.unsqueeze(0).to(device) for p in pos_paths]
                neg_paths_t = [p.unsqueeze(0).to(device) for p in neg_paths]
                pos_embeds = torch.stack([path_encoder(p) for p in pos_paths_t])
                neg_embeds = torch.stack([path_encoder(p) for p in neg_paths_t])
                pos_vec = path_attn(pos_embeds)
                neg_vec = path_attn(neg_embeds)
                user_emb = node_embeds[node_to_idx[u]]
                pos_item_emb = node_embeds[node_to_idx[i]]
                neg_item_emb = node_embeds[node_to_idx[neg_i]]
                gnn_vec_pos = user_emb + pos_item_emb
                gnn_vec_neg = user_emb + neg_item_emb
                if gnn_vec_pos.dim() == 1: gnn_vec_pos = gnn_vec_pos.unsqueeze(0)
                if gnn_vec_neg.dim() == 1: gnn_vec_neg = gnn_vec_neg.unsqueeze(0)
                if pos_vec.dim() == 1: pos_vec = pos_vec.unsqueeze(0)
                if neg_vec.dim() == 1: neg_vec = neg_vec.unsqueeze(0)
                pos_score = fusion(gnn_vec_pos, pos_vec)
                neg_score = fusion(gnn_vec_neg, neg_vec)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
                batch_loss.append(loss)
            if batch_loss:
                loss = torch.stack(batch_loss).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            idx += batch_size
            step += 1
        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}")

# Efficient Negative Sampling Evaluation
def evaluate_dynamic_hybrid_neg_sampling(
    test_df, G, gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col,
    Ks=[1,5,10,20], num_negs=50, max_users=100, device='cuda'
):
    user_positive = defaultdict(list)
    user_scores = defaultdict(dict)
    users = test_df[user_col].astype(str).unique()[:max_users]
    all_items = test_df[item_col].astype(str).unique()
    gnn_model.eval(); path_encoder.eval(); path_attn.eval(); fusion.eval()
    all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
    edge_index = get_edge_index(G, node_to_idx)
    with torch.no_grad():
        node_embeds = gnn_model(node_features.to(device), edge_index.to(device))
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
                paths = sample_paths_with_features(G, u, item, node_to_idx, node_features)
                paths_t = [p.unsqueeze(0).to(device) for p in paths]
                path_embeds = torch.stack([path_encoder(p) for p in paths_t])
                path_vec = path_attn(path_embeds)
                user_emb = node_embeds[node_to_idx[u]]
                item_emb = node_embeds[node_to_idx[item]]
                gnn_vec = user_emb + item_emb
                if gnn_vec.dim() == 1: gnn_vec = gnn_vec.unsqueeze(0)
                if path_vec.dim() == 1: path_vec = path_vec.unsqueeze(0)
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

    print("\n--- Ranking Metrics (neg-sampling, dynamic hybrid) ---")
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
        
# Run the Pipeline
# Record the start time before the step
start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Initial KG ---
G = build_kg_with_features(train_df)
G, _ = dynamic_enrich_with_communities(G)
all_nodes, node_to_idx, node_features = get_node_idx_and_features(G)
node_features = node_features.to(device)
edge_index = get_edge_index(G, node_to_idx).to(device)

# --- Models ---
gnn_model = GNNModel(node_features.shape[1], hidden_dim=32, out_dim=16).to(device)
path_encoder = PathEncoder(node_features.shape[1], emb_dim=16, hidden_dim=16).to(device)
path_attn = PathAttention(hidden_dim=16).to(device)
fusion = FusionMLP(in_dim=32, hidden_dim=16).to(device)

# --- Training ---
train_dynamic_hybrid(
    train_df, G, gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col,
    epochs=5, batch_size=64, device=device,
    update_interval=1000
)

# --- Test KG/Features ---
G_test = build_kg_with_features(test_df)
G_test, _ = dynamic_enrich_with_communities(G_test)
# You may also just use latest train KG for eval if you want
# But for proper split, use test KG
evaluate_dynamic_hybrid_neg_sampling(
    test_df, G_test, gnn_model, path_encoder, path_attn, fusion,
    user_col, item_col, label_col, Ks=[1,5,10,20], num_negs=50, max_users=100, device=device
)

# Record the end time after the step
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"The Method 04 took: {elapsed_time:.4f} seconds")
