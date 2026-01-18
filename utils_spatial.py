import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# --- Spatial Analysis Helper (v5.5) ---
def generate_spatial_cluster_summary(df, cluster_col, x_col, y_col, label_map=None):
    """
    クラスタの空間配置（重心位置と近接関係）を分析し、テキスト説明を生成する。
    """
    try:
        if df.empty or cluster_col not in df.columns or x_col not in df.columns:
            return "空間データなし"

        # 1. 重心の計算
        centroids = df.groupby(cluster_col)[[x_col, y_col]].mean()
        
        # 2. 近接性の分析
        spatial_desc = []
        spatial_desc.append("【クラスタ配置と近接関係】")
        
        coords = centroids.values
        labels = centroids.index.tolist()
        
        if len(labels) < 2:
            return "単一クラスタのため配置分析なし"
            
        dist_matrix = squareform(pdist(coords))
        
        # 各クラスタについて、最近傍を見つける
        for i, label_id in enumerate(labels):
            if label_id == -1: continue # Skip noise if present (though usually noise centroid is meaningless)
            
            # Get distances to other clusters
            dists = dist_matrix[i]
            # argsortでインデックスを取得。自身（距離0）を除外
            nearest_indices = np.argsort(dists)[1:3] # Top 2 nearest
            
            # IDを名前にマップ
            current_name = label_map.get(label_id, f"Cluster {label_id}") if label_map else f"Cluster {label_id}"
            
            neighbors = []
            for n_idx in nearest_indices:
                n_id = labels[n_idx]
                n_dist = dists[n_idx]
                n_name = label_map.get(n_id, f"Cluster {n_id}") if label_map else f"Cluster {n_id}"
                neighbors.append(f"{n_name}")
            
            if neighbors:
                spatial_desc.append(f"- 「{current_name}」の近傍: {', '.join(neighbors)}")

        # 3. 全体的な広がり（オプション）
        # x_range = df[x_col].max() - df[x_col].min()
        # y_range = df[y_col].max() - df[y_col].min()
        
        return "\n".join(spatial_desc)

    except Exception as e:
        return f"空間分析計算エラー: {e}"
