# ==================================================================
# utils_spatial.py — APOLLO v8.0.0 空間分析ヘルパー
# patiroha.generate_spatial_summary への委譲
# ==================================================================

import patiroha


def generate_spatial_cluster_summary(df, cluster_col, x_col, y_col, label_map=None):
    """
    クラスタの空間配置（重心位置と近接関係）を分析し、テキスト説明を生成する。
    patiroha.generate_spatial_summary に委譲。
    """
    return patiroha.generate_spatial_summary(
        df=df,
        cluster_col=cluster_col,
        x_col=x_col,
        y_col=y_col,
        label_map=label_map,
    )
