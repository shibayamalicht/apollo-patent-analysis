import os
# Numba/TBB の「Attempted to fork from a non-main thread, the TBB library may be in
# an invalid state...」警告を抑止する。Streamlit はスクリプトを非メインスレッドで実行し、
# その中で UMAP/HDBSCAN（Numba）や joblib(loky) 並列が fork するため、TBB スレッド層が
# 毎回この警告を stderr へ出す（コンパイル済みCからの出力なので warnings では消せない）。
# fork-safe な workqueue 層に切り替えると TBB の atfork ハンドラが登録されず警告も消える。
# ※ patiroha（UMAP/HDBSCAN→Numba）を import する前に設定する必要がある。
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
import streamlit as st
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import json
import pandas as pd
import numpy as np
import patiroha
# ↑の NUMBA_THREADING_LAYER env は「numba が import される前」に設定されている必要があるが、
# Home.py は patiroha(→numba) を utils より先に import するため env だけでは間に合わない。
# patiroha 読込後に numba.config の属性を直接 workqueue へ固定する（最初の並列実行前に効くので
# import 順に依存せず確実）。これで TBB の「fork from non-main thread」警告が出なくなる。
try:
    import numba as _numba
    _numba.config.THREADING_LAYER = "workqueue"
except Exception:
    pass
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

# ==================================================================
# --- 端末ログのノイズ抑止（Streamlit の Arrow 変換 INFO ログ） ---
# ==================================================================
# Streamlit は混合型（数値と文字列が混在する）カラムを含む DataFrame を表示する際、
# Arrow 変換に一度失敗してから自動修正する。表示自体は正常だが、その都度
#   "Serialization of dataframe to Arrow table was unsuccessful ... Applying automatic fixes ..."
# という INFO ログを端末へ大量に出す（出願人/発明者名がカラムになるクロス表などで頻発）。
# 該当 logger（streamlit.dataframe_util）だけレベルを WARNING に上げ、この無害な INFO を
# 抑止する（他モジュールの警告・エラーは残す）。utils は全ページが import するため一度で全体に効く。
import logging as _logging
_logging.getLogger("streamlit.dataframe_util").setLevel(_logging.WARNING)

# ==================================================================
# --- 0z. 文埋め込みモデル (SBERT) の選択設定 ---
# ==================================================================
# Mission Control「フェーズ4」のラジオボタンで切り替え可能な文埋め込みモデル。
#   fast   : 軽量で速い（前バージョンのモデル）。重さで処理が止まる場合の回避用。
#   quality: 多言語 E5。日本語特許＋英語論文を高品質に埋め込むが重く時間がかかる。
# prefix が設定されたモデル（E5）は接頭辞 "passage: " が必須なので、
# sentence-transformers の default_prompt_name で全 encode に自動付与する
# （MiniLM 等は接頭辞不要なので付与しない＝付けると逆に品質低下）。
SBERT_MODELS = {
    "fast": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "label": "高速（MiniLM・384次元・軽量／前バージョン）",
        "prefix": None,
        "dim": 384,
    },
    "quality": {
        "name": "intfloat/multilingual-e5-base",
        "label": "高精度（multilingual-e5-base・768次元・低速／初回DLあり）",
        "prefix": "passage: ",
        "dim": 768,
    },
}
# 既定は「高速」。高精度 E5 は重くて処理が止まる環境があるため、安全側を既定にする。
DEFAULT_SBERT_KEY = "fast"

# 後方互換（旧コード・メタ表示用）。既定モデルを指す。
SBERT_MODEL_NAME = SBERT_MODELS[DEFAULT_SBERT_KEY]["name"]
SBERT_MODEL_LABEL = SBERT_MODELS[DEFAULT_SBERT_KEY]["label"]


def _sbert_cfg_by_name(model_name):
    """モデル名から設定（prefix 等）を引く。未登録名は接頭辞なしでそのまま使う。"""
    for cfg in SBERT_MODELS.values():
        if cfg["name"] == model_name:
            return cfg
    return {"name": model_name, "label": model_name, "prefix": None}


def current_sbert_model_name():
    """解析に使われた（または選択中の）文埋め込みモデル名を返す。

    優先順: 前処理で実際に使用したモデル(sbert_model_used) > フェーズ4の現在の選択
    (sbert_model_name) > 既定。前処理後にラジオだけ変えても、レポート/インサイトには
    実際に解析へ使ったモデルが記録される。
    """
    return (st.session_state.get("sbert_model_used")
            or st.session_state.get("sbert_model_name")
            or SBERT_MODEL_NAME)


def current_sbert_model_info():
    """現在のモデル名・次元・ラベルをまとめた dict を返す（CAPCOM/メタ用）。"""
    name = current_sbert_model_name()
    cfg = _sbert_cfg_by_name(name)
    return {"name": name, "dim": cfg.get("dim"), "label": cfg.get("label", name)}


def build_sbert_embedder(model_name=None):
    """指定モデルで patiroha.SBERTEmbedder を構築して返す（model_name 省略時は既定）。

    - prefix を持つモデル（E5）は接頭辞を default_prompt_name 経由で全 encode に自動付与
      （patiroha 本体は非改変。MiniLM 等 prefix=None のモデルには付与しない）。
    - GPU(CUDA/MPS) があればそのデバイスに載せる。環境変数 APOLLO_FORCE_CPU=1 で CPU 強制。
    - 設定に失敗しても patiroha 既定構築のまま安全側で返す（解析は止めない）。
    """
    if not model_name:
        model_name = SBERT_MODEL_NAME
    cfg = _sbert_cfg_by_name(model_name)
    embedder = patiroha.SBERTEmbedder(model_name)
    try:
        model = getattr(embedder, "_model", None)
        if model is not None:
            # 接頭辞が必要なモデルのみ既定プロンプトとして登録（以後の encode に自動付与）
            prefix = cfg.get("prefix")
            if prefix:
                existing = dict(getattr(model, "prompts", {}) or {})
                existing["passage"] = prefix
                model.prompts = existing
                model.default_prompt_name = "passage"

            # デバイス選択（CUDA > MPS > CPU）
            force_cpu = os.environ.get("APOLLO_FORCE_CPU", "").strip() not in ("", "0", "false", "False")
            if not force_cpu:
                try:
                    import torch
                    device = None
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                        device = "mps"
                    if device:
                        embedder._model = model.to(device)
                except Exception:
                    pass  # GPU 失敗時は CPU のまま
    except Exception:
        pass  # 設定失敗時は patiroha 既定のまま安全側で返す
    return embedder


@st.cache_resource
def load_sbert_embedder(model_name=None):
    """文埋め込みモデルを構築してアプリ全体で共有する（特許=Home / 学術=NEBULA）。

    st.cache_resource は引数（model_name）ごとにキャッシュするため、モデルを切り替えても
    それぞれ 1 インスタンスに保たれる（同じモデルなら Home と NEBULA で実体を共有）。
    """
    if not model_name:
        model_name = SBERT_MODEL_NAME
    return build_sbert_embedder(model_name)


# ==================================================================
# --- 0y. HDBSCAN パラメータ自動掃引（クラスタ数の適正化） ---
# ==================================================================
def suggest_target_k(n):
    """母集団規模 N から「ちょうど良い」目標クラスタ数の目安を返す（少なすぎ/多すぎ回避）。"""
    if not n or n < 8:
        return 2
    return int(min(25, max(4, round(n ** 0.33))))


def sweep_hdbscan_params(embedding, target_k=None, k_min=None, k_max=None,
                         metric='euclidean', cluster_selection_method='eom',
                         max_trials=28, use_validity=True, progress_callback=None):
    """2D 埋め込みに対し HDBSCAN の (min_cluster_size, min_samples) を掃引し、
    「クラスタ数が適正（少なすぎ/多すぎでない）」かつ「クラスタリング品質が高い」組を返す。

    2 パス方式:
      ① 全候補を高速フィットしてクラスタ数・ノイズ率を得る（母集団規模 N に適応した候補グリッド）。
      ② 目標範囲に収まる候補（無ければ目標 k に近い数件）だけ DBCV（HDBSCAN の relative_validity_、
         密度ベースクラスタリング専用の妥当性指標）を計算する。
      最良 = 範囲内 かつ DBCV 最大（同点は目標 k への近さ・低ノイズで決定）。

    DBCV を主基準にすることで「目標 k に機械的に当てる」だけでなく、データの密度構造として
    妥当な分割を選ぶ（純粋な統計指標ではないが、密度クラスタリングでは標準的な検証指標）。

    Args:
        embedding: shape (N, 2) の 2D 座標（UMAP 等）。
        target_k: 目標クラスタ数（None なら N から自動）。
        k_min, k_max: 許容クラスタ数の範囲（None なら target_k から自動）。
        max_trials: ①で掃引する組み合わせの上限。
        use_validity: True なら②で DBCV を計算して品質基準に使う。
        progress_callback: 0.0-1.0 の進捗コールバック（任意）。

    Returns dict:
        min_cluster_size, min_samples, n_clusters, noise_ratio, validity(DBCV|None),
        labels(ndarray), target_k, k_min, k_max, trials(list[dict] / labels 抜き)
    """
    import numpy as np
    import hdbscan

    emb = np.asarray(embedding, dtype=float)
    n = len(emb)

    def _fit(mcs, ms, with_mst=False):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, int(mcs)), min_samples=max(1, int(ms)),
            metric=metric, cluster_selection_method=cluster_selection_method,
            gen_min_span_tree=with_mst)
        lab = clusterer.fit_predict(emb)
        kk = len(set(lab.tolist()) - {-1})
        nz = float((lab == -1).mean()) if n else 0.0
        rv = None
        if with_mst:
            try:
                rv = float(clusterer.relative_validity_)
                if rv != rv:  # NaN
                    rv = -1.0
            except Exception:
                rv = -1.0
        return lab, kk, nz, rv

    if target_k is None:
        target_k = suggest_target_k(n)
    if k_min is None:
        k_min = max(2, target_k // 2)
    if k_max is None:
        k_max = max(k_min + 1, target_k * 2)

    # N が小さすぎる場合は素直に既定で 1 回だけ
    if n < 8:
        lab, kk, nz, _ = _fit(2, 1)
        return {'min_cluster_size': 2, 'min_samples': 1, 'n_clusters': kk,
                'noise_ratio': nz, 'validity': None, 'labels': lab, 'target_k': target_k,
                'k_min': k_min, 'k_max': k_max, 'trials': []}

    # 掃引候補（母集団規模 N に適応）
    mcs_set = set()
    for frac in (0.004, 0.008, 0.015, 0.025, 0.04, 0.06):
        mcs_set.add(max(2, int(round(n * frac))))
    mcs_set.update({5, 10, 15, 25})
    mcs_candidates = sorted(m for m in mcs_set if 2 <= m <= max(2, n // 2))

    grid = []
    for mcs in mcs_candidates:
        ms_set = {1, max(1, mcs // 5), max(1, mcs // 2), mcs}
        for ms in sorted(s for s in ms_set if 1 <= s <= mcs):
            grid.append((mcs, ms))
    if len(grid) > max_trials:
        step = len(grid) / max_trials
        grid = [grid[int(i * step)] for i in range(max_trials)]

    # --- ① 高速フィット（MSTなし）でクラスタ数・ノイズを得る ---
    trials = []
    for i, (mcs, ms) in enumerate(grid):
        try:
            lab, kk, nz, _ = _fit(mcs, ms, with_mst=False)
            trials.append({'min_cluster_size': int(mcs), 'min_samples': int(ms),
                           'n_clusters': kk, 'noise_ratio': nz, 'validity': None, 'labels': lab})
        except Exception:
            pass
        if progress_callback:
            try:
                progress_callback(0.7 * (i + 1) / max(1, len(grid)))
            except Exception:
                pass

    if not trials:
        lab, kk, nz, _ = _fit(15, 10)
        return {'min_cluster_size': 15, 'min_samples': 10, 'n_clusters': kk,
                'noise_ratio': nz, 'validity': None, 'labels': lab, 'target_k': target_k,
                'k_min': k_min, 'k_max': k_max, 'trials': []}

    # --- 候補の絞り込み: 範囲内優先（無ければ目標kに近い数件）。DBCV計算を少数に限定 ---
    valid_trials = [t for t in trials if t['n_clusters'] >= 2]
    in_range = [t for t in valid_trials if k_min <= t['n_clusters'] <= k_max]
    base = in_range if in_range else (valid_trials if valid_trials else trials)
    shortlist = sorted(base, key=lambda t: (abs(t['n_clusters'] - target_k), t['noise_ratio']))[:6]

    # --- ② 絞り込んだ候補のみ DBCV（relative_validity_）を計算 ---
    if use_validity:
        for j, t in enumerate(shortlist):
            try:
                _, _, _, rv = _fit(t['min_cluster_size'], t['min_samples'], with_mst=True)
                t['validity'] = rv
            except Exception:
                t['validity'] = -1.0
            if progress_callback:
                try:
                    progress_callback(0.7 + 0.3 * (j + 1) / max(1, len(shortlist)))
                except Exception:
                    pass

    def _score(t):
        k = t['n_clusters']
        if k < 2:
            return -1e9
        in_rng = k_min <= k <= k_max
        rv = t['validity'] if t['validity'] is not None else 0.0
        # 範囲内を最優先 → DBCV(品質)最大 → 目標kへの近さ → 低ノイズ
        return (1000.0 if in_rng else 0.0) + rv * 20.0 - abs(k - target_k) * 1.0 - t['noise_ratio'] * 5.0

    pool = shortlist if shortlist else trials
    best = max(pool, key=_score)
    return {
        'min_cluster_size': best['min_cluster_size'],
        'min_samples': best['min_samples'],
        'n_clusters': best['n_clusters'],
        'noise_ratio': best['noise_ratio'],
        'validity': best.get('validity'),
        'labels': best['labels'],
        'target_k': target_k, 'k_min': k_min, 'k_max': k_max,
        'trials': [{k: v for k, v in t.items() if k != 'labels'} for t in trials],
    }


# DBCV（クラスタ品質指標）の解説。自動最適化モードで表示する。
DBCV_HELP_MD = """\
**DBCV（クラスタ品質）とは？**

DBCV（Density-Based Clustering Validation）は「クラスタがどれだけ **密で・よく分離** しているか」を表す、
密度ベースクラスタリング専用の品質指標です（HDBSCAN の `relative_validity_`）。**範囲は −1〜+1、高いほど良い**。

| DBCV | 目安 |
|---|---|
| > 0.5 | 非常に明確（合成データ並み。実テキストでは稀） |
| 0.3〜0.5 | 良好 |
| 0.15〜0.3 | まずまず（実特許 × UMAP2D ではこの辺が "普通に良い"） |
| 0.05〜0.15 | 弱い構造 |
| ≦ 0 | 構造が乏しい／不良 → クラスタ数を見直す価値あり |

**実用の目安**: 0.3 前後あれば上々／0.15 以上で実用／0 以下なら要再検討。

ただしこれは **同じデータ内でパラメータを比べる相対指標** で、絶対的な合格ラインはありません。
本ツールは 2D（UMAP）上でクラスタリングするため、実特許では **0.2〜0.4 でも十分良い** ことが多いです。
数字を追いかけず、最後は **クラスタ数が用途に合うか・各クラスタのラベルや代表特許が筋が通っているか** で
判断してください（DBCV はタイブレーカー）。
"""


def render_dbcv_help(key=None):
    """DBCV（クラスタ品質指標）の見方を折りたたみ（popover）で表示する共通部品。

    expander の入れ子禁止に抵触しないよう、collapsible には popover を使う。
    """
    try:
        with st.popover("品質DBCVの見方（クラスタ品質指標とは）"):
            st.markdown(DBCV_HELP_MD)
    except Exception:
        # popover 非対応環境などのフォールバック
        st.caption("品質DBCV: −1〜+1（高いほど良い）。0.3前後で上々／0.15以上で実用／0以下は要再検討。")


# ==================================================================
# --- 0. APOLLO Standard 色定数 ---
# ==================================================================
APOLLO_COLORS = px.colors.qualitative.G10      # Plotly 離散カラーパレット (10色)
APOLLO_BG = "#ffffff"                          # 背景色 (paper_bgcolor / plot_bgcolor)
APOLLO_TEXT = "#333333"                        # 文字色 (font.color)
APOLLO_ACCENT = "#003366"                      # アクセント色 (H1タイトル等)
APOLLO_TEMPLATE = "plotly_white"               # Plotly テンプレート

# ==================================================================
# --- 0b. ステータス パステルカラー (参考_AIS 準拠。低彩度で資料に載せやすい配色) ---
# ==================================================================
APOLLO_STATUS_PASTEL = {
    'granted':   "#AFCBEA",  # 登録/有効
    'published': "#C8DDF2",  # 公開
    'pending':   "#F2D7A6",  # 出願中
    'examining': "#EAC58D",  # 審査中
    'rejected':  "#EFB7B5",  # 拒絶
    'appeal':    "#CFC0E4",  # 審判・異議係属中
    'withdrawn': "#D6DCE8",  # 取下げ
    'expired':   "#C8CED8",  # 失効/消滅/放棄
}
# 未分類ステータス用フォールバック循環色 (パステル)
APOLLO_STATUS_FALLBACK_PASTEL = [
    "#AFCBEA", "#BFDCC8", "#F2D7A6", "#EBC4D6", "#CFC7E8",
    "#BFE0DF", "#E7C3B3", "#D5E3B7", "#CCD3DD", "#B8C7E6",
]

# ==================================================================
# --- 0c. 俯瞰図（ランドスケープ）共通スタイル ---
# ==================================================================
# 俯瞰図ファミリー（俯瞰図分析のメイン/ドリルダウン・環境分析の学術ランドスケープ）の
# 見た目をここで一元定義する。点・勢力圏・ラベルは必ず同じ色を使う（離散固定色）。
#
# 12色の並び順は、色覚多様性シミュレーション（Machado 2009）+ OKLab 色差の
# maximin 探索で決定（隣接する連番同士の見分けやすさを最大化・赤と緑は非隣接）。
# 13〜24番目のクラスタは同じ色相の暗色版（OKLab で明度を下げた2周目）、以降は循環。
LANDSCAPE_COLORS = [
    "#4E79A7", "#59A14F", "#B07AA1", "#F28E2B", "#499894", "#B6992D",
    "#5B8DB8", "#E15759", "#76B7B2", "#9C755F", "#EDC948", "#D37295",
]
LANDSCAPE_COLORS_2ND = [
    "#2D537B", "#37782E", "#855678", "#C16900", "#26706D", "#8C7200",
    "#3A668C", "#AE3339", "#538E89", "#72513E", "#C0A025", "#A44F6E",
]
LANDSCAPE_NOISE_STYLE = dict(color="#E2E3E8", size=3, opacity=0.55)  # ノイズ点（背景に沈める）
LANDSCAPE_POINT_SIZE = 6
LANDSCAPE_POINT_OPACITY = 0.88
LANDSCAPE_LABEL_TOP_N = 5  # ラベルを強調（拡大）する件数上位クラスタの数


def landscape_color(seq_idx):
    """クラスタの連番（0始まり・ノイズ除く）→ 固定色。12超は暗色2周目、24超は循環。"""
    block = (int(seq_idx) // 12) % 2
    palette = LANDSCAPE_COLORS if block == 0 else LANDSCAPE_COLORS_2ND
    return palette[int(seq_idx) % 12]


def landscape_color_map(cluster_ids):
    """クラスタID列 → {cid: HEX色}。ノイズ(-1)は除外し、ID昇順に連番で割り当てる。"""
    ids = sorted({int(c) for c in cluster_ids} - {-1})
    return {cid: landscape_color(i) for i, cid in enumerate(ids)}


def hex_to_rgba(hex_color, alpha):
    """'#RRGGBB' → 'rgba(r,g,b,a)'。既に rgba/rgb 形式ならそのまま返す。"""
    if not str(hex_color).startswith("#"):
        return hex_color
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def smooth_hull(points, expand=1.13, iterations=3):
    """点群(N,2)の凸包を、重心から expand 倍に膨らませて角を丸めた閉曲線にする。

    クラスタ領域を「ギザギザの多角形」ではなく有機的な「勢力圏」として描くための
    形状。膨張と平滑化のぶん実際の境界より広い「目安」である点に注意。

    Returns:
        (M, 2) ndarray（閉曲線・末尾に先頭点を複製済み）| None（3点未満・失敗時）
    """
    try:
        arr = np.asarray(points, dtype=float)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) < 3:
            return None
        poly = arr[ConvexHull(arr).vertices]
    except Exception:
        return None
    center = poly.mean(axis=0)
    poly = center + (poly - center) * expand
    for _ in range(int(iterations)):  # Chaikin corner cutting（角の丸め）
        n = len(poly)
        nxt = []
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            nxt.append(0.75 * a + 0.25 * b)
            nxt.append(0.25 * a + 0.75 * b)
        poly = np.asarray(nxt)
    return np.vstack([poly, poly[:1]])


def add_landscape_region(fig, points, color):
    """平滑化した勢力圏（塗り7%・輪郭線35%/1px）を1クラスタ分追加する。"""
    poly = smooth_hull(points)
    if poly is None:
        return
    fig.add_trace(go.Scatter(
        x=poly[:, 0], y=poly[:, 1], mode="lines", fill="toself",
        fillcolor=hex_to_rgba(color, 0.07),
        line=dict(color=hex_to_rgba(color, 0.35), width=1),
        hoverinfo="skip", showlegend=False,
    ))


def add_landscape_label(fig, x, y, text, color, emphasized=False, x_range=None, y_range=None):
    """ピル型クラスタラベル（●色ドット+文字・白地・薄枠）を追加する。

    emphasized=True（件数上位クラスタ）は文字を一段大きくし、地図の
    都市名のような大小の階層を作る。ドラッグ移動（editable）は従来どおり可能。
    x_range / y_range（(min, max)）を渡すと、端に近いクラスタのラベルの
    アンカーを内側に倒し、プロット外への見切れを抑える。
    """
    xanchor, yanchor = "center", "middle"
    try:
        if x_range and x_range[1] > x_range[0]:
            _tx = (float(x) - x_range[0]) / (x_range[1] - x_range[0])
            if _tx < 0.10:
                xanchor = "left"
            elif _tx > 0.90:
                xanchor = "right"
        if y_range and y_range[1] > y_range[0]:
            _ty = (float(y) - y_range[0]) / (y_range[1] - y_range[0])
            if _ty < 0.08:
                yanchor = "bottom"
            elif _ty > 0.92:
                yanchor = "top"
    except Exception:
        pass
    fig.add_annotation(
        x=x, y=y,
        text=f"<span style='color:{color}'>●</span> {text}",
        showarrow=False, xanchor=xanchor, yanchor=yanchor,
        font=dict(size=14 if emphasized else 11.5, color="#1F2430"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(31,36,48,0.12)", borderwidth=1, borderpad=5,
    )


def landscape_top_clusters(size_by_cid, top_n=None):
    """件数上位 top_n クラスタのID集合（ラベル強調用）。"""
    n = LANDSCAPE_LABEL_TOP_N if top_n is None else int(top_n)
    return set(sorted(size_by_cid, key=size_by_cid.get, reverse=True)[:n])


def landscape_density_bins(x_values, y_values, mesh_n, pad_bins=2):
    """全クラスタ共通の密度メッシュ (xbins, ybins) を返す。

    メッシュ範囲をデータの min/max ちょうどにすると、端にあるクラスタの
    等高線がメッシュ境界でスパッと切れて見える。pad_bins ビン分だけ外側に
    余白を持たせ、すそ野が自然に減衰して描かれるようにする。

    Returns:
        (xbins dict, ybins dict) — go.Histogram2dContour の xbins/ybins に渡す形式
    """
    x0, x1 = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    y0, y1 = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    n = max(int(mesh_n), 1)
    sx = max((x1 - x0) / n, 1e-6)
    sy = max((y1 - y0) / n, 1e-6)
    p = max(int(pad_bins), 0)
    xbins = dict(start=x0 - p * sx, end=x1 + p * sx, size=sx)
    ybins = dict(start=y0 - p * sy, end=y1 + p * sy, size=sy)
    return xbins, ybins


def landscape_bin_edges(bins):
    """landscape_density_bins が返す bins dict → np.histogram2d 用のビン境界配列。

    絶対スケール（共通 zmax）の最大ビン件数を、描画と同じメッシュで数えるために使う。
    """
    return np.arange(bins['start'], bins['end'] + bins['size'] * 0.5, bins['size'])


def add_landscape_density(fig, x_values, y_values, color, zmax=None, xbins=None, ybins=None,
                          line_width=0, line_color=None):
    """クラスタ1つ分のソフト密度（等高線風5段・地形図スタイル）を追加する。

    zmax を指定すると絶対スケール（全クラスタ/全表示で共通の濃さ基準）になり、
    未指定なら各クラスタ独立の自動スケールになる。
    xbins/ybins（dict(start, end, size)）を指定すると全クラスタ共通のメッシュで
    集計する（未指定は各クラスタのデータ範囲による自動ビン）。

    line_width > 0 で等高線の輪郭線を描く（既定は塗りだけ）。点が多い図では塗りが
    点に覆われて地形が読めなくなるため、線を出して等高線図として成立させたいときに使う。
    line_color 未指定時は色をそのまま薄めた線になる。
    """
    params = dict(
        x=x_values, y=y_values, ncontours=5,
        colorscale=[[0, "rgba(255,255,255,0)"], [1, hex_to_rgba(color, 0.38)]],
        showscale=False, hoverinfo="skip",
        line=dict(width=line_width, color=line_color or hex_to_rgba(color, 0.45)),
        contours=dict(coloring="fill", showlines=bool(line_width)), showlegend=False,
    )
    if zmax:
        params.update(dict(zauto=False, zmin=0, zmax=zmax))
    if xbins:
        params["xbins"] = xbins
    if ybins:
        params["ybins"] = ybins
    fig.add_trace(go.Histogram2dContour(**params))


# ==================================================================
# --- 0d. モジュール アイコン (NASA等の実写ベース 円形アイコン) ---
# ==================================================================
# 各モジュールの絵文字を assets/icons/<key>_icon.png に写実化したもの。
# 画像が見つからない場合は絵文字へフォールバックする。
_MODULE_EMOJI = {
    "home": "", "earth": "", "core": "", "saturnv": "",
    "mega": "", "explorer": "", "crew": "", "eagle": "",
    "record": "", "nebula": "", "capcom": "",
}


def _icon_path(key):
    """assets/icons/<key>_icon.png の絶対パスを返す（cwd 非依存）。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "icons", f"{key}_icon.png")


def module_icon(key):
    """st.set_page_config(page_icon=...) 用。実写アイコンのパスを返し、
    無ければ絵文字へフォールバックする。"""
    p = _icon_path(key)
    return p if os.path.exists(p) else _MODULE_EMOJI.get(key, "")


def module_header(key, title, subtitle=None):
    """円形の実写アイコン＋タイトルを1行で描画する（st.title の置換）。
    key は assets/icons/ 配下のファイル名キー（例: "earth"）。"""
    import base64
    p = _icon_path(key)
    try:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        # <img> は Streamlit の画像CSS（height:auto 等）や行ボックスの影響で
        # 上端が欠けることがあるため、固定サイズ div の背景画像として描画する。
        icon = (f'<div style="width:52px;height:52px;flex:0 0 auto;border-radius:50%;'
                f'background-image:url(data:image/png;base64,{b64});'
                f'background-size:cover;background-position:center;'
                f'box-shadow:0 1px 5px rgba(0,0,0,0.28)"></div>')
    except Exception:
        icon = (f'<div style="font-size:2.4rem;line-height:1;flex:0 0 auto">'
                f'{_MODULE_EMOJI.get(key, "")}</div>')
    sub = (f'<div style="font-size:0.95rem;color:#64748b;margin-top:.15rem">{subtitle}</div>'
           if subtitle else "")
    # 固定ヘッダー（[data-testid="stHeader"]、高さ約60px・不透明白）が本文先頭に
    # 重なるため、タイトルが欠けないよう上側に十分な余白を確保する（.block-container の
    # padding-top:2rem だけではヘッダー高さに足りない）。margin 相殺を避け padding を使う。
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;padding:1.5rem 0 .55rem">{icon}'
        f'<div style="min-width:0"><div style="font-size:2.25rem;font-weight:700;'
        f'line-height:1.15;margin:0">{title}</div>{sub}</div></div>',
        unsafe_allow_html=True)

# --- 地図/マップ用の鮮やかなステータス配色（視認性重視） ---
# パステル(資料用)は小さな点では見分けにくいため、マップでは彩度の高いこちらを使う。
APOLLO_STATUS_VIVID = {
    'granted':   "#2E86DE",  # 登録/有効 — 濃い青
    'published': "#74B9FF",  # 公開 — 明るい青
    'pending':   "#F39C12",  # 出願中 — オレンジ
    'examining': "#E67E22",  # 審査中 — 濃いオレンジ
    'rejected':  "#E74C3C",  # 拒絶 — 赤
    'appeal':    "#8E44AD",  # 審判・異議係属中 — 濃い紫
    'withdrawn': "#9B59B6",  # 取下げ — 紫
    'expired':   "#7F8C8D",  # 失効/消滅/放棄 — グレー
}
APOLLO_STATUS_FALLBACK_VIVID = [
    "#2E86DE", "#27AE60", "#F39C12", "#E84393", "#8E44AD",
    "#16A085", "#E67E22", "#2ECC71", "#34495E", "#3498DB",
]


def classify_status(s):
    """ステータス文字列をセマンティックカテゴリに分類する（日英キーワード）。

    APOLLO 全モジュールで色を統一するための共通分類。拒絶を pending より先に
    判定する（「拒絶査定後の出願」等の誤判定回避）。
    """
    s_lower = str(s).lower()
    # 「査定」単体は不可（「拒絶査定」も含むため）。granted は「特許査定」のみ拾う。
    if any(k in s_lower for k in ['granted', 'registered', 'active', 'allowed', 'allowance',
                                  '登録', '有効', '権利存続', '存続', '権利継続', '特許査定']):
        return 'granted'
    # 審判・異議系（J-PlatPatの「査定不服」「異議申立中・無効審判中」等）。
    # 「拒絶査定不服審判」を rejected に誤分類しないよう rejected より先に判定する。
    if any(k in s_lower for k in ['appeal', 'opposition', 'invalidation', '審判', '査定不服', '異議']):
        return 'appeal'
    if any(k in s_lower for k in ['rejected', 'refused', 'denied', '拒絶', '却下']):
        return 'rejected'
    if any(k in s_lower for k in ['withdrawn', 'withdraw', '取下', '取り下げ']):
        return 'withdrawn'
    if any(k in s_lower for k in ['expired', 'lapsed', 'abandoned', 'dead', '消滅', '失効', '放棄', '満了']):
        return 'expired'
    if any(k in s_lower for k in ['examining', 'examination', 'review', '審査', '審理']):
        return 'examining'
    if any(k in s_lower for k in ['published', 'publication', '公開', '公表']):
        return 'published'
    if any(k in s_lower for k in ['pending', 'application', 'filed', 'filing', '出願', '係属']):
        return 'pending'
    return None


def status_color_for(value, fallback_idx=0, vivid=False):
    """ステータス値 → 色。vivid=True で地図用の鮮やかな配色、False で資料用パステル。

    未分類はフォールバック循環色を返す。
    """
    category = classify_status(value)
    if vivid:
        if category and category in APOLLO_STATUS_VIVID:
            return APOLLO_STATUS_VIVID[category]
        return APOLLO_STATUS_FALLBACK_VIVID[fallback_idx % len(APOLLO_STATUS_FALLBACK_VIVID)]
    if category and category in APOLLO_STATUS_PASTEL:
        return APOLLO_STATUS_PASTEL[category]
    return APOLLO_STATUS_FALLBACK_PASTEL[fallback_idx % len(APOLLO_STATUS_FALLBACK_PASTEL)]


# ステータス列が NaN の行を集計・表示に含めるための共通ラベル。
# J-PlatPat は 1997 年（平成9年）より前の案件にステージ情報を付与しない等、欠損は正常に起こる。
# groupby は NaN をそのまま落とすため、内訳系の集計では本ラベルで埋めてから集計する。
STATUS_UNKNOWN_LABEL = "(ステータス不明)"


def build_status_color_map(statuses):
    """ステータス値のリスト（ソート済み推奨）→ {status: パステル色} の配色マップ。

    分類済みカテゴリは固定色、未分類は順にフォールバックパステルを割り当てる。
    全タブ・全モジュールで色が統一される。
    """
    color_map = {}
    fallback_idx = 0
    for s in statuses:
        # ステータス不明ラベルは中立グレー固定（フォールバック1色目が「登録」と同色のため、
        # 循環色に任せると不明が登録済みに見えてしまう）
        if s == STATUS_UNKNOWN_LABEL:
            color_map[s] = "#C4C4C4"
            continue
        category = classify_status(s)
        if category and category in APOLLO_STATUS_PASTEL:
            color_map[s] = APOLLO_STATUS_PASTEL[category]
        else:
            color_map[s] = APOLLO_STATUS_FALLBACK_PASTEL[fallback_idx % len(APOLLO_STATUS_FALLBACK_PASTEL)]
            fallback_idx += 1
    return color_map


def compute_grant_rate_stats(df, group_col, status_col, denom="all", min_count=1, expired_as_granted=True):
    """グループ（出願人・IPC 等）別に「出願数」と「権利化率」を算出する。

    classify_status でステータスを正規化する。「権利化＝登録に到達したか」を測る指標。
    - expired_as_granted=True（既定）: 失効（満了・放棄・消滅）も「一度は登録された＝権利化成功」として
      登録側に算入する。失効はマイナス要素にしない（＝過去に権利化に成功した事実を評価）。
    - expired_as_granted=False: 現在有効（権利継続・登録・有効・存続）のみを権利化成功とみなす（現存権利率）。
    - 審判・異議系（appeal）のうち登録後の争い（異議申立・無効審判）は「現在も登録が生きている」ため
      権利化成功・処分確定の両方に常に算入する。登録前の争い（拒絶査定不服審判等）は審査中と同じ
      未確定扱い（denom="all" のときのみ分母に入る）。区別は生のステータス文字列で行う。

    権利化率の定義は denom で切り替える（分子=権利化成功 success）:
    - denom="all":     権利化率 = success / 全件（係属・審査・公開も分母。母集団全体の権利化度）
    - denom="decided": 権利化率 = success / 処分確定（登録+失効+拒絶+取下）。審査/係属/公開は分母から除外

    group_col の各セルがリスト（applicant_main / ipc_normalized 等）の場合は自動で explode する。
    min_count 未満のグループは除外。戻り値は total 降順の DataFrame。
    列: group, total, granted, granted_eff(権利化成功数), rejected, examining, pending, published,
        withdrawn, expired, appeal(審判・異議係属中), other, decided, grant_rate(%)。
    """
    if (df is None or df.empty or not group_col or group_col not in df.columns
            or not status_col or status_col not in df.columns):
        return pd.DataFrame()

    work = df[[group_col, status_col]].copy()
    # リスト列（出願人/IPC は1セルに複数値）なら展開して1行=1(グループ,特許)にする
    if work[group_col].apply(lambda v: isinstance(v, (list, tuple))).any():
        work = work.explode(group_col)
    work[group_col] = work[group_col].astype(str).str.strip()
    work = work[(work[group_col] != "") & (~work[group_col].str.lower().isin(["nan", "none", "n/a"]))]
    if work.empty:
        return pd.DataFrame()

    work["_cat"] = work[status_col].map(classify_status)
    cats = ["granted", "rejected", "examining", "pending", "published", "withdrawn", "expired", "appeal"]
    # 登録後の審判・異議（異議申立・無効審判）は「現在も登録されている」ため権利化成功に数える。
    # 登録前の審判（拒絶査定不服等）は未確定のまま。両者は生のステータス文字列で区別する
    work["_appeal_post"] = (work["_cat"] == "appeal") & work[status_col].astype(str).str.contains(
        "異議|無効|opposition|invalidation", case=False, na=False)

    rows = []
    for g, sub in work.groupby(group_col):
        total = int(len(sub))
        if total < min_count:
            continue
        counts = {c: int((sub["_cat"] == c).sum()) for c in cats}
        appeal_post = int(sub["_appeal_post"].sum())
        other = total - sum(counts.values())
        # 処分確定 = 登録 + 失効 + 拒絶 + 取下 + 登録後審判（登録済みの権利への争いなので確定側）
        decided = (counts["granted"] + counts["expired"] + counts["rejected"]
                   + counts["withdrawn"] + appeal_post)
        # 権利化成功（分子）。失効を「一度登録された＝成功」に含めるかを切替。
        # 登録後審判は現在も登録が生きているため expired_as_granted の設定によらず成功側
        success = counts["granted"] + appeal_post + (counts["expired"] if expired_as_granted else 0)
        if denom == "decided":
            rate = (success / decided * 100.0) if decided > 0 else None
        else:
            rate = (success / total * 100.0) if total > 0 else None
        rows.append({
            "group": g, "total": total, "granted": counts["granted"], "granted_eff": int(success),
            "rejected": counts["rejected"], "examining": counts["examining"],
            "pending": counts["pending"], "published": counts["published"],
            "withdrawn": counts["withdrawn"], "expired": counts["expired"],
            "appeal": counts["appeal"],
            "other": int(other), "decided": int(decided),
            "grant_rate": round(rate, 1) if rate is not None else None,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values("total", ascending=False).reset_index(drop=True)


# ==================================================================
# --- 0e. J-PlatPat CSV 取込サポート ---
# ==================================================================
# J-PlatPat（特許情報プラットフォーム）の検索結果CSVを、カラム紐付けの手作業
# なしでそのまま読み込むための検出・正規化・結合ヘルパー。
# いずれも Streamlit 非依存の純関数（ヘッドレスで単体テスト可能）。

# J-PlatPat CSV を見分ける特徴カラム（4つすべて揃えば J-PlatPat 形式とみなす）
JPLAT_SIGNATURE_COLUMNS = {"文献番号", "出願人/権利者", "FI", "ステージ"}

# J-PlatPat カラム → 分析フィールドの対応（「要約」列は要約なしDLだと存在しない。
# IPC列は無くFIのみのため、分類コードは fi スロットに割り当てる）
JPLAT_COL_MAP = {
    'title': '発明の名称', 'abstract': '要約', 'claim': None,
    'app_num': '出願番号', 'date': '出願日', 'applicant': '出願人/権利者',
    'ipc': None, 'fi': 'FI', 'pub_number': '公開番号', 'status': 'ステージ',
    'inventor': None, 'fterm': None, 'doc_url': '文献URL',
}
# J-PlatPat 用の区切り文字（FI は prepare_jplat_df で ';' 区切りに正規化して合わせる）
JPLAT_DELIMITERS = {'applicant': ',', 'inventor': ';', 'ipc': ';', 'fterm': ';'}

_FI_HEAD_RE = re.compile(r'^[A-H]\d')  # FI記号の先頭（セクション記号+類番号）


def clean_header_columns(columns):
    """カラム名から BOM・前後空白を除去したリストを返す（UTF-8 BOM 付きCSV対策）。"""
    return [str(c).replace('\ufeff', '').strip() for c in columns]


def is_jplatpat_columns(columns):
    """カラム名の並びが J-PlatPat CSV のシグネチャを満たすか。"""
    return JPLAT_SIGNATURE_COLUMNS.issubset(set(clean_header_columns(columns)))


def strip_gaiji_marks(value):
    """J-PlatPat の外字代替表記の記号 ▲▼ を除去する（中の文字は残す）。

    例: '深▲セン▼市華宝新能源' → '深セン市華宝新能源'
    """
    if not isinstance(value, str):
        return value
    return value.replace('▲', '').replace('▼', '')


def normalize_jplat_fi_cell(value):
    """J-PlatPat の FI セルを ';' 区切りに正規化する。

    FI は記号の内部にもカンマを含む（例: 'H02J3/00,170' で1つのFI）ため、
    単純なカンマ分割では展開記号が独立した断片になってしまう。
    先頭が英字+数字のトークンを新しいFIの開始とみなし、それ以外
    （展開記号など）は直前のFIに連結し直す。

    例: 'H02J3/00,170,H02J3/32' → 'H02J3/00,170;H02J3/32'
    """
    if not isinstance(value, str) or not value.strip():
        return ''
    codes = []
    for token in value.split(','):
        token = token.strip()
        if not token:
            continue
        if _FI_HEAD_RE.match(token) or not codes:
            codes.append(token)
        else:
            codes[-1] = codes[-1] + ',' + token
    return ';'.join(codes)


def clean_jplat_abstract(value):
    """J-PlatPat の要約セルから定型部分を除去する。

    - 先頭の「(57)【要約】」ヘッダ
    - 「【選択図】…」（選択図の図番号の指示。テキスト分析には不要）
    【課題】【解決手段】は内容の区切りとして意味を持つため残す。
    """
    if not isinstance(value, str):
        return value
    v = re.sub(r'^\s*[（(]\s*57\s*[）)]\s*【\s*要約\s*】\s*', '', value)
    v = re.sub(r'【\s*選択図\s*】.*', '', v, flags=re.S)
    return v.strip()


def prepare_jplat_df(df):
    """J-PlatPat CSV の DataFrame を分析用に正規化した複製を返す。

    - カラム名の BOM・前後空白を除去
    - 全文字列セルから外字記号 ▲▼ を除去
    - FI 列を ';' 区切りに正規化（JPLAT_DELIMITERS['ipc'] と対応）
    - 要約列の定型部分（「(57)【要約】」ヘッダ・「【選択図】…」）を除去
    """
    df = df.copy()
    df.columns = clean_header_columns(df.columns)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(strip_gaiji_marks)
    if 'FI' in df.columns:
        df['FI'] = df['FI'].map(normalize_jplat_fi_cell)
    if '要約' in df.columns:
        df['要約'] = df['要約'].map(clean_jplat_abstract)
    return df


def merge_patent_files(dfs, names, dedupe_col='文献番号'):
    """同じ形式の特許リスト（分割ダウンロード）を1つの DataFrame に結合する。

    カラム構成が一致するファイル同士のみ結合を許す（要約あり・なしの混在は
    列数が違うためここで検出されエラーになる）。重複は dedupe_col（既定は
    J-PlatPat の文献番号）があればその値で、なければ全列一致で除去する。

    Args:
        dfs: DataFrame のリスト
        names: 各 DataFrame の表示名（ファイル名）のリスト
    Returns:
        (結合後DataFrame | None, info dict)
        info = {'n_files', 'per_file': [(name, 行数)], 'n_rows_total',
                'n_dup_removed', 'n_rows_final', 'error': None | メッセージ}
    """
    info = {'n_files': len(dfs), 'per_file': [(n, len(d)) for n, d in zip(names, dfs)],
            'n_rows_total': 0, 'n_dup_removed': 0, 'n_rows_final': 0, 'error': None}
    if not dfs:
        info['error'] = "ファイルがありません。"
        return None, info
    base_cols = clean_header_columns(dfs[0].columns)
    for name, d in zip(names[1:], dfs[1:]):
        cols = clean_header_columns(d.columns)
        if cols != base_cols:
            only_base = [c for c in base_cols if c not in cols]
            only_this = [c for c in cols if c not in base_cols]
            detail = []
            if only_base:
                detail.append(f"「{names[0]}」だけにある列: {'・'.join(only_base[:5])}")
            if only_this:
                detail.append(f"「{name}」だけにある列: {'・'.join(only_this[:5])}")
            info['error'] = (
                f"「{names[0]}」と「{name}」のカラム構成が一致しません（同じ条件でダウンロードした"
                f"ファイル同士を選んでください。要約あり・なしの混在は結合できません）。"
                + ("　" + " / ".join(detail) if detail else ""))
            return None, info
    merged = pd.concat(dfs, ignore_index=True)
    merged.columns = clean_header_columns(merged.columns)
    info['n_rows_total'] = len(merged)
    if dedupe_col and dedupe_col in merged.columns:
        merged = merged.drop_duplicates(subset=[dedupe_col], keep='first').reset_index(drop=True)
    else:
        merged = merged.drop_duplicates(keep='first').reset_index(drop=True)
    info['n_dup_removed'] = info['n_rows_total'] - len(merged)
    info['n_rows_final'] = len(merged)
    return merged, info


def classification_code_column(col_map=None):
    """分析に使う分類コードの実カラム名を返す（IPC/FI 選択を反映）。

    col_map['ipc']=IPC列・col_map['fi']=FI列の2スロット制。両方割り当て時は
    「分析に使う分類コード」の選択（session_state['code_pref']）に従い、
    片方だけならその列を返す。どちらも無ければ None。
    """
    cm = col_map if col_map is not None else st.session_state.get('col_map', {})
    ipc_col, fi_col = cm.get('ipc'), cm.get('fi')
    if ipc_col and fi_col:
        return fi_col if st.session_state.get('code_pref') == 'FI' else ipc_col
    return fi_col or ipc_col


def classification_code_label(col_map=None):
    """画面表示用の分類コードの呼び名（'IPC' または 'FI'）を返す。

    分析に使っている実効列（classification_code_column）が FI 側なら 'FI'。
    旧セッション（'fi' スロットなしで ipc スロットに FI 列を割り当てたもの）は
    列名が 'FI' かどうかで判定する。表示ラベル専用で、内部キー名 'ipc' は変えない。
    """
    cm = col_map if col_map is not None else st.session_state.get('col_map', {})
    col = classification_code_column(cm)
    if not col:
        return 'IPC'
    if cm.get('fi') and col == cm.get('fi'):
        return 'FI'
    return 'FI' if str(col).strip().upper() == 'FI' else 'IPC'


def combine_text_columns(df, col_map, keys=('title', 'abstract', 'claim')):
    """col_map の割り当て済みテキスト列だけを空白区切りで結合した Series を返す。

    要約・請求項がないデータ（J-PlatPat 等）では割り当てのある列のみ使う。
    1列も割り当てがない場合は空文字の Series を返す。
    """
    parts = [df[col_map[k]].fillna('').astype(str)
             for k in keys if col_map.get(k) and col_map[k] in df.columns]
    if not parts:
        return pd.Series([''] * len(df), index=df.index)
    combined = parts[0]
    for s in parts[1:]:
        combined = combined + ' ' + s
    return combined


# ==================================================================
# --- 1. フォント設定 (共通) ---
# ==================================================================
def get_japanese_font_path():
    """OSを判定して適切な日本語フォントパスを返す"""
    system = platform.system()
    font_paths = []
    
    if system == "Darwin": # Mac
        font_paths = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans W3.ttc",
            "/System/Library/Fonts/Hiragino Kaku Gothic ProN.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc" 
        ]
    elif system == "Windows": # Windows
        font_paths = [
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/yugothr.ttc",
            "C:/Windows/Fonts/YuGothR.ttc"
        ]
    else: # Linux (Streamlit Cloudなど)
        font_paths = [
            "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto/NotoSansCJKjp-Regular.otf"
        ]
        
    for path in font_paths:
        if os.path.exists(path): return path
    return None


@st.cache_data(show_spinner=False)
def compute_wordcloud_array(words_tuple, font_path=None, width=800, height=400, max_words=100):
    """WordCloud 画像（RGB numpy配列）を生成してメモリキャッシュする。

    重い generate_from_frequencies をメモリキャッシュし、同じ語頻度・フォントなら再計算しない。
    ローカルストレージは使わず Streamlit のメモリキャッシュ（persist 無効）に保持する。

    Args:
        words_tuple: tuple[(word, count), ...]（ハッシュ可能化のためソート済みタプルで渡す）
        font_path: 日本語フォントパス
    Returns:
        np.ndarray: RGB画像配列（matplotlib の imshow にそのまま渡せる）
    """
    from wordcloud import WordCloud
    word_freq = dict(words_tuple)
    wc = WordCloud(
        width=width, height=height, background_color='white',
        font_path=font_path, collocations=False, max_words=max_words
    ).generate_from_frequencies(word_freq)
    return wc.to_array()


def configure_matplotlib_font():
    """Matplotlibのフォント設定を適用する"""
    font_path = get_japanese_font_path()
    if font_path:
        try:
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            return font_path
        except:
            pass
    return None

# ==================================================================
# --- 2. ストップワード (共通) — patiroha委譲 ---
# ==================================================================

def get_stopwords(mode='patent'):
    """ストップワード取得 (patiroha委譲)"""
    return patiroha.get_stopwords(mode)

def get_patent_stopwords():
    """特許分析用のストップワード"""
    return patiroha.get_stopwords("patent")

def get_npl_stopwords():
    """NPL分析用のストップワード"""
    return patiroha.get_stopwords("npl")

# ==================================================================
# --- 3. 統計指標 (patiroha委譲) ---
# ==================================================================
def calculate_hhi(counts):
    """ヘルフィンダール・ハーシュマン指数 (HHI) を計算し、公取委基準で判定する (patiroha委譲)"""
    result = patiroha.calculate_hhi(counts)
    return result.value, result.status

def calculate_cagr_slope(df_subset, year_col='year'):
    """年平均成長率(CAGR)とトレンド(Slope)を計算する (patiroha委譲)"""
    result = patiroha.calculate_cagr(df_subset, year_col=year_col)
    return result.growth_rate, result.trend

def best_patent_ref(row, col_map):
    """代表特許の識別子を返す。
    公開番号(pub_number)が非空ならそれを優先、無ければ出願番号(app_num)、
    どちらも無ければ 'N/A' を返す。
    CAPCOMレポートで実在番号を引用させ、番号捏造（特開2023-XXXXXX 等）を防ぐための統一ヘルパー。
    ※カード表示(render_paper_card)は「出願番号優先」の別仕様のため、ここは使わない。
    """
    for key in ('pub_number', 'app_num'):   # 公開番号を優先、無ければ出願番号
        c = col_map.get(key)
        if c and c in row.index:
            v = row.get(c)
            if v is not None and pd.notna(v) and str(v).strip() not in ('', 'N/A', 'nan'):
                return str(v).strip()
    return 'N/A'

@st.cache_data(show_spinner=False)
def generate_rich_summary(df_target, title_col='title', abstract_col='abstract', n_representatives=5):
    """
    VOYAGER v5.1用の高解像度サマリを生成する (Cached)
    - 統計情報 (HHI, CAGR, Trend)
    - 代表特許 (Centroid Distance)
    """
    summary = {
        "stats": {},
        "representatives": []
    }
    
    # 1. 統計情報の計算 (年次推移がある場合)
    if 'year' in df_target.columns:
        cagr, trend = calculate_cagr_slope(df_target)
        summary['stats']['cagr'] = f"{cagr:.1%}" if cagr is not None else "N/A"
        summary['stats']['trend'] = trend if trend else "N/A"

    # 2. HHI (市場集中度) の計算
    try:
        # 出願人情報 ('applicant_main') を利用して市場集中度を算出
        if 'applicant_main' in df_target.columns:
            all_apps = [a for sublist in df_target['applicant_main'] for a in sublist]
            counts = pd.Series(all_apps).value_counts().tolist()
            hhi, hhi_status = calculate_hhi(counts)
            summary['stats']['hhi'] = hhi
            summary['stats']['hhi_status'] = hhi_status
    except: pass
        
    # 3. 代表特許の抽出 (Centroid Distance)
    if 'sbert_embeddings' in st.session_state and not df_target.empty:
        try:
            # df_targetのindexを使ってembeddingsを抽出
            # 前提: df_mainのindexがresetされておらず、embeddingsと1対1対応していること
            valid_indices = [i for i in df_target.index if i < len(st.session_state.sbert_embeddings)]
            
            if valid_indices:
                vectors = st.session_state.sbert_embeddings[valid_indices]
                centroid = np.mean(vectors, axis=0)
                
                # 重心との距離計算 (Cosine Similarity相当)
                dots = np.dot(vectors, centroid)
                
                # 上位N件のインデックスを取得
                top_n_local_indices = np.argsort(dots)[::-1][:n_representatives]
                top_global_indices = [valid_indices[i] for i in top_n_local_indices]
                
                # データ取得
                reps = []
                invalid_count = 0
                
                # Column mapping for enhanced info
                col_map = st.session_state.get('col_map', {})
                app_col = col_map.get('applicant', 'applicant')
                ipc_col = classification_code_column(col_map)
                # 番号は best_patent_ref(row, col_map) で公開番号→出願番号の順に取得

                for idx in top_global_indices:
                    try:
                        row = st.session_state.df_main.loc[idx]
                        t_val = str(row.get(title_col, ''))
                        # 要約列がないデータ（abstract_col=None）は要約なしで代表特許を出す
                        a_val = str(row.get(abstract_col, '')) if abstract_col else ''

                        # Enhanced Info
                        y_val = str(row.get('year', 'N/A'))
                        app_val = _extract_field_value(row, app_col, max_len=30) if app_col else "N/A"
                        ipc_val = _extract_field_value(row, ipc_col, max_len=40) if ipc_col else "N/A"
                        num_val = best_patent_ref(row, col_map)

                        # Check validity
                        if (not t_val or t_val == 'nan') and (not a_val or a_val == 'nan'):
                             invalid_count += 1
                             title = "No Title"
                             abstract = "No Abstract"
                        else:
                             title = t_val if t_val and t_val != 'nan' else "No Title"
                             abstract = a_val if a_val and a_val != 'nan' else "No Abstract"

                        title = title.replace('\n', ' ')
                        abstract = abstract.replace('\n', ' ')[:200] + "..."

                        reps.append(f"- [{num_val}]【{title}】 (出願: {y_val}, {app_val}, IPC:{ipc_val}) {abstract}")

                        summary.setdefault('representatives_raw', []).append({
                            'title': title,
                            'abstract': abstract,
                            'year': y_val,
                            'applicant': app_val,
                            'ipc': ipc_val,
                            'number': num_val
                        })
                    except: pass
                
                # If mostly invalid, don't show
                if len(reps) > 0 and (invalid_count / len(reps)) > 0.5:
                     summary['representatives'] = [] # Suppress
                     summary['representatives_raw'] = []
                else:
                     summary['representatives'] = reps

        except Exception as e:
            summary['error'] = str(e)

    return summary

def get_cluster_representatives(df_subset, cluster_col='cluster', n_reps=3):
    """
    データフレーム内の各クラスタについて、重心に近い代表特許を抽出する (patiroha委譲)。
    Returns:
        dict: {cluster_id: ["- 【Title】(Applicant): Abstract...", ...]}
    """
    reps_dict = {}

    unique_clusters = sorted(df_subset[cluster_col].unique())
    embeddings = st.session_state.get('sbert_embeddings')

    col_map = st.session_state.get('col_map', {})
    title_col = col_map.get('title', 'title')
    abs_col = col_map.get('abstract', 'abstract')
    app_col = col_map.get('applicant', 'applicant')
    ipc_col = classification_code_column(col_map)
    # 番号は best_patent_ref(row, col_map) で公開番号→出願番号の順に取得

    for cid in unique_clusters:
        if cid == -1: continue

        df_c = df_subset[df_subset[cluster_col] == cid]
        if df_c.empty: continue

        try:
            target_indices = []

            # patiroha.find_representatives でセントロイドベース抽出
            if embeddings is not None and len(embeddings) >= df_subset.index.max():
                valid_indices = [i for i in df_c.index if i < len(embeddings)]
                if valid_indices:
                    vectors = embeddings[valid_indices]
                    # vectors と整合する df を渡す（valid_indices で絞り込んだ df_c）
                    reps = patiroha.find_representatives(vectors, df_c.loc[valid_indices], n=n_reps)
                    # reps の index は渡した df 内の iloc 位置 → 元の df ラベルへ変換
                    for r in reps:
                        if hasattr(r, 'index') and r.index is not None:
                            target_indices.append(valid_indices[r.index])

            # フォールバック: head選択
            if not target_indices:
                target_indices = df_c.index[:n_reps].tolist()

            cluster_reps = []
            for idx in target_indices:
                row = st.session_state.df_main.loc[idx]

                t_val = str(row.get(title_col, 'No Title')).replace('\n', ' ')
                a_val = str(row.get(abs_col, 'No Abstract')).replace('\n', ' ')[:200] + "..."
                app_val = _extract_field_value(row, app_col, max_len=30)
                y_val = str(row.get('year', 'N/A'))
                ipc_val = _extract_field_value(row, ipc_col, max_len=40) if ipc_col else "N/A"
                num_val = best_patent_ref(row, col_map)

                cluster_reps.append(f"  * [{num_val}]【{t_val}】({app_val}, {y_val}, IPC:{ipc_val}): {a_val}")

            reps_dict[cid] = cluster_reps

        except Exception as e:
            continue

    return reps_dict


def _extract_field_value(row, col_name, max_len=30):
    """行からフィールド値を安全に抽出するヘルパー"""
    if not col_name or col_name not in row.index:
        return "N/A"
    val = row[col_name]
    if isinstance(val, list):
        clean_vals = [str(x).strip() for x in val if x and str(x).lower() != 'nan']
        result = ", ".join(clean_vals)
    elif pd.isna(val):
        return "N/A"
    else:
        result = str(val)
    if len(result) > max_len:
        result = result[:max_len] + "..."
    return result

def get_keyword_centric_representatives(df_target, top_keywords, n_reps=10):
    """
    ネットワークの主要キーワードを多く含む特許を抽出する (Keyword-Centric)。
    Args:
        df_target: 対象DataFrame
        top_keywords: ネットワークの中心性/頻度が高いキーワードリスト (list of str)
        n_reps: 抽出件数 (User request: significantly increase)
    Returns:
        list of dict: [{'title':..., 'applicant':..., 'abstract':..., 'score':...}]
    """
    if df_target.empty or not top_keywords:
        return []

    try:
        # Avoid side effects
        df = df_target.copy()
        
        col_map = st.session_state.get('col_map', {})
        t_col = col_map.get('title', 'title')
        a_col = col_map.get('abstract', 'abstract')
        app_col = col_map.get('applicant', 'applicant')
        ipc_col = classification_code_column(col_map)
        # 番号は best_patent_ref(row, col_map) で公開番号→出願番号の順に取得
        y_col = 'year'

        # 上位30キーワードでスコアリング
        target_kws = top_keywords[:30]
        import re
        safe_kws = [re.escape(k) for k in target_kws if k and isinstance(k, str)]
        if not safe_kws: return []

        pattern = "|".join(safe_kws)

        # スコア計算（タイトル重み: x2, 要約: x1）
        score_series = pd.Series(0, index=df.index)
        if t_col in df.columns:
            score_series += df[t_col].astype(str).str.count(pattern) * 2
        if a_col in df.columns:
            score_series += df[a_col].astype(str).str.count(pattern)

        df['_kw_score'] = score_series

        # スコア > 0 のみ抽出してソート
        df_sorted = df[df['_kw_score'] > 0].sort_values(by='_kw_score', ascending=False).head(n_reps)

        reps = []
        for _, row in df_sorted.iterrows():
            title = str(row.get(t_col, 'No Title')).replace('\n', ' ')
            abstract = str(row.get(a_col, 'No Abstract')).replace('\n', ' ')[:200] + "..."
            year = str(row.get(y_col, '-'))

            app_val = "N/A"
            if app_col and app_col in row.index:
                val = row[app_col]
                if isinstance(val, list): app_val = ",".join([str(x) for x in val if x])
                else: app_val = str(val)
            if len(app_val) > 30: app_val = app_val[:30] + "..."

            # IPC
            ipc_val = "N/A"
            if ipc_col and ipc_col in row.index:
                val = row[ipc_col]
                if isinstance(val, list): ipc_val = ",".join([str(x) for x in val if x])[:40]
                else: ipc_val = str(val)[:40]

            # 番号（公開番号優先、無ければ出願番号）
            num_val = best_patent_ref(row, col_map)

            reps.append({
                'title': title,
                'abstract': abstract,
                'year': year,
                'applicant': app_val,
                'ipc': ipc_val,
                'number': num_val,
                'score': row['_kw_score'],
            })

        return reps

    except Exception as e:
        pass
        return []

def render_snapshot_button(title, description, key, fig=None, data_summary=None, figs=None, allow_multiple=True, group=None):
    """
    グラフやデータをVOYAGER用に保存するボタンを表示する
    Args:
        title (str): スナップショットのタイトル
        description (str): スナップショットの説明
        key (str): このビュー固有の識別キー（保存済み判定に使う）。
                   ドリルダウン等「同じマップでビューが切り替わる」場合は、対象（クラスタ等）を含めて
                   ビューごとに一意にすること（例 saturn_drill_snap_<対象>）。静的キーにすると、
                   1つ撮った後に切り替えても「保存済み」のまま出てしまう。
        fig (Figure): 単一のFigure (レガシーサポート)
        data_summary (dict): 分析データのサマリー
        figs (list): Figureのリスト (複数画像スナップショット用)
        allow_multiple (bool): 後方互換のため残置（現在は未使用）
        group (str): 「このマップで何枚撮ったか」を数える系列プレフィックス。未指定なら key を使う。
                     ドリルダウンでは group をマップ共通の接頭辞（例 saturn_drill_snap）にすると、
                     クラスタ別 key を跨いで枚数を集計できる。
    """
    if 'snapshots' not in st.session_state:
        st.session_state['snapshots'] = []

    # 保存済みかチェック（この「現在のビュー」= key 完全一致が保存済みか）
    is_saved = any(s['id'] == key for s in st.session_state['snapshots'])

    # このマップ系列（group プレフィックスで束ねる）で保存済みの枚数
    group_prefix = group or key
    _map_count = sum(1 for s in st.session_state['snapshots']
                     if str(s.get('id', '')).startswith(group_prefix))

    btn_label = "保存済み" if is_saved else "レポート用に保存"
    btn_type = "secondary" if is_saved else "primary"

    # ボタン直下にこのマップの保存枚数を表示する（st.columns を使わない＝列ネスト制限を避ける）
    _snap_clicked = st.button(btn_label, key=f"snap_btn_{key}", type=btn_type, disabled=is_saved)

    # 同じマップから複数枚撮れるようにする「別カット追加」ボタン。
    # 1枚目は id=key で保存し、以後その完全一致ボタンは無効化される。表示オプション
    # （上位N・配色・軸・対象など）を変えてもう1枚撮りたい場合は、このボタンで現在の表示を
    # id=key__sN（連番）として追加保存する（既存スナップショットを上書きしない）。
    _add_clicked = False
    if is_saved:
        _add_clicked = st.button(
            "現在の表示を別カットとして追加", key=f"snap_add_{key}", type="secondary",
            help="表示オプション（上位N・配色・軸など）を変えた別バージョンを、もう1枚スナップショットとして追加します。")
    if _map_count > 0:
        st.caption(f"このマップで {_map_count} 枚 保存済み")

    if _snap_clicked or _add_clicked:
        if _add_clicked:
            # 次の空き連番 key__sN を探して別カットとして追加（上書き防止）
            _n = 2
            while any(s.get('id') == f"{key}__s{_n}" for s in st.session_state['snapshots']):
                _n += 1
            save_key = f"{key}__s{_n}"
            save_title = f"{title}（別カット {_n}）"
        else:
            save_key = key
            save_title = title
        # モジュール名の決定 (data_summary['module']があれば優先)
        module_name = st.session_state.get('current_page', 'Unknown')
        if data_summary and isinstance(data_summary, dict) and 'module' in data_summary:
            module_name = data_summary['module']

        snapshot_data = {
            'id': save_key,
            'title': save_title,
            'description': description,
            'data_summary': data_summary,
            'module': module_name,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'images': []
        }
        
        # 対象フィギュアの統合
        target_figs = []
        if figs:
            target_figs = figs
        elif fig:
            target_figs = [fig]
            
        import io
        
        # --- ヘルパー: 1つのfigをバイト列に変換 ---
        def convert_fig_to_bytes(target_fig):
            if target_fig is None: return None, None
            img_bytes = None
            error_msg = None
            
            try:
                # Plotly
                if hasattr(target_fig, 'to_image'):
                    try:
                        # --- スマート解像度 & アスペクト比 ---
                        base_width = 1600
                        use_width = base_width
                        use_height = 1000 # デフォルトフォールバック
                        
                        is_saturn_v = module_name == 'Saturn V'
                        
                        try:
                            if is_saturn_v:
                                # SATURN V: アスペクト比を計算
                                xaxis = target_fig.layout.xaxis
                                yaxis = target_fig.layout.yaxis
                                if xaxis.range and yaxis.range:
                                    x_range = xaxis.range[1] - xaxis.range[0]
                                    y_range = yaxis.range[1] - yaxis.range[0]
                                    if x_range > 0 and y_range > 0:
                                        ratio = x_range / y_range
                                        calc_height = base_width / ratio
                                        calc_height = max(600, min(calc_height, 2400))
                                        use_height = int(calc_height)
                                    else:
                                        use_height = int(base_width * 0.618)
                                else:
                                    use_height = 1000
                            else:
                                # 標準ワイドフォーマット (16:9)
                                use_height = int(base_width * 9 / 16)
                        except:
                            use_height = 1000

                        # --- レポート/スライド用に「クリーン版」へ整えてから画像化 ---
                        # 白背景・大きな文字・大きいクラスタラベルにする。タイトル/要点/出典は
                        # CAPCOM のスライド/レポート側で付与する（二重表示を避けるため焼き込まない）。
                        _themed = apply_report_theme(
                            target_fig, width=use_width, height=use_height)
                        img_bytes = _themed.to_image(format="png", width=use_width, height=use_height, scale=3.0)
                    except Exception as e:
                        error_msg = f"Plotly Image Error (Kaleido): {str(e)}"
                
                # Matplotlib
                elif hasattr(target_fig, 'savefig'):
                    try:
                        buf = io.BytesIO()
                        target_fig.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        img_bytes = buf.getvalue()
                    except Exception as e:
                        error_msg = f"Matplotlib Image Error: {str(e)}"
                        
            except Exception as e:
                error_msg = f"General Image Error: {str(e)}"
                
            return img_bytes, error_msg

        # Process all figures
        for f in target_figs:
            ib, err = convert_fig_to_bytes(f)
            if ib:
                snapshot_data['images'].append(ib)
            if err:
                snapshot_data.setdefault('image_errors', []).append(err)
                st.warning(f"Image conversion warning: {err}")

        # レガシー互換性: メインの'image'を最初の有効な画像に設定
        if snapshot_data['images']:
            snapshot_data['image'] = snapshot_data['images'][0]
        else:
            snapshot_data['image'] = None

        st.session_state['snapshots'].append(snapshot_data)

        # --- CAPCOM出力フック ---
        try:
            import capcom
            if capcom.is_active():
                # PNG画像を保存
                for idx, img_bytes in enumerate(snapshot_data.get('images', [])):
                    if img_bytes:
                        suffix = idx if len(snapshot_data.get('images', [])) > 1 else None
                        capcom.save_snapshot_image(save_key, img_bytes, index=suffix)

                # 1スナップショット操作 = +1（画像数ではなく操作回数をカウント）
                capcom.increment_snap_count()

                # metadata.jsonを更新（全スナップショット）
                capcom.save_metadata(st.session_state['snapshots'])
        except Exception as e:
            pass

        st.rerun()

    #     状態はボタンラベル「Snapshot Saved」＋横の枚数表示で示す。
    #     別ビュー（別クラスタ等）を撮りたい場合は、呼び出し側がビュー別の key を渡すことで
    #     「Save Snapshot」が再度有効になる（ドリルダウンはクラスタ別 key）。

    # --- スライド/レポート用 高解像度PNG 書き出し（全マップ共通・任意） ---
    # Plotly図がある場合のみ。要点欄は空のままにすれば画像に焼き込まず、
    # CAPCOM(AI)が画像を見て要点を執筆する運用にできる。
    # 複数の図を渡された場合（例: NEBULA の急上昇キーワード棒グラフ＋共起ネットワーク）は
    #   各図に書き出しボタンを出す。
    _all_figs = []
    for _f in (([fig] if fig is not None else []) + (list(figs) if figs else [])):
        if _f is not None and not any(_f is _x for _x in _all_figs):
            _all_figs.append(_f)
    _plotly_figs = [f for f in _all_figs if hasattr(f, 'to_image')]
    _mod = ''
    if data_summary and isinstance(data_summary, dict):
        _mod = data_summary.get('module', '') or ''
    _multi = len(_plotly_figs) > 1
    for _i, _pf in enumerate(_plotly_figs):
        _fig_title = ''
        try:
            _fig_title = (_pf.layout.title.text or '') if _pf.layout.title else ''
        except Exception:
            _fig_title = ''
        if _multi and _fig_title:
            _lbl = f"「{_fig_title}」をスライド/レポート用PNGに書き出す"
        elif _multi:
            _lbl = f"図{_i + 1}/{len(_plotly_figs)} をスライド/レポート用PNGに書き出す"
        else:
            _lbl = "このグラフをスライド/レポート用PNGに書き出す"
        try:
            render_report_png_button(
                _pf, key=f"{key}_rep{_i}",
                default_title=(title or ''),
                default_subtitle=_mod,
                default_caption='',
                label=_lbl,
            )
        except Exception:
            pass

# ==================================================================
# --- 5. AI アシスタント (共通) ---
# ==================================================================
def parse_label_response(text: str) -> dict:
    """LLM 応答をパースして {cluster_id: label} 辞書を返す。

    対応形式（自動判別）:
      - TSV:          `0\\tラベル`（タブ / カンマ / コロン / 全角コロン区切り）
      - JSON:         `{"0": "ラベル", "1": "ラベル"}`
      - Markdown 表:  `| 0 | ラベル | ...`
      - 平文:         `0: ラベル` `0. ラベル` `0 - ラベル`

    部分応答にも対応（対象クラスタ ID が全部揃っていなくてよい）。
    ラベル末尾の改行・クォートは自動で除去。

    Returns:
        dict[int, str]: {cluster_id: label} の辞書。パース失敗時は空 dict。
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    s = text.strip()

    # Markdown コードブロック除去（```json ... ``` / ``` ... ```）
    s = re.sub(r'^```(?:json|tsv|csv|txt|markdown)?\s*\n?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\n?\s*```\s*$', '', s)
    s = s.strip()

    # 1. JSON 形式を優先的に試行
    if s.startswith('{') or s.startswith('['):
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                out = {}
                for k, v in data.items():
                    try:
                        cid = int(str(k).strip())
                        label = str(v).strip().strip('"\'「」『』')
                        if label:
                            out[cid] = label
                    except (ValueError, TypeError):
                        continue
                if out:
                    return out
        except json.JSONDecodeError:
            pass

    # 2. Markdown 表形式（パイプ `|` で始まる行が複数ある場合）
    pipe_lines = [ln for ln in s.split('\n') if ln.strip().startswith('|')]
    if len(pipe_lines) >= 2:
        out = {}
        for line in pipe_lines:
            line = line.strip()
            # 区切り行スキップ: |---|---|
            if re.match(r'^\|[\s\-:|]+\|$', line):
                continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            if len(cells) >= 2:
                try:
                    cid = int(cells[0])
                    label = cells[1].strip().strip('"\'「」『』')
                    if label and label.lower() not in ('id', 'cluster id', 'ラベル', 'label', '提案ラベル'):
                        out[cid] = label
                except ValueError:
                    continue
        if out:
            return out

    # 3. TSV / 平文形式: 各行を「ID [区切り文字] ラベル」として解析
    out = {}
    for line in s.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 区切り文字: タブ / カンマ / コロン / 全角コロン / ハイフン / ピリオド
        m = re.match(r'^(\d+)\s*[\t\.,:：\-—]\s*(.+)$', line)
        if not m:
            continue
        try:
            cid = int(m.group(1))
            label = m.group(2).strip().strip('"\'「」『』')
            # CSV の場合は末尾クォートも除去
            label = label.rstrip(',').strip()
            if label:
                out[cid] = label
        except ValueError:
            continue

    return out


def generate_ai_cluster_prompt(df_source, cluster_col, target_cols, tfidf_matrix, feature_names, n_samples=5, domain="patent"):
    """クラスタごとの代表文献を抽出し、命名用プロンプトを生成する（c-TF-IDF方式）

    domain: "patent"（特許・既定）/ "academic"（学術論文）。役割・呼称・反映観点を切り替える
            （NEBULA 学術ランドスケープでは "academic" を渡す。特許用語で誤らせないため）。
    """
    from sklearn.metrics.pairwise import euclidean_distances
    if df_source.empty: return "データがありません。"

    unique_clusters = sorted([c for c in df_source[cluster_col].unique() if c != -1])
    if not unique_clusters: return "有効なクラスタがありません。"

    # embeddingカラムの特定
    if 'umap_x' in df_source.columns and 'umap_y' in df_source.columns:
        embedding_cols = ['umap_x', 'umap_y']
    elif 'drill_x' in df_source.columns and 'drill_y' in df_source.columns:
        embedding_cols = ['drill_x', 'drill_y']
    elif 'acad_umap_x' in df_source.columns and 'acad_umap_y' in df_source.columns:
        embedding_cols = ['acad_umap_x', 'acad_umap_y']
    elif 'x' in df_source.columns and 'y' in df_source.columns:
        embedding_cols = ['x', 'y']
    else:
        return "埋め込み座標が見つかりません。"

    # c-TF-IDFでクラスタ判別キーワードを一括抽出
    ctfidf_keywords = {}
    try:
        # テキスト列を結合
        text_col_candidates = [c for c in target_cols if c and c in df_source.columns]
        if text_col_candidates:
            texts = df_source[text_col_candidates[0]].fillna('')
            for c in text_col_candidates[1:]:
                texts = texts + ' ' + df_source[c].fillna('')
            # safe_auto_label を top_n=10 で呼んでキーワード部分を抽出 (Janome 例外耐性付き)
            label_map_10 = safe_auto_label(
                texts, df_source[cluster_col].values,
                method='c-tfidf', top_n=10,
                label_format="{terms}",  # IDプレフィックスなし
            )
            ctfidf_keywords = label_map_10
    except Exception:
        pass

    sampled_docs = []

    for cid in unique_clusters:
        c_df = df_source[df_source[cluster_col] == cid]
        if c_df.empty: continue

        # c-TF-IDFキーワード
        keywords_str = ctfidf_keywords.get(cid, "")

        # 重心計算 → 重心に近い代表文献を抽出
        coords = c_df[embedding_cols].values
        centroid = coords.mean(axis=0)
        dists = euclidean_distances(coords, centroid.reshape(1, -1)).flatten()
        top_indices = np.argsort(dists)[:n_samples]

        docs = []
        for idx in top_indices:
            row = c_df.iloc[idx]
            text_parts = []
            for col in target_cols:
                if col and col in row and pd.notna(row[col]):
                    val = str(row[col]).replace('\n', ' ')
                    text_parts.append(val)
            docs.append(f"  - {' '.join(text_parts)}")

        sampled_docs.append(f"Cluster {cid}:\n[特徴語(c-TF-IDF)] {keywords_str}\n[代表文献]\n" + "\n".join(docs))

    sampled_docs_str = "\n\n".join(sampled_docs)

    # ドメイン別の役割・呼称・反映観点（特許＝既定 / 学術＝NEBULA 学術ランドスケープ）
    if domain == "academic":
        _role = "あなたは熟練した学術文献アナリスト（研究動向の分析専門）です。"
        _list_name = "クラスタごとの特徴語と代表的な論文リスト"
        _reflect = "研究テーマ・対象技術・手法を反映させてください。"
    else:
        _role = "あなたは熟練した特許情報アナリストです。"
        _list_name = "クラスタごとの特徴語と代表的特許リスト"
        _reflect = "技術的特徴や解決課題を反映させてください。"

    prompt = f"""
{_role}
以下の「{_list_name}」を分析し、各クラスタの内容を端的に表す**「短い説明ラベル（日本語）」**を提案してください。

# 制約事項
- ラベルは**20文字以内**の日本語で記述してください。
- 専門用語を適切に使用し、{_reflect}
- 出力は下記の **TSV 形式（タブ区切り、1 行 1 クラスタ）** を推奨します。
  解説・前置き・コードブロックは不要です。
- 全クラスタを一度に回答する必要はありません。**一部のクラスタだけの応答も受け付けます**
  （例: 「10-30 だけ再提案して」のような部分応答も OK）。

# 出力フォーマット（TSV: クラスタID<TAB>ラベル）
0	全固体電池の固体電解質
1	画像認識による異常検知
2	カーボンニュートラル燃料製造

# 補足: 以下の形式でも取り込み可能です（TSV が困難な場合のみ使用）
- JSON: `{{"0": "ラベル", "1": "ラベル"}}`
- Markdown 表: `| 0 | ラベル |`
- 平文: `0: ラベル` / `0. ラベル` / `0 - ラベル`

# クラスタデータ
{sampled_docs_str}
"""
    return prompt

def render_ai_label_assistant(df_source, cluster_col, label_map_key, col_map, tfidf_matrix, feature_names, widget_key_prefix=None, domain="patent"):
    """AIラベルサジェストUI (共通部品)

    domain: "patent"（既定）/ "academic"。命名プロンプトの役割・呼称を切り替える
            （NEBULA 学術ランドスケープでは "academic" を渡し、特許用語を出させない）。

    LLM 応答は TSV / JSON / Markdown 表 / 平文 いずれも自動判別。
    部分応答（一部クラスタだけ）も受け付け、既存マップに **追記マージ** します。
    取り込み結果は session_state に保存され、下段の data_editor の「AI 提案」列に
    そのまま表示されます。
    """
    with st.expander("ラベルをAIにきれいにしてもらう（チャットAI用キット・任意）"):
        st.caption(
            "自動ラベルは特徴語の羅列なので、普段お使いのチャットAI（ChatGPT / Claude / Gemini 等）に"
            "読みやすい名前を付けてもらえます。手順は3つだけ。APIキーは不要です。"
        )

        st.markdown("**1 命名依頼プロンプトを作ってコピー**")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            n_samples_ai = st.number_input(
                "1クラスタあたりの例示件数", min_value=1, value=5,
                key=f"ai_n_samples_{label_map_key}",
                help="プロンプトに含める代表特許の件数です。通常は変更不要。")

        if st.button("プロンプトを作る", key=f"ai_gen_btn_{label_map_key}"):
            target_cols = [col_map.get('title'), col_map.get('abstract')]
            prompt = generate_ai_cluster_prompt(df_source, cluster_col, target_cols, tfidf_matrix, feature_names, n_samples=n_samples_ai, domain=domain)
            st.session_state[f"ai_prompt_{label_map_key}"] = prompt

        if f"ai_prompt_{label_map_key}" in st.session_state:
            st.code(st.session_state[f"ai_prompt_{label_map_key}"], language="markdown")
            st.caption("枠の右上のコピーボタンで丸ごとコピーできます。")

        st.markdown("**2 チャットAIに貼り付けて送信**")
        st.caption("AIが各クラスタの名前を提案してくれます（形式はタブ区切り・JSON・表・平文のどれでもOK）。")

        st.markdown("**3 AIの回答をここに貼り戻して適用**")
        st.caption(
            "一部のクラスタだけ貼っても大丈夫（既存のラベルに追記されます）。"
            "同じクラスタ番号は新しい名前で上書きされます。"
        )
        tsv_input = st.text_area(
            "AIの回答を貼り付け:",
            height=150,
            key=f"ai_json_input_{label_map_key}",
            placeholder="例:\n0\t全固体電池の固体電解質\n1\t画像認識による異常検知\n...",
        )

        if st.button("貼り付けた名前を適用する", key=f"ai_apply_btn_{label_map_key}", type="primary"):
            try:
                parsed = parse_label_response(tsv_input)
                if not parsed:
                    st.warning("応答を解析できませんでした。TSV / JSON / Markdown 表 / 平文 のいずれかで貼り付けてください。")
                    return

                # 追記マージ: 既存 current_map を保持しつつ、新エントリを上書き
                current_map = st.session_state.get(label_map_key, {}) or {}
                unique_cids = set(df_source[cluster_col].unique().tolist())
                # AI 提案そのものを session_state に保存（data_editor の AI 提案列で参照）
                ai_suggest_key = f"ai_suggestions_{label_map_key}"
                existing_suggest = st.session_state.get(ai_suggest_key, {}) or {}

                count_updated = 0
                count_added = 0
                for cid, raw_label in parsed.items():
                    # 対象クラスタに存在する ID のみ採用
                    if cid not in current_map and cid not in unique_cids:
                        continue
                    new_val = f"[{cid}] {raw_label}"
                    if cid in current_map:
                        count_updated += 1
                    else:
                        count_added += 1
                    current_map[cid] = new_val
                    existing_suggest[cid] = new_val

                    # 旧 text_input 形式の互換: widget の session_state も強制更新
                    if widget_key_prefix:
                        w_key = f"{widget_key_prefix}_{cid}"
                        if w_key in st.session_state:
                            st.session_state[w_key] = new_val

                st.session_state[label_map_key] = current_map
                st.session_state[ai_suggest_key] = existing_suggest

                # data_editor 形式のテーブルも同期（AI 提案列を反映するため再生成）
                table_key = f"{widget_key_prefix}_editor_df" if widget_key_prefix else None
                if table_key and table_key in st.session_state:
                    # 既存 DataFrame に AI 提案を反映
                    _edf = st.session_state[table_key]
                    if 'AI 提案' in _edf.columns:
                        _edf['AI 提案'] = _edf['クラスタID'].map(
                            lambda cid: existing_suggest.get(int(cid), _edf.loc[_edf['クラスタID'] == cid, 'AI 提案'].iloc[0] if len(_edf.loc[_edf['クラスタID'] == cid]) > 0 else '')
                        )
                        st.session_state[table_key] = _edf

                # 派生ラベル列の更新（各モジュール別）
                if label_map_key == "saturnv_labels_map" and 'df_main' in st.session_state:
                    st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(current_map)
                elif label_map_key == "drill_labels_map" and 'df_drilldown_result' in st.session_state:
                    st.session_state.df_drilldown_result['drill_cluster_label'] = st.session_state.df_drilldown_result['drill_cluster'].map(current_map)
                elif label_map_key == "mega_drill_labels_map" and 'df_drilldown' in st.session_state:
                    st.session_state.df_drilldown['label'] = st.session_state.df_drilldown['cluster_id'].map(current_map)
                    st.session_state.sbert_sub_cluster_map_auto = current_map

                st.success(
                    f"取り込み完了: 新規 **{count_added}** 件 / 上書き **{count_updated}** 件 "
                    f"(合計 {count_added + count_updated} 件、未対応 {len(parsed) - count_added - count_updated} 件)"
                )
                st.rerun()

            except Exception as e:
                st.error(f"取り込みエラー: {e}")

def create_label_editor_ui(original_map, current_map, key_prefix, max_individual_widgets=30):
    """手動ラベル編集UI機能 (共通)

    クラスタ数が max_individual_widgets を超える場合は自動的に st.data_editor
    (テーブル形式編集) に切り替わり、Streamlit の WebSocket メッセージ制限や
    "Bad message format / Tried to use Session..." 系エラーを回避する。

    Args:
        original_map: {cluster_id: original_label} 初回ラベル
        current_map: {cluster_id: current_label} 現在のラベル（編集済み含む）
        key_prefix: session_state のキー接頭辞
        max_individual_widgets: これを超えたら data_editor 形式に切替（デフォルト 30）

    Returns:
        dict: {cluster_id: new_label} 編集後のラベル辞書
    """
    st.caption(
        "ラベルは下の欄で**そのまま書き換えられます**（書き換えると図に反映されます）。"
        "AIにまとめて名付けてほしいときは、上の「ラベルをAIにきれいにしてもらう」をどうぞ。"
    )
    widgets_dict = {}
    sorted_ids = sorted([cid for cid in original_map.keys() if cid != -1])
    # "(該当なし)" を除いた有効クラスタ数
    valid_ids = [cid for cid in sorted_ids if original_map.get(cid, "") != "(該当なし)"]

    # --- 大規模クラスタ対応: テーブル形式 (st.data_editor) ---
    if len(valid_ids) > max_individual_widgets:
        import pandas as pd

        # AI 提案を session_state から取得（render_ai_label_assistant が書き込む）
        ai_suggest_key = f"ai_suggestions_{key_prefix.replace('_label', '_labels_map') if key_prefix.endswith('_label') else key_prefix}"
        # キー規約が複数あるので候補を試す
        ai_suggestions = {}
        for candidate in [
            f"ai_suggestions_{key_prefix.replace('_label', '_labels_map')}",
            f"ai_suggestions_{key_prefix}_labels_map",
            f"ai_suggestions_{key_prefix}",
        ]:
            if candidate in st.session_state:
                ai_suggestions = st.session_state[candidate] or {}
                break

        st.caption(
            f"クラスタ数が {len(valid_ids)} 個と多いため、テーブル形式で編集します"
            f"（text_input を大量生成すると Streamlit のメッセージ制限を超えるため）。"
            f"『編集後ラベル』列のセルをクリックして編集してください。"
        )
        st.caption(
            "**操作のコツ**: セルをダブルクリック or クリック後 **Enter** で編集 → **Tab** で次セルへ。"
            " Excel からの **コピー&ペースト** も可能（複数セル同時貼付対応）。"
        )

        table_key = f"{key_prefix}_editor_df"

        # 初回のみ DataFrame を session_state に格納
        if table_key not in st.session_state:
            rows = [
                {
                    'クラスタID': cid,
                    '元ラベル': original_map.get(cid, ""),
                    'AI 提案': ai_suggestions.get(cid, ""),
                    '編集後ラベル': current_map.get(cid, original_map.get(cid, "")),
                }
                for cid in valid_ids
            ]
            st.session_state[table_key] = pd.DataFrame(rows)
        else:
            # 2 回目以降: AI 提案列を session_state から再同期（取り込み後の反映）
            _df_existing = st.session_state[table_key]
            if 'AI 提案' not in _df_existing.columns:
                _df_existing['AI 提案'] = ""
            if ai_suggestions:
                _df_existing['AI 提案'] = _df_existing['クラスタID'].map(
                    lambda cid: ai_suggestions.get(int(cid), "")
                )
                st.session_state[table_key] = _df_existing

        # --- 編集UI をフラグメント化 ---
        # data_editor のセル編集は @st.fragment 内で処理することで「フラグメントのみ再実行」になり、
        # 同一ページ上の重いランドスケープ/クラスタ動態マップを再送しない。これにより 91 クラスタ等で
        # 起きていた "Bad message format / Tried to use SessionInfo before it was initialized" を防ぐ。
        # 一括操作ボタン（明示操作）は st.rerun() で全再描画する。
        st.caption("ラベル編集中はマップを再描画しません（軽量化）。編集後、スライダー操作やページ更新でマップに反映されます。")

        @st.fragment
        def _label_table_fragment():
            # 一括操作ボタン
            btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])
            with btn_col1:
                if st.button("AI 提案 → 編集後ラベルへ一括コピー", key=f"{key_prefix}_bulk_copy_ai", use_container_width=True):
                    _df = st.session_state[table_key].copy()
                    mask = _df['AI 提案'].astype(str).str.strip().astype(bool)
                    _df.loc[mask, '編集後ラベル'] = _df.loc[mask, 'AI 提案']
                    st.session_state[table_key] = _df
                    st.toast(f"{mask.sum()} 件の AI 提案を編集後ラベルへコピーしました", icon=":material/download:")
                    st.rerun()
            with btn_col2:
                if st.button("↩ 編集後ラベルを元ラベルへリセット", key=f"{key_prefix}_bulk_reset", use_container_width=True):
                    _df = st.session_state[table_key].copy()
                    _df['編集後ラベル'] = _df['元ラベル']
                    st.session_state[table_key] = _df
                    st.toast("↩ 元ラベルに戻しました", icon="↩")
                    st.rerun()
            with btn_col3:
                st.caption("テーブル右上の虫眼鏡アイコンで絞り込み検索、列ヘッダークリックでソート可")

            edited = st.data_editor(
                st.session_state[table_key],
                disabled=['クラスタID', '元ラベル', 'AI 提案'],
                hide_index=True,
                use_container_width=True,
                num_rows='fixed',
                key=f"{key_prefix}_data_editor",
                column_config={
                    'クラスタID': st.column_config.NumberColumn('ID', width='small'),
                    '元ラベル': st.column_config.TextColumn('元ラベル', width='medium'),
                    'AI 提案': st.column_config.TextColumn('AI 提案', width='medium', help="AIラベルサジェストの結果。「編集後ラベル」列にドラッグしてコピー、または上の一括コピーボタンを使用"),
                    '編集後ラベル': st.column_config.TextColumn('編集後ラベル', width='large', help="この列のセルをクリックして編集してください"),
                },
            )
            # 編集内容を session_state に反映（フラグメント外はこれを読む）
            st.session_state[table_key] = edited

        _label_table_fragment()

        # 戻り値の辞書を構築（全再描画時のみ実行。最新の編集は session_state[table_key] にある）
        edited_df = st.session_state[table_key]
        for _, row in edited_df.iterrows():
            cid = int(row['クラスタID'])
            widgets_dict[cid] = str(row['編集後ラベル']) if row['編集後ラベル'] else original_map.get(cid, "")

    # --- 少数クラスタ: text_input 形式 ---
    else:
        for cluster_id in valid_ids:
            orig_label = original_map.get(cluster_id, "")
            curr_label = current_map.get(cluster_id, orig_label)
            col1, col2 = st.columns([2, 3])
            with col1: st.markdown(f":green[{orig_label}]")
            with col2:
                key = f"{key_prefix}_{cluster_id}"
                if key not in st.session_state:
                    st.session_state[key] = curr_label
                # value引数を指定せず、key経由でsession_stateの値を使用させる
                new_label = st.text_input(f"Edit {cluster_id}", label_visibility="collapsed", key=key)
                widgets_dict[cluster_id] = new_label

    # ノイズクラスタ (-1) は編集不可で別途表示
    if -1 in original_map:
        orig_noise = original_map[-1]
        curr_noise = current_map.get(-1, orig_noise)
        col1, col2 = st.columns([2, 3])
        with col1: st.markdown(f":green[{orig_noise}]")
        with col2:
            st.text_input(f"noise_label", value=curr_noise, disabled=True, key=f"{key_prefix}_noise")
            widgets_dict[-1] = curr_noise
    return widgets_dict

def render_cluster_dynamics_map(
    df, cluster_col, cluster_labels_map, year_col='year',
    cagr_window=5, unique_key='dynamics'
):
    """
    クラスタ動態マップを生成する。
    X=累積件数、Y=CAGR、バブル=シェア、4象限表示。

    Args:
        df: DataFrame with cluster and year columns
        cluster_col: cluster ID column name
        cluster_labels_map: dict mapping cluster_id -> label string
        year_col: year column name
        cagr_window: years for CAGR calculation (default 5)
        unique_key: unique key for Streamlit widgets

    Returns:
        tuple: (plotly Figure, dynamics_data dict for CAPCOM export)
    """
    # Filter out noise (cluster == -1)
    df_valid = df[df[cluster_col] != -1].copy() if -1 in df[cluster_col].values else df.copy()

    clusters = sorted(df_valid[cluster_col].unique())
    if len(clusters) == 0:
        return None, None

    # Calculate per-cluster metrics
    total_patents = len(df_valid)
    years = df_valid[year_col].dropna()
    if years.empty:
        return None, None
    max_year = int(years.max())
    min_cagr_year = max_year - cagr_window

    cluster_data = []
    for cid in clusters:
        df_c = df_valid[df_valid[cluster_col] == cid]
        cumulative = len(df_c)
        share = cumulative / total_patents if total_patents > 0 else 0
        label = cluster_labels_map.get(cid, f"Cluster {cid}")

        # CAGR calculation
        years_c = df_c[year_col].dropna()
        if years_c.empty:
            cagr = 0.0
        else:
            recent = len(df_c[df_c[year_col] > min_cagr_year])
            past = len(df_c[df_c[year_col] <= min_cagr_year])
            if past > 0 and recent > 0:
                n_years = cagr_window
                cagr = (recent / past) ** (1 / n_years) - 1
            elif past == 0 and recent > 0:
                cagr = 1.0  # New cluster (100% growth)
            else:
                cagr = -0.5  # Declining

        cluster_data.append({
            'cluster_id': cid,
            'label': label,
            'cumulative_count': cumulative,
            'cagr': round(cagr, 4),
            'share': round(share, 4),
        })

    if not cluster_data:
        return None, None

    # Calculate thresholds (median)
    x_vals = [d['cumulative_count'] for d in cluster_data]
    y_vals = [d['cagr'] for d in cluster_data]
    import numpy as np
    x_threshold = float(np.median(x_vals))
    y_threshold = float(np.median(y_vals))

    # Assign quadrants
    quadrant_names = {
        (True, True): '成長リーダー',
        (False, True): '新興クラスタ',
        (True, False): '成熟クラスタ',
        (False, False): 'ニッチ/衰退',
    }
    for d in cluster_data:
        x_high = d['cumulative_count'] >= x_threshold
        y_high = d['cagr'] >= y_threshold
        d['quadrant'] = quadrant_names[(x_high, y_high)]

    # Build Plotly figure
    quadrant_colors = {
        '成長リーダー': '#2E7D32',
        '新興クラスタ': '#1565C0',
        '成熟クラスタ': '#E65100',
        'ニッチ/衰退': '#78909C',
    }

    fig = go.Figure()

    for qname, qcolor in quadrant_colors.items():
        q_items = [d for d in cluster_data if d['quadrant'] == qname]
        if not q_items:
            continue
        fig.add_trace(go.Scatter(
            x=[d['cumulative_count'] for d in q_items],
            y=[d['cagr'] for d in q_items],
            mode='markers+text',
            marker=dict(
                size=[max(15, d['share'] * 200) for d in q_items],
                color=qcolor,
                opacity=0.7,
                line=dict(width=1, color='white'),
            ),
            text=[d['label'] for d in q_items],
            textposition='top center',
            textfont=dict(size=10),
            name=qname,
            hovertemplate=(
                '<b>%{text}</b><br>'
                '累積件数: %{x}<br>'
                'CAGR: %{y:.1%}<br>'
                '<extra>%{fullData.name}</extra>'
            ),
        ))

    # Add quadrant lines
    fig.add_hline(y=y_threshold, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=x_threshold, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    x_range = max(x_vals) - min(x_vals) if len(x_vals) > 1 else max(x_vals)
    x_min_plot = min(x_vals) - x_range * 0.1
    x_max_plot = max(x_vals) + x_range * 0.1
    y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else abs(max(y_vals))
    y_min_plot = min(y_vals) - y_range * 0.15
    y_max_plot = max(y_vals) + y_range * 0.15

    annotations = [
        dict(x=x_max_plot * 0.95, y=y_max_plot * 0.95, text="成長リーダー", showarrow=False,
             font=dict(size=12, color='rgba(46,125,50,0.4)'), xanchor='right', yanchor='top'),
        dict(x=x_min_plot * 1.05 if x_min_plot > 0 else x_threshold * 0.1, y=y_max_plot * 0.95,
             text="新興クラスタ", showarrow=False,
             font=dict(size=12, color='rgba(21,101,192,0.4)'), xanchor='left', yanchor='top'),
        dict(x=x_max_plot * 0.95, y=y_min_plot * 1.05 if y_min_plot < 0 else y_threshold * 0.1,
             text="成熟クラスタ", showarrow=False,
             font=dict(size=12, color='rgba(230,81,0,0.4)'), xanchor='right', yanchor='bottom'),
        dict(x=x_min_plot * 1.05 if x_min_plot > 0 else x_threshold * 0.1,
             y=y_min_plot * 1.05 if y_min_plot < 0 else y_threshold * 0.1,
             text="ニッチ/衰退", showarrow=False,
             font=dict(size=12, color='rgba(120,144,156,0.4)'), xanchor='left', yanchor='bottom'),
    ]

    fig.update_layout(
        title=dict(text='クラスタ動態マップ', font=dict(size=16)),
        xaxis=dict(title='累積件数', gridcolor='#f0f0f0'),
        yaxis=dict(title=f'CAGR (直近{cagr_window}年)', tickformat='.0%', gridcolor='#f0f0f0'),
        plot_bgcolor='white',
        annotations=annotations,
        height=800,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    fig.update_layout(
        template=APOLLO_TEMPLATE,
        paper_bgcolor=APOLLO_BG,
    )

    # Build CAPCOM export data
    quadrant_summary = {}
    for qname in quadrant_colors:
        q_items = [d for d in cluster_data if d['quadrant'] == qname]
        quadrant_summary[qname] = {
            'count': len(q_items),
            'total_patents': sum(d['cumulative_count'] for d in q_items),
        }

    dynamics_data = {
        'x_axis': 'cumulative_count',
        'y_axis': 'cagr',
        'cagr_window_years': cagr_window,
        'x_threshold': x_threshold,
        'y_threshold': y_threshold,
        'clusters': cluster_data,
        'quadrant_summary': quadrant_summary,
    }

    # --- 象限別拡大図を生成 ---
    quadrant_figs = {}
    for qname, qcolor in quadrant_colors.items():
        q_items = [d for d in cluster_data if d['quadrant'] == qname]
        if not q_items:
            continue
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            x=[d['cumulative_count'] for d in q_items],
            y=[d['cagr'] for d in q_items],
            mode='markers+text',
            marker=dict(
                size=[max(20, d['share'] * 300) for d in q_items],
                color=qcolor,
                opacity=0.8,
                line=dict(width=1, color='white'),
            ),
            text=[d['label'] for d in q_items],
            textposition='top center',
            textfont=dict(size=11),
            name=qname,
            hovertemplate=(
                '<b>%{text}</b><br>'
                '累積件数: %{x}<br>'
                'CAGR: %{y:.1%}<br>'
                f'<extra>{qname}</extra>'
            ),
        ))
        fig_q.update_layout(
            title=dict(text=f'{qname} ({len(q_items)}クラスタ)', font=dict(size=14)),
            xaxis=dict(title='累積件数', gridcolor='#f0f0f0'),
            yaxis=dict(title=f'CAGR (直近{cagr_window}年)', tickformat='.0%', gridcolor='#f0f0f0'),
            plot_bgcolor='white',
            height=400,
            showlegend=False,
        )
        quadrant_figs[qname] = fig_q

    return fig, dynamics_data, quadrant_figs


def _build_cluster_dynamics_prompt(dynamics_data, module_name, cagr_window):
    """クラスタ動態マップ（累積件数×CAGRの4象限）専用のAIインサイト用プロンプトを構築する。

    Args:
        dynamics_data: render_cluster_dynamics_map の返り値 dict
                       （'clusters' と 'quadrant_summary' を持つ）
        module_name: 表示用モジュール名（Saturn V / EAGLE / NEBULA Academic 等）
        cagr_window: CAGR計算期間（年）

    Returns:
        str: 生成したプロンプト。データ不足時は空文字。
    """
    if not dynamics_data or not isinstance(dynamics_data, dict):
        return ""
    clusters = dynamics_data.get('clusters', [])
    if not clusters:
        return ""

    import utils_ai

    # --- クラスタ別 動態テーブル（累積件数の降順・正式日本語呼称のみ） ---
    rows = sorted(clusters, key=lambda d: d.get('cumulative_count', 0) or 0, reverse=True)
    df_dyn = pd.DataFrame([{
        'クラスタ': d.get('label', f"Cluster {d.get('cluster_id')}"),
        '累積件数': d.get('cumulative_count', 0),
        'CAGR': f"{(d.get('cagr', 0) or 0) * 100:.0f}%",
        'シェア': f"{(d.get('share', 0) or 0) * 100:.1f}%",
        '象限': d.get('quadrant', ''),
    } for d in rows])
    dyn_table = utils_ai.df_to_markdown(df_dyn, index=False) if not df_dyn.empty else "（クラスタなし）"

    # --- 象限サマリー ---
    qs = dynamics_data.get('quadrant_summary', {})
    qs_line = " / ".join(
        f"{q}: {v.get('count', 0)}クラスタ({v.get('total_patents', 0)}件)"
        for q, v in qs.items()) if qs else "（集計なし）"

    total = sum((d.get('cumulative_count', 0) or 0) for d in clusters)

    metadata = {
        '分析対象': module_name,
        'クラスタ数': len(clusters),
        'CAGR計算期間': f"直近{cagr_window}年",
        '母集団件数': total,
        '累積件数しきい値（中央値）': round(dynamics_data.get('x_threshold', 0) or 0, 1),
        'CAGRしきい値（中央値）': f"{(dynamics_data.get('y_threshold', 0) or 0) * 100:.0f}%",
    }

    role = (
        "あなたは技術ライフサイクルとポートフォリオ戦略の専門家です。"
        "クラスタ動態マップ（累積件数×成長率の4象限）から各技術領域のライフサイクル局面を読み解き、"
        "注力・育成・刈り取り・撤退の戦略示唆を導いてください。"
    )
    context = (
        "クラスタ動態マップは、各技術クラスタを「累積件数（X軸＝技術の蓄積規模・出願の累積量）」と"
        f"「CAGR（Y軸＝直近{cagr_window}年の年平均成長率）」で配置し、両軸の中央値で4象限に分類した散布図です。"
        "バブルの大きさは母集団内シェアを表します。4象限の意味は次の通り:\n"
        "- **成長リーダー**（規模大×高成長）: 主戦場かつ伸びているコア技術領域\n"
        "- **新興クラスタ**（規模小×高成長）: 件数は少ないが急成長中の次世代・先行投資候補\n"
        "- **成熟クラスタ**（規模大×低成長）: 蓄積は厚いが伸びが鈍化した成熟・定番領域\n"
        "- **ニッチ/衰退**（規模小×低成長）: 規模も成長も小さい周辺・縮小領域"
    )
    data_summary = (
        f"分析対象: {module_name} / 全{len(clusters)}クラスタ / 母集団{total}件\n"
        f"[象限サマリー] {qs_line}\n\n"
        f"[クラスタ別 動態テーブル]\n{dyn_table}"
    )
    instructions = """\
以下の観点で分析してください:
1. **ポートフォリオ全体像**: 4象限のどこにクラスタ・件数が偏っているかから、技術ポートフォリオ全体の成熟度・成長性を診断する
2. **成長リーダー**: 該当クラスタを名指しし、主戦場としての位置づけと競争激化の可能性を述べる
3. **新興クラスタ**: 該当クラスタを名指しし、次世代の有望株として先行投資・参入の妥当性を論じる
4. **成熟・衰退クラスタ**: 該当クラスタを名指しし、刈り取り・効率化・撤退・再活性化のいずれが適切かを判断する
5. **急成長領域**: CAGR上位のクラスタを特定し、その背景にある技術・市場要因を考察する
6. **戦略示唆**: 注力すべき領域と縮小・撤退を検討すべき領域を、具体的なクラスタ名と数値で提言する

各主張には必ずクラスタ名と具体的な数値（累積件数・CAGR・シェア・象限のいずれか）を1つ以上含めてください。"""
    constraints = (
        f"CAGRは直近{cagr_window}年の件数比から年率換算した簡易指標であり、"
        "新規クラスタ（過去ゼロ→直近あり）は便宜的に+100%、衰退クラスタは-50%等の固定値になる場合がある。"
        "また件数の小さいクラスタはCAGRの振れ幅が大きいため、シェア・累積件数と併せて慎重に解釈すること。"
    )
    return utils_ai.generate_ai_insight_prompt(
        role=role,
        context=context,
        data_summary=data_summary,
        instructions=instructions,
        constraints=constraints,
        output_format="Markdown形式。見出し付きの構造化された分析レポート。",
        metadata=metadata,
    )


def render_cluster_dynamics_section(
    df, cluster_col, cluster_labels_map, year_col='year',
    cagr_window=5, unique_key='dynamics',
    module_name='SaturnV'
):
    """
    クラスタ動態マップのUI全体を描画する。
    CAGRウィンドウ設定UI + 全体図 + 象限別拡大図 + スナップショット対応。
    """
    st.markdown("### クラスタ動態マップ")

    # --- CAGR ウィンドウ設定 ---
    years = df[year_col].dropna()
    if years.empty:
        st.info("年データがないためクラスタ動態マップを表示できません。")
        return None
    data_span = int(years.max()) - int(years.min())
    max_window = max(2, min(data_span, 15))
    if max_window <= 2:
        # 年幅が狭い場合 slider は min_value==max_value となり例外を投げるため、固定値で続行
        cagr_window_actual = max_window
        st.caption(f"データ期間が短い（{data_span}年）ため、CAGR 計算期間は {max_window} 年で固定します。")
    else:
        default_window = min(cagr_window, max_window)
        cagr_window_actual = st.slider(
            "CAGR計算期間（直近N年）", min_value=2, max_value=max_window,
            value=default_window, key=f"cagr_window_{unique_key}",
            help=f"データ期間: {int(years.min())}〜{int(years.max())}年（{data_span}年間）"
        )

    result = render_cluster_dynamics_map(
        df, cluster_col, cluster_labels_map,
        year_col=year_col, cagr_window=cagr_window_actual,
        unique_key=unique_key,
    )
    if result is None or result[0] is None:
        st.info("クラスタ動態マップを生成できませんでした。")
        return None

    fig, dynamics_data, quadrant_figs = result

    # --- 全体図 ---
    st.plotly_chart(fig, use_container_width=True)

    # スナップショット（全体図）
    summary_lines = []
    for qname, qinfo in dynamics_data['quadrant_summary'].items():
        summary_lines.append(f"- {qname}: {qinfo['count']}クラスタ ({qinfo['total_patents']}件)")
    summary_text = "\n".join(summary_lines)

    # --- AIインサイト用プロンプトを構築（スナップショットにも文脈として同梱） ---
    dynamics_prompt = _build_cluster_dynamics_prompt(
        dynamics_data, module_name, cagr_window_actual)

    # スナップショットには AIインサイトの文脈を同梱（CAPCOM/VOYAGER 用）。
    # 返り値 dynamics_data（CAPCOM JSON出力）は汚さないようコピーに付与する。
    snap_summary = dict(dynamics_data)
    if dynamics_prompt:
        snap_summary['ai_insight_context'] = dynamics_prompt

    render_snapshot_button(
        title=f"{module_name} クラスタ動態マップ",
        description=f"X=累積件数, Y=CAGR(直近{cagr_window_actual}年)\n{summary_text}",
        key=f"snap_{unique_key}_overview",
        fig=fig,
        data_summary=snap_summary,
    )

    # --- AIインサイト（クラスタ動態マップ専用） ---
    if dynamics_prompt:
        import utils_ai
        utils_ai.render_ai_insight_button(dynamics_prompt, f"{unique_key}_insight")

    # --- 象限別拡大図 ---
    if quadrant_figs:
        with st.expander("象限別 拡大図", expanded=False):
            # 2列レイアウト
            cols = st.columns(2)
            quadrant_order = ['成長リーダー', '新興クラスタ', '成熟クラスタ', 'ニッチ/衰退']
            for i, qname in enumerate(quadrant_order):
                if qname in quadrant_figs:
                    with cols[i % 2]:
                        st.plotly_chart(quadrant_figs[qname], use_container_width=True)

                        # 象限内クラスタの一覧テーブル
                        q_items = [d for d in dynamics_data['clusters'] if d['quadrant'] == qname]
                        if q_items:
                            import pandas as pd
                            df_q = pd.DataFrame(q_items)[['label', 'cumulative_count', 'cagr', 'share']]
                            df_q.columns = ['クラスタ', '累積件数', 'CAGR', 'シェア']
                            df_q['CAGR'] = df_q['CAGR'].apply(lambda x: f"{x:.1%}")
                            df_q['シェア'] = df_q['シェア'].apply(lambda x: f"{x:.1%}")
                            st.dataframe(df_q, use_container_width=True, hide_index=True)

    return dynamics_data

def update_fig_layout(fig, title, height=1000, width=800, show_axes=False, show_legend=True):
    """Plotlyのレイアウトを統一的に更新する (APOLLO Standard 配色固定)"""
    # Sanitize title to remove implicit/explicit HTML tags (e.g. <b>)
    if isinstance(title, str):
        title = re.sub(r'<[^>]+>', '', title)

    layout_params = dict(
        template=APOLLO_TEMPLATE,
        title=dict(text=title, font=dict(size=20, color=APOLLO_TEXT, family="Helvetica Neue", weight="normal")),
        paper_bgcolor=APOLLO_BG,
        plot_bgcolor=APOLLO_BG,
        # 基本フォントを大きめに（軸の数値・凡例・ホバーを読みやすく）
        font=dict(color=APOLLO_TEXT, family="Helvetica Neue", size=15),
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#eee", borderwidth=1,
            font=dict(size=14)
        )
    )

    if not show_legend:
        layout_params['showlegend'] = False

    if not show_axes:
        layout_params['xaxis'] = dict(visible=False, showgrid=False, zeroline=False, showticklabels=False)
        layout_params['yaxis'] = dict(
            visible=False, showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=1
        )
    else:
        if "width" in layout_params:
            del layout_params["width"]

        layout_params['xaxis'] = dict(
            visible=True, showgrid=False, zeroline=False, showline=False, showticklabels=True,
            tickfont=dict(size=14), title=dict(font=dict(size=16))
        )
        layout_params['yaxis'] = dict(
            visible=True, showgrid=True, gridcolor='#eee', zeroline=False, showline=False, showticklabels=True,
            tickfont=dict(size=14), title=dict(font=dict(size=16))
        )

    fig.update_layout(**layout_params)
    return fig


def apply_report_theme(fig, title=None, subtitle=None, caption=None, source=None,
                       width=1600, height=900):
    """Plotly fig を「レポート/スライド用」に整えたコピーを返す（元のfigは変更しない）。

    分析用マップ（小フォント・密・インタラクティブ前提）を、スライドや報告書に貼れる
    静的ビジュアルへ変換する: 白背景・大きな文字・タイトル/サブタイトル・要点キャプション・
    出典フッター・固定16:9寸法。軸の範囲やトレースはそのまま保持する。

    Args:
        fig: Plotly Figure（go.Figure / px の戻り値）
        title (str): 大見出し。None ならタイトル無し
        subtitle (str): 小見出し（母集団件数や手法など）
        caption (str): 図の下に置く「要点」テキスト（最重要メッセージ）
        source (str): 出典フッター（母集団・取得日・DB名等）
        width, height (int): 出力寸法（既定 1600x900 = 16:9）

    Returns:
        go.Figure: レポート用に整えたコピー（失敗時は元のfigをそのまま返す）
    """
    try:
        import plotly.graph_objects as go
        rep = go.Figure(fig)  # ディープコピー（元figに影響を与えない）

        ann = []
        # 上部のマージン（タイトル/サブタイトル用）・下部（キャプション/出典用）を確保
        top_margin = 70
        if title:
            top_margin = 120 if subtitle else 100
        bottom_margin = 48
        if caption:
            bottom_margin += 64
        if source:
            bottom_margin += 30

        if title:
            ann.append(dict(
                text=f"<b>{title}</b>", xref="paper", yref="paper",
                x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                showarrow=False, align="left",
                font=dict(size=34, color="#1a1a2e", family="Helvetica Neue"),
                yshift=44 if subtitle else 24,
            ))
        if subtitle:
            ann.append(dict(
                text=subtitle, xref="paper", yref="paper",
                x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                showarrow=False, align="left",
                font=dict(size=19, color="#6b6b78", family="Helvetica Neue"),
                yshift=16,
            ))
        if caption:
            ann.append(dict(
                text=f"<b>要点：</b>{caption}", xref="paper", yref="paper",
                x=0.0, y=0.0, xanchor="left", yanchor="top",
                showarrow=False, align="left",
                font=dict(size=20, color="#1a1a2e", family="Helvetica Neue"),
                yshift=-30,
            ))
        if source:
            ann.append(dict(
                text=source, xref="paper", yref="paper",
                x=0.0, y=0.0, xanchor="left", yanchor="top",
                showarrow=False, align="left",
                font=dict(size=14, color="#9a9aa5", family="Helvetica Neue"),
                yshift=-(74 if caption else 34),
            ))

        # 既存の注釈（クラスタラベル等）に新しい注釈を追加する
        existing = list(rep.layout.annotations) if rep.layout.annotations else []
        # 既存注釈（クラスタラベル等）の文字を読みやすく拡大する。ただしラベルが多い図では
        # 大きくすると重なって潰れるため、ラベル数に応じて控えめにする（適応的サイズ）。
        _n_lab = len(existing)
        if _n_lab <= 6:
            _target = 22
        elif _n_lab <= 12:
            _target = 16
        elif _n_lab <= 20:
            _target = 13
        else:
            _target = 12
        for _a in existing:
            try:
                _cur = _a.font.size if (_a.font is not None and _a.font.size) else 11
                _a.font.size = max(_cur, _target)
                if _n_lab <= 6 and _a.borderpad is not None:
                    _a.borderpad = max(_a.borderpad, 6)
                elif _n_lab > 12 and _a.borderpad is not None:
                    _a.borderpad = min(_a.borderpad, 2)  # 箱を詰めて重なりを軽減
            except Exception:
                pass
        rep.update_layout(
            annotations=existing + ann,
            title=None,  # 既存タイトルは消し、上記の大見出し注釈に置き換える
            paper_bgcolor="white",
            plot_bgcolor="white",
            width=width, height=height,
            margin=dict(l=56, r=56, t=top_margin, b=bottom_margin),
            font=dict(family="Helvetica Neue", color="#1a1a2e", size=18),
            showlegend=rep.layout.showlegend,
        )
        # 軸の数値・タイトルを大きく（表示されている軸のみ影響）
        rep.update_xaxes(tickfont=dict(size=16), title=dict(font=dict(size=18)))
        rep.update_yaxes(tickfont=dict(size=16), title=dict(font=dict(size=18)))

        # --- レポート画像でテキストラベルが端で見切れないようにする ---
        # テキスト付きトレース（ネットワークのノード名等）は cliponaxis=False で軸外（余白）にも描画。
        # さらに、そういう図は自動範囲の軸にデータ範囲＋18%余白を明示設定して端のノードを内側へ寄せ、
        # ラベルの居場所を確保する。マーカーのみ/カテゴリ軸の図（棒・ヒートマップ等）は対象外。
        def _collect_num(seq):
            out = []
            if seq is None:
                return out
            for _v in seq:
                try:
                    _f = float(_v)
                except (TypeError, ValueError):
                    continue
                if _f == _f:  # NaN 除外
                    out.append(_f)
            return out

        _has_text = False
        _xs, _ys = [], []
        for _tr in rep.data:
            _mode = getattr(_tr, 'mode', None)
            if _mode and 'text' in str(_mode):
                _has_text = True
                try:
                    _tr.cliponaxis = False
                except Exception:
                    pass
            _xs.extend(_collect_num(getattr(_tr, 'x', None)))
            _ys.extend(_collect_num(getattr(_tr, 'y', None)))
        if _has_text:
            # 対数軸では range は log10 単位で解釈される。生値で範囲を設定すると軸が壊れ、
            # CAPCOM 同梱図が画面表示（autorange）と食い違う（例: 権利化率マップの対数X軸）。
            # → 対数軸は範囲拡張せず autorange のままにし、画面表示と一致させる。
            _x_is_log = (getattr(rep.layout.xaxis, 'type', None) == 'log')
            _y_is_log = (getattr(rep.layout.yaxis, 'type', None) == 'log')
            if _xs and getattr(rep.layout.xaxis, 'range', None) is None and not _x_is_log:
                _x0, _x1 = min(_xs), max(_xs)
                if _x1 > _x0:
                    _xp = (_x1 - _x0) * 0.18
                    rep.update_xaxes(range=[_x0 - _xp, _x1 + _xp], autorange=False)
            if _ys and getattr(rep.layout.yaxis, 'range', None) is None and not _y_is_log:
                _y0, _y1 = min(_ys), max(_ys)
                if _y1 > _y0:
                    _yp = (_y1 - _y0) * 0.18
                    rep.update_yaxes(range=[_y0 - _yp, _y1 + _yp], autorange=False)
        return rep
    except Exception:
        return fig


def fig_to_report_png(fig, title=None, subtitle=None, caption=None, source=None,
                      width=1600, height=900, scale=2):
    """fig をレポート用テーマで整えて高解像度 PNG バイト列にする（kaleido）。

    Returns:
        bytes | None: PNG バイト列（kaleido 不在や失敗時は None）
    """
    try:
        themed = apply_report_theme(
            fig, title=title, subtitle=subtitle, caption=caption, source=source,
            width=width, height=height)
        return themed.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def save_curated_to_capcom(fig, snap_id, cache_token="", width=1600, height=900):
    """整理版ランドスケープ等のスライド向け fig を、CAPCOM アクティブ時に
    クリーン高解像度 PNG として CAPCOM ストア（ZIP）に同梱する。

    方針:
    - VOYAGER スナップショットには登録しない（CAPCOM 専用同梱。DLボタンとも独立）。
    - kaleido での PNG 生成は重いため、cache_token（整理版の設定値=上位N・領域スタイル等）が
      変わらない限り再生成しない。生成済み PNG は session_state にキャッシュする。
    - CAPCOM 非アクティブ・fig 無し・生成失敗時は何もしない（安全に no-op）。

    Args:
        fig: 対象 Plotly Figure（クリーン版で書き出す）
        snap_id (str): CAPCOM 内のスナップショットID（例 'saturnv_curated_landscape'）
        cache_token (str): 再生成要否を判定するトークン（整理版のパラメータを連結した文字列）
        width, height (int): 出力寸法
    """
    if fig is None:
        return
    try:
        import capcom
        if not capcom.is_active():
            return
    except Exception:
        return

    state_key = f"_capcom_curated_{snap_id}"
    cached = st.session_state.get(state_key)
    if not cached or cached.get('token') != cache_token:
        png = fig_to_report_png(
            fig, title=None, subtitle=None, caption=None, source=None,
            width=width, height=height, scale=2)
        if not png:
            return
        cached = {'token': cache_token, 'png': png}
        st.session_state[state_key] = cached

    try:
        capcom.save_snapshot_image(snap_id, cached['png'])
    except Exception:
        pass


def render_report_png_button(fig, key, default_title="", default_subtitle="",
                             default_caption="", source=None, width=1600, height=900,
                             label="スライド/レポート用に書き出す（高解像度PNG）"):
    """マップを「スライド用クリーン高解像度PNG」として書き出す UI（ダウンロード専用）。

    方針: APOLLO 本体では要点・見出し・出典を画像に焼き込まない（クリーン版のみ）。
    タイトル・要点・出典は下流（CAPCOM のレポート/スライド生成）で付与する運用に統一したため、
    本 UI にテキスト入力欄は設けない。引数 default_title/subtitle/caption・source は後方互換のため
    受け取るが画像には反映しない（クリーン出力）。
    生成は重い（kaleido）ため、ボタン押下時のみ実行し session_state にキャッシュする。
    ローカルストレージは使わず session_state のみ。

    Args:
        fig: 対象 Plotly Figure
        key (str): 一意キー（ウィジェット衝突回避用）
        default_title/subtitle/caption, source: 後方互換用（未使用）
        width, height (int): 出力寸法のフォールバック
        label (str): 開閉チェックボックスの表示名（何を書き出すか区別するため）
    """
    if fig is None:
        return
    # expander は使わない（他の expander 内に置かれてもネストエラーにならないよう checkbox で開閉）
    if st.checkbox(label, key=f"rep_show_{key}"):
        st.caption("クリーンな高解像度画像を書き出します。**要点・見出し・出典はレポート/スライド生成側（CAPCOM）で付与**するため、ここでは入力しません。")
        _ratio = st.selectbox("寸法", ["16:9 (1600×900)", "4:3 (1600×1200)", "正方形 (1400×1400)"],
                              key=f"rep_ratio_{key}")
        _w, _h = (1600, 900)
        if _ratio.startswith("4:3"):
            _w, _h = 1600, 1200
        elif _ratio.startswith("正方形"):
            _w, _h = 1400, 1400

        gen_key = f"_report_png_bytes_{key}"
        if st.button("クリーン画像を生成", key=f"rep_gen_{key}", type="primary"):
            with st.spinner("高解像度画像を生成中（数秒）..."):
                _png = fig_to_report_png(
                    fig, title=None, subtitle=None, caption=None, source=None,
                    width=_w, height=_h, scale=2)
            if _png:
                st.session_state[gen_key] = _png
            else:
                st.warning("画像生成に失敗しました（kaleido 未導入の可能性）。")
        if st.session_state.get(gen_key):
            st.image(st.session_state[gen_key], caption="プレビュー（クリーン版）", use_container_width=True)
            st.download_button(
                "PNG をダウンロード", st.session_state[gen_key],
                file_name=f"APOLLO_{key}.png", mime="image/png", key=f"rep_dl_{key}")


def _clean_cluster_label(lbl, max_terms=2):
    """クラスタラベル "[6] 繊維状セルロース, 分散液, 微細繊維状" → "繊維状セルロース、分散液" に整える。"""
    import re
    s = re.sub(r'^\s*\[?\d+\]?\s*', '', str(lbl))
    parts = [p.strip() for p in re.split(r'[,、]', s) if p.strip()]
    return '、'.join(parts[:max_terms]) if parts else (s if s and s != 'nan' else '')


def _hex_to_rgba(color, alpha=0.2):
    """色(hex / rgb / 名前)を rgba(...) 文字列にする。失敗時は淡い灰青。"""
    try:
        c = str(color).strip()
        if c.startswith('#'):
            c = c.lstrip('#')
            if len(c) == 3:
                c = ''.join(ch * 2 for ch in c)
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        if c.startswith('rgb'):
            inner = c[c.find('(') + 1:c.find(')')]
            parts = [p.strip() for p in inner.split(',')][:3]
            return f"rgba({parts[0]},{parts[1]},{parts[2]},{alpha})"
    except Exception:
        pass
    return f"rgba(120,120,160,{alpha})"


def _convex_hull_xy(pts):
    """点群(N,2)の凸包を閉じたポリゴン(xs, ys)で返す。3点未満や失敗時は None。"""
    try:
        arr = np.asarray(pts, dtype=float)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) < 3:
            return None
        hull = ConvexHull(arr)
        idx = list(hull.vertices) + [hull.vertices[0]]
        return arr[idx, 0].tolist(), arr[idx, 1].tolist()
    except Exception:
        return None


def build_curated_landscape(df, cluster_col='cluster', label_col='cluster_label',
                            x_col='umap_x', y_col='umap_y', top_n=8,
                            noise_value=-1, max_label_terms=2, height=820,
                            region_style='hull', density_bins=30):
    """件数上位クラスタのみを大きく示す「整理版ランドスケープ」fig を返す（スライド/レポート用）。

    全クラスタを密にラベルすると重なって読めないため、要点（主要クラスタ）を絞って見せる。
    region_style で領域の描き方を選べる:
      - 'hull'    : 各上位クラスタを凸包ポリゴン（塗り）で領域表示
      - 'density' : 全点の密度マップ（等高線）を背景に、上位クラスタを件数バブル＋ラベルで表示
      - 'points'  : 各上位クラスタを色付き点で表示（領域なし）
    いずれも件数比例バブル＋件数/%付き大ラベルを重ねる。

    Args:
        df: 点データ（umap座標・クラスタ・ラベルを含む DataFrame）
        cluster_col/label_col/x_col/y_col: 列名
        top_n (int): ラベル表示する上位クラスタ数
        noise_value: ノイズクラスタの値（除外）
        max_label_terms (int): ラベルに残す単語数
        height (int): 図の高さ
        region_style (str): 'hull' | 'density' | 'points'

    Returns:
        go.Figure: 整理版ランドスケープ（apply_report_theme で装飾→PNG化できる）
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    if df is None or len(df) == 0 or x_col not in df.columns or y_col not in df.columns:
        return fig

    valid = df[df[cluster_col] != noise_value] if cluster_col in df.columns else df
    if len(valid) == 0:
        valid = df
    total = len(valid)

    # 背景レイヤー
    if region_style == 'density':
        # 全点の密度マップ。配色はメインマップ（Saturn V 密度モード）と同じ白→青に合わせる。
        _density_colorscale = [
            [0.0, "rgba(255, 255, 255, 0)"],
            [0.1, "rgba(225, 245, 254, 0.3)"],
            [0.4, "rgba(129, 212, 250, 0.6)"],
            [1.0, "rgba(2, 119, 189, 0.9)"],
        ]
        fig.add_trace(go.Histogram2dContour(
            x=df[x_col], y=df[y_col], colorscale=_density_colorscale, showscale=False,
            nbinsx=density_bins, nbinsy=density_bins,
            contours=dict(coloring='fill', showlines=True),
            line=dict(width=0.5, color='rgba(0,0,0,0.2)'), hoverinfo='skip'))
    else:
        # 全点を淡いグレーで（クラスタの広がりの文脈）
        fig.add_trace(go.Scattergl(
            x=df[x_col], y=df[y_col], mode='markers',
            marker=dict(size=4, color='#dcdce4'), opacity=0.4,
            hoverinfo='skip', showlegend=False, name='全体'))

    if cluster_col not in valid.columns:
        fig.update_layout(height=height, plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    sizes = valid[cluster_col].value_counts()
    top_ids = list(sizes.head(top_n).index)
    max_cnt = int(sizes.iloc[0]) if len(sizes) else 1
    palette = APOLLO_COLORS
    # 配色をメインマップと一致させる: ソートしたクラスタID順（ノイズ-1も順序に含む）で
    # APOLLO_COLORS を割り当てる。これで同じクラスタが両ビューで同じ色になる。
    try:
        all_cids = sorted(df[cluster_col].dropna().unique().tolist())
    except Exception:
        all_cids = list(top_ids)

    def _color_for(_cid):
        try:
            return palette[all_cids.index(_cid) % len(palette)]
        except Exception:
            return palette[0]

    ann = []
    for i, cid in enumerate(top_ids):
        sub = valid[valid[cluster_col] == cid]
        if len(sub) == 0:
            continue
        color = _color_for(cid)
        cnt = len(sub)
        pct = (cnt / total * 100) if total else 0
        cx = float(sub[x_col].mean()); cy = float(sub[y_col].mean())
        lbl = ''
        if label_col in sub.columns:
            lbl = _clean_cluster_label(sub[label_col].iloc[0], max_label_terms)
        if not lbl:
            lbl = f"クラスタ{cid}"

        if region_style == 'hull':
            # 凸包でクラスタ領域を示す（メインマップ「クラスタ領域 (Clusters)」準拠・やや濃いめの塗り）
            hull = _convex_hull_xy(sub[[x_col, y_col]].values)
            if hull is not None:
                fig.add_trace(go.Scatter(
                    x=hull[0], y=hull[1], mode='lines', fill='toself',
                    fillcolor=_hex_to_rgba(color, 0.25), line=dict(color=color, width=2),
                    hoverinfo='skip', showlegend=False))
            _label_yshift = 0  # 領域の中心にラベル（バブルは出さない）
        elif region_style == 'points':
            # 散布図: クラスタ点 ＋ 件数に比例した重心バブル
            fig.add_trace(go.Scattergl(
                x=sub[x_col], y=sub[y_col], mode='markers',
                marker=dict(size=7, color=color), opacity=0.55,
                hoverinfo='skip', showlegend=False, name=lbl))
            bsize = 26 + 64 * float(np.sqrt(cnt) / np.sqrt(max_cnt)) if max_cnt > 0 else 30
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode='markers',
                marker=dict(size=bsize, color=color, opacity=0.9,
                            line=dict(width=2, color='white')),
                hoverinfo='skip', showlegend=False))
            _label_yshift = int(bsize / 2) + 16  # バブルの上にラベル
        else:
            # density: 背景の密度マップで領域を示す。重心バブルは出さず、中心にラベルのみ。
            _label_yshift = 0

        ann.append(dict(
            x=cx, y=cy, text=f"<b>{lbl}</b><br>{cnt:,}件 ・ {pct:.0f}%",
            showarrow=False, align='center',
            font=dict(size=18, color='#1a1a2e', family="Helvetica Neue"),
            bgcolor='rgba(255,255,255,0.88)', bordercolor=color, borderwidth=2, borderpad=5,
            yshift=_label_yshift))

    fig.update_layout(
        annotations=ann,
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1, constrain="domain"),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20), height=height, showlegend=False)
    return fig


# ==================================================================
# --- 5. 高度なテキスト処理 — patiroha委譲 ---
# ==================================================================

def normalize_text(text):
    """テキスト正規化 (patiroha委譲)"""
    return patiroha.normalize_text(text)

def apply_ngram_filters(text):
    """N-gramフィルタ適用 (patiroha委譲)"""
    return patiroha.apply_ngram_filters(text)

def load_tokenizer():
    """トークナイザ読み込み — patirohaが内部管理するため不要"""
    pass


def _sanitize_texts_for_janome(texts, stopwords=None, max_text_len=8000):
    """Janome 経由の patiroha API（build_tfidf / auto_label 等）に渡す前のサニタイズ。

    特許テキストの一部（超長文・特殊文字・制御文字等）で Janome が
    `IndexError: list index out of range` を吐く問題に対応する。

    3 段階の防御:
    1. 入力サニタイズ: 非 str / 空文字 / None を空文字に置換
    2. 超長文切り詰め: 8000 文字超を切り詰め（Janome lattice サイズ制約）
    3. 個別検証: 各テキストを単独で tokenize 試行し、失敗するものを空文字に置換

    Args:
        texts (Iterable): 入力テキストのリスト/イテラブル
        stopwords: tokenize_for_tfidf に渡すストップワード
        max_text_len (int): 1 テキストの最大文字数

    Returns:
        list[str]: サニタイズ済みテキストリスト（元と同じ長さ）
    """
    import patiroha

    safe = []
    for t in texts:
        # 1. 入力サニタイズ
        if not isinstance(t, str) or not t.strip():
            safe.append("")
            continue

        # 2. 超長文切り詰め
        if len(t) > max_text_len:
            t = t[:max_text_len]

        # 3. 個別検証: tokenize できるかチェック
        try:
            _ = list(patiroha.tokenize_for_tfidf(t, stopwords=stopwords))
            safe.append(t)
        except Exception:
            # tokenize 失敗 → 空文字に置換してスキップ
            safe.append("")

    return safe


def safe_build_tfidf(texts, stopwords=None, max_text_len=8000, **build_kwargs):
    """patiroha.build_tfidf を Janome 例外耐性付きで呼ぶラッパー。

    楽観実行（まず素のまま build_tfidf を試行）し、Janome が IndexError 等で
    クラッシュした場合のみ `_sanitize_texts_for_janome` で事前サニタイズして再試行する。
    正常データでは 1 回のトークナイズで済み、二重トークナイズによる性能劣化を避ける。

    Args:
        texts (list[str]): 入力テキストのリスト
        stopwords: patiroha.build_tfidf に渡すストップワード
        max_text_len (int): 1 テキストの最大文字数（Janome lattice サイズ制約、失敗時のみ適用）
        **build_kwargs: patiroha.build_tfidf に渡す追加引数（min_df / max_df 等）

    Returns:
        (tfidf_matrix, feature_names): patiroha.build_tfidf の戻り値
    """
    import patiroha
    # 超長文だけ軽量に切り詰めてから 1 回トークナイズ（二重トークナイズ回避＋速度維持）
    text_list = [t[:max_text_len] if isinstance(t, str) and len(t) > max_text_len else t for t in texts]
    try:
        return patiroha.build_tfidf(text_list, stopwords=stopwords, **build_kwargs)
    except Exception:
        # Janome がクラッシュする異常データのときだけサニタイズして再試行
        safe_texts = _sanitize_texts_for_janome(text_list, stopwords=stopwords, max_text_len=max_text_len)
        return patiroha.build_tfidf(safe_texts, stopwords=stopwords, **build_kwargs)


def build_tfidf_from_tokens(token_lists, min_df=5, max_df=0.80, max_features=None):
    """事前トークナイズ済みのトークン列から TF-IDF 行列を構築する（二重トークナイズ回避）。

    `extract_keywords` で既に抽出済みの複合名詞リストを再利用して TF-IDF を構築するため、
    Janome 形態素解析を再実行しない（前処理 Phase 5 の二重トークナイズを 1 パスに削減）。
    `patiroha.build_tfidf` と同じ min_df/max_df クランプ・縮退コーパスのフォールバックを踏襲し、
    戻り値の形式も合わせる。トークンは抽出済みのため `analyzer=identity` で再分割・小文字化を抑止し、
    複合名詞の表層をそのまま特徴語にする。

    Args:
        token_lists (list[list[str]]): 各文書のトークン列（utils.extract_keywords の結果）
        min_df (int): 最小文書頻度
        max_df (float): 最大文書頻度（割合）
        max_features (int|None): 最大特徴数（None で無制限）

    Returns:
        (tfidf_matrix, feature_names): patiroha.build_tfidf と同形式の戻り値
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    # None / 非リストは空リストに正規化（NaN 等を吸収）
    docs = [t if isinstance(t, list) else [] for t in token_lists]

    # min_df を max_df と両立可能な範囲にクランプ（patiroha.build_tfidf と同じ対策）
    n_docs = len(docs)
    max_df_docs = int(max_df * n_docs) if isinstance(max_df, float) else max_df
    effective_min_df = min(min_df, max(1, max_df_docs))

    # 入力は既にトークン化済み → analyzer=identity で再トークナイズ/小文字化を回避
    def _identity(doc):
        return doc

    vectorizer = TfidfVectorizer(
        analyzer=_identity,
        min_df=effective_min_df,
        max_df=max_df,
        max_features=max_features,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
    except ValueError:
        # 縮退コーパス（全語が希少 等）は許容設定にフォールバック
        vectorizer = TfidfVectorizer(
            analyzer=_identity, min_df=1, max_df=1.0, max_features=max_features
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
        except ValueError as e:
            raise ValueError(
                "build_tfidf_from_tokens: 有効な語彙がありません（全文書が空）。"
            ) from e
    feature_names = np.array(vectorizer.get_feature_names_out())
    return tfidf_matrix, feature_names


def safe_auto_label(texts, labels, stopwords=None, max_text_len=8000, **label_kwargs):
    """patiroha.auto_label を Janome 例外耐性付きで呼ぶラッパー。

    `patiroha.auto_label` は内部で `[tokenize_for_tfidf(t) for t in texts]` を実行するため、
    `safe_build_tfidf` と同じ Janome IndexError リスクがある。楽観実行し、クラッシュした
    場合のみ事前サニタイズして再試行する（二重トークナイズによる性能劣化を回避）。

    Args:
        texts (list[str]): クラスタリング対象テキスト
        labels: クラスタラベル配列（texts と同じ長さ、-1 = ノイズ）
        stopwords: ストップワード（None なら patent デフォルト）
        max_text_len (int): 1 テキストの最大文字数（失敗時のみ適用）
        **label_kwargs: patiroha.auto_label に渡す追加引数（method / top_n / label_format 等）

    Returns:
        dict[int, str]: クラスタ ID → ラベル文字列の辞書
    """
    import patiroha
    # 超長文だけ軽量に切り詰めてから 1 回トークナイズ（二重トークナイズ回避＋速度維持）
    text_list = [t[:max_text_len] if isinstance(t, str) and len(t) > max_text_len else t for t in texts]
    try:
        return patiroha.auto_label(text_list, labels, stopwords=stopwords, **label_kwargs)
    except Exception:
        # Janome がクラッシュする異常データのときだけサニタイズして再試行
        safe_texts = _sanitize_texts_for_janome(text_list, stopwords=stopwords, max_text_len=max_text_len)
        return patiroha.auto_label(safe_texts, labels, stopwords=stopwords, **label_kwargs)


def extract_keywords(text, tokenizer=None, stopwords=None, top_n=None, clean_html=False):
    """
    テキストから特徴語（名詞・複合名詞）を抽出する (patiroha委譲)
    Janome の内部エラー (IndexError 等) と異常入力を吸収するロバストラッパー。

    Args:
        text (str): 対象テキスト
        tokenizer: 未使用 (後方互換性のため維持)
        stopwords (list/set): ストップワードのリスト
        top_n: 未使用 (後方互換性のため維持)
        clean_html (bool): Trueの場合、HTML/XMLタグを除去する (NPL推奨)
    Returns:
        list: 抽出されたキーワードのリスト (失敗時・異常入力時は空リスト)
    """
    # 入力サニタイズ: None / NaN / 非文字列を弾く
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []

    # Janome の lattice サイズ制約を回避。
    # 実測で 8000 文字超で IndexError (lattice.enodes のインデックス超過) を誘発することがある。
    # キーワード抽出では十分なサイズなので切り詰める。
    MAX_LEN = 8000
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN]

    if stopwords is None:
        stopwords = patiroha.get_stopwords()

    try:
        return patiroha.extract_keywords(text, stopwords=stopwords, clean_html=clean_html)
    except Exception:
        # Janome / patiroha 内部の予期せぬエラーは空リストで吸収し、分析全体を止めない
        return []


def extract_keywords_batch(texts, stopwords=None, clean_html=False, n_jobs=None,
                           min_parallel=600, min_total_chars=1_500_000):
    """テキスト列からキーワードを一括抽出する。処理が十分重く、かつマルチコアがある環境では
    プロセス並列で高速化する（Janome は pure-Python=GIL律速のため、スレッドでなくプロセス並列が効く）。

    出力は逐次版（extract_keywords を1件ずつ）と完全に同一。並列が割に合わない/使えない場合は
    自動的に逐次へフォールバックするため、少コア環境（クラウド）や短文データでも安全・無駄がない。
    ワーカーは streamlit を読み込まない軽量モジュール(apollo_kw_worker)を使い、起動を軽く保つ。

    Args:
        texts: テキストの列（list/Series）
        stopwords: ストップワード（None なら patent デフォルト）
        clean_html (bool): HTML 除去（NPL 推奨）
        n_jobs (int|None): 並列数。None で自動（論理コア-1、上限8）
        min_parallel (int): これ未満の件数は逐次（並列オーバーヘッド回避）
        min_total_chars (int): 総文字数がこれ未満なら逐次（軽い処理での逆効果を防ぐ）

    Returns:
        list[list[str]]: 各テキストのキーワードリスト（入力と同じ順序・長さ）
    """
    texts = list(texts)
    n = len(texts)

    def _serial():
        return [extract_keywords(t, stopwords=stopwords, clean_html=clean_html) for t in texts]

    # 件数・総文字数のどちらかが小さければ逐次（並列の起動コストが見合わない）
    total_chars = sum(len(t) for t in texts if isinstance(t, str))
    if n < min_parallel or total_chars < min_total_chars:
        return _serial()
    try:
        import os
        cores = os.cpu_count() or 1
        if n_jobs is None:
            n_jobs = max(1, min(cores - 1, 8))
        if n_jobs <= 1:
            return _serial()
        from joblib import Parallel, delayed
        from apollo_kw_worker import extract_keywords_worker
        sw = list(stopwords) if stopwords is not None else None
        results = Parallel(n_jobs=n_jobs, backend='loky', batch_size='auto')(
            delayed(extract_keywords_worker)(t, sw, clean_html) for t in texts
        )
        if results is not None and len(results) == n:
            return list(results)
    except Exception:
        pass
    # 並列が使えない/失敗した場合は逐次フォールバック（出力は同一）
    return _serial()


# ==================================================================
# --- 特許詳細ポップアップ (チャート/マップのクリック → st.dialog) ---
# 参考_AIS の Patent Detail Viewer 相当。個別特許点のクリックで詳細をモーダル表示。
# ==================================================================
def disable_selection_fade(fig):
    """選択時に非選択点が薄くなる Plotly の既定挙動を無効化する（クリック後もハイライトが残らない）。

    各 trace の selected/unselected の marker opacity を元の opacity に固定することで、
    点をクリック選択しても見た目が変化しない（＝閉じても元に戻ったように見える）。
    """
    try:
        for tr in fig.data:
            mk = getattr(tr, 'marker', None)
            op = getattr(mk, 'opacity', None) if mk is not None else None
            if op is None:
                op = 1.0
            tr.update(selected=dict(marker=dict(opacity=op)),
                      unselected=dict(marker=dict(opacity=op)))
    except Exception:
        pass
    return fig


def _render_patent_card(row, col_map):
    """1 件の特許をカード形式で描画する（公報番号｜タイトル / 出願日 / 出願人 / IPC・ステータス / 要約）。"""
    # 番号は 出願番号 → 正規化出願番号 を最優先で表示する（ユーザーは出願番号カラムを
    # 確実に割り当てており、公報番号を混ぜたくないため）。公報番号は出願番号が全く無い
    # 場合の最終フォールバックに留める。
    no = "N/A"
    for _nc in (col_map.get('app_num'), 'app_num_main', col_map.get('pub_number')):
        if not _nc:
            continue
        _v = _extract_field_value(row, _nc, max_len=40)
        if _v and _v not in ("N/A", "nan", ""):
            no = _v
            break
    title = _extract_field_value(row, col_map.get('title'), max_len=120) if col_map.get('title') else ""
    date = _extract_field_value(row, col_map.get('date'), max_len=40) if col_map.get('date') else "-"
    app = _extract_field_value(row, col_map.get('applicant'), max_len=80) if col_map.get('applicant') else "-"
    _code_col = classification_code_column(col_map)
    ipc = _extract_field_value(row, _code_col, max_len=60) if _code_col else ""
    status = _extract_field_value(row, col_map.get('status'), max_len=40) if col_map.get('status') else ""
    abst = _extract_field_value(row, col_map.get('abstract'), max_len=300) if col_map.get('abstract') else "-"
    # 原文リンク（文献URL列の割り当てがある場合。URLは切り詰めずそのまま使う）
    url = str(row.get(col_map.get('doc_url'), '') or '').strip() if col_map.get('doc_url') else ""
    header = f"**{no}**" + (f"　｜　{title}" if title and title not in ('N/A', '') else "")
    if url.startswith('http'):
        header += f"　[ 原文を開く]({url})"
    st.markdown(header)
    meta = f"{date}　・　 {app}"
    if ipc and ipc != "N/A":
        meta += f"　・　{classification_code_label(col_map)}: {ipc}"
    if status and status != "N/A":
        meta += f"　・　{status}"
    st.caption(meta)
    if abst and abst != "N/A":
        st.markdown(f"> {abst}")
    st.divider()


def show_patent_detail_dialog(indices, title="特許詳細", per_page=3, df=None, card_renderer=None):
    """レコード（特許/論文）のインデックスリストを st.dialog でカード表示する。

    5 件/ページのページネーション付き。df 未指定なら df_main（特許）。
    card_renderer(row, col_map) を渡せば論文等の別カードに差し替え可能。
    ローカルストレージは使わず session_state でページ管理。
    """
    df = st.session_state.get('df_main') if df is None else df
    if df is None or not indices:
        return
    col_map = st.session_state.get('col_map', {})
    renderer = card_renderer or _render_patent_card

    @st.dialog("" + str(title), width="large")
    def _dlg():
        # ページネーションは @st.fragment で局所再実行（ページ送りでページ全体を再実行しないので高速）
        @st.fragment
        def _content():
            page_key = "_patent_dlg_page"
            total = len(indices)
            total_pages = max(1, (total + per_page - 1) // per_page)
            page = min(max(0, st.session_state.get(page_key, 0)), total_pages - 1)
            # ページ送りUIを最上部に置く（開いた瞬間に先頭が見える）
            if total_pages > 1:
                c1, c2, c3 = st.columns([1, 1.4, 1])
                with c1:
                    if st.button("← 前へ", disabled=(page <= 0), key="_pat_prev", use_container_width=True):
                        st.session_state[page_key] = page - 1
                        st.rerun(scope="fragment")
                with c2:
                    st.markdown(
                        f"<div style='text-align:center; padding-top:6px; font-weight:700;'>{page + 1} / {total_pages} ページ（全 {total} 件）</div>",
                        unsafe_allow_html=True)
                with c3:
                    if st.button("次へ →", disabled=(page >= total_pages - 1), key="_pat_next", use_container_width=True):
                        st.session_state[page_key] = page + 1
                        st.rerun(scope="fragment")
            else:
                st.caption(f"該当 {total} 件")
            st.divider()
            for idx in indices[page * per_page:(page + 1) * per_page]:
                try:
                    row = df.loc[idx]
                except Exception:
                    continue
                renderer(row, col_map)
        _content()
    _dlg()


def handle_map_click(selection, state_key, title="クリックした特許", df=None, card_renderer=None):
    """plotly_chart(on_select="rerun") の戻り値から customdata（特許インデックス）を取り出し、
    前回クリックと異なる場合のみ特許詳細ダイアログを開く（リランの無限ループを回避）。

    Args:
        selection: st.plotly_chart(on_select="rerun") の戻り値
        state_key: チャート固有のキー（前回クリック記録用、ページ間で一意にする）
        title: ダイアログのタイトル
    """
    if not selection:
        return
    pts = (selection.get("selection") or {}).get("points") or []
    indices = []
    for p in pts:
        cd = p.get("customdata")
        if cd is None:
            continue
        indices.append(cd[0] if isinstance(cd, (list, tuple)) else cd)
    if not indices:
        return
    last_key = "_last_click_" + state_key
    sig = tuple(indices)
    if st.session_state.get(last_key) == sig:
        return  # 同じ点の再クリック → 二重表示しない
    st.session_state[last_key] = sig
    st.session_state["_patent_dlg_page"] = 0  # 新規クリックでページをリセット
    show_patent_detail_dialog(indices, title=title, df=df, card_renderer=card_renderer)


def render_paper_card(row, col_map=None):
    """学術論文 1 件をカード形式で描画する（NEBULA の学術マップ用）。"""
    title = str(row.get('unified_title', '') or '') or '(タイトル不明)'
    src = str(row.get('unified_source', '') or '')
    yr = str(row.get('year', '') or '')
    lbl = str(row.get('acad_cluster_label', '') or '')
    st.markdown(f"**{title}**")
    meta = []
    if src and src != 'nan':
        meta.append(f"{src}")
    if yr and yr != 'nan':
        meta.append(f"{yr}")
    if lbl and lbl != 'nan':
        meta.append(f"クラスタ: {lbl}")
    if meta:
        st.caption("　・　".join(meta))
    st.divider()


def handle_chart_click_to_records(selection, state_key, resolver, title="該当特許"):
    """集計チャートのクリック点 → resolver(point)=(indices, dlg_title) で該当レコードを解決し
    ダイアログを開く。MEGA/ATLAS/CORE の「カテゴリ → 該当特許群」用。

    Args:
        selection: st.plotly_chart(on_select="rerun") の戻り値
        state_key: チャート固有のキー
        resolver: 関数 point(dict) -> (indices:list, dlg_title:str|None)
    """
    if not selection:
        return
    pts = (selection.get("selection") or {}).get("points") or []
    if not pts:
        return
    try:
        indices, dlg_title = resolver(pts[0])
    except Exception:
        return
    if indices is None or len(indices) == 0:
        return
    last_key = "_last_click_" + state_key
    sig = (str(dlg_title), tuple(list(indices)[:100]))
    if st.session_state.get(last_key) == sig:
        return
    st.session_state[last_key] = sig
    st.session_state["_patent_dlg_page"] = 0
    show_patent_detail_dialog(list(indices), title=dlg_title or title)
