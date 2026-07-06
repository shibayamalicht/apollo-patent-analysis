import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import warnings
import unicodedata
import re
import json
import traceback

from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
import utils
import utils_ai
import patiroha

# ==================================================================
# --- 1. ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO v9 | CORE", 
    page_icon=utils.module_icon("core"), 
    layout="wide"
)

st.session_state['current_page'] = 'CORE'

pio.templates.default = "plotly_white"
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. デザインテーマ設定 & 共通CSS ---
# ==================================================================




# ==================================================================
# --- 3. ヘルパー関数 & リソースロード ---
# ==================================================================
@st.cache_resource
def load_tokenizer_core(): return Tokenizer()
t = load_tokenizer_core()

if "stopwords" in st.session_state and st.session_state["stopwords"]:
    STOP_WORDS = st.session_state["stopwords"]
else:
    STOP_WORDS = patiroha.get_stopwords()

@st.cache_data
def _core_text_preprocessor(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'[（(][^）)]{1,80}[）)]', ' ', text)
    text = re.sub(r'(?:図|Fig|FIG|fig)[. 　]*\d+', ' ', text)
    text = re.sub(r'[!！?"“”#$%＆&\'()（）*＋+,\-．.\:：;；<=>?？@\[\]［］\\^_`{|}~〜〜／/]', ' ', text)
    return text

@st.cache_data
def advanced_tokenize_core(text):
    text = _core_text_preprocessor(text)
    if not text: return ""
    # 防御層: 超長文で Janome の IndexError を避ける
    if len(text) > 8000:
        text = text[:8000]
    try:
        tokens = list(t.tokenize(text))
    except Exception:
        return ""
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token1 = tokens[i]
        base1 = token1.base_form if token1.base_form != '*' else token1.surface
        if base1 in STOP_WORDS: i += 1; continue
        pos1 = token1.part_of_speech.split(',')
        if len(base1) < 2 and pos1[0] != '名詞': i += 1; continue
        if pos1[0] == '名詞' and (len(pos1) > 1 and pos1[1] == '数'): i += 1; continue
        if (i + 1) < len(tokens):
            token2 = tokens[i+1]
            base2 = token2.base_form if token2.base_form != '*' else token2.surface
            pos2 = token2.part_of_speech.split(',')
            if pos1[0] == '名詞' and pos2[0] == '名詞' and base2 not in STOP_WORDS and (len(pos2) > 1 and pos2[1] != '数'):
                processed_tokens.append(base1 + base2); i += 2; continue
        if pos1[0] == '名詞' or (pos1[0] in ['動詞', '形容詞'] and len(pos1)>1 and pos1[1] == '自立'):
            processed_tokens.append(base1)
        i += 1
    return " ".join(processed_tokens)

# --- CORE 検索エンジン (Recursive Descent Parser) ---
class LogicNode:
    def evaluate(self, text): raise NotImplementedError

class AndNode(LogicNode):
    def __init__(self, children): self.children = children
    def evaluate(self, text): return all(c.evaluate(text) for c in self.children)

class OrNode(LogicNode):
    def __init__(self, children): self.children = children
    def evaluate(self, text): return any(c.evaluate(text) for c in self.children)

class RegexNode(LogicNode):
    def __init__(self, pattern): 
        try: self.pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        except: self.pattern = None
    def evaluate(self, text): return bool(self.pattern.search(text)) if self.pattern else False

class CoreLogicParser:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def tokenize(self, rule_str):
        # Tokenize: (, ), +, *, nearN, adjN, or literals
        raw_tokens = re.findall(r'\(|\)|' r'\bnear\d+\b|' r'\badj\d+\b|' r'[\+\*]' r'|' r'[^\(\)\+\*\s]+', rule_str, re.IGNORECASE)
        self.tokens = [t.strip() for t in raw_tokens if t.strip()]
        self.pos = 0

    def parse(self, rule_str):
        self.tokenize(rule_str)
        if not self.tokens: return None
        node = self.expression()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at end: {self.tokens[self.pos]}")
        return node

    def expression(self):
        # Expression -> Term { + Term }  (OR)
        nodes = [self.term()]
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '+':
            self.pos += 1
            nodes.append(self.term())
        return OrNode(nodes) if len(nodes) > 1 else nodes[0]

    def term(self):
        # Term -> Factor { * Factor } (AND)
        nodes = [self.factor()]
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '*':
            self.pos += 1
            nodes.append(self.factor())
        return AndNode(nodes) if len(nodes) > 1 else nodes[0]

    def factor(self):
        # Factor -> Atom { (nearN|adjN) Atom }

        # To support (A+B) near C, we need to compile sub-parts to regex strings if possible.
        # Limitation: near/adj can only apply to "Regex-compatible" nodes (Leaf or OR of Leafs). NO ANDs allowed inside near/adj.
        
        left = self.atom()
        
        while self.pos < len(self.tokens) and re.match(r'^(near|adj)\d+$', self.tokens[self.pos], re.IGNORECASE):
            op = self.tokens[self.pos].lower()
            self.pos += 1
            right = self.atom()
            
            # recursive constraint check: Left and Right must be convertible to Regex String
            l_rex = self.to_regex_string(left)
            r_rex = self.to_regex_string(right)
            n = int(re.findall(r'\d+', op)[0])
            
            if op.startswith('near'): pattern = r'(?:{}.{{0,{}}}?{}|{}.{{0,{}}}?{})'.format(l_rex, n, r_rex, r_rex, n, l_rex)
            else: pattern = r'{}.{{0,{}}}?{}'.format(l_rex, n, r_rex) # adj
            
            left = RegexNode(pattern)
            
        return left

    def atom(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '(':
            self.pos += 1
            node = self.expression()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ')':
                self.pos += 1
                return node
            else:
                raise ValueError("Missing closing parenthesis")
        elif self.pos < len(self.tokens):
            t = self.tokens[self.pos]
            self.pos += 1
            # Literal
            norm = unicodedata.normalize('NFKC', t).lower()
            return RegexNode(re.escape(norm))
        else:
            raise ValueError("Unexpected end of rule")

    def to_regex_string(self, node):
        # Helper to convert a Node back to regex string if it contains only OR/Literal
        if isinstance(node, RegexNode): 
            # Pattern inside RegexNode is already compiled or string? 
            # In our class, it's compiled. We need the source string. 
            # Implementation trick: Store source pattern in RegexNode
            if hasattr(node, 'pattern') and node.pattern: return node.pattern.pattern
            # If it was constructed blindly? 
            # Let's modifying RegexNode to store source.
            return "" 
        if isinstance(node, OrNode):
            parts = [self.to_regex_string(c) for c in node.children]
            return r'(?:' + '|'.join(parts) + r')'
        if isinstance(node, AndNode):
             raise ValueError("Cannot use AND (*) inside a NEAR/ADJ condition. Use OR (+) only.")
        return ""

# Patch RegexNode to store source for recursion
class RegexNode(LogicNode):
    def __init__(self, pattern): 
        self.source = pattern
        try: self.pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        except: self.pattern = None
    def evaluate(self, text): return bool(self.pattern.search(text)) if self.pattern else False

@st.cache_resource
def parse_core_rule(rule_str):
    try:
        parser = CoreLogicParser()
        return parser.parse(rule_str)
    except Exception:
        # cache 中はエラー表示を抑制。呼び出し側で None を判定する。
        return None

@st.cache_data
def prepare_axis_data_core(df, axis_col_name, delimiter):
    df_processed = df.copy()
    if axis_col_name not in df_processed.columns: return pd.DataFrame()
    df_processed[axis_col_name] = df_processed[axis_col_name].fillna('N/A')
    if axis_col_name == 'year':
        df_processed[axis_col_name] = df_processed[axis_col_name].apply(lambda x: str(int(x)) if pd.notna(x) else 'N/A')
    if delimiter:
        df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.split(delimiter)
        df_processed = df_processed.explode(axis_col_name)
    df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.strip().replace('', 'N/A')
    return df_processed

@st.cache_data
def convert_df_to_csv_core(df): return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')



# ==================================================================
# --- 4. アプリケーション初期化 & UI構成 ---
# ==================================================================
utils.render_sidebar()

utils.module_header("core", "CORE")
st.markdown("AND/OR/NEAR/ADJ論理式で特許を分類し、分類軸間のクロス集計で技術ポートフォリオの構造を明らかにします。")

if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。"); st.stop()
else:
    df_main = st.session_state.df_main
    col_map = st.session_state.col_map

if "core_classification_rules" not in st.session_state: st.session_state.core_classification_rules = {}
if "core_df_classified" not in st.session_state: st.session_state.core_df_classified = None
if "core_current_axis" not in st.session_state: st.session_state.core_current_axis = ""
if "core_reanalyze_result" not in st.session_state: st.session_state.core_reanalyze_result = ""

# ==================================================================
# --- 4b. クラスタリング高度化（数理的土台・最適k・SBERT・HDBSCAN） ---
# ==================================================================
# 設計方針（数理情報学）:
#  - テキストのクラスタリングは生TF-IDF＋ユークリッドだと文書長バイアスが乗るため
#    L2正規化して球面k-means（=コサイン）にする。
#  - 高次元の呪い対策に TruncatedSVD(LSA) で50–100次元へ圧縮し、圧縮後も再正規化。
#  - 最適kは silhouette / Calinski-Harabasz / Davies-Bouldin で評価し、推奨kを提示。

@st.cache_data(show_spinner=False)
def core_build_tfidf_features(texts_tuple, min_df=2, svd_components=80):
    """テキスト群 → L2正規化（+LSA次元圧縮）済みの密ベクトル行列を返す。
    返り値: (X[np.ndarray], info[dict])"""
    toks = [advanced_tokenize_core(t) for t in texts_tuple]
    # min_df が大きすぎて語彙が空になる小規模データへのフォールバック
    for _mdf in (min_df, 1):
        try:
            vec = TfidfVectorizer(min_df=_mdf, max_df=0.9, token_pattern=r"(?u)\b\w+\b")
            X = vec.fit_transform(toks)
            if X.shape[1] > 0:
                break
        except ValueError:
            X = None
    if X is None or X.shape[1] == 0:
        return np.zeros((len(texts_tuple), 1)), {'mode': 'tfidf', 'n_features': 0, 'svd': False}
    n_feat = int(X.shape[1])
    X = sk_normalize(X)  # L2正規化 → 球面k-means
    used_svd = False
    if svd_components and n_feat > svd_components and X.shape[0] > svd_components + 1:
        try:
            X = TruncatedSVD(n_components=int(svd_components), random_state=42).fit_transform(X)
            X = sk_normalize(X)  # 圧縮後も再正規化
            used_svd = True
        except Exception:
            X = X.toarray()
    else:
        X = X.toarray()
    return np.asarray(X, dtype=float), {'mode': 'tfidf', 'n_features': n_feat, 'svd': used_svd}


def core_prepare_features(texts, feature_mode, embeddings=None, min_df=2, svd_components=80):
    """クラスタリング用の特徴量（L2正規化済み密行列）を返す。
    feature_mode: 'tfidf' / 'sbert'（sbert は Home 前処理で計算済みの埋め込みを再利用）。"""
    if feature_mode == 'sbert' and embeddings is not None and len(embeddings) == len(texts):
        X = sk_normalize(np.asarray(embeddings, dtype=float))
        return X, {'mode': 'sbert', 'n_features': int(X.shape[1]), 'svd': False}
    return core_build_tfidf_features(tuple(texts), min_df=min_df, svd_components=svd_components)


def core_find_optimal_k(X, k_min=2, k_max=15, sample_size=2000):
    """k を走査し silhouette/Calinski-Harabasz/Davies-Bouldin を算出。
    返り値: (scores_df, recommended_k[int])。X は L2正規化済み前提（euclidean≈cosine）。"""
    n = int(X.shape[0])
    k_max = max(k_min, min(int(k_max), n - 1))
    rows = []
    for k in range(int(k_min), k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        except Exception:
            continue
        labels = km.labels_
        if len(set(labels)) < 2:
            continue
        try:
            sil = float(silhouette_score(X, labels, metric='euclidean',
                                         sample_size=min(int(sample_size), n), random_state=42))
        except Exception:
            sil = float('nan')
        try: ch = float(calinski_harabasz_score(X, labels))
        except Exception: ch = float('nan')
        try: db = float(davies_bouldin_score(X, labels))
        except Exception: db = float('nan')
        rows.append({'k': k, 'silhouette': sil, 'Calinski-Harabasz': ch,
                     'Davies-Bouldin': db, 'inertia': float(km.inertia_)})
    scores = pd.DataFrame(rows)
    if scores.empty or not scores['silhouette'].notna().any():
        return scores, int(k_min)
    rec_k = int(scores.loc[scores['silhouette'].idxmax(), 'k'])
    return scores, rec_k


def core_cluster(X, method='kmeans', k=8, min_cluster_size=15,
                 auto_sweep=False, target_k=None, sweep_info=None):
    """クラスタリング実行。返り値: (labels, centers or None, actual_method)。
    method='hdbscan' は、高次元のまま HDBSCAN にかけると密度推定が破綻し全件ノイズに
    なりやすいため、APOLLO 他モジュールと同様に UMAP で低次元化してから HDBSCAN する
    （patiroha.build_landscape を利用）。有効クラスタが得られなければ KMeans にフォールバック。

    auto_sweep=True のとき、build_landscape で得た 2D UMAP 座標上で HDBSCAN の
    (min_cluster_size, min_samples) を掃引し、クラスタ数が適正かつ DBCV 品質の高い組を選ぶ。
    決定した値・クラスタ数等は sweep_info（dict）に書き戻す。"""
    if method == 'hdbscan':
        try:
            mcs = max(2, int(min_cluster_size))
            res = patiroha.build_landscape(
                np.asarray(X, dtype=float), method='hdbscan',
                min_cluster_size=mcs, min_samples=max(1, mcs // 3),
                umap_metric='cosine', random_state=42)
            if auto_sweep:
                # 2D UMAP 座標上で 2 パラメータを掃引（UMAP は build_landscape で 1 回のみ）
                _sw = utils.sweep_hdbscan_params(np.asarray(res.coords, dtype=float), target_k=target_k)
                labels = np.asarray(_sw['labels'])
                if sweep_info is not None:
                    sweep_info.update({
                        'mcs': _sw['min_cluster_size'], 'ms': _sw['min_samples'],
                        'k': _sw['n_clusters'], 'noise': _sw['noise_ratio'],
                        'target_k': _sw['target_k'], 'validity': _sw.get('validity')})
            else:
                labels = np.asarray(res.labels)
            if len([1 for l in set(int(x) for x in labels) if l != -1]) >= 1:
                return labels, None, 'hdbscan'
        except Exception:
            pass
        method = 'kmeans_fallback'  # 全件ノイズ / 失敗
    k = max(2, min(int(k), X.shape[0]))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    return km.labels_, km.cluster_centers_, ('kmeans_fallback' if method == 'kmeans_fallback' else 'kmeans')


def core_select_representatives(X, labels, cluster_id, n, centers=None, diversity=0.3):
    """クラスタ内の代表文献インデックスを返す（重心コサイン近傍＋MMRで多様性確保）。"""
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    sub = X[idx]
    center = centers[cluster_id] if centers is not None else sub.mean(axis=0)
    cn = float(np.linalg.norm(center)) + 1e-9
    sims = (sub @ center) / cn  # 重心へのコサイン（X は正規化済み）
    order = list(np.argsort(-sims))
    lam = 1.0 - float(diversity)
    selected = []
    while order and len(selected) < int(n):
        if not selected:
            selected.append(order.pop(0)); continue
        best_j, best_score = 0, -1e18
        for j, c in enumerate(order):
            div = max(float(sub[c] @ sub[s]) for s in selected)  # 既選択との最大類似
            score = lam * float(sims[c]) - (1 - lam) * div
            if score > best_score:
                best_score, best_j = score, j
        selected.append(order.pop(best_j))
    return idx[selected].tolist()


# ==================================================================
# --- 5. CORE アプリケーション ---
# ==================================================================
current_phase = st.radio("フェーズ選択:", ["フェーズ 1: AIアシスタント (クラスタリング)", "フェーズ 2: 分類ルール定義", "フェーズ 3: 分類実行", "フェーズ 4: 特許マップ作成"], horizontal=True, key="core_phase_selector",
    help="CORE分析の作業段階を切り替えます。フェーズ1＝特許をクラスタリングしてAIに分類設計を依頼するプロンプトを作る（任意）、フェーズ2＝分類ルール（論理式）を定義・編集、フェーズ3＝定義したルールで全特許を分類実行、フェーズ4＝分類結果を2軸でクロス集計して特許マップを描画します。")
st.markdown("---")

# --- フェーズ 1: AIアシスタント ---
if current_phase.startswith("フェーズ 1"):
    st.subheader("フェーズ 1: AIによる分類サジェスト (オプション)")
    st.caption("特許母集団をクラスタリングして代表文献を抽出し、LLMに『技術×課題×解決手段』のMECE分類設計を依頼するプロンプトを生成します。")
    col_map_options = [v for k, v in col_map.items() if k in ['title', 'abstract', 'claim']]
    target_column = st.selectbox("分析対象カラム:", options=col_map_options, key="core_target_col")

    # --- 特徴量・クラスタ手法・サンプル数 ---
    _emb = st.session_state.get('sbert_embeddings')
    _emb_ok = _emb is not None and len(_emb) == len(df_main)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        _feat_opts = ["TF-IDF（語彙）"] + (["SBERT（意味）"] if _emb_ok else [])
        feature_label = st.radio("特徴量:", _feat_opts, key="core_feature_mode",
            help="クラスタリングに使う特徴ベクトルの種類です。TF-IDF＝単語の出現頻度ベース（語の一致を重視）、SBERT＝文章の意味ベクトル（言い回しが違っても意味が近ければ近接）。意味的なまとまりはSBERT、語の一致重視はTF-IDFが得意です。")
        feature_mode = 'sbert' if feature_label.startswith("SBERT") else 'tfidf'
        if not _emb_ok:
            st.caption("※SBERT（意味ベクトル）を使うには Mission Control で前処理を実行してください。")
    with fc2:
        method_label = st.radio("クラスタ手法:", ["KMeans（最適k自動）", "HDBSCAN（k不要）"], key="core_cluster_method",
            help="KMeans＝指定した数(k)に必ず分割し全件どこかに所属、HDBSCAN＝密度ベースで数を自動決定し外れ値はノイズとして分離。数を固定したいならKMeans、外れ値を分けたい・数を決めたくないならHDBSCAN。")
        cluster_method = 'hdbscan' if method_label.startswith("HDBSCAN") else 'kmeans'
    with fc3:
        ai_n_w = st.number_input("各クラスタの代表サンプル数 (N)", min_value=1, value=5, key="core_n",
            help="各クラスタから代表として抽出する特許の件数です。多いほど内容を確認しやすくなります。")

    # --- 手法別パラメータ ---
    if cluster_method == 'kmeans':
        kc1, kc2 = st.columns(2)
        with kc1: _k_min = st.number_input("探索 k 下限", min_value=2, max_value=30, value=2, key="core_kmin",
            help="KMeansの最適クラスタ数を自動探索する範囲（kの下限）です。この範囲でスコアが最良のkを推奨します。")
        with kc2: _k_max = st.number_input("探索 k 上限", min_value=3, max_value=40, value=15, key="core_kmax",
            help="KMeansの最適クラスタ数を自動探索する範囲（kの上限）です。この範囲でスコアが最良のkを推奨します。")
        # 分析設定のシグネチャ（特徴量・探索範囲）。これが変わると過去の推奨kは無効化する
        _optk_sig = (feature_mode, int(_k_min), int(_k_max))
        if st.button("① 最適クラスタ数を分析", key="core_run_optk"):
            with st.spinner("特徴量を構築し、最適クラスタ数を探索中..."):
                try:
                    _texts = df_main[target_column].astype(str).fillna('').tolist()
                    _X, _info = core_prepare_features(_texts, feature_mode, embeddings=_emb)
                    _scores, _reck = core_find_optimal_k(_X, k_min=int(_k_min), k_max=int(_k_max))
                    st.session_state['core_optk_scores'] = _scores
                    st.session_state['core_optk_reck'] = int(_reck)
                    st.session_state['core_optk_info'] = _info
                    st.session_state['core_optk_sig'] = _optk_sig
                    # 推奨kをスライダへ自動反映（範囲内にクランプ）
                    st.session_state['core_chosen_k'] = int(max(2, min(_reck, int(_k_max))))
                except Exception as e:
                    st.error(f"最適k分析エラー: {e}")
        _scores = st.session_state.get('core_optk_scores')
        _reck = st.session_state.get('core_optk_reck')
        # 設定が一致しているときだけ有効（特徴量/手法/範囲を変えたら stale）
        _fresh = st.session_state.get('core_optk_sig') == _optk_sig
        _default_k = 8
        if _scores is not None and not _scores.empty and _fresh:
            _info = st.session_state.get('core_optk_info', {})
            _dim = 'LSA圧縮' if _info.get('svd') else f"{_info.get('n_features','?')}次元"
            st.success(f"推奨 k = **{_reck}**（silhouette 最大）　｜　特徴量: {_info.get('mode','tfidf')} / {_dim}")
            _figk = go.Figure()
            _figk.add_trace(go.Scatter(x=_scores['k'], y=_scores['silhouette'],
                name='Silhouette（高い=良）', mode='lines+markers', yaxis='y1'))
            _figk.add_trace(go.Scatter(x=_scores['k'], y=_scores['Davies-Bouldin'],
                name='Davies-Bouldin（低い=良）', mode='lines+markers', yaxis='y2', line=dict(dash='dot')))
            _figk.add_vline(x=int(_reck), line=dict(color='#e53935', dash='dash'))
            _figk.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10),
                yaxis=dict(title='Silhouette'),
                yaxis2=dict(title='Davies-Bouldin', overlaying='y', side='right'),
                xaxis=dict(title='クラスタ数 k', dtick=1), legend=dict(orientation='h'))
            st.plotly_chart(_figk, use_container_width=True)
            with st.expander("評価指標テーブル（silhouette / Calinski-Harabasz / Davies-Bouldin）"):
                st.dataframe(_scores, use_container_width=True, hide_index=True)
            _default_k = int(_reck)
        elif _scores is not None and not _scores.empty and not _fresh:
            st.warning("特徴量／探索範囲を変更しました。最新の推奨kを得るには「① 最適クラスタ数を分析」を再実行してください（下のスライダは手動指定のままです）。")
        else:
            st.info("「① 最適クラスタ数を分析」で推奨kとスコア曲線を表示できます（未実行でも下のスライダで手動指定可）。")
        # スライダは session_state 駆動（① が推奨kを書き込む）。範囲外をクランプしてから生成
        st.session_state.setdefault('core_chosen_k', int(_default_k))
        st.session_state['core_chosen_k'] = int(max(2, min(int(st.session_state['core_chosen_k']), max(3, int(_k_max)))))
        chosen_k = st.slider("使用するクラスタ数 k（推奨を上書き可）",
            min_value=2, max_value=max(3, int(_k_max)), key="core_chosen_k",
            help="実際に分割するクラスタ数(k)です。自動推奨値を上書きできます。大きいほど細かく、小さいほど大まかに分かれます。")
        min_cluster_size = 15
    else:
        chosen_k = 8
        core_hdbscan_auto = st.checkbox(
            "🤖 自動最適化（最小クラスタサイズ・最小サンプル数を掃引してクラスタ数を適正化）",
            key="core_hdbscan_auto",
            help="母集団に合わせて HDBSCAN の2パラメータを自動掃引し、クラスタ数が少なすぎ(2,3)も多すぎもしない"
                 "設定を DBCV（密度クラスタリングの妥当性指標）基準で選びます。手動調整が不要になります。")
        min_cluster_size = st.number_input("最小クラスタサイズ (HDBSCAN)", min_value=2,
            value=max(15, len(df_main) // 50), key="core_hdbscan_mcs", disabled=core_hdbscan_auto,
            help="この件数未満のまとまりはノイズ（萌芽候補）として分類対象から除外されます。")
        core_target_k = None
        if core_hdbscan_auto:
            core_target_k = st.number_input(
                "目標クラスタ数（目安）", min_value=2, max_value=60,
                value=utils.suggest_target_k(len(df_main)), key="core_hdbscan_target_k",
                help="この数に近づくよう2パラメータを掃引します（母集団規模から自動初期化）。"
                     "結果のクラスタ数や粒度に満足できない場合は、この値を増減して「② 分類サジェスト用プロンプトを生成」を押し直してください。"
                     "実際の値は密度構造に依存するため目標ちょうどにならないこともあります（品質 DBCV を優先）。")
            utils.render_dbcv_help()
            if st.session_state.get('core_hdbscan_auto_result'):
                _ar = st.session_state['core_hdbscan_auto_result']
                _rv = _ar.get('validity')
                _rv_txt = f"・品質DBCV={_rv:.2f}" if isinstance(_rv, (int, float)) else ""
                st.info(
                    f"🤖 前回の自動決定: 最小クラスタサイズ=**{_ar['mcs']}** / 最小サンプル数=**{_ar['ms']}** "
                    f"→ クラスタ数 **{_ar['k']}**・ノイズ {_ar['noise'] * 100:.1f}%（目標≈{_ar['target_k']}{_rv_txt}）。"
                    f"　数が合わない/粒度が好みでない場合は上の「目標クラスタ数」を変えて再生成してください。")

    use_mece = st.checkbox("MECEモード (カテゴリ数を自動決定)", value=True, key="core_use_mece",
        help="ONにすると、各分類軸のカテゴリ数を「モレなくダブりなく(MECE)」最適になるようAIに任せます。OFFにすると、技術・課題・解決手段それぞれのカテゴリ数を手動で指定できます。")
    
    if not use_mece:
        st.markdown("<b>生成する分類の数 (手動設定):</b>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1: ai_cat_count_tech = st.number_input("技術分類:", min_value=1, value=6, key="core_cat_tech",
            help="技術の観点で生成する分類カテゴリの数です。AIへの依頼文に反映され、この個数のカテゴリ案が提案されます。大きいほど細かく分かれます。")
        with col_c2: ai_cat_count_prob = st.number_input("課題分類:", min_value=1, value=6, key="core_cat_prob",
            help="課題（解決したい問題）の観点で生成する分類カテゴリの数です。AIへの依頼文に反映され、この個数のカテゴリ案が提案されます。大きいほど細かく分かれます。")
        with col_c3: ai_cat_count_sol = st.number_input("解決手段分類:", min_value=1, value=6, key="core_cat_sol",
            help="解決手段（手法・構成）の観点で生成する分類カテゴリの数です。AIへの依頼文に反映され、この個数のカテゴリ案が提案されます。大きいほど細かく分かれます。")

    if st.button("② 分類サジェスト用プロンプトを生成", key="core_run_ai", type="primary"):
        with st.spinner("クラスタリング & 代表文献を抽出中..."):
            try:
                texts_raw = df_main[target_column].astype(str).fillna('')
                X, _info2 = core_prepare_features(texts_raw.tolist(), feature_mode, embeddings=_emb)
                _core_si = {}
                _core_auto = (cluster_method == 'hdbscan' and core_hdbscan_auto)
                labels, centers, _actual = core_cluster(
                    X, method=cluster_method, k=int(chosen_k), min_cluster_size=int(min_cluster_size),
                    auto_sweep=_core_auto,
                    target_k=(int(core_target_k) if (_core_auto and core_target_k) else None),
                    sweep_info=_core_si)
                if _core_si:
                    st.session_state['core_hdbscan_auto_result'] = _core_si
                    _rv = _core_si.get('validity')
                    _rv_txt = f"・品質DBCV={_rv:.2f}" if isinstance(_rv, (int, float)) else ""
                    st.info(f"🤖 自動決定: 最小クラスタサイズ={_core_si['mcs']} / 最小サンプル数={_core_si['ms']} "
                            f"→ クラスタ数 {_core_si['k']}・ノイズ {_core_si['noise'] * 100:.1f}%{_rv_txt}")
                cluster_ids = [c for c in sorted(set(int(l) for l in labels)) if c != -1]
                if not cluster_ids:
                    st.error("有効なクラスタが得られませんでした。最小クラスタサイズを下げる、"
                             "またはKMeans（最適k自動）をお試しください。")
                    st.stop()
                if _actual == 'kmeans_fallback':
                    st.warning("HDBSCANで有効なクラスタが得られなかったため、KMeansにフォールバックしました"
                               "（最小クラスタサイズを下げると改善する場合があります）。")
                _noise_n = int((labels == -1).sum())
                sampled_docs = []
                for ci in cluster_ids:
                    reps = core_select_representatives(X, labels, ci, int(ai_n_w), centers=centers)
                    if not reps:
                        continue
                    sampled_docs.append(f"\n--- クラスタ {ci} （{int((labels == ci).sum())}件） ---\n" +
                        "\n".join([f"・{_core_text_preprocessor(texts_raw.iloc[idx])}" for idx in reps]))
                _n_clusters = len(cluster_ids)
                _feat_desc = "SBERT（意味ベクトル）" if feature_mode == 'sbert' else "TF-IDF（語彙）"
                _method_desc = (f"HDBSCAN法で{_n_clusters}個のクラスタに分類（ノイズ{_noise_n}件は萌芽候補として除外）"
                                if _actual == 'hdbscan'
                                else f"KMeans法でk={_n_clusters}個のクラスタに分類")
                
                if use_mece:
                    instruction_text = (
                        "この特許母集団全体を網羅的に分類するための、**「技術分類」「課題分類」「解決手段分類」**の3つの分類軸について、**分類定義**（分類名、定義、CORE論理式のセット）を設計してください。\n"
                        "\n# 重要: MECE (Mutually Exclusive, Collectively Exhaustive) の原則\n"
                        "- 生成する各分類軸内のカテゴリは、相互に排他的（ダブりがない）であり、かつ全体として網羅的（モレがない）であるように設計してください。\n"
                        "- 各軸のカテゴリ数は、MECEを満たすのに最適だとあなたが判断する数（目安として5〜10個程度）にしてください。"
                    )
                else:
                    instruction_text = "\n".join([
                        "この特許母集団全体を網羅的に分類するための、以下の3つの分類軸について、指定された個数で**分類定義**を設計してください。",
                        f"- **技術分類**: {ai_cat_count_tech}個",
                        f"- **課題分類**: {ai_cat_count_prob}個",
                        f"- **解決手段分類**: {ai_cat_count_sol}個"
                    ])

                sampled_docs_str = "".join(sampled_docs)

                prompt = f"""
あなたは優秀な特許情報ストラテジストです。
以下の「代表文献サンプル」は、ある特許母集団（{len(df_main)}件）を{_feat_desc}特徴量を用いて{_method_desc}し、各クラスタから代表的な文献の「{target_column}」を{ai_n_w}件ずつ抽出したものです。

# 依頼内容
{instruction_text}

以下の形式の **JSONデータのみ** を出力してください。解説は不要です。
JSONをコピーしてシステムにそのままインポートします。

# JSONフォーマット (厳守)
{{
  "技術分類": [
    {{
      "name": "カテゴリ名 (例: CO2分離膜)",
      "definition": "カテゴリの定義...",
      "rule": "CORE論理式 (例: (CO2 + 二酸化炭素) * (膜 + メンブレン))"
    }},
    ...
  ],
  "課題分類": [ ... ],
  "解決手段分類": [ ... ]
}}

# CORE論理式文法 (厳守)
- `A + B` (OR): A または B
- `A * B` (AND): A かつ B (順序問わず)
- `()`: 括弧を使って優先順位を制御できます。入れ子も可能です。
    - 例: `(水素 * (吸蔵 + 貯蔵) * (合金 + 材料))`
    - 例: `(A * B) + (C * D)`
- `A nearN B` (近傍): AとBがN文字以内 (順序不問)。
- `A adjN B` (順序指定): A→BがN文字以内。
- **重要:** `near` や `adj` の条件の内部には `*` (AND) を含めることはできません（`+` (OR) は可能）。
    - OK: `(A + B) near10 C`
    - NG: `(A * B) near10 C`

# 最重要ルール (キーワード拡張と表記ゆれ)
- サンプルに存在するキーワードをそのまま使うだけでは不十分です。
- AIの知識を活用し、そのキーワードの**類義語、関連語、上位/下位概念、特許特有の表現、表記ゆれ（カタカナ、ひらがな、漢字）**を、あなたの知識ベースから網羅的に想起してください。
- **特許用語の網羅:** （例: 「保持」→「担持」「固着」「係止」など、特許で使われる言い換えを網羅）
- **概念の階層化:** 上位概念（例: 「車両」）と下位概念（例: 「自動車」「二輪車」）の両方を含め、取りこぼしを防ぎます。
- **カタカナ:** キーワードにカタカナを使用する場合は、**必ず全角（例: `ポリマー`）**を使用し、**半角（例: `ﾎﾟﾘﾏｰ`）は絶対に使用しないでください**。

# 代表文献サンプル
{sampled_docs_str}
"""
                st.success("プロンプトを生成しました。右上のコピーボタンでコピーしてください。")
                st.code(prompt, language='markdown')
            except Exception as e: st.error(f"エラー: {e}")

# --- フェーズ 2: 分類ルール定義 ---
elif current_phase.startswith("フェーズ 2"):
    st.subheader("フェーズ 2: 分類ルール定義")
    
    tab_manual, tab_json = st.tabs(["手動追加・修正", "JSON一括インポート"])
    existing = list(st.session_state.core_classification_rules.keys())
    
    with tab_manual:
        is_edit_mode = "core_edit_target" in st.session_state and st.session_state.core_edit_target is not None
        
        mode = st.radio("軸の指定:", ["新規作成", "既存に追加"], horizontal=True, index=1 if is_edit_mode else 0,
            help="このルールをどの分類軸に登録するかを決めます。新規作成＝新しい分類軸（例: 課題分類）を作ってそこに登録、既存に追加＝すでに作成済みの分類軸へカテゴリを追加します。")
        
        if mode == "既存に追加" and existing:
            default_idx = 0
            if is_edit_mode:
                try: default_idx = existing.index(st.session_state.core_edit_target["axis"])
                except: pass
            elif st.session_state.core_current_axis in existing:
                try: default_idx = existing.index(st.session_state.core_current_axis)
                except: pass
            axis = st.selectbox("追加/修正先の軸:", existing, index=default_idx)
        else:
            axis = st.text_input("新規軸名:", value=st.session_state.core_edit_target["axis"] if is_edit_mode else "", placeholder="例: 課題分類")
            
        c_name = st.text_input("分類名:", value=st.session_state.core_edit_target["cat"] if is_edit_mode else "", placeholder="例: 耐久性向上")
        c_def = st.text_area("定義:", value=st.session_state.core_edit_target["def"] if is_edit_mode else "", height=68)
        c_rule = st.text_input("論理式:", value=st.session_state.core_edit_target["rule"] if is_edit_mode else "", placeholder="(耐久性 + 寿命) * 向上")
        
        btn_label = "ルールを更新" if is_edit_mode else "ルールを追加"
        
        if st.button(btn_label, key="add_manual"):
            if all([axis, c_name, c_rule]):
                try:
                    parse_core_rule(c_rule)
                    if axis not in st.session_state.core_classification_rules:
                        st.session_state.core_classification_rules[axis] = {}
                    st.session_state.core_classification_rules[axis][c_name] = {'rule': c_rule, 'definition': c_def}
                    st.session_state.core_current_axis = axis
                    if is_edit_mode: del st.session_state.core_edit_target
                    st.success(f"{btn_label}しました: {c_name}")
                    st.rerun()
                except Exception as e: st.error(f"文法エラー: {e}")
        
        if is_edit_mode:
            if st.button("編集をキャンセル"):
                del st.session_state.core_edit_target
                st.rerun()
    
    with tab_json:
        st.markdown("AIが生成したJSONをここに貼り付けてください。既存のルールは維持され、新しい軸が追加されます。")
        json_input = st.text_area("JSON入力:", height=300)
        if st.button("JSONを一括インポート"):
            try:
                cleaned_json = re.sub(r'^```json\s*|\s*```$', '', json_input.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned_json)
                count = 0
                for axis_name, categories in data.items():
                    if axis_name not in st.session_state.core_classification_rules:
                        st.session_state.core_classification_rules[axis_name] = {}
                    for cat in categories:
                        name = cat.get('name'); rule = cat.get('rule'); defn = cat.get('definition', '')
                        if name and rule:
                            st.session_state.core_classification_rules[axis_name][name] = {'rule': rule, 'definition': defn}
                            count += 1
                st.success(f"{count} 個のルールをインポートしました！")
                st.rerun()
            except Exception as e: st.error(f"JSONパースエラー: {e}")

    st.markdown("---")
    st.subheader("現在のルール一覧")
    
    if st.button("全ルールを削除", type="primary"):
        st.session_state.core_classification_rules = {}
        st.rerun()
        
    for ax, cats in st.session_state.core_classification_rules.items():
        with st.expander(f"軸: {ax} ({len(cats)}件)"):
            for cn, cd in cats.items():
                r = cd['rule'] if isinstance(cd, dict) else cd[0]
                d = cd.get('definition', '') if isinstance(cd, dict) else ""
                
                c1, c2, c3 = st.columns([1, 4, 1])
                with c1:
                    if st.button("編集", key=f"edit_{ax}_{cn}"):
                        st.session_state.core_edit_target = {"axis": ax, "cat": cn, "rule": r, "def": d}
                        st.rerun()
                with c2:
                    st.text(f"【{cn}】 {r}")
                with c3:
                    if st.button("削除", key=f"del_{ax}_{cn}"):
                        del st.session_state.core_classification_rules[ax][cn]
                        if not st.session_state.core_classification_rules[ax]:
                            del st.session_state.core_classification_rules[ax]
                        st.rerun()

# --- フェーズ 3: 分類実行 ---
elif current_phase.startswith("フェーズ 3"):
    st.subheader("フェーズ 3: 分類実行")
    
    st.info("※ 探索範囲は自動的に「発明の名称 + 要約 + 請求項」の結合テキストとなります。")
    
    if st.button("すべての分類を実行", type="primary"):
        if not st.session_state.core_classification_rules:
            st.error("ルールがありません。")
        else:
            with st.spinner("実行中..."):
                try:
                    df_res = df_main.copy()
                    
                    search_cols = []
                    if col_map.get('title') in df_res.columns: search_cols.append(df_res[col_map['title']].fillna(''))
                    if col_map.get('abstract') in df_res.columns: search_cols.append(df_res[col_map['abstract']].fillna(''))
                    if col_map.get('claim') in df_res.columns: search_cols.append(df_res[col_map['claim']].fillna(''))
                    
                    if not search_cols:
                        st.error("検索対象のテキスト列（発明の名称／要約／請求項）が割り当てられていません。Mission Control でカラムマッピングを確認してください。")
                        st.stop()
                    combined_text = search_cols[0]
                    for s in search_cols[1:]:
                        combined_text = combined_text + " " + s
                    
                    rules = st.session_state.core_classification_rules
                    compiled_rule_nodes = {}
                    for ax, cats in rules.items():
                        compiled_rule_nodes[ax] = []
                        for cn, cd in cats.items():
                            r_str = cd['rule'] if isinstance(cd, dict) else cd[0]
                            # Try parse
                            node = parse_core_rule(r_str)
                            if node:
                                compiled_rule_nodes[ax].append((cn, node))
                            else:
                                st.warning(f"Failed to parse rule for {cn}: {r_str}")

                    def apply_rules(text, ax_nodes):
                        text = _core_text_preprocessor(str(text))
                        hits = []
                        for c_name, node in ax_nodes:
                            if node.evaluate(text):
                                hits.append(c_name)
                        return ";".join(hits) if hits else "その他"

                    bar = st.progress(0)
                    for i, ax in enumerate(rules.keys()):
                        df_res[ax] = combined_text.apply(lambda x: apply_rules(x, compiled_rule_nodes[ax]))
                        bar.progress((i+1)/len(rules))
                    
                    st.session_state.core_df_classified = df_res
                    st.success("完了！")
                    
                    st.subheader("分類結果サマリー")
                    cols = st.columns(len(rules))
                    for i, ax in enumerate(rules.keys()):
                        with cols[i]:
                            st.markdown(f"**{ax}**")
                            counts = df_res[ax].str.split(';').explode().value_counts()
                            st.dataframe(counts)

                    csv_core = convert_df_to_csv_core(df_res)
                    st.download_button("分類結果CSVをダウンロード", csv_core, "CORE_classified.csv", "text/csv")

                    # CAPCOM: patents.csv更新（CORE分類列を追加）
                    try:
                        import capcom
                        if capcom.is_active():
                            capcom.save_patents_csv()
                    except Exception as e:
                        st.caption(f"⚠️ WARN 分類結果(patents.csv) の CAPCOM 保存に失敗しました（要確認）: {e}")

                except Exception as e: st.error(f"エラー: {e}")

    st.markdown("---")
    st.subheader("🔍 未分類データの再分析 (『その他』を減らす)")
    if st.session_state.core_df_classified is not None:
        rules = st.session_state.core_classification_rules
        if rules:
            col_re1, col_re2 = st.columns(2)
            with col_re1: reanalyze_axis = st.selectbox("再分析する軸を選択:", list(rules.keys()), key="core_reanalyze_axis")
            
            # 特徴量・クラスタ手法（Phase1 と同じ選択肢を提供）
            _emb_re = st.session_state.get('sbert_embeddings')
            _emb_re_ok = _emb_re is not None and len(_emb_re) == len(df_main)
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                _rfeat_opts = ["TF-IDF（語彙）"] + (["SBERT（意味）"] if _emb_re_ok else [])
                re_feature_label = st.radio("特徴量:", _rfeat_opts, key="re_feature_mode",
                    help="クラスタリングに使う特徴ベクトルの種類です。TF-IDF＝単語の出現頻度ベース（語の一致を重視）、SBERT＝文章の意味ベクトル（言い回しが違っても意味が近ければ近接）。意味的なまとまりはSBERT、語の一致重視はTF-IDFが得意です。")
                re_feature_mode = 'sbert' if re_feature_label.startswith("SBERT") else 'tfidf'
            with rc2:
                re_method_label = st.radio("クラスタ手法:",
                    ["KMeans（k自動）", "KMeans（k手動）", "HDBSCAN（k不要）"], key="re_cluster_method",
                    help="KMeans＝指定した数(k)に必ず分割し全件どこかに所属、HDBSCAN＝密度ベースで数を自動決定し外れ値はノイズとして分離。KMeans（k自動）はスコア最良のkを自動選択、KMeans（k手動）はkを自分で指定します。")
            with rc3:
                re_n = st.number_input("1トピックあたりのサンプル数 (N)", min_value=1, value=3, key="re_n",
                    help="各トピック（『その他』グループ）から代表として抽出する特許の件数です。多いほど内容を確認しやすくなります。")
            # 手法別パラメータ
            if re_method_label.startswith("KMeans（k手動"):
                re_k = st.number_input("抽出トピック数 (K)", min_value=2, value=5, key="re_k",
                    help="『その他』から抽出する技術トピックの数(K)です。大きいほど細かく分かれます。")
                re_mcs = 15
            elif re_method_label.startswith("HDBSCAN"):
                re_k = None
                re_mcs = st.number_input("最小クラスタサイズ (HDBSCAN)", min_value=2, value=5, key="re_hdbscan_mcs",
                    help="『その他』は少数のことが多いので小さめが目安。未満のまとまりはノイズ扱い。")
            else:  # k自動
                re_k = None
                re_mcs = 15

            re_mece = st.checkbox("MECEモード (自動)", value=True, key="re_mece",
                help="ONにすると、追加する新カテゴリの数を「モレなくダブりなく(MECE)」最適になるようAIに任せます。OFFにすると、追加するカテゴリ数を手動で指定できます。")
            re_cnt = 3 if re_mece else st.number_input("追加するカテゴリ数", value=3, key="re_cnt",
                help="『その他』を分析して新たに提案させる分類カテゴリの数です。AIへの依頼文に反映され、この個数の新カテゴリ案が提案されます。")

            if st.button("『その他』を分析して新ルールを提案", key="core_btn_reanalyze"):
                try:
                    df_c = st.session_state.core_df_classified
                    others_df = df_c[df_c[reanalyze_axis] == 'その他']
                    if others_df.empty:
                        st.info("『その他』はありません。")
                    else:
                        with st.spinner(f"『その他』({len(others_df)}件) を分析中..."):
                            search_cols = []
                            if col_map.get('title') in others_df.columns: search_cols.append(others_df[col_map['title']].fillna(''))
                            if col_map.get('abstract') in others_df.columns: search_cols.append(others_df[col_map['abstract']].fillna(''))
                            if col_map.get('claim') in others_df.columns: search_cols.append(others_df[col_map['claim']].fillna(''))
                            if not search_cols:
                                st.error("検索対象のテキスト列が割り当てられていません。Mission Control でカラムマッピングを確認してください。")
                                st.stop()
                            texts = search_cols[0]
                            for s in search_cols[1:]: texts = texts + " " + s
                            
                            # 特徴量を構築（SBERT 選択時は others_df に対応する埋め込みを抽出）
                            _emb_sub = None
                            if re_feature_mode == 'sbert' and _emb_re_ok:
                                _pos = df_main.index.get_indexer(others_df.index)
                                if (_pos >= 0).all() and len(_pos) == len(others_df):
                                    _emb_sub = np.asarray(_emb_re)[_pos]
                            _eff_feat = 'sbert' if _emb_sub is not None else 'tfidf'
                            _Xr, _ = core_prepare_features(texts.tolist(), _eff_feat, embeddings=_emb_sub)

                            s_docs = []
                            _labels_r, _centers_r = None, None
                            if re_method_label.startswith("HDBSCAN"):
                                _labels_r, _centers_r, _act_r = core_cluster(_Xr, method='hdbscan', min_cluster_size=int(re_mcs))
                                if _act_r == 'kmeans_fallback':
                                    st.warning("HDBSCANで有効クラスタが得られず、KMeansにフォールバックしました（最小クラスタサイズを下げてください）。")
                            else:
                                if re_k is None:  # k自動
                                    _kmax = min(15, max(2, len(others_df) - 1))
                                    _sc_r, _rk_r = core_find_optimal_k(_Xr, k_min=2, k_max=_kmax)
                                    _use_k = int(_rk_r)
                                    st.caption(f"自動選択クラスタ数 k = {_use_k}（silhouette最大）")
                                else:  # k手動
                                    _use_k = min(int(re_k), len(others_df))
                                if _use_k < 2:
                                    reps = list(range(min(int(re_n), len(texts))))
                                    s_docs.append("\n--- その他グループ 0 ---\n" + "\n".join(
                                        [f"・{_core_text_preprocessor(texts.iloc[idx])}" for idx in reps]))
                                else:
                                    _labels_r, _centers_r, _ = core_cluster(_Xr, method='kmeans', k=_use_k)
                            if _labels_r is not None:
                                for i in sorted(set(int(l) for l in _labels_r)):
                                    if i == -1:
                                        continue
                                    reps = core_select_representatives(_Xr, _labels_r, i, int(re_n), centers=_centers_r)
                                    if not reps:
                                        continue
                                    s_docs.append(f"\n--- その他グループ {i} （{int((_labels_r == i).sum())}件） ---\n" + "\n".join(
                                        [f"・{_core_text_preprocessor(texts.iloc[idx])}" for idx in reps]))
                            
                            s_docs_str = "".join(s_docs)
                            exist_rules = [f"- {cat}: {d['rule']}" for cat, d in rules[reanalyze_axis].items()]
                            exist_rules_str = "\n".join(exist_rules)
                            
                            instruction_part = "MECEを意識し、カテゴリ数は自動で最適化してください。" if re_mece else f"**{re_cnt}個** の新しいカテゴリを追加してください。"
                            
                            p_re = f"""
あなたは特許情報ストラテジストです。
現在、分類軸「{reanalyze_axis}」を作成中ですが、以下の「既存の分類」に当てはまらない特許が「その他」として残っています。

# 既存の分類リスト
{exist_rules_str}

# 依頼内容
以下の「未分類特許のサンプル」を分析し、**既存の分類とは概念的に重複しない、新しい分類カテゴリ**を提案してください。
出力は **JSON形式のみ** としてください。
{instruction_part}

# JSONフォーマット
{{
  "{reanalyze_axis}": [
    {{
      "name": "新カテゴリ名",
      "definition": "...",
      "rule": "論理式 (Allowed: `(A * B) + C`, `(A+B) near10 C` etc)"
    }}, ...
  ]
}}

# 未分類特許のサンプル
{s_docs_str}
"""
                            st.session_state.core_reanalyze_result = p_re
                except Exception as e: st.error(f"エラー: {e}")

        if st.session_state.core_reanalyze_result:
            st.success("再分析プロンプトを生成しました。"); st.code(st.session_state.core_reanalyze_result, language='markdown')

# --- フェーズ 4: 特許マップ ---
elif current_phase.startswith("フェーズ 4"):
    st.subheader("フェーズ 4: 特許マップ作成")
    
    if st.session_state.core_df_classified is None:
        st.warning("先に分類を実行してください。")
    else:
        df_c = st.session_state.core_df_classified
        axes = list(st.session_state.core_classification_rules.keys())
        meta_axes = []
        if 'year' in df_c.columns: meta_axes.append('出願年')
        if col_map.get('applicant') in df_c.columns: meta_axes.append('出願人')
        all_axes = axes + meta_axes
        
        c1, c2, c3 = st.columns(3)
        with c1: x_ax = st.selectbox("X軸", all_axes, index=0,
            help="クロス集計マトリクスの横軸に置く項目です。分類軸（技術分類・課題分類など）のほか、出願年・出願人も選べます。X軸とY軸の組み合わせ（例: 技術×課題）で見たい切り口を決めます。")
        with c2: y_ax = st.selectbox("Y軸", all_axes, index=min(1, len(all_axes)-1),
            help="クロス集計マトリクスの縦軸に置く項目です。分類軸（技術分類・課題分類など）のほか、出願年・出願人も選べます。X軸とY軸の組み合わせ（例: 技術×課題）で見たい切り口を決めます。")
        with c3: chart_type = st.radio("グラフタイプ", ["ヒートマップ", "バブルチャート"],
            help="クロス集計の表示形式を切り替えます。ヒートマップ＝各セルの件数を色の濃淡と数値で表示（全体の濃淡を一覧しやすい）、バブルチャート＝件数を円の大きさで表示（多い組み合わせが直感的にわかる）。集計値は同じで見せ方だけが変わります。")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1: exclude_other = st.checkbox("「その他」を除外する", value=True,
            help="どのルールにも当てはまらなかった「その他」の特許をマトリクスから除くかどうかです。ONにすると分類済みの構造が見やすくなり、OFFにすると未分類がどれだけ残っているかも確認できます。")
        
        if st.button("描画 (分析実行)", type="primary"):
            st.session_state.core_phase4_run = True

        if st.session_state.get("core_phase4_run"):
            st.markdown("---")
            
            # --- 1. グローバルな軸の順序を決定 (母集団全体で固定) ---
            # 出願人カラムの区切り文字（Home.py フェーズ2 のユーザー設定。既定は ;）
            _app_delim = st.session_state.get('delimiters', {}).get('applicant', ';')
            def get_col_data_global(target_df, ax_name):
                if ax_name == '出願年': return target_df['year'].fillna(0).astype(int).astype(str), None
                if ax_name == '出願人': return target_df[col_map['applicant']].fillna('Unknown'), _app_delim
                if ax_name in axes: return target_df[ax_name], ';'
                return None, None

            # 全体データでクロス集計して軸順序を決定
            x_data_g, x_sep_g = get_col_data_global(df_c, x_ax)
            y_data_g, y_sep_g = get_col_data_global(df_c, y_ax)
            temp_df_g = pd.DataFrame({'X': x_data_g, 'Y': y_data_g})
            
            if x_sep_g: temp_df_g['X'] = temp_df_g['X'].astype(str).str.split(x_sep_g); temp_df_g = temp_df_g.explode('X')
            if y_sep_g: temp_df_g['Y'] = temp_df_g['Y'].astype(str).str.split(y_sep_g); temp_df_g = temp_df_g.explode('Y')
            
            temp_df_g = temp_df_g.replace({'nan': np.nan, 'None': np.nan}).dropna()
            if exclude_other:
                temp_df_g = temp_df_g[(temp_df_g['X'] != 'その他') & (temp_df_g['Y'] != 'その他')]
            
            # --- 軸が出願人の場合のフィルタリング (Top N / 手動) ---
            if '出願人' in [x_ax, y_ax]:
                # 全データの出願人頻度を計算
                app_s_all = df_c[col_map['applicant']].fillna('Unknown').astype(str).str.split(_app_delim)
                app_counts_all = app_s_all.explode().str.strip().value_counts()
                
                st.markdown("##### 👥 出願人軸の表示設定")
                app_filter_mode = st.radio("表示モード:", ["上位指定 (Top N)", "手動選択 (Manual)"], horizontal=True, key="core_app_axis_mode",
                    help="出願人軸に表示する企業の選び方です。上位指定＝出願件数の多い順に上位N社を自動表示、手動選択＝表示したい出願人を自分で選びます。表示対象が変わるだけで件数の集計方法は変わりません。")
                
                target_apps_set = set()
                if app_filter_mode == "上位指定 (Top N)":
                    top_n_val = st.number_input("表示件数 (上位N社):", min_value=5, max_value=200, value=10, step=5, key="core_app_axis_n",
                        help="出願人軸に上位何社まで表示するかです（表示数のみ変わり、分析結果は変わりません）。")
                    target_apps_set = set(app_counts_all.head(top_n_val).index)
                    st.info(f"上位 {top_n_val} 社を表示します（全 {len(app_counts_all)} 社中）")
                else:
                    target_apps_set = set(st.multiselect("表示する出願人を選択:", app_counts_all.index.tolist(), default=app_counts_all.head(10).index.tolist(), key="core_app_axis_manual"))
                
                # フィルタリング適用
                if x_ax == '出願人':
                    temp_df_g = temp_df_g[temp_df_g['X'].isin(target_apps_set)]
                if y_ax == '出願人':
                    temp_df_g = temp_df_g[temp_df_g['Y'].isin(target_apps_set)]
            
            if temp_df_g.empty:
                st.warning("有効なデータがありません。")
            else:
                ct_g = pd.crosstab(temp_df_g['Y'], temp_df_g['X'])
                
                # Sorting Global
                if x_ax == '出願年': x_ord_global = sorted(ct_g.columns, key=lambda x: int(x) if x.isdigit() else x)
                else: x_ord_global = ct_g.sum(axis=0).sort_values(ascending=False).index.tolist()

                if y_ax == '出願年': y_ord_global = sorted(ct_g.index, key=lambda x: int(x) if x.isdigit() else x)
                else: y_ord_global = ct_g.sum(axis=1).sort_values(ascending=False).index.tolist()

                # 分類軸は、件数0でクロス集計に現れない「定義済みカテゴリ」も漏れなく表示する。
                # （X軸とY軸でカテゴリ数が異なっても、0件カテゴリが脱落しないようにする。）
                if x_ax in st.session_state.core_classification_rules:
                    for _c in st.session_state.core_classification_rules[x_ax].keys():
                        if _c not in x_ord_global:
                            x_ord_global.append(_c)
                if y_ax in st.session_state.core_classification_rules:
                    for _c in st.session_state.core_classification_rules[y_ax].keys():
                        if _c not in y_ord_global:
                            y_ord_global.append(_c)

                # --- Sorting Adjustment: Force 'Others' to the end ---
                if 'その他' in x_ord_global:
                    x_ord_global.remove('その他')
                    x_ord_global.append('その他')
                
                if 'その他' in y_ord_global:
                    y_ord_global.remove('その他')
                    y_ord_global.append('その他')

                # --- 2. 出願人選択 (プルダウン) ---
                target_applicant_options = ["全体 (Overall)"]
                app_name_map = {} # label -> app_name
                
                if col_map.get('applicant') in df_c.columns:
                    app_s = df_c[col_map['applicant']].fillna('Unknown').astype(str).str.split(_app_delim)
                    app_exploded = app_s.explode().str.strip()
                    top_apps = app_exploded.value_counts() # All applicants
                    
                    for app_name, count in top_apps.items():
                        if app_name and app_name != 'nan':
                            label = f"{app_name} ({count})"
                            target_applicant_options.append(label)
                            app_name_map[label] = app_name
                
                selected_app_label = st.selectbox("出願人で絞り込み (Focus Applicant):", target_applicant_options)

                # --- 3. データフィルタリング ---
                if selected_app_label == "全体 (Overall)":
                    df_target = df_c
                else:
                    target_app_name = app_name_map[selected_app_label]
                    mask = df_c[col_map['applicant']].fillna('').astype(str).apply(lambda x: target_app_name in [s.strip() for s in x.split(_app_delim)])
                    df_target = df_c[mask]
                
                st.markdown(f"**分析対象: {selected_app_label}**")

                # --- クリック→該当特許群ポップアップ用 resolver ---
                # クロス集計セル/バブルのクリック点 (x=X軸カテゴリ, y=Y軸カテゴリ) から
                # 描画対象 df をフィルタし、該当特許の (ラベル) インデックス群を返す。
                # 分類軸は ';' split で部分一致（複数分類対応）。出願人は区切り文字で分割。出願年は year 一致。
                def _core_axis_cat_mask(target_df, ax_name, cat):
                    """指定軸 ax_name のカテゴリ cat に該当する行の boolean マスクを返す。"""
                    cat = str(cat).strip()
                    if ax_name == '出願年':
                        if 'year' not in target_df.columns:
                            return pd.Series(False, index=target_df.index)
                        # 軸カテゴリは get_col_data_global と同じ fillna(0).astype(int).astype(str) で生成される
                        year_str = target_df['year'].fillna(0).astype(int).astype(str)
                        return year_str == cat
                    if ax_name == '出願人':
                        col = col_map.get('applicant')
                        if not col or col not in target_df.columns:
                            return pd.Series(False, index=target_df.index)
                        return target_df[col].fillna('Unknown').astype(str).apply(
                            lambda v: cat in [p.strip() for p in v.split(_app_delim)]
                        )
                    if ax_name in axes and ax_name in target_df.columns:
                        # 分類軸: ';' 連結の複数分類に対する部分一致
                        return target_df[ax_name].fillna('').astype(str).apply(
                            lambda v: cat in [p.strip() for p in v.split(';')]
                        )
                    return pd.Series(False, index=target_df.index)

                def _core_handle_chart_click(selection, sub_df, wrapper_key, chart_state_key):
                    def _resolver(point):
                        x_cat = point.get('x')
                        y_cat = point.get('y')
                        if x_cat is None or y_cat is None:
                            return None, None
                        mask_x = _core_axis_cat_mask(sub_df, x_ax, x_cat)
                        mask_y = _core_axis_cat_mask(sub_df, y_ax, y_cat)
                        matched = sub_df[mask_x & mask_y]
                        indices = matched.index.tolist()
                        dlg_title = f"{x_ax}: {x_cat} × {y_ax}: {y_cat}（{len(indices)} 件）"
                        return indices, dlg_title
                    utils.handle_chart_click_to_records(
                        selection, chart_state_key, _resolver, title="該当特許"
                    )

                # --- 4. 描画関数 (Global Axisを適用) ---
                def render_core_chart(sub_df, wrapper_key):
                    # Local Data Prep
                    x_d, x_s = get_col_data_global(sub_df, x_ax)
                    y_d, y_s = get_col_data_global(sub_df, y_ax)
                    t_df = pd.DataFrame({'X': x_d, 'Y': y_d})
                    if x_s: t_df['X'] = t_df['X'].astype(str).str.split(x_s); t_df = t_df.explode('X')
                    if y_s: t_df['Y'] = t_df['Y'].astype(str).str.split(y_s); t_df = t_df.explode('Y')
                    
                    t_df = t_df.replace({'nan': np.nan, 'None': np.nan}).dropna()
                    if exclude_other:
                        t_df = t_df[(t_df['X'] != 'その他') & (t_df['Y'] != 'その他')]
                    
                    # Create Crosstab
                    ct_local = pd.crosstab(t_df['Y'], t_df['X'])
                    
                    # Reindex with Global Orders (Forces matrix structure)
                    ct_final = ct_local.reindex(index=y_ord_global, columns=x_ord_global).fillna(0)

                    # チャートキーはチャート種別ごとに分ける（固定）。ヒートマップとバブルは図構造が
                    # 全く異なるため、同じキーを使い回すと切替時に古い選択状態が残ってクリックが
                    # 効かなくなる。クリックのたびにキーを変えるのは Streamlit の "setIn" エラーに
                    # なるため不可。種別ごとの静的キーが正解。
                    _chart_slug = 'hm' if chart_type == "ヒートマップ" else 'bub'
                    _chart_key = f"core_chart_{wrapper_key}_{_chart_slug}"

                    if chart_type == "ヒートマップ":
                        # 元の px.imshow の見た目（全幅・塗りつぶし・矩形セル）を維持しつつ、
                        # 透明なクリック用 scatter を重ねてセルクリック→ポップアップを実現する。
                        # hovermode='closest' + hoverdistance=-1 により、セル内のどこをクリックしても
                        # 最寄りセルが選択される（透明マーカーは位置ベースで検出されるため不可視でも可）。
                        fig = px.imshow(
                            ct_final,
                            labels=dict(x=x_ax, y=y_ax, color="件数"),
                            x=ct_final.columns, y=ct_final.index,
                            aspect="auto", color_continuous_scale='YlGnBu', text_auto=True)
                        # マス間の白線（グリッド）は付けない。塗りつぶし側のホバーは切り、クリックは透明層に任せる
                        fig.update_traces(xgap=0, ygap=0, hoverinfo='skip', selector=dict(type='heatmap'))
                        # 透明クリックレイヤ（各セル中心にマーカー）
                        _ox, _oy, _oc = [], [], []
                        for _yi in ct_final.index:
                            for _xi in ct_final.columns:
                                _ox.append(_xi); _oy.append(_yi); _oc.append(int(ct_final.loc[_yi, _xi]))
                        fig.add_trace(go.Scatter(
                            x=_ox, y=_oy, mode='markers',
                            marker=dict(symbol='square', size=40, color='rgba(0,0,0,0)'),
                            customdata=_oc, showlegend=False, name='',
                            hovertemplate=f"{x_ax}: %{{x}}<br>{y_ax}: %{{y}}<br>件数: %{{customdata}}<extra></extra>",
                            selected=dict(marker=dict(opacity=0)),
                            unselected=dict(marker=dict(opacity=0)),
                        ))
                        fig.update_layout(
                            height=max(600, len(ct_final) * 40),
                            yaxis=dict(title=y_ax, showgrid=False, zeroline=False),
                            xaxis=dict(title=x_ax, side='bottom', showgrid=False, zeroline=False),
                            plot_bgcolor='white',
                            hovermode='closest', hoverdistance=-1,
                        )
                        st.caption("セルをクリックすると、該当する特許がポップアップ表示されます。")
                        @st.fragment
                        def _core_click_heatmap():
                            selection = st.plotly_chart(fig, use_container_width=True, config={'editable': False}, key=_chart_key, on_select="rerun", selection_mode="points")
                            _core_handle_chart_click(selection, sub_df, wrapper_key, _chart_key)
                        _core_click_heatmap()

                    else: # バブルチャート
                        ct_long = ct_final.reset_index().melt(id_vars='Y', var_name='X', value_name='Count')
                        ct_long = ct_long[ct_long['Count'] > 0] 
                        
                        atlas_colors = utils.APOLLO_COLORS
                        
                        fig = px.scatter(
                            ct_long, x='X', y='Y', size='Count', color='Y',
                            size_max=40, color_discrete_sequence=atlas_colors,
                            category_orders={'X': x_ord_global, 'Y': y_ord_global}
                        )
                        # 行間に余裕を持たせバブルの重なり（詰まり）を解消、凡例は不要（Y軸で判別可）、グリッド線も除去
                        fig.update_layout(
                            height=max(650, len(y_ord_global) * 78),
                            showlegend=False,
                            plot_bgcolor='white',
                            hovermode='closest',
                        )

                        # Y軸: Plotlyの散布図は通常下から上だが、行列形式（上から下）にするために範囲を明示。
                        # 行・列を読み取りやすいよう薄いグリッド線を表示する（バブルがどのセルかを判別しやすくする）。
                        fig.update_yaxes(
                            categoryorder='array',
                            categoryarray=y_ord_global,
                            title=y_ax,
                            type='category',
                            showgrid=True, gridcolor='#e3e6ea', gridwidth=1, zeroline=False,
                            range=[len(y_ord_global) - 0.5, -0.5] # 上から下の行列形式
                        )
                        fig.update_xaxes(
                            categoryorder='array',
                            categoryarray=x_ord_global,
                            title=x_ax,
                            side='bottom',
                            type='category',
                            showgrid=True, gridcolor='#e3e6ea', gridwidth=1, zeroline=False,
                            range=[-0.5, len(x_ord_global) - 0.5] # Ensure full width
                        )
                        @st.fragment
                        def _core_click_bubble():
                            utils.disable_selection_fade(fig)
                            selection = st.plotly_chart(fig, use_container_width=True, config={'editable': False}, key=_chart_key, on_select="rerun", selection_mode="points")
                            _core_handle_chart_click(selection, sub_df, wrapper_key, _chart_key)
                        _core_click_bubble()


                    # --- Snapshot (Dynamic) ---
                    # AI/レポート向けマトリクスを markdown テーブルで生成（テーブルとして読みやすくする）。
                    # 大規模時は件数上位 20×20 に絞ってトークンを抑制。
                    if ct_final.shape[0] > 20 or ct_final.shape[1] > 20:
                        top_rows = ct_final.sum(axis=1).sort_values(ascending=False).head(20).index
                        top_cols = ct_final.sum(axis=0).sort_values(ascending=False).head(20).index
                        _ct_ai = ct_final.loc[top_rows, top_cols]
                        note = f"(件数上位20×20に圧縮 / 元サイズ {ct_final.shape})"
                    else:
                        _ct_ai = ct_final
                        note = ""
                    # 行=Y軸カテゴリ、列=X軸カテゴリ、セル=件数 の markdown テーブル
                    matrix_md = utils_ai.df_to_markdown(_ct_ai.astype(int), index=True, index_label=y_ax)

                    snap_data = {
                        'module': 'CORE',
                        'type': 'matrix',
                        'axes': {'x': x_ax, 'y': y_ax},
                        'matrix_data': matrix_md,
                        'matrix_context': f"Analysis Target: {selected_app_label}",
                        'stats': {
                            'total_count': int(ct_final.sum().sum()),
                            'max_value': int(ct_final.max().max())
                        },
                        'chart_data': f"クロス集計（{y_ax} × {x_ax}） 対象: {selected_app_label}{(' ' + note) if note else ''}\n{matrix_md}"
                    }

                    utils.render_snapshot_button(
                        title=f"CORE Map ({selected_app_label}): {x_ax} vs {y_ax}",
                        description=f"COREマトリクス分析: {x_ax} (X) × {y_ax} (Y)。対象: {selected_app_label}。",
                        key=f"core_snap_{x_ax}_{y_ax}_{selected_app_label}",
                        fig=fig,
                        data_summary=snap_data
                    )

                    # --- AI Insight: 分類結果の戦略的解釈 ---
                    _meta_core = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_c, col_map=col_map)
                    _meta_core['分析対象'] = selected_app_label
                    _meta_core['X軸'] = x_ax
                    _meta_core['Y軸'] = y_ax
                    _meta_core['グラフタイプ'] = chart_type
                    _meta_core['マトリクスサイズ'] = f"{ct_final.shape[0]}行 × {ct_final.shape[1]}列"
                    _meta_core['合計件数'] = int(ct_final.sum().sum())
                    # 分類ルール・軸別件数は AI が読みやすいよう markdown で別掲する
                    # （メタデータに dict で入れると json.dumps された塊が延々と1段落に繋がって読みにくいため）。
                    _extra_lines = ["# 分類ルール（CORE論理式）"]
                    for _ax_name, _ax_rules in st.session_state.core_classification_rules.items():
                        _extra_lines.append(f"\n## {_ax_name}")
                        for _cat, _d in _ax_rules.items():
                            _rule = (_d.get('rule', '') if isinstance(_d, dict) else str(_d)).replace('\n', ' ').strip()
                            _extra_lines.append(f"- **{_cat}**: `{_rule}`")
                    # 軸別の合計件数（マトリクスの行/列マージン）を表で
                    for _ax_name in [y_ax, x_ax]:
                        if _ax_name in st.session_state.core_classification_rules:
                            _df_sum = (ct_final.sum(axis=1 if _ax_name == y_ax else 0)
                                       .astype(int).sort_values(ascending=False).reset_index())
                            _df_sum.columns = [_ax_name, '件数']
                            _extra_lines.append(f"\n# {_ax_name}別の合計件数")
                            _extra_lines.append(utils_ai.df_to_markdown(_df_sum, index=False))
                    _extra_core = "\n".join(_extra_lines)

                    _core_prompt = utils_ai.generate_ai_insight_prompt(
                        role="特許分類・技術戦略の専門家として、ルールベース分類によるクロス集計結果を戦略的に分析してください。",
                        context=f"""\
{chart_type}によるクロス集計マトリクスを表示しています。
- X軸: {x_ax}
- Y軸: {y_ax}
- 対象: {selected_app_label}
各セルの値は該当する特許の件数を示します（下記マトリクスは markdown テーブル形式）。""",
                        data_summary=matrix_md,
                        instructions="""\
以下の観点で分析してください:
1. **注力領域**: 件数の集中パターンから、注力している技術-課題の組み合わせを特定
2. **空白領域**: 件数がゼロまたは少ないセルの意味（未開拓 or 不要な組み合わせ）
3. **バランス**: 分類間の件数のばらつきと「その他」比率の評価
4. **競合比較示唆**: 出願人別分析の場合、各社の技術ポートフォリオ特性の違い
5. **戦略提言**: データから導かれる研究開発・知財戦略の方向性

各主張には必ず具体的な数値を1つ以上含めてください。""",
                        metadata=_meta_core,
                        extra_content=_extra_core,
                        constraints="下記「分類ルール（CORE論理式）」でカテゴリの意味を正確に理解した上で分析すること。",
                        output_format="Markdown形式。見出し付きの構造化された分析レポート。"
                    )
                    utils_ai.render_ai_insight_button(_core_prompt, f"core_matrix_insight_{x_ax}_{y_ax}_{selected_app_label}")


                render_core_chart(df_target, "main_display")

                # CAPCOM data/ JSON出力（CORE分類結果）
                try:
                    import capcom
                    if capcom.is_active():
                        # ルール情報の構造化
                        rules_json = {}
                        for ax_name, ax_cats in st.session_state.core_classification_rules.items():
                            rules_json[ax_name] = {}
                            for cat_name, cat_info in ax_cats.items():
                                rules_json[ax_name][cat_name] = {
                                    "rule": cat_info.get('rule', ''),
                                    "definition": cat_info.get('definition', ''),
                                    "count": int(df_c[ax_name].fillna('').astype(str).str.split(';').explode().eq(cat_name).sum()) if ax_name in df_c.columns else 0
                                }
                            # 「その他」カテゴリの件数
                            if ax_name in df_c.columns:
                                other_count = int((df_c[ax_name] == 'その他').sum())
                                if other_count > 0:
                                    rules_json[ax_name]['その他'] = {"rule": "(未分類)", "definition": "いずれのルールにもマッチしなかった特許", "count": other_count}

                        core_json = {
                            "metadata": {
                                "module": "CORE",
                                "total_patents": len(df_c),
                                "axes": list(rules_json.keys())
                            },
                            "rules": rules_json
                        }
                        capcom.save_data("core_classification.json", core_json)
                except Exception as e:
                    st.caption(f"⚠️ WARN クラス分類マトリクス の CAPCOM 保存に失敗しました（要確認）: {e}")

        # CSVダウンロード
        st.markdown("---")
        csv_core = convert_df_to_csv_core(df_c)
        st.download_button("分類結果付き全データCSVをダウンロード", csv_core, "CORE_classified_full.csv", "text/csv")