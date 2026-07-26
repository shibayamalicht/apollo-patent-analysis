# ==================================================================
# APOLLO — CREW（出願人・発明者ネットワーク）
# 旧 APOLLO pages/6_🔗_CREW.py の移植。
# session_state キー・CAPCOM 保存スキーマ
# (crew_network.json / crew_wordcloud.json / スナップショット key) は無改変。
# PatentAnalyzer は発明者/出願人の区切り文字を分離指定できるよう拡張し、
# 発明者カラムが無いデータ（J-PlatPat 等）では企業アライアンスのみ動作する。
# ==================================================================
import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import itertools
import unicodedata
import re
import string
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from wordcloud import WordCloud
import os
import utils
import utils_ai
import patiroha
import apollo_ui

utils.configure_matplotlib_font()

# ==============================================================================
#  表示用の指標名辞書
#  内部値（DataFrameカラム名・CAPCOM保存値）は旧来のまま変えず、
#  画面表示だけを「平易語（専門語）」形式にする。
# ==============================================================================
_METRIC_LABELS = {
    'コミュニティ (派閥)': '派閥（コミュニティ検出）',
    '技術トピック': '技術トピック',
    '媒介中心性': '橋渡し度（媒介中心性）',
    '技術ブローカー': '越境度（技術ブローカー）',
    '生産性スコア': '少人数での多産度（生産性スコア）',
    '急上昇スコア': '直近の伸び（急上昇スコア）',
}

# 指標一覧テーブルの表示用カラム名（CSV出力・CAPCOM保存は元のカラム名のまま）
_DISPLAY_COLS = {
    '媒介中心性': '橋渡し度（媒介中心性）',
    '技術ブローカー': '越境度（技術ブローカー）',
    '次数': '直接の繋がり数（次数中心性）',
    '生産性スコア': '少人数での多産度（生産性スコア）',
    '急上昇スコア': '直近の伸び（急上昇スコア）',
    'コミュニティ': '派閥ID（コミュニティ）',
    'PrimaryTopic': '主要トピック',
}


# ==============================================================================
#  分析ロジック (Analyzer Class) — 旧 CREW から無改変で移植
# ==============================================================================
class PatentAnalyzer:
    def __init__(self, df, col_inv, col_date, col_assignee=None, embeddings=None):
        self.df = df.copy()
        # 元データのインデックスを保持（ベクトル参照用）
        self.df['__orig_idx'] = range(len(self.df))

        self.col_inv = col_inv
        self.col_date = col_date
        self.col_assignee = col_assignee
        self.embeddings_global = embeddings  # Mission Control から受け取った全件ベクトル (正規化済み想定)

        self.remove_chars = ['▲', '▼']
        self.sep_char = ';'
        self.app_sep_char = ';'

        # キャッシュ (ベクトル計算結果などを保持)
        self.cache_inventor = {'embeddings': None, 'topics': {}, 'names': [], 'name_to_idx': {}, 'processed_corpus': []}
        self.cache_corporate = {'embeddings': None, 'topics': {}, 'names': [], 'name_to_idx': {}, 'processed_corpus': []}
        self.current_mode = 'inventor'

    def preprocess(self, sep_char=';', app_sep_char=None):
        self.sep_char = sep_char
        # 出願人は発明者と区切り文字が異なることがある（例: J-PlatPat の出願人は ',' 区切り）
        self.app_sep_char = app_sep_char if app_sep_char else sep_char
        def clean(name):
            if not isinstance(name, str): return ""
            for char in self.remove_chars: name = name.replace(char, "")
            return name.strip().replace("　", "")

        # リスト化処理（発明者カラムが無いデータでは空リストにし、企業アライアンスのみ利用可能）
        if self.col_inv:
            self.df['inventors_list'] = self.df[self.col_inv].astype(str).apply(
                lambda x: [clean(n) for n in x.split(self.sep_char) if clean(n)]
            )
        else:
            self.df['inventors_list'] = [[] for _ in range(len(self.df))]

        if self.col_assignee:
            self.df[self.col_assignee] = self.df[self.col_assignee].fillna("不明")
            self.df['applicants_list'] = self.df[self.col_assignee].astype(str).apply(
                lambda x: [clean(n) for n in x.split(self.app_sep_char) if clean(n)]
            )

        if self.col_date and self.col_date != '(なし)':
            self.df['dt'] = pd.to_datetime(self.df[self.col_date], errors='coerce')
            self.df['year'] = self.df['dt'].dt.year
        else:
            self.df['year'] = 2024

    def get_applicant_info(self):
        if not self.col_assignee: return [], {}
        all_applicants = [a for sublist in self.df['applicants_list'] for a in sublist]
        counts = pd.Series(all_applicants).value_counts()
        return counts.index.tolist(), counts.to_dict()

    def switch_mode(self, mode):
        self.current_mode = mode

    def get_current_cache(self):
        return self.cache_inventor if self.current_mode == 'inventor' else self.cache_corporate

    def _compute_vectors_for_mode(self, mode):
        """
        Mission Control のベクトル(embeddings_global)を参照し、
        発明者/出願人ごとに「関与特許のベクトル平均」を計算して正規化する。
        """
        cache = self.cache_inventor if mode == 'inventor' else self.cache_corporate
        target_col = 'inventors_list' if mode == 'inventor' else 'applicants_list'

        if cache['embeddings'] is not None: return True
        if mode == 'inventor' and (not self.col_inv): return False
        if mode == 'corporate' and (not self.col_assignee): return False
        if self.embeddings_global is None: return False

        # 1. 展開 (Explode)
        # text_for_tfidf が無い場合はtitle+abstractで代替
        text_col = 'text_for_tfidf'
        if text_col not in self.df.columns:
            col_map = st.session_state.get('col_map', {})
            t_col = col_map.get('title', '')
            a_col = col_map.get('abstract', '')
            self.df[text_col] = (
                self.df[t_col].fillna('') if t_col and t_col in self.df.columns else ''
            ) + ' ' + (
                self.df[a_col].fillna('') if a_col and a_col in self.df.columns else ''
            )

        exploded = self.df.explode(target_col)[['__orig_idx', target_col, text_col]]
        exploded = exploded.dropna(subset=[target_col])
        exploded = exploded[exploded[target_col] != ""]

        # 2. グルーピング
        grouped = exploded.groupby(target_col)['__orig_idx'].apply(list)
        names = grouped.index.tolist()

        # テキストも結合 (TF-IDF用)
        text_grouped = exploded.groupby(target_col)[text_col].apply(lambda x: ' '.join(x.astype(str)))

        # 3. ベクトル平均化 & 再正規化
        n_samples = len(names)
        dim = self.embeddings_global.shape[1]

        vectors = np.zeros((n_samples, dim), dtype=np.float32)
        processed_corpus = []

        prog_bar = st.progress(0)
        step = max(1, n_samples // 10)

        with st.spinner(f"[{mode}] ベクトルを集約中 (Mean Pooling)..."):
            for i, name in enumerate(names):
                if i % step == 0: prog_bar.progress(i / n_samples)

                indices = grouped[name]
                if indices:
                    vecs = self.embeddings_global[indices]
                    mean_vec = np.mean(vecs, axis=0)
                    vectors[i] = mean_vec

                processed_corpus.append(text_grouped[name])

            prog_bar.empty()

            # 再正規化
            vectors = normalize(vectors, norm='l2')

        cache['names'] = names
        cache['name_to_idx'] = {n: i for i, n in enumerate(names)}
        cache['processed_corpus'] = processed_corpus
        cache['embeddings'] = vectors
        return True

    def precompute_all(self):
        if self.col_inv: self._compute_vectors_for_mode('inventor')
        if self.col_assignee: self._compute_vectors_for_mode('corporate')

    def calculate_topics_multi(self, n_clusters=5):
        cache = self.get_current_cache()
        if cache['embeddings'] is None: return {}, {}

        num_samples = cache['embeddings'].shape[0]
        actual_k = min(n_clusters, num_samples)
        if actual_k < 1: return {}, {}

        if actual_k in cache['topics']: return cache['topics'][actual_k]

        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        kmeans.fit(cache['embeddings'])

        # トピック名生成 (TF-IDF)
        df_cluster = pd.DataFrame({'label': kmeans.labels_, 'text': cache['processed_corpus']})
        topic_names = {}

        for i in range(actual_k):
            texts = df_cluster[df_cluster['label'] == i]['text']
            if texts.empty:
                topic_names[i] = f"Topic {i}"
                continue
            try:
                # 日本語を形態素解析し、ストップワード・定型句(【選択図】【課題】【解決手段】等)を
                # 除去してからTF-IDFにかける。生テキスト直接 + \b\w+\b では日本語を分割できず、
                # 括弧内の定型見出しばかりが topic に選ばれてしまうため。
                tokenized = [' '.join(utils.extract_keywords(str(t)[:3000])) for t in texts]
                tokenized = [t for t in tokenized if t.strip()]
                if not tokenized:
                    topic_names[i] = f"Topic {i}"
                    continue
                tfidf = TfidfVectorizer(max_features=3)
                tfidf_matrix = tfidf.fit_transform(tokenized)
                feature_names = tfidf.get_feature_names_out()
                sums = tfidf_matrix.sum(axis=0)
                data = []
                for col, term in enumerate(feature_names):
                    data.append((term, sums[0, col]))
                ranking = sorted(data, key=lambda x: x[1], reverse=True)
                top_words = [x[0] for x in ranking[:3]]
                topic_names[i] = "-".join(top_words) if top_words else f"Topic {i}"
            except:
                topic_names[i] = f"Topic {i}"

        dists = kmeans.transform(cache['embeddings'])
        multi_topic_map = {}
        primary_topic_map = {}
        names = cache['names']

        for idx, name in enumerate(names):
            sorted_indices = np.argsort(dists[idx])
            top_indices = sorted_indices[:2]
            primary_id = top_indices[0]
            primary_topic_map[name] = topic_names[primary_id]
            labels_str = [topic_names[i] for i in top_indices]
            multi_topic_map[name] = "; ".join(labels_str)

        cache['topics'][actual_k] = (multi_topic_map, primary_topic_map)
        return multi_topic_map, primary_topic_map

    def _get_filtered_df(self, year_range=None, applicants=None):
        df_sub = self.df.copy()
        if year_range is not None:
            start_year, end_year = year_range
            df_sub = df_sub[(df_sub['year'] >= start_year) & (df_sub['year'] <= end_year)]

        target_set = set(applicants) if applicants else None
        if target_set:
            target_col = 'applicants_list'
            mask = df_sub[target_col].apply(lambda x: not set(x).isdisjoint(target_set))
            df_sub = df_sub[mask]

        return df_sub

    def build_graph_at_year(self, year_range=None, applicants=None):
        """
        グラフ構築時に、必要なエッジについてのみ距離を計算 (内積 = Cosine Similarity)
        """
        G = nx.Graph()
        df_subset = self._get_filtered_df(year_range, applicants)
        target_col = 'inventors_list' if self.current_mode == 'inventor' else 'applicants_list'
        cache = self.get_current_cache()

        embeddings = cache.get('embeddings')
        name_to_idx = cache.get('name_to_idx')

        if target_col not in df_subset.columns: return G

        # 1. 共起カウント
        for item_list in df_subset[target_col]:
            if len(item_list) > 1:
                for u, v in itertools.combinations(item_list, 2):
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
            elif len(item_list) == 1:
                if not G.has_node(item_list[0]):
                    G.add_node(item_list[0])

        # 2. 距離計算 (オンデマンド)
        if embeddings is not None and name_to_idx is not None:
            for u, v in G.edges():
                if u in name_to_idx and v in name_to_idx:
                    idx_u = name_to_idx[u]
                    idx_v = name_to_idx[v]
                    sim = np.dot(embeddings[idx_u], embeddings[idx_v])
                    dist = 1.0 - sim
                    dist = max(0.0, min(1.0, dist))
                    G[u][v]['distance'] = dist
                else:
                    G[u][v]['distance'] = 0.5

        return G

    def apply_filters(self, G, min_app_count, min_edge_weight, remove_isolated, year_range=None, applicants=None):
        df_subset = self._get_filtered_df(year_range, applicants)
        target_col = 'inventors_list' if self.current_mode == 'inventor' else 'applicants_list'

        exploded = df_subset.explode(target_col)
        app_counts = exploded[target_col].value_counts().to_dict()

        nodes_to_keep = [n for n in G.nodes() if app_counts.get(n, 0) >= min_app_count]
        G_sub = G.subgraph(nodes_to_keep).copy()

        if min_edge_weight > 1:
            edges_to_remove = [(u, v) for u, v, d in G_sub.edges(data=True) if d.get('weight', 0) < min_edge_weight]
            G_sub.remove_edges_from(edges_to_remove)

        if remove_isolated:
            isolates = list(nx.isolates(G_sub))
            G_sub.remove_nodes_from(isolates)

        return G_sub

    def calculate_all_metrics(self, G, year_range=None, n_topics=5, applicants=None, recent_years=3):
        if len(G.nodes) == 0: return pd.DataFrame()

        betweenness = nx.betweenness_centrality(G)
        degree = dict(G.degree())

        try:
            communities_list = nx.community.louvain_communities(G, seed=42)
            partition = {}
            for i, comm in enumerate(communities_list):
                for node in comm:
                    partition[node] = i
        except Exception:
            partition = {n: 0 for n in G.nodes()}

        topic_str_map, primary_topic_map = self.calculate_topics_multi(n_topics)

        df_sub = self._get_filtered_df(year_range, applicants)
        target_col = 'inventors_list' if self.current_mode == 'inventor' else 'applicants_list'
        exploded = df_sub.explode(target_col)
        app_counts_total = exploded[target_col].value_counts()

        # Rising Score
        if year_range is not None:
            start_year, end_year = year_range
            recent_threshold = end_year - recent_years

            df_recent = df_sub[df_sub['year'] > recent_threshold]
            exploded_recent = df_recent.explode(target_col)
            app_counts_recent = exploded_recent[target_col].value_counts()

            df_past = df_sub[df_sub['year'] <= recent_threshold]
            exploded_past = df_past.explode(target_col)
            app_counts_past = exploded_past[target_col].value_counts()
        else:
            app_counts_recent = pd.Series()
            app_counts_past = pd.Series()

        rising_scores = {}
        for n in G.nodes():
            rec = app_counts_recent.get(n, 0)
            past = app_counts_past.get(n, 0)
            rising_scores[n] = rec / (past + 1)

        brokerage = {}
        for n in G.nodes():
            score = sum([G[n][neighbor].get('distance', 0.5) for neighbor in G.neighbors(n)])
            brokerage[n] = score

        df = pd.DataFrame({
            '出願数': app_counts_total,
            '媒介中心性': pd.Series(betweenness),
            '技術ブローカー': pd.Series(brokerage),
            '急上昇スコア': pd.Series(rising_scores),
            '次数': pd.Series(degree),
            'コミュニティ': pd.Series(partition),
            '技術トピック': pd.Series(topic_str_map),
            'PrimaryTopic': pd.Series(primary_topic_map)
        }).fillna(0)

        df = df[df.index.isin(G.nodes())]
        df['生産性スコア'] = df['出願数'] / (df['次数'] + 1)

        return df

    def get_keywords(self, name, top_n=20):
        cache = self.get_current_cache()
        try:
            idx = cache['names'].index(name)
            # 形態素解析で複合名詞を抽出（定型句・ストップワードを除去）。
            # 生テキストの単純splitだと日本語が分割できず定型見出しが上位に来てしまう。
            words = utils.extract_keywords(str(cache['processed_corpus'][idx])[:6000])
            return pd.Series(words).value_counts().head(top_n).to_dict()
        except:
            return {}

    def get_inventor_trends(self, year_range=None, applicants=None):
        df_sub = self._get_filtered_df(year_range, applicants)
        years = sorted(df_sub['year'].dropna().unique())
        stats = []
        seen_inventors = set()

        for y in years:
            df_y = df_sub[df_sub['year'] == y]
            invs_in_year = set(df_y.explode('inventors_list')['inventors_list'].dropna().unique())
            invs_in_year = {i for i in invs_in_year if i}
            new_invs = invs_in_year - seen_inventors
            existing_invs = invs_in_year & seen_inventors
            stats.append({'出願年': int(y), '新規': len(new_invs), '継続': len(existing_invs)})
            seen_inventors.update(invs_in_year)
        return pd.DataFrame(stats)

    def get_inventor_timeline(self, year_range=None, applicants=None, top_n=30):
        df_sub = self._get_filtered_df(year_range, applicants)
        exploded = df_sub.explode('inventors_list')
        exploded = exploded[exploded['inventors_list'].astype(str).str.strip() != ""]
        exploded = exploded.dropna(subset=['inventors_list'])

        top_inventors = exploded['inventors_list'].value_counts().head(top_n).index.tolist()
        df_top = exploded[exploded['inventors_list'].isin(top_inventors)]

        timeline = df_top.groupby(['inventors_list', 'year']).size().reset_index(name='出願数')
        return timeline.rename(columns={'inventors_list': '発明者', 'year': '出願年'})


# ==============================================================================
#  メインUI
# ==============================================================================
def render():
    apollo_ui.page_header(
        "出願人・発明者ネットワーク", "CREW",
        "共同出願・発明者の共著関係から、協業構造と媒介中心性を分析します",
    )

    # --- 共通データガード ---
    apollo_ui.require_data()

    # --- CREW 固有の追加前提: 共有DFとAI意味解析ベクトル ---
    df = None
    if st.session_state.get('shared_df') is not None:
        df = st.session_state['shared_df']
    elif st.session_state.get('df_main') is not None:
        df = st.session_state['df_main']
        st.session_state['shared_df'] = df

    embeddings = st.session_state.get('sbert_embeddings')
    df_main = df  # AI Insight用の変数名統一

    if df is None or embeddings is None:
        st.warning("このモジュールには AI意味解析（SBERT）のベクトルが必要ですが、まだ用意されていません。", icon=":material/rocket_launch:")
        st.markdown("「Mission Control」で **前処理を実行** すると使えるようになります。")
        pages = st.session_state.get("ap_pages", {})
        if "mission_control" in pages:
            st.page_link(pages["mission_control"], label="データ準備を開く", icon=":material/anchor:")
        st.stop()

    # --- Analyzer初期化 ---
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    # ==========================================================================
    #  おすすめ実行バー + 詳細設定
    #  （バーを先に確保し、中のボタンは詳細設定ウィジェットの値が確定した後に描く）
    # ==========================================================================
    run_bar = st.container(border=True)

    with st.expander("詳細設定（対象カラムと分析モード）", expanded=False):
        col1, col2 = st.columns([1, 1])

        cols = df.columns.tolist()
        opt_cols = ['(なし)'] + cols
        col_map = st.session_state.get('col_map', {})

        def get_idx(key_name, options, default=0):
            c = col_map.get(key_name)
            if c and c in options: return options.index(c)
            return default

        with col1:
            col_inv = st.selectbox("発明者カラム (任意)", opt_cols, index=get_idx('inventor', opt_cols),
                                   help="発明者名が入ったカラムです。データに発明者情報が無い場合は「(なし)」のままにしてください"
                                        "（J-PlatPatのCSVには発明者が含まれないため自動で「(なし)」になります）。"
                                        "その場合は企業アライアンスのみ分析できます。")
            col_assignee = st.selectbox("出願人カラム (任意)", opt_cols, index=get_idx('applicant', opt_cols))
            col_date = st.selectbox("出願日カラム", opt_cols, index=get_idx('date', opt_cols))

        with col2:
            st.info("※テキスト分析には「Mission Control」の前処理で生成した文埋め込みを使用します。")

    # 分析モードは実行バーで選ぶ（詳細設定に埋めない）。col_inv / col_assignee の確定後に描くため、
    # バー枠は先に確保し、この位置で中身を描画する。
    with run_bar:
        c_msg, c_btn = st.columns([2.2, 1])
        with c_msg:
            _mode_opts = []
            if col_inv != '(なし)': _mode_opts.append("発明者ネットワーク")
            if col_assignee != '(なし)': _mode_opts.append("企業アライアンス")
            if _mode_opts:
                mode = st.radio(
                    "誰のネットワークを見ますか？（分析モード）:",
                    _mode_opts,
                    horizontal=True, key="fleet_mode_radio",
                    help="ネットワークを誰の共同関係で作るかです。発明者＝個人どうしの協働関係、"
                         "企業アライアンス＝企業間の共同出願（提携）関係を可視化します。")
                if col_inv == '(なし)':
                    st.caption("このデータには発明者カラムがないため、**企業アライアンス**（企業間の共同出願ネットワーク）のみ分析できます。")
                else:
                    st.caption("対象カラムは Mission Control のマッピングから自動設定済みです。まず「発明者ネットワーク」で全体像を掴むのがおすすめです。")
            else:
                mode = None
                st.warning("発明者カラム・出願人カラムのどちらも設定されていません。上の詳細設定でカラムを割り当ててください。", icon=":material/rocket_launch:")
        with c_btn:
            run_clicked = st.button("ネットワーク分析を実行", type="primary", use_container_width=True,
                                    disabled=(mode is None))

    if run_clicked and mode is not None:
        real_inv = col_inv if col_inv != '(なし)' else None
        real_assignee = col_assignee if col_assignee != '(なし)' else None
        real_date = col_date if col_date != '(なし)' else None

        analyzer = PatentAnalyzer(df, real_inv, real_date, real_assignee, embeddings)
        delimiters = st.session_state.get('delimiters', {})
        sep = delimiters.get('inventor', ';')
        app_sep = delimiters.get('applicant', ';')

        analyzer.preprocess(sep_char=sep, app_sep_char=app_sep)
        analyzer.switch_mode('inventor' if mode == "発明者ネットワーク" else 'corporate')

        with st.spinner("ネットワーク構築 & ベクトル集約中..."):
            analyzer.precompute_all()
            st.session_state.analyzer = analyzer
        st.success("分析が完了しました。下の結果をご覧ください。")

    # ==========================================================================
    #  表示フィルタと結果
    # ==========================================================================
    if st.session_state.analyzer and mode is not None:
        analyzer = st.session_state.analyzer
        analyzer.switch_mode('inventor' if mode == "発明者ネットワーク" else 'corporate')

        with st.expander("表示フィルタ（期間・出願人・粒度・色分け）", expanded=True):
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                years = sorted(analyzer.df['year'].dropna().unique())
                year_range = None
                if years:
                    year_range = st.slider("対象期間", int(min(years)), int(max(years)), (int(min(years)), int(max(years))),
                                           help="分析対象とする出願年の範囲です。")

                app_opts, app_counts = analyzer.get_applicant_info()
                sel_apps = st.multiselect(
                    "出願人で絞り込み",
                    app_opts,
                    format_func=lambda x: f"{x} ({app_counts.get(x, 0)}件)"
                )

            with c_f2:
                min_node = st.number_input("最小出願件数 (ノード)", 1, 100, 1,
                                           help="ネットワークに含める最小の出願件数です。これ未満の小規模なノード（人/企業）を除外し、主要プレイヤーに絞ります。")
                min_edge = st.number_input("最小共願回数 (エッジ)", 1, 20, 1,
                                           help="2者を線（エッジ）で結ぶ最小の共同出願回数です。大きくすると強い協働関係だけが残り、小さくすると弱いつながりも線になり密になります。")
                recent_years_win = st.number_input("急上昇判定期間 (年)", 1, 10, 3,
                                                   help="「急上昇」プレイヤーを判定する直近の期間（年数）です。この期間での活動の伸びを評価します。")

            with c_f3:
                n_topics = st.slider("分類トピック数 (KMeans)", 3, 20, 6,
                                     help="発明者/企業を技術トピックへ分類する数(k)です。KMeansでこの数のグループに分けます。大きいほど細かく分かれます。")

                # 内部値は旧来のまま（CAPCOM保存・並べ替え判定と互換）。表示だけ平易語+括弧形式。
                color_mode = st.selectbox("色分け基準", [
                    'コミュニティ (派閥)',
                    '技術トピック',
                    '媒介中心性',
                    '技術ブローカー',
                    '生産性スコア',
                    '急上昇スコア'
                ], format_func=lambda x: _METRIC_LABELS.get(x, x),
                   help="ネットワーク図のノードを何の指標で色分けするかです。コミュニティ＝共願が密な派閥、技術トピック＝主に扱う技術領域、媒介中心性＝派閥を橋渡しする結節点、技術ブローカー＝複数領域をまたぐ越境者、生産性スコア＝少人数連携での多産度、急上昇スコア＝直近の活動の伸びを表します。")

        # データ構築
        G = analyzer.build_graph_at_year(year_range, applicants=sel_apps)
        G_filtered = analyzer.apply_filters(G, min_node, min_edge, True, year_range, applicants=sel_apps)

        if len(G_filtered.nodes) == 0:
            st.warning("条件に合致するデータがありません。フィルタを緩めてください。")
        else:
            metrics_df = analyzer.calculate_all_metrics(G_filtered, year_range, n_topics, applicants=sel_apps, recent_years=recent_years_win)

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["ネットワーク図", "指標ランキング", "派閥×技術トピック", "推移・タイムライン", "個別詳細"])

            with tab1:
                st.markdown("**共起ネットワーク図**")
                pos = nx.spring_layout(G_filtered, k=0.8, seed=42)
                edge_x, edge_y = [], []
                for edge in G_filtered.edges():
                    x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#ccc'), hoverinfo='none', mode='lines')

                node_x, node_y, node_txt, node_col, node_sz, node_custom = [], [], [], [], [], []
                for node in G_filtered.nodes():
                    x, y = pos[node]
                    node_x.append(x); node_y.append(y)
                    rec = metrics_df.loc[node]
                    node_txt.append(node)
                    node_custom.append([node, int(rec['出願数']), rec['コミュニティ'], rec['PrimaryTopic']])

                    if color_mode == 'コミュニティ (派閥)': val = rec['コミュニティ']
                    elif color_mode == '技術トピック': val = hash(rec['PrimaryTopic']) % 20
                    elif color_mode == '媒介中心性': val = rec['媒介中心性']
                    elif color_mode == '技術ブローカー': val = rec['技術ブローカー']
                    elif color_mode == '生産性スコア': val = rec['生産性スコア']
                    else: val = rec['急上昇スコア']

                    node_col.append(val)
                    node_sz.append(np.log1p(rec['出願数']) * 10 + 5)

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text', text=node_txt, textposition="top center",
                    customdata=node_custom,
                    hovertemplate="<b>%{customdata[0]}</b><br>出願数: %{customdata[1]}件<br>派閥ID: %{customdata[2]}<br>主要トピック: %{customdata[3]}<extra></extra>",
                    marker=dict(showscale=True, colorscale='Viridis', color=node_col, size=node_sz, line_width=1))

                fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0), height=600))
                fig_net.update_xaxes(visible=False); fig_net.update_yaxes(visible=False)
                # 幅いっぱいに広げるためにwidth=Noneを渡す (凝縮防止)
                utils.update_fig_layout(fig_net, "共起ネットワーク図", height=600, width=None)
                fig_net.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                # アスペクト比制限を解除 (隠された軸設定による制限を回避)
                fig_net.update_yaxes(scaleanchor=None, scaleratio=None)
                st.plotly_chart(fig_net, use_container_width=True, config={'editable': False})

                # --- 読み方カード ---
                _hw_node = '発明者（個人）' if mode == "発明者ネットワーク" else '出願人（企業）'
                apollo_ui.howto(
                    f"<b>丸（ノード）は{_hw_node}</b>、<b>線（エッジ）は共同出願（共願）の関係</b>を表します。"
                    "<b>丸の大きさ＝出願数</b>、色は上で選んだ色分け基準の値です。"
                    "<b>線が多く集まる丸はハブ</b>（協業の中心プレイヤー）で、離れたかたまり同士を少ない線で繋ぐ丸は"
                    "<b>橋渡し役（媒介中心性が高い）</b>— 抜けると分断が起きるキーパーソンです。"
                    "線のない丸は単独出願が中心という意味で、重要度が低いとは限りません。"
                )

                # --- スナップショット（CAPCOM/VOYAGER 用）+ ネットワークデータの CAPCOM 出力 ---
                # color_mode（色分け基準）ごとに、その指標の「読み方」をスナップショットへ埋め込む。
                # 各指標で別々にスナップショットを撮れば、指標固有の解釈が CAPCOM/レポートに渡る。
                # ※ 以下の文言はスナップショット説明文用（CAPCOM/レポートに渡る）。画面表示には流用しない。
                _crew_metric_guide = {
                    'コミュニティ (派閥)': "色=協働グループ（派閥）。同色は共願が密な派閥。アライアンス構造・派閥の分断/結合を読む。",
                    '技術トピック': "色=各プレイヤーが主に扱う技術領域。どの技術に誰が集まるか（人材・技術の分布）を読む。",
                    '媒介中心性': "高=異なる派閥を橋渡しする結節点。除去すると分断が起きるキーパーソン。採用・提携・引き抜き防止の重点人物。",
                    '技術ブローカー': "高=複数の技術コミュニティをまたぐ越境者。技術融合の担い手。新規融合の起点候補。",
                    '生産性スコア': "出願数/(次数+1)。高=少人数連携で多産（効率的）。低=大人数だが出願少。R&D効率の差を読む。",
                    '急上昇スコア': "高=直近で活動を急拡大するプレイヤー。新興の要注意株。早期警戒・新興競合の発見に使う。",
                }
                _crew_guide = _crew_metric_guide.get(color_mode, '')
                # 数値指標ならその指標で並べ替え、カテゴリ指標（派閥/技術トピック）は媒介中心性で代表化
                _sort_col = color_mode if (color_mode in metrics_df.columns and color_mode not in ('コミュニティ (派閥)', '技術トピック')) else '媒介中心性'
                # :material/warning: トークン対策: 全ノードを載せると肥大化するため上位15ノード・主要数値列のみに切り取る
                #   （技術トピック等の長文列は除外。全件は画面の指標ランキングタブ/CSVで参照可能）
                _CREW_TOP = 15
                _keep_cols = [c for c in ['出願数', 'コミュニティ', '媒介中心性', '技術ブローカー', '生産性スコア', '急上昇スコア'] if c in metrics_df.columns]
                _crew_top = metrics_df.sort_values(_sort_col, ascending=False).head(_CREW_TOP)
                _crew_top_view = _crew_top[_keep_cols].copy()
                _crew_top_view.insert(0, 'ノード', _crew_top.index.astype(str))
                _crew_snap = {'module': 'CREW', 'network_mode': mode, 'metric_guide': _crew_guide}
                try:
                    _crew_snap['chart_data'] = utils_ai.df_to_markdown(_crew_top_view.round(3), index=False)
                except Exception:
                    pass
                try:
                    import capcom
                    if capcom.is_active():
                        # 指標別トップ15を全て保存（最後に選んだ色分け基準に依存せず、
                        # 各指標のトップ層が常に揃うようにする。各レコードは全指標カラムを含む）
                        _metric_cols = [c for c in ['媒介中心性', '技術ブローカー', '生産性スコア', '急上昇スコア']
                                        if c in metrics_df.columns]
                        _top_by_metric = {}
                        for _mc in _metric_cols:
                            _t = metrics_df.sort_values(_mc, ascending=False).head(_CREW_TOP)
                            _tv = _t[_keep_cols].copy()
                            _tv.insert(0, 'ノード', _t.index.astype(str))
                            _top_by_metric[_mc] = _tv.round(3).to_dict('records')
                        capcom.save_data("crew_network.json", {
                            "mode": mode,
                            "color_mode": color_mode,
                            "metric_guide": _crew_guide,
                            "node_count": int(len(metrics_df)),
                            "shown_top": int(len(_crew_top_view)),
                            "sorted_by": _sort_col,
                            "top_nodes": _crew_top_view.round(3).to_dict('records'),
                            "top_by_metric": _top_by_metric,
                        })
                except Exception as e:
                    st.caption(f":material/warning: WARN 共起ネットワーク の CAPCOM 保存に失敗しました（要確認）: {e}")
                # キー設計（2つのセレクタを性質で分ける）:
                #   ・分析モード（発明者ネットワーク / 企業アライアンス）は「別グラフ」なので key に含めて
                #     独立した base スナップショットにする（切替で新規保存ボタンになる）。
                #   ・色分け基準（color_mode）は同一グラフの「配色オプション」なので key に含めない。
                #     → 一度撮った後に色分けを変えると「:material/add:別カット追加」ボタンになり、配色違いを別カット（__sN）
                #       として積める（ボタンの説明文「配色などを変えた別バージョン」と一致）。各カットは保存時点の
                #       色分けの読み方（description/data_summary）を保持するので CAPCOM 側の情報は失われない。
                #   ・group="crew_network" で「このマップで N 枚」をモード/配色を跨いで通算表示（ドリルダウン系と同様）。
                _safe_mode = 'inv' if mode == "発明者ネットワーク" else 'corp'
                _safe_cm = color_mode.replace(' ', '_').replace('(', '').replace(')', '')  # AIインサイトのキー用（色分け別にプロンプトが変わる）
                utils.render_snapshot_button(
                    title=f"共起ネットワーク図［{mode}・色分け: {color_mode}］",
                    description=f"{mode}の協働（共願）ネットワーク。ノード色={color_mode}・サイズ=出願数。【{color_mode}の読み方】{_crew_guide}",
                    key=f"crew_network_{_safe_mode}", fig=fig_net, data_summary=_crew_snap,
                    group="crew_network")
                apollo_ui.snapshot_hint()

                # --- AI Insight: 協働ネットワーク分析 ---
                _node_label = '発明者' if mode == "発明者ネットワーク" else '出願人（企業）'
                _crew_meta = utils_ai.build_common_metadata(
                    df_main=st.session_state.get('df_main'), df_filtered=None, col_map=st.session_state.get('col_map'))
                _crew_meta['分析単位'] = _node_label
                _crew_meta['ネットワーク規模'] = f"ノード{int(len(metrics_df))}・上位{int(len(_crew_top_view))}表示"
                _crew_meta['色分け基準'] = color_mode
                _crew_meta['並べ替え指標'] = _sort_col
                try:
                    _n_comm = int(metrics_df['コミュニティ'].nunique())
                    _crew_meta['コミュニティ数'] = _n_comm
                except Exception:
                    pass
                _crew_prompt = utils_ai.generate_ai_insight_prompt(
                    role=f"技術経営・ネットワーク分析の専門家として、{_node_label}の協働（共願）ネットワークから組織・人材戦略を分析してください。",
                    context=(f"共起ネットワーク図を表示しています。ノード={_node_label}、エッジ=共願（共同出願）関係、"
                             f"ノードサイズ=出願数、ノード色={color_mode}。下記データは{_sort_col}の上位ノードと各指標です。"
                             f"指標の意味: 媒介中心性=異なる派閥を橋渡しする結節点（除去で分断が起きるキーパーソン）、"
                             f"技術ブローカー=複数コミュニティをまたぐ越境者（技術融合の担い手）、"
                             f"生産性スコア=出願数/(次数+1)で少人数連携の多産度、急上昇スコア=直近の活動急拡大度、"
                             f"コミュニティ=共願が密な派閥。【現在の色分け『{color_mode}』の読み方】{_crew_guide}"),
                    data_summary=_crew_snap.get('chart_data', ''),
                    instructions="""\
以下の観点で分析してください:
1. **キーパーソン/中核プレイヤー**: 媒介中心性・出願数の上位ノードを特定し、ネットワーク上の役割（ハブ/橋渡し/専門特化）を論じる
2. **派閥・アライアンス構造**: コミュニティ（派閥）の数と規模から、協働クラスタの分断/結合・系列構造を読み解く
3. **技術ブローカー**: コミュニティをまたぐ越境プレイヤーを特定し、技術融合・オープンイノベーションの担い手を示す
4. **R&D効率と新興プレイヤー**: 生産性スコア（少人数で多産か）と急上昇スコア（新興の要注意株）の観点で注目ノードを抽出
5. **戦略的示唆**: 採用・提携・引き抜き防止・新興競合の早期警戒など、人材・組織戦略へのインプリケーション

各主張には必ずノード名と具体的な数値を1つ以上含めてください。""",
                    metadata=_crew_meta,
                    constraints="出願数の多さは必ずしも質や影響力と一致しない点、ネットワーク指標は表示上位ノードに基づく点に注意すること。",
                    output_format="Markdown形式。見出し付きの構造化された分析レポート。"
                )
                utils_ai.render_ai_insight_button(_crew_prompt, f"crew_network_{_safe_mode}_{_safe_cm}_insight")

            with tab2:
                st.dataframe(
                    metrics_df.sort_values('媒介中心性', ascending=False).rename(columns=_DISPLAY_COLS),
                    use_container_width=True)
                st.caption("橋渡し度（媒介中心性）の高い順に表示しています。CSVは元の列名（媒介中心性・次数 等）のまま出力されます。")
                st.download_button("CSVダウンロード", metrics_df.to_csv().encode('utf-8'), "crew_metrics.csv", "text/csv")

            with tab3:
                st.markdown("**派閥（コミュニティ）× 技術トピック**")
                df_hm = metrics_df[['コミュニティ', '技術トピック']].copy()
                df_hm['技術トピック'] = df_hm['技術トピック'].astype(str).str.split('; ')
                df_exp = df_hm.explode('技術トピック')
                df_exp = df_exp[df_exp['技術トピック'].str.strip() != ""]

                # ヒートマップデータ
                ct = pd.crosstab(df_exp['コミュニティ'], df_exp['技術トピック'])

                if not ct.empty:
                    # 1. 軸を文字列化 (隙間防止) & 小数点削除 (float -> int -> str)
                    ct.index = ct.index.astype(float).astype(int).astype(str)
                    ct.columns = ct.columns.astype(str)

                    # 2. 並び替え
                    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
                    ct = ct[ct.sum().sort_values(ascending=False).index]

                    # 3. テキストデータ作成 (0を空文字に)
                    text_df = ct.astype(int).astype(str)
                    text_df[ct == 0] = ""

                    # 4. カラーマップ (0を薄いグレーに)
                    custom_colorscale = [[0.0, "#f9f9f9"], [0.01, "#eff3ff"], [1.0, "#08519c"]]

                    # 5. 描画
                    fig_hm = px.imshow(
                        ct,
                        aspect="auto",
                        color_continuous_scale=custom_colorscale,
                        labels=dict(x="技術トピック", y="コミュニティID", color="人数")
                    )
                    # 6. テキスト適用 & グリッド線 (xgap/ygap)
                    fig_hm.update_traces(text=text_df, texttemplate="%{text}", xgap=1, ygap=1)

                    fig_hm.update_xaxes(type='category', side="bottom")
                    fig_hm.update_yaxes(type='category', autorange="reversed")

                    utils.update_fig_layout(fig_hm, "コミュニティ × 技術トピック", height=600, show_axes=True)
                    st.plotly_chart(fig_hm, use_container_width=True, config={'editable': False})
                    apollo_ui.howto(
                        "<b>行＝派閥（コミュニティ）、列＝技術トピック</b>で、数字はその組み合わせに属するプレイヤー数です。"
                        "<b>1つの行に色が集中</b>していればその派閥は特定技術に特化、"
                        "<b>横に広く色が付く行</b>は複数技術をまたぐ派閥です。"
                        "同じ列に複数の派閥が並ぶ場合、同一技術を別グループが競って開発している可能性を示します。"
                    )
                    # （スナップショットはネットワーク図のみ採取。ヒートマップは対象外）

                else: st.info("データなし")

            with tab4:
                if mode == "発明者ネットワーク":
                    st.markdown("##### 発明者数推移")
                    df_tr = analyzer.get_inventor_trends(year_range=year_range, applicants=sel_apps)
                    if not df_tr.empty:
                        fig_tr = px.bar(df_tr, x='出願年', y=['継続', '新規'], labels={'value':'人数'}, color_discrete_map={'新規':'#EF553B','継続':'#636EFA'})
                        utils.update_fig_layout(fig_tr, "発明者数推移", height=400, show_axes=True)
                        st.plotly_chart(fig_tr, use_container_width=True, config={'editable': False})
                        apollo_ui.howto(
                            "<b>棒の高さ＝その年に出願した発明者の人数</b>で、<b>赤＝その年に初めて登場した新規人材</b>、"
                            "青＝以前から活動する継続人材です。<b>新規が細り続ける</b>場合は人材流入の停滞、"
                            "新規の急増は研究体制の拡大や新規参入のサインです。"
                        )

                    st.markdown("##### タイムライン")
                    col_tl, _ = st.columns([1, 2])
                    with col_tl: top_n_tl = st.slider("トップ表示数", 5, 50, 20,
                                                      help="ランキングで上位何件まで表示するかです（表示数のみ変わり、分析結果は変わりません）。")
                    df_time = analyzer.get_inventor_timeline(year_range=year_range, applicants=sel_apps, top_n=top_n_tl)
                    if not df_time.empty:
                        # 出願数が多い順に上から表示
                        total_counts = df_time.groupby('発明者')['出願数'].sum().sort_values(ascending=False)
                        sorted_inventors = total_counts.index.tolist()

                        fig_time = px.scatter(
                            df_time, x='出願年', y='発明者', size='出願数', color='発明者',
                            category_orders={'発明者': sorted_inventors}
                        )
                        fig_time.update_layout(showlegend=False, height=max(400, top_n_tl * 25))
                        utils.update_fig_layout(fig_time, "タイムライン", height=max(400, top_n_tl * 25), show_axes=True)
                        st.plotly_chart(fig_time, use_container_width=True, config={'editable': False})
                        apollo_ui.howto(
                            "<b>横軸＝出願年、縦＝発明者（出願数の多い順）</b>で、<b>丸の大きさ＝その年の出願数</b>です。"
                            "長く続く行はその分野の中核人材、<b>途中で途切れた行</b>は活動休止・異動・退職の可能性を示します。"
                            "最近になって現れた行は新戦力です。"
                        )

                else: st.info("企業アライアンスモードでは発明者の推移・タイムラインは表示されません。分析モードを「発明者ネットワーク」に切り替えるとご覧いただけます。")

            with tab5:
                nodes_sorted = metrics_df.sort_values('出願数', ascending=False).index.tolist()
                sel = st.selectbox("詳しく見るプレイヤーを選択", nodes_sorted,
                                   help="出願数の多い順に並んでいます。選ぶと、そのプレイヤーの得意キーワードと各指標を表示します。")
                if sel:
                    c1, c2 = st.columns(2)
                    kws = analyzer.get_keywords(sel, 50)
                    with c1:
                        if kws:
                            wc = WordCloud(width=600, height=400, background_color='white', font_path=utils.get_japanese_font_path(), regexp=r"[\w']+").generate_from_frequencies(kws)
                            fig, ax = plt.subplots(); ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); st.pyplot(fig)
                            # CAPCOM: CREWワードクラウドデータ保存
                            try:
                                import capcom
                                if capcom.is_active():
                                    wc_data = {
                                        "metadata": {"module": "CREW", "title": sel, "type": "node_keywords"},
                                        "word_frequencies": dict(kws)
                                    }
                                    capcom.save_data(f"crew_wordcloud.json", wc_data)
                            except Exception as e:
                                st.caption(f":material/warning: WARN ワードクラウド の CAPCOM 保存に失敗しました（要確認）: {e}")
                    with c2:
                        if kws: st.dataframe(pd.DataFrame(list(kws.items()), columns=['キーワード', '頻度']), height=200)
                        st.write(metrics_df.loc[sel].rename(_DISPLAY_COLS))
