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
import japanize_matplotlib
import utils

# ==============================================================================
#  1. Page Config (Must be first)
# ==============================================================================
st.set_page_config(
    page_title="APOLLO | CREW", 
    page_icon="🔗", 
    layout="wide"
)

# ==============================================================================
#  2. Font Setup
# ==============================================================================


# --- Custom CSS ---
st.markdown("""
<style>

    .stButton>button { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #003366; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
#  3. Sidebar Navigation
# ==============================================================================
utils.render_sidebar()

# ==============================================================================
#  4. Analysis Logic
# ==============================================================================
class PatentAnalyzer:
    def __init__(self, df, col_inv, col_date, col_assignee=None, embeddings=None):
        self.df = df.copy()
        # 元データのインデックスを保持（ベクトル参照用）
        self.df['__orig_idx'] = range(len(self.df))
        
        self.col_inv = col_inv
        self.col_date = col_date
        self.col_assignee = col_assignee
        self.embeddings_global = embeddings # Mission Controlから受け取った全件ベクトル (正規化済み想定)
        
        self.remove_chars = ['▲', '▼']
        self.sep_char = ';'
        
        # キャッシュ (ベクトル計算結果などを保持)
        self.cache_inventor = {'embeddings': None, 'topics': {}, 'names': [], 'name_to_idx': {}, 'processed_corpus': []}
        self.cache_corporate = {'embeddings': None, 'topics': {}, 'names': [], 'name_to_idx': {}, 'processed_corpus': []}
        self.current_mode = 'inventor'

    def preprocess(self, sep_char=';'):
        self.sep_char = sep_char
        def clean(name):
            if not isinstance(name, str): return ""
            for char in self.remove_chars: name = name.replace(char, "")
            return name.strip().replace("　", "")
        
        # リスト化処理
        self.df['inventors_list'] = self.df[self.col_inv].astype(str).apply(
            lambda x: [clean(n) for n in x.split(self.sep_char) if clean(n)]
        )
        
        if self.col_assignee:
            self.df[self.col_assignee] = self.df[self.col_assignee].fillna("不明")
            self.df['applicants_list'] = self.df[self.col_assignee].astype(str).apply(
                lambda x: [clean(n) for n in x.split(self.sep_char) if clean(n)]
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
        Mission Controlのベクトル(embeddings_global)を参照し、
        発明者/出願人ごとに「関与特許のベクトル平均」を計算して正規化する。
        """
        cache = self.cache_inventor if mode == 'inventor' else self.cache_corporate
        target_col = 'inventors_list' if mode == 'inventor' else 'applicants_list'
        
        if cache['embeddings'] is not None: return True
        if mode == 'corporate' and (not self.col_assignee): return False
        if self.embeddings_global is None: return False

        # 1. 展開 (Explode)
        exploded = self.df.explode(target_col)[['__orig_idx', target_col, 'text_for_tfidf']]
        exploded = exploded.dropna(subset=[target_col])
        exploded = exploded[exploded[target_col] != ""]
        
        # 2. グルーピング
        grouped = exploded.groupby(target_col)['__orig_idx'].apply(list)
        names = grouped.index.tolist()
        
        # テキストも結合 (TF-IDF用)
        text_grouped = exploded.groupby(target_col)['text_for_tfidf'].apply(lambda x: ' '.join(x.astype(str)))
        
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
        self._compute_vectors_for_mode('inventor')
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
                tfidf = TfidfVectorizer(max_features=3, token_pattern=r"(?u)\b\w+\b")
                tfidf_matrix = tfidf.fit_transform(texts)
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
            words = cache['processed_corpus'][idx].split()
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
#  5. Main UI Logic
# ==============================================================================
st.title("🔗 CREW")
st.markdown("##### Co-occurrence Relationship Exploration Web")

st.markdown("""
CREW（共起関係探索ウェブ） は、発明者や出願人のつながり（共起ネットワーク）を可視化し、組織内のハブ人材や技術コミュニティを特定するモジュールです。
Mission Controlで計算されたベクトルを活用し、高速に解析を行います。
""")

# --- データ連携チェック ---
df = None
if st.session_state.get('shared_df') is not None:
    df = st.session_state['shared_df']
elif st.session_state.get('df_main') is not None:
    df = st.session_state['df_main']
    st.session_state['shared_df'] = df

embeddings = st.session_state.get('sbert_embeddings')

if df is None or embeddings is None:
    st.error("⚠️ データまたはベクトル情報が読み込まれていません")
    st.markdown("Mission Control に戻って「分析エンジン起動」を実行してください。")
    st.stop()

# --- Analyzer初期化 ---
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# ==============================================================================
#  6. Settings & Filters
# ==============================================================================

st.markdown("### Analysis Settings")
with st.expander("分析パラメータ設定", expanded=True):
    col1, col2 = st.columns([1, 1])
    
    cols = df.columns.tolist()
    opt_cols = ['(なし)'] + cols
    col_map = st.session_state.get('col_map', {})

    def get_idx(key_name, options, default=0):
        c = col_map.get(key_name)
        if c and c in options: return options.index(c)
        return default

    with col1:
        col_inv = st.selectbox("発明者カラム", cols, index=get_idx('inventor', cols))
        col_assignee = st.selectbox("出願人カラム (任意)", opt_cols, index=get_idx('applicant', opt_cols))
        col_date = st.selectbox("出願日カラム", opt_cols, index=get_idx('date', opt_cols))
        
    with col2:
        st.info("※テキスト分析にはMission Controlで生成されたベクトルを使用します。")
        mode = st.radio("分析モード", ["発明者ネットワーク", "企業アライアンス"] if col_assignee != '(なし)' else ["発明者ネットワーク"], horizontal=True)

    if st.button("🔄 分析実行 (Initialize / Update)", type="primary"):
        real_assignee = col_assignee if col_assignee != '(なし)' else None
        real_date = col_date if col_date != '(なし)' else None
        
        analyzer = PatentAnalyzer(df, col_inv, real_date, real_assignee, embeddings)
        delimiters = st.session_state.get('delimiters', {})
        sep = delimiters.get('inventor', ';')
        
        analyzer.preprocess(sep_char=sep)
        analyzer.switch_mode('inventor' if mode == "発明者ネットワーク" else 'corporate')
        
        with st.spinner("ネットワーク構築 & ベクトル集約中..."):
            analyzer.precompute_all()
            st.session_state.analyzer = analyzer
        st.success("分析完了！")

# --- Filters ---
if st.session_state.analyzer:
    analyzer = st.session_state.analyzer
    analyzer.switch_mode('inventor' if mode == "発明者ネットワーク" else 'corporate')
    
    st.markdown("### Filters & Visualization")
    with st.expander("表示フィルタ設定", expanded=True):
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            years = sorted(analyzer.df['year'].dropna().unique())
            year_range = None
            if years:
                year_range = st.slider("対象期間", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
            
            app_opts, app_counts = analyzer.get_applicant_info()
            sel_apps = st.multiselect(
                "出願人で絞り込み", 
                app_opts,
                format_func=lambda x: f"{x} ({app_counts.get(x, 0)}件)"
            )

        with c_f2:
            min_node = st.number_input("最小出願件数 (ノード)", 1, 100, 1)
            min_edge = st.number_input("最小共願回数 (エッジ)", 1, 20, 1)
            recent_years_win = st.number_input("急上昇判定期間 (年)", 1, 10, 3)
        
        with c_f3:
            n_topics = st.slider("分類トピック数 (KMeans)", 3, 20, 6)
            
            color_mode = st.selectbox("色分け基準", [
                'コミュニティ (派閥)', 
                '技術トピック', 
                '媒介中心性', 
                '技術ブローカー', 
                '生産性スコア',    
                '急上昇スコア'
            ])

    # データ構築
    G = analyzer.build_graph_at_year(year_range, applicants=sel_apps)
    G_filtered = analyzer.apply_filters(G, min_node, min_edge, True, year_range, applicants=sel_apps)
    
    if len(G_filtered.nodes) == 0:
        st.warning("条件に合致するデータがありません。フィルタを緩めてください。")
    else:
        metrics_df = analyzer.calculate_all_metrics(G_filtered, year_range, n_topics, applicants=sel_apps, recent_years=recent_years_win)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Network", "Metrics", "Heatmap", "Trends", "Details"])

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
                hovertemplate="<b>%{customdata[0]}</b><br>出願数: %{customdata[1]}件<br>ComID: %{customdata[2]}<br>Topic: %{customdata[3]}<extra></extra>",
                marker=dict(showscale=True, colorscale='Viridis', color=node_col, size=node_sz, line_width=1))
            
            fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0), height=600))
            fig_net.update_xaxes(visible=False); fig_net.update_yaxes(visible=False)
            # Pass width=None to allow the network graph to fill the container width (prevent clumping)
            utils.update_fig_layout(fig_net, "共起ネットワーク図", height=600, width=None, theme_config=utils.get_theme_config("APOLLO Standard"))
            fig_net.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            # Remove aspect ratio constraint (set by utils for hidden axes) to fill width
            fig_net.update_yaxes(scaleanchor=None, scaleratio=None)
            st.plotly_chart(fig_net, use_container_width=True, config={'editable': False})
            
        with tab2:
            st.dataframe(metrics_df.sort_values('媒介中心性', ascending=False), use_container_width=True)
            st.download_button("CSVダウンロード", metrics_df.to_csv().encode('utf-8'), "crew_metrics.csv", "text/csv")

        with tab3:
            st.markdown("**コミュニティ × 技術トピック**")
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
                
                utils.update_fig_layout(fig_hm, "コミュニティ × 技術トピック", height=600, theme_config=utils.get_theme_config("APOLLO Standard"), show_axes=True)
                st.plotly_chart(fig_hm, use_container_width=True, config={'editable': False})
            else: st.info("データなし")

        with tab4:
            if mode == "発明者ネットワーク":
                st.markdown("##### 発明者数推移")
                df_tr = analyzer.get_inventor_trends(year_range=year_range, applicants=sel_apps)
                if not df_tr.empty:
                    fig_tr = px.bar(df_tr, x='出願年', y=['継続', '新規'], labels={'value':'人数'}, color_discrete_map={'新規':'#EF553B','継続':'#636EFA'})
                    utils.update_fig_layout(fig_tr, "発明者数推移", height=400, theme_config=utils.get_theme_config("APOLLO Standard"), show_axes=True)
                    st.plotly_chart(fig_tr, use_container_width=True, config={'editable': False})
                
                st.markdown("##### タイムライン")
                col_tl, _ = st.columns([1, 2])
                with col_tl: top_n_tl = st.slider("トップ表示数", 5, 50, 20)
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
                    utils.update_fig_layout(fig_time, "タイムライン", height=max(400, top_n_tl * 25), theme_config=utils.get_theme_config("APOLLO Standard"), show_axes=True)
                    st.plotly_chart(fig_time, use_container_width=True, config={'editable': False})
            else: st.info("企業モードではタイムライン非表示")

        with tab5:
            nodes_sorted = metrics_df.sort_values('出願数', ascending=False).index.tolist()
            sel = st.selectbox("対象選択", nodes_sorted)
            if sel:
                c1, c2 = st.columns(2)
                kws = analyzer.get_keywords(sel, 50)
                with c1:
                    if kws:
                        wc = WordCloud(width=600, height=400, background_color='white', font_path=utils.get_japanese_font_path(), regexp=r"[\w']+").generate_from_frequencies(kws)
                        fig, ax = plt.subplots(); ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); st.pyplot(fig)
                with c2:
                    if kws: st.dataframe(pd.DataFrame(list(kws.items()), columns=['Word', 'Freq']), height=200)
                    st.write(metrics_df.loc[sel])