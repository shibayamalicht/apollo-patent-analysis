import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re
import string
import os
import platform
import unicodedata
from collections import Counter
from itertools import combinations
import datetime

import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import japanize_matplotlib
import utils

# ==================================================================
# --- 1. 設定・ヘルパー関数 ---
# ==================================================================
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="APOLLO | Explorer",
    page_icon="🧭",
    layout="wide"
)



FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except:
        pass





def update_fig_layout(fig, title, height=600, theme_config=None, show_legend=True):
    if theme_config:
        # Sanitize title
        if isinstance(title, str):
            title = re.sub(r'<[^>]+>', '', title)

        layout_params = dict(
            template=theme_config["plotly_template"],
            title=dict(text=title, font=dict(size=18, color=theme_config["text_color"], weight="normal")),
            paper_bgcolor=theme_config["bg_color"], plot_bgcolor=theme_config["bg_color"],
            font=dict(color=theme_config["text_color"], family="Helvetica Neue"),
            height=height, margin=dict(l=20, r=20, t=60, b=20)
        )
        if not show_legend:
            layout_params['showlegend'] = False
            
        fig.update_layout(**layout_params)
    return fig

# ==================================================================
# --- 2. テキスト処理 ---
# ==================================================================
@st.cache_resource
def load_tokenizer_explorer(): return Tokenizer()
t = load_tokenizer_explorer()

if "stopwords" in st.session_state and st.session_state["stopwords"]:
    stopwords = st.session_state["stopwords"]
else:
    stopwords = utils.get_stopwords()

_ngram_rows = [
    ("参照符号付き要素", r"[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[（(]\s*[0-9０-９A-Za-z]+[A-Za-z]?\s*[）)]", "regex", 1),
    ("参照符号付き要素", r"(?:上記|前記)?[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[0-9０-９A-Za-z]+[A-Za-z]?", "regex", 1),
    ("参照符号付き要素", r"[A-Z]+[0-9]+", "regex", 1),
    ("見出し・章句","一実施形態において","literal",1), ("見出し・章句","他の実施形態において","literal",1), ("見出し・章句","別の実施形態において","literal",1),
    ("見出し・章句","本明細書において","literal",1), ("見出し・章句","本明細書では","literal",1), ("見出し・章句","本発明の一側面","literal",1),
    ("見出し・章句","一実施例において","literal",1), ("見出し・章句","他の実施例において","literal",1), ("見出し・章句","好ましい態様として","literal",2),
    ("見出し・章句","好適には","literal",2), ("見出し・章句","用語の定義","literal",2), ("見出し・章句","図示しない","literal",2),
    ("図表参照", r"図[ 　]*[０-９0-9]+に示す", "regex", 1), ("図表参照", r"表[ 　]*[０-９0-9]+に示す", "regex", 1),
    ("図表参照", r"式[ 　]*[０-９0-9]+に示す", "regex", 1), ("図表参照", r"請求項[ 　]*[０-９0-9]+", "regex", 1),
    ("図表参照", r"(?:【|\[)\s*[０-９0-9]{4,5}\s*(?:】|\])", "regex", 1), ("図表参照", r"[（(][０-９0-9]+[）)]", "regex", 2),
    ("図表参照", r"第\s*[０-９0-9]+の?実施形態", "regex", 2), ("図表参照", r"段落\s*[０-９0-9]+", "regex", 2),
    ("図表参照", r"図[ 　]*[０-９0-9]+[A-Za-z]?", "regex", 2), ("定義導入", r"以下、[^、。]+を[^、。]+と称する", "regex", 1),
    ("定義導入", r"以下、[^、。]+を[^、。]+という", "regex", 1), ("機能句","してもよい","literal",1), ("機能句","であってもよい","literal",1),
    ("機能句","することができる","literal",1), ("機能句","行うことができる","literal",1), ("機能句","に限定されない","literal",1),
    ("機能句","に限られない","literal",1), ("機能句","一例として","literal",2), ("機能句","例示的には","literal",2),
    ("参照句","前述のとおり","literal",2), ("参照句","前述の通り","literal",2), ("参照句","後述するように","literal",2),
    ("参照句","後述のとおり","literal",2), ("範囲表現", r"少なくとも(?:一|１)つ", "regex", 2), ("範囲表現", "少なくとも一部", "literal", 2),
    ("範囲表現", r"複数の(?:実施形態|構成|要素)", "regex", 3), ("課題句", r"(?:上記|前記)の?課題", "regex", 1),
    ("接続・論理","一方で","literal",3), ("接続・論理","他方で","literal",3), ("接続・論理","すなわち","literal",3),
    ("接続・論理","したがって","literal",3), ("接続・論理","しかしながら","literal",3), ("接続・論理","例えば","literal",3),
    ("接続・論理","具体的には","literal",3), ("補助句","以下に説明する","literal",3), ("補助句","前記のとおり","literal",3),
    ("補助句","これにより","literal",3), ("補助句","このように","literal",3)
]

_ngram_compiled = sorted(_ngram_rows, key=lambda x: (x[3], -len(x[1]) if x[2]=="literal" else -50))
_ngram_compiled = [(cat, re.compile(pat) if ptype == "regex" else pat, ptype, pri) for cat, pat, ptype, pri in _ngram_compiled]

def normalize_text(text):
    if not isinstance(text, str): text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("µ", "μ")
    text = re.sub(r"\s+", " ", text)
    return text

def apply_ngram_filters(text):
    for cat, pat, ptype, pri in _ngram_compiled:
        if ptype == "literal":
            if pat in text: text = text.replace(pat, "")
        else:
            text = pat.sub("", text)
    return text

@st.cache_data
def extract_compound_nouns(text):
    text = normalize_text(text)
    text = apply_ngram_filters(text) 
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)

    tokens = t.tokenize(text)
    words, compound_word = [], ''
    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        if pos == '名詞':
            compound_word += token.surface
        else:
            if (len(compound_word) > 1 and
                compound_word not in stopwords and
                not re.fullmatch(r'[\d０-９]+', compound_word) and
                not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
                not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
                not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
                not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
                words.append(compound_word)
            compound_word = ''
            
    if (len(compound_word) > 1 and
        compound_word not in stopwords and
        not re.fullmatch(r'[\d０-９]+', compound_word) and
        not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
        not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
        not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
        not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
        words.append(compound_word)
    return words

def generate_wordcloud_and_list(words, title, top_n=20, font_path=None):
    if not words: return None
    word_freq = Counter(words)
    try:
        wc = WordCloud(
            width=800, height=400, background_color='white',
            font_path=font_path, collocations=False,
            max_words=100
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=20)
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")

# ==================================================================
# --- 3. UI & アプリケーション ---
# ==================================================================

utils.render_sidebar()

st.title("🧭 Explorer")
st.markdown("""
Explorer (戦略的キーワード探索) は、特許文書内の専門用語を抽出し、市場全体のトレンド変遷や競合他社との戦略的差異を多角的に分析するモジュールです。
急上昇ワードの特定、時系列での技術推移、そして企業間のポジショニング比較を通じて、次の一手となるインサイトを発掘します。
""")

col_theme, _ = st.columns([1, 3])
with col_theme:
    selected_theme = st.selectbox("表示テーマ:", ["APOLLO Standard", "Modern Presentation"], key="exp_theme")
theme_config = utils.get_theme_config(selected_theme)
st.markdown(f"<style>{theme_config['css']}</style>", unsafe_allow_html=True)

# データロード
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。Mission Controlでデータをロードしてください。"); st.stop()

df_main = st.session_state.df_main
col_map = st.session_state.col_map
delimiters = st.session_state.delimiters

# 出願人リスト生成
app_counts = pd.Series()
if col_map['applicant'] in df_main.columns:
    if 'applicant_main' in df_main.columns:
        app_series = df_main['applicant_main'].explode().dropna()
    else:
        app_series = df_main[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
    
    app_counts = app_series[app_series != ''].value_counts()

sorted_applicants = app_counts.index.tolist()
app_count_dict = app_counts.to_dict()

# 前処理
# 前処理 ("explorer_keywords"カラムがない場合のみ実行)
if 'explorer_keywords' not in df_main.columns:
    with st.spinner("Explorer: テキスト解析とキーワード抽出を実行中..."):
        df_main['explorer_text'] = df_main[col_map['title']].fillna('') + ' ' + df_main[col_map['abstract']].fillna('')
        df_main['explorer_keywords'] = df_main['explorer_text'].apply(extract_compound_nouns)
        st.session_state.df_main = df_main

# モード選択
selected_tab = st.radio(
    "分析モードを選択:",
    ["Global Overview", "Trend Analysis", "Comparative Strategy", "Context Search (KWIC)"],
    horizontal=True
)

st.markdown("---")

# ==================================================================
# --- 4. Global Overview (全体俯瞰) ---
# ==================================================================
if selected_tab == "Global Overview":
    st.subheader("Global Overview")
    
    top_n_cloud = st.number_input("ワードクラウド単語数", 10, 100, 50, key="go_cloud_n")
    all_keywords = [w for sublist in df_main['explorer_keywords'] for w in sublist]
    word_counts = Counter(all_keywords)
    
    if not word_counts:
        st.warning("有効なキーワードがありません。")
    else:
        st.markdown("##### 1. 全体ワードクラウド")
        generate_wordcloud_and_list(all_keywords, "全体ワードクラウド", top_n_cloud, FONT_PATH)

        st.markdown("##### 2. 全体共起ネットワーク (技術クラスター)")
        col_net1, col_net2 = st.columns(2)
        with col_net1: global_net_top_n = st.slider("抽出単語数 (Top N)", 30, 100, 60, key="global_net_n")
        with col_net2: global_net_threshold = st.slider("共起閾値 (Jaccard)", 0.01, 0.3, 0.05, 0.01, key="global_net_th")

        with st.spinner("全体ネットワーク計算中..."):
            c_all = Counter(all_keywords)
            top_nodes_global = [w for w, c in c_all.most_common(global_net_top_n)]
            pair_counts_global = Counter()
            for kws in df_main['explorer_keywords']:
                valid_w = [w for w in set(kws) if w in top_nodes_global]
                if len(valid_w) >= 2:
                    for pair in combinations(sorted(valid_w), 2): pair_counts_global[pair] += 1
            
            G_global = nx.Graph()
            for w in top_nodes_global: G_global.add_node(w, size=c_all[w])
            for (u, v), c in pair_counts_global.items():
                weight = c / (c_all[u] + c_all[v] - c)
                if weight >= global_net_threshold: G_global.add_edge(u, v, weight=weight)
            G_global.remove_nodes_from(list(nx.isolates(G_global)))
            
            if G_global.number_of_nodes() > 0:
                communities = community.greedy_modularity_communities(G_global)
                community_map = {}; pos_global = nx.spring_layout(G_global, k=0.8, seed=42)
                for i, comm in enumerate(communities):
                    for node in comm: community_map[node] = i
                
                edge_x, edge_y = [], []
                for edge in G_global.edges():
                    x0, y0 = pos_global[edge[0]]; x1, y1 = pos_global[edge[1]]
                    edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                
                edge_trace_g = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
                node_x, node_y, node_color, node_text = [], [], [], []
                for node in G_global.nodes():
                    node_x.append(pos_global[node][0]); node_y.append(pos_global[node][1])
                    node_color.append(community_map.get(node, 0))
                    node_text.append(f"{node} ({c_all[node]}件)")
                
                node_trace_g = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text',
                    text=[n for n in G_global.nodes()], textposition="top center",
                    hovertext=node_text, hoverinfo="text",
                    marker=dict(showscale=False, colorscale='Turbo', color=node_color, size=[np.log(c_all[n]+1)*8 for n in G_global.nodes()], line_width=1)
                )
                fig_net_g = go.Figure(data=[edge_trace_g, node_trace_g])
                fig_net_g.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                update_fig_layout(fig_net_g, "Global Co-occurrence Network", height=700, theme_config=theme_config)
                st.plotly_chart(fig_net_g, use_container_width=True, config={'editable': False})
                
                # --- Snapshot (Global Network) ---
                # 1. Community Structure
                comm_summary = []
                for i in range(len(communities)):
                    comm_words = list(communities[i])[:5] 
                    comm_summary.append(f"Group {i+1}: {', '.join(comm_words)}")
                
                # 2. Hubs (Degree Centrality)
                deg_centrality = nx.degree_centrality(G_global)
                sorted_hubs = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                hubs_str = ", ".join([f"{n}({val:.2f})" for n, val in sorted_hubs])

                # 3. Strongest Edges
                sorted_edges = sorted(G_global.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
                edges_str = ", ".join([f"{u}-{v}" for u, v, d in sorted_edges])

                # Combine with Rich Summary
                snap_data = utils.generate_rich_summary(df_main, title_col=col_map['title'], abstract_col=col_map['abstract'])
                snap_data['module'] = 'Explorer'
                snap_data['network_stats'] = {
                    "communities": "; ".join(comm_summary),
                    "hubs": hubs_str,
                    "edges": edges_str
                }
                
                utils.render_snapshot_button(
                    title="全体共起ネットワーク (技術クラスター)",
                    description="全期間を通じた技術用語の共起関係とクラスター構造。",
                    key="exp_global_net_snap",
                    fig=fig_net_g,
                    data_summary=snap_data
                )
            else: st.warning("条件に一致する共起関係が見つかりませんでした。")

# ==================================================================
# --- 5. Trend Analysis (時系列分析) ---
# ==================================================================
elif selected_tab == "Trend Analysis":
    st.subheader("Trend Analysis")
    
    target_filter = st.selectbox(
        "分析対象:", 
        ["全体 (Market)"] + sorted_applicants, 
        format_func=lambda x: f"{x} (全{len(df_main)}件)" if x == "全体 (Market)" else f"{x} ({app_count_dict.get(x, 0)}件)",
        key="trend_target"
    )
    
    if target_filter == "全体 (Market)":
        df_target = df_main
    else:
        if 'applicant_main' in df_main.columns:
            mask = df_main['applicant_main'].apply(lambda x: isinstance(x, list) and target_filter in x)
        else:
            mask = df_main[col_map['applicant']].fillna('').str.contains(re.escape(target_filter))
        df_target = df_main[mask]
        
    st.info(f"分析対象: {target_filter} ({len(df_target)}件)")
    
    if df_target.empty:
        st.warning("データがありません。")
    else:
        current_year = int(df_target['year'].max())
        min_year = int(df_target['year'].min())
        
        interval_years = st.slider("期間の粒度 (年)", 1, 10, 5, key="ta_interval")
        
        periods = []
        c_end = current_year
        while c_end >= min_year:
            c_start = c_end - interval_years + 1
            real_start = max(min_year, c_start)
            periods.append((real_start, c_end))
            c_end -= interval_years
            if c_end < min_year: break
        
        # 1. 急上昇キーワード
        st.markdown(f"##### 1. 急上昇キーワード (Growth Rate)")
        if len(periods) > 1:
            st.caption(f"比較期間: [{periods[0][0]}-{periods[0][1]}] vs [{periods[1][0]}-{periods[1][1]}]")
            df_recent = df_target[(df_target['year'] >= periods[0][0]) & (df_target['year'] <= periods[0][1])]
            df_past = df_target[(df_target['year'] >= periods[1][0]) & (df_target['year'] <= periods[1][1])]
            
            c_recent = Counter([w for sublist in df_recent['explorer_keywords'] for w in sublist])
            c_past = Counter([w for sublist in df_past['explorer_keywords'] for w in sublist])
            
            growth_data = []
            min_freq = max(2, len(df_recent) * 0.01)
            for word, count_r in c_recent.items():
                if count_r < min_freq: continue
                count_p = c_past.get(word, 0)
                growth_rate = (count_r - count_p) / (count_p + 1)
                growth_data.append({"Keyword": word, "Growth Rate": growth_rate})
            
            df_growth = pd.DataFrame(growth_data).sort_values("Growth Rate", ascending=False).head(20)
            if not df_growth.empty:
                fig_growth = px.bar(df_growth, x="Growth Rate", y="Keyword", orientation='h', color="Growth Rate", color_continuous_scale="Reds")
                fig_growth.update_layout(yaxis={'categoryorder':'total ascending'})
                update_fig_layout(fig_growth, "Growth Rate Top 20", height=500, theme_config=theme_config)
                st.plotly_chart(fig_growth, use_container_width=True, config={'editable': False})
                
                utils.render_snapshot_button(
                    title="急上昇キーワード (Growth Rate)",
                    description=f"直近期間 [{periods[0][0]}-{periods[0][1]}] で急増したキーワード。",
                    key="exp_growth_snap",
                    fig=fig_growth,
                    data_summary=df_growth.to_string(index=False)
                )
        else:
            st.warning("比較対象となる過去のデータ期間が不足しています。")

        # 2. 時系列マルチ・ワードクラウド
        st.markdown(f"##### 2. 時系列ワードクラウド (Time-Lapse)")
        cols = st.columns(3)
        for i, (start, end) in enumerate(periods):
            with cols[i % 3]:
                df_p = df_target[(df_target['year'] >= start) & (df_target['year'] <= end)]
                kws_p = [w for sublist in df_p['explorer_keywords'] for w in sublist]
                st.markdown(f"**{start} - {end}** ({len(df_p)}件)")
                if kws_p: generate_wordcloud_and_list(kws_p, f"{start}-{end}", 30, FONT_PATH)
            
        # 3. トレンド・ネットワーク
        st.markdown(f"##### 3. トレンド・共起ネットワーク (赤=急上昇 / 青=停滞)")
        col_net1, col_net2 = st.columns(2)
        with col_net1: ta_net_n = st.slider("抽出単語数", 30, 100, 60, key="ta_net_n")
        with col_net2: ta_net_th = st.slider("共起閾値", 0.01, 0.3, 0.05, 0.01, key="ta_net_th")
        
        all_target_kw = [w for sublist in df_target['explorer_keywords'] for w in sublist]
        c_all = Counter(all_target_kw)
        top_nodes = [w for w, c in c_all.most_common(ta_net_n)]
        
        pair_counts = Counter()
        for kws in df_target['explorer_keywords']:
            valid_w = [w for w in set(kws) if w in top_nodes]
            if len(valid_w) >= 2:
                for pair in combinations(sorted(valid_w), 2): pair_counts[pair] += 1
                
        G = nx.Graph()
        for w in top_nodes: G.add_node(w, size=c_all[w])
        for (u, v), c in pair_counts.items():
            weight = c / (c_all[u] + c_all[v] - c)
            if weight >= ta_net_th: G.add_edge(u, v, weight=weight)
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=0.8, seed=42)
            node_colors, node_texts = [], []
            
            c_rec_net = Counter([w for sublist in df_target[df_target['year'] >= periods[0][0]]['explorer_keywords'] for w in sublist])
            if len(periods) > 1:
                c_pst_net = Counter([w for sublist in df_target[(df_target['year'] >= periods[1][0]) & (df_target['year'] <= periods[1][1])]['explorer_keywords'] for w in sublist])
            else:
                c_pst_net = Counter()

            for node in G.nodes():
                gr = (c_rec_net.get(node, 0) - c_pst_net.get(node, 0)) / (c_pst_net.get(node, 0) + 1)
                node_colors.append(gr)
                node_texts.append(f"{node}<br>Growth: {gr:.2f}")
            
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
            node_trace = go.Scatter(
                x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                mode='markers+text', text=list(G.nodes()), textposition="top center",
                hovertext=node_texts, hoverinfo="text",
                marker=dict(showscale=True, colorscale='RdBu_r', color=node_colors, size=[np.log(G.nodes[n]['size']+1)*8 for n in G.nodes()], line_width=1, colorbar=dict(title="Growth"))
            )
            fig_net = go.Figure(data=[edge_trace, node_trace])
            fig_net.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
            update_fig_layout(fig_net, "Trend Network", height=700, theme_config=theme_config)
            st.plotly_chart(fig_net, use_container_width=True, config={'editable': False})

            # --- Snapshot (Trend Network) ---
            trend_summary = [f"{n}: Growth {c_rec_net.get(n,0)/(c_pst_net.get(n,0)+1):.2f}" for n in list(G.nodes())[:15]]
            
            # Hubs & Edges
            deg_centrality = nx.degree_centrality(G)
            sorted_hubs = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            hubs_str = ", ".join([f"{n}({val:.2f})" for n, val in sorted_hubs])
            
            sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
            edges_str = ", ".join([f"{u}-{v}" for u, v, d in sorted_edges])

            snap_data = utils.generate_rich_summary(df_target, title_col=col_map['title'], abstract_col=col_map['abstract'])
            snap_data['module'] = 'Explorer'
            snap_data['network_stats'] = {
                "hubs": hubs_str,
                "edges": edges_str,
                "notes": "Nodes colored by Growth Rate (Red=High, Blue=Low)."
            }

            utils.render_snapshot_button(
                title="トレンド・共起ネットワーク",
                description="技術用語の共起関係に成長率（赤=急上昇）を重ね合わせたマップ。",
                key="exp_trend_net_snap",
                fig=fig_net,
                data_summary=snap_data
            )

# ==================================================================
# --- 6. Comparative Strategy (競合比較) ---
# ==================================================================
elif selected_tab == "Comparative Strategy":
    st.subheader("Comparative Strategy")
    
    c1, c2 = st.columns(2)
    with c1: 
        my_comp = st.selectbox(
            "自社 (My Company)", 
            ["(選択してください)"] + sorted_applicants, 
            format_func=lambda x: x if x == "(選択してください)" else f"{x} ({app_count_dict.get(x, 0)}件)",
            key="comp_my"
        )
    with c2: 
        target_comp = st.selectbox(
            "競合他社 (Competitor)", 
            ["(選択してください)"] + sorted_applicants, 
            format_func=lambda x: x if x == "(選択してください)" else f"{x} ({app_count_dict.get(x, 0)}件)",
            key="comp_target"
        )
    
    if my_comp != "(選択してください)" and target_comp != "(選択してください)":
        def get_keywords_for_app(app_name):
            if 'applicant_main' in df_main.columns:
                mask = df_main['applicant_main'].apply(lambda x: isinstance(x, list) and app_name in x)
            else:
                mask = df_main[col_map['applicant']].fillna('').str.contains(re.escape(app_name))
            return [w for sublist in df_main[mask]['explorer_keywords'] for w in sublist]

        words_my = get_keywords_for_app(my_comp)
        words_target = get_keywords_for_app(target_comp)
        c_my = Counter(words_my)
        c_tgt = Counter(words_target)
        
        # 1. Tornado Chart
        st.markdown("##### 1. キーワード出現頻度比較 (Tornado Chart)")
        all_keys = set(list(c_my.keys()) + list(c_tgt.keys()))
        valid_keys = [k for k in all_keys if (c_my[k] + c_tgt[k]) >= 3]
        
        tornado_data = []
        for k in valid_keys:
            tornado_data.append({
                "Keyword": k, "My Count": -c_my[k], # Left (Negative)
                "Competitor Count": c_tgt[k],       # Right (Positive)
                "My Abs": c_my[k], 
                "Total": c_my[k] + c_tgt[k]
            })
        df_tornado = pd.DataFrame(tornado_data).sort_values("Total", ascending=True).tail(30)
        
        if not df_tornado.empty:
            max_val = max(df_tornado["My Abs"].max(), df_tornado["Competitor Count"].max())
            range_x = [-max_val * 1.1, max_val * 1.1]
            tick_vals = [-max_val, -max_val/2, 0, max_val/2, max_val]
            tick_text = [str(int(abs(v))) for v in tick_vals]

            fig_tornado = go.Figure()
            fig_tornado.add_trace(go.Bar(y=df_tornado["Keyword"], x=df_tornado["My Count"], orientation='h', name=my_comp, marker_color=theme_config["color_sequence"][0]))
            fig_tornado.add_trace(go.Bar(y=df_tornado["Keyword"], x=df_tornado["Competitor Count"], orientation='h', name=target_comp, marker_color=theme_config["color_sequence"][1]))
            
            fig_tornado.update_layout(
                barmode='relative', bargap=0.1, 
                xaxis=dict(
                    title="出現件数 (左: 自社 / 右: 競合)", 
                    tickmode='array', tickvals=tick_vals, ticktext=tick_text,
                    range=range_x,
                    showline=True, linewidth=1, linecolor='black'
                ),
                yaxis=dict(side='right', showline=True, linewidth=1, linecolor='black'),
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                margin=dict(r=150, l=20, b=100)
            )
            update_fig_layout(fig_tornado, "Tornado Chart", height=800, theme_config=theme_config)
            st.plotly_chart(fig_tornado, use_container_width=True, config={'editable': False})
            st.info("左側 (青系) が自社、右側 (赤/オレンジ系) が競合の出現数を示します。")
            
            utils.render_snapshot_button(
                title=f"Tornado Chart ({my_comp} vs {target_comp})",
                description="キーワード出現頻度の直接比較。左右への突出が各社の特徴を示す。",
                key="exp_tornado_snap",
                fig=fig_tornado,
                data_summary=df_tornado[['Keyword', 'My Abs', 'Competitor Count']].tail(20).to_string(index=False)
            )

        # 2. ワードクラウド
        st.markdown("##### 2. 企業別ワードクラウド")
        c_wc1, c_wc2 = st.columns(2)
        with c_wc1:
            st.markdown(f"**{my_comp}**")
            if words_my: generate_wordcloud_and_list(words_my, my_comp, 30, FONT_PATH)
        with c_wc2:
            st.markdown(f"**{target_comp}**")
            if words_target: generate_wordcloud_and_list(words_target, target_comp, 30, FONT_PATH)
            
        # 3. 支配率ネットワーク
        st.markdown(f"##### 3. 支配率ネットワーク (青=自社優勢 / 赤=競合優勢)")
        col_cs1, col_cs2 = st.columns(2)
        with col_cs1: cs_net_n = st.slider("抽出単語数", 30, 100, 60, key="cs_net_n")
        with col_cs2: cs_net_th = st.slider("共起閾値", 0.01, 0.3, 0.05, 0.01, key="cs_net_th")
        
        combined_keywords = words_my + words_target
        c_combined = Counter(combined_keywords)
        top_nodes = [w for w, c in c_combined.most_common(cs_net_n)]
        
        if 'applicant_main' in df_main.columns:
            mask_2 = df_main['applicant_main'].apply(lambda x: isinstance(x, list) and (my_comp in x or target_comp in x))
        else:
            mask_2 = df_main[col_map['applicant']].fillna('').str.contains(re.escape(my_comp) + "|" + re.escape(target_comp))
        df_2 = df_main[mask_2]
        
        pair_counts = Counter()
        for kws in df_2['explorer_keywords']:
            valid_w = [w for w in set(kws) if w in top_nodes]
            if len(valid_w) >= 2:
                for pair in combinations(sorted(valid_w), 2): pair_counts[pair] += 1
                
        G = nx.Graph()
        for w in top_nodes: G.add_node(w, size=c_combined[w])
        for (u, v), c in pair_counts.items():
            weight = c / (c_combined[u] + c_combined[v] - c)
            if weight >= cs_net_th: G.add_edge(u, v, weight=weight)
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=0.8, seed=42)
            node_colors, node_texts = [], []
            for node in G.nodes():
                m = c_my[node]; t = c_tgt[node]
                if m + t == 0: dom = 0.5
                else: dom = m / (m + t)
                node_colors.append(dom)
                node_texts.append(f"{node}<br>{my_comp}: {m}<br>{target_comp}: {t}")
            
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
            node_trace = go.Scatter(
                x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                mode='markers+text', text=list(G.nodes()), textposition="top center",
                hovertext=node_texts, hoverinfo="text",
                marker=dict(showscale=True, colorscale='RdBu', color=node_colors, size=[np.log(G.nodes[n]['size']+1)*8 for n in G.nodes()], line_width=1, colorbar=dict(title="Dominance"))
            )
            fig_net = go.Figure(data=[edge_trace, node_trace])
            fig_net.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
            update_fig_layout(fig_net, "Dominance Network", height=700, theme_config=theme_config)
            st.plotly_chart(fig_net, use_container_width=True, config={'editable': False})

            # --- Snapshot (Dominance Network) ---
            dom_summary = []
            for n in list(G.nodes())[:20]: # Top nodes
                m = c_my[n]; t = c_tgt[n]
                dom_val = m/(m+t) if (m+t)>0 else 0.5
                dom_summary.append(f"{n}: {my_comp}={m}, {target_comp}={t} (Dom={dom_val:.2f})")

            # Hubs & Edges
            deg_centrality = nx.degree_centrality(G)
            sorted_hubs = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            hubs_str = ", ".join([f"{n}({val:.2f})" for n, val in sorted_hubs])
            
            sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
            edges_str = ", ".join([f"{u}-{v}" for u, v, d in sorted_edges])

            snap_data = utils.generate_rich_summary(df_2, title_col=col_map['title'], abstract_col=col_map['abstract'])
            snap_data['module'] = 'Explorer'
            snap_data['network_stats'] = {
                "hubs": hubs_str,
                "edges": edges_str,
                "notes": f"Dominance: {my_comp} vs {target_comp}. Blue favorable to {my_comp}, Red favorable to {target_comp}."
            }

            utils.render_snapshot_button(
                title=f"Dominance Network ({my_comp} vs {target_comp})",
                description="共起ネットワーク上での両社の優劣（支配率）分布。",
                key="exp_dom_net_snap",
                fig=fig_net,
                data_summary=snap_data
            )

# ==================================================================
# --- 7. Context Search (文脈検索) ---
# ==================================================================
elif selected_tab.startswith("Context Search"):
    st.subheader("Context Search (KWIC: KeyWord In Context)")
    search_kw = st.text_input("検索したいキーワードを入力してください:", "")
    
    if search_kw:
        mask = df_main['explorer_text'].str.contains(re.escape(search_kw), na=False)
        df_hit = df_main[mask]
        st.write(f"ヒット件数: {len(df_hit)} 件")
        
        if not df_hit.empty:
            def highlight_text(text, kw):
                if pd.isna(text): return ""
                matches = [m.start() for m in re.finditer(re.escape(kw), text)]
                if not matches: return text[:100] + "..."
                snippets = []
                for idx in matches[:3]: 
                    start = max(0, idx - 40); end = min(len(text), idx + len(kw) + 40)
                    snippet = text[start:end].replace(kw, f"**{kw}**")
                    snippets.append(f"...{snippet}...")
                return " / ".join(snippets)

            for i, row in df_hit.head(20).iterrows():
                with st.expander(f"{row[col_map['title']]} ({row['year']}) - {row.get(col_map['applicant'], '')}"):
                    if col_map['abstract'] and pd.notna(row[col_map['abstract']]):
                        st.markdown(highlight_text(row[col_map['abstract']], search_kw))
                    st.caption(f"出願番号: {row.get(col_map['app_num'], 'N/A')}")