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
import utils
import utils_ai
import patiroha
utils.configure_matplotlib_font()

# ==================================================================
# --- 1. 設定・ヘルパー関数 ---
# ==================================================================
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="APOLLO v8 | Explorer",
    page_icon="🧭",
    layout="wide"
)

st.session_state['current_page'] = 'Explorer'



FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except:
        pass





def update_fig_layout(fig, title, height=600, show_legend=True):
    # タイトルのサニタイズ
    if isinstance(title, str):
        title = re.sub(r'<[^>]+>', '', title)

    layout_params = dict(
        template=utils.APOLLO_TEMPLATE,
        title=dict(text=title, font=dict(size=18, color=utils.APOLLO_TEXT, weight="normal")),
        paper_bgcolor=utils.APOLLO_BG, plot_bgcolor=utils.APOLLO_BG,
        font=dict(color=utils.APOLLO_TEXT, family="Helvetica Neue"),
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
    stopwords = patiroha.get_stopwords()

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



def generate_wordcloud_and_list(words, title, top_n=20, font_path=None, capcom_key=None):
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

        # CAPCOM: ワードクラウドデータ保存
        if capcom_key:
            try:
                import capcom
                if capcom.is_active():
                    import io
                    wc_data = {
                        "metadata": {"module": "Explorer", "title": title, "top_n": top_n},
                        "word_frequencies": {w: c for w, c in word_freq.most_common(100)}
                    }
                    capcom.save_data(f"{capcom_key}_wordcloud.json", wc_data)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    capcom.save_snapshot_image(f"{capcom_key}_wordcloud", buf.read())
            except Exception:
                pass

        # VOYAGERスナップショットボタン
        if capcom_key:
            utils.render_snapshot_button(
                title=f"ワードクラウド: {title}",
                description=f"TF-IDFワードクラウド（上位{top_n}語）",
                key=f"{capcom_key}_wordcloud",
                fig=fig,
                data_summary={
                    "module": "Explorer",
                    "type": "wordcloud",
                    "title": title,
                    "top_words": [{"word": w, "freq": c} for w, c in word_freq.most_common(top_n)]
                }
            )
    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")

# ==================================================================
# --- 3. UI & アプリケーション ---
# ==================================================================

utils.render_sidebar()

st.title("🧭 Explorer")
st.markdown("キーワード共起ネットワーク・急上昇トレンド・競合比較トルネードチャートで、技術キーワードの戦略的ポジションを分析します。")
st.markdown("""
Explorer (戦略的キーワード探索) は、特許文書内の専門用語を抽出し、市場全体のトレンド変遷や競合他社との戦略的差異を多角的に分析するモジュールです。
急上昇ワードの特定、時系列での技術推移、そして企業間のポジショニング比較を通じて、次の一手となるインサイトを発掘します。
""")

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

# 前処理 (Home.pyで作成された 'explorer_keywords' を使用。存在しない場合のみ計算)
target_col_keywords = 'explorer_keywords'
if target_col_keywords not in df_main.columns:
    with st.spinner("Explorer: テキスト解析とキーワード抽出を実行中..."):
        df_main['explorer_text'] = df_main[col_map['title']].fillna('') + ' ' + df_main[col_map['abstract']].fillna('')
        df_main[target_col_keywords] = df_main['explorer_text'].apply(
            lambda x: utils.extract_keywords(x, stopwords=stopwords)
        )
        st.session_state.df_main = df_main

# explorer_textが無い場合の自己修復
if 'explorer_text' not in df_main.columns:
    df_main['explorer_text'] = df_main[col_map['title']].fillna('') + ' ' + df_main[col_map['abstract']].fillna('')

# モード選択
selected_tab = st.radio(
    "分析モードを選択:",
    ["全体俯瞰 (Global Overview)", "トレンド分析 (Trend Analysis)", "競合比較戦略 (Comparative Strategy)", "文脈検索 (Context Search/KWIC)"],
    horizontal=True
)

st.markdown("---")

# ==================================================================
# --- 4. Global Overview (全体俯瞰) ---
# ==================================================================
if selected_tab == "全体俯瞰 (Global Overview)":
    st.subheader("Global Overview")
    
    top_n_cloud = st.number_input("ワードクラウド単語数", 10, 100, 50, key="go_cloud_n")
    all_keywords = [w for sublist in df_main['explorer_keywords'] for w in sublist]
    word_counts = Counter(all_keywords)
    
    if not word_counts:
        st.warning("有効なキーワードがありません。")
    else:
        st.markdown("##### 1. 全体ワードクラウド")
        generate_wordcloud_and_list(all_keywords, "全体ワードクラウド", top_n_cloud, FONT_PATH, capcom_key="explorer_global")

        st.markdown("##### 2. 全体共起ネットワーク (技術クラスター)")
        col_net1, col_net2 = st.columns(2)
        with col_net1: global_net_top_n = st.slider("抽出単語数 (Top N)", 30, 100, 70, key="global_net_n")
        with col_net2: global_net_threshold = st.slider("共起閾値 (Jaccard)", 0.01, 0.3, 0.03, 0.01, key="global_net_th")

        with st.spinner("全体ネットワーク計算中..."):
            c_all = Counter(all_keywords)
            top_nodes_global = [w for w, c in c_all.most_common(global_net_top_n)]
            # patirohaで共起グラフ構築 (上位キーワードに絞った文書リスト)
            keyword_lists_filtered = [
                [w for w in kws if w in top_nodes_global]
                for kws in df_main['explorer_keywords']
            ]
            G_global = patiroha.build_cooccurrence_graph(
                keyword_lists_filtered,
                top_n=global_net_top_n,
                threshold=global_net_threshold,
                similarity="jaccard",
            )
            # 共起回数を保持（スナップショット用）
            pair_counts_global = Counter()
            for kws in df_main['explorer_keywords']:
                valid_w = [w for w in set(kws) if w in top_nodes_global]
                if len(valid_w) >= 2:
                    for pair in combinations(sorted(valid_w), 2): pair_counts_global[pair] += 1

            if G_global.number_of_nodes() > 0:
                # patiroha.detect_communities は dict[str, int]（ノード名→コミュニティID）を返す
                community_map = patiroha.detect_communities(G_global, algorithm="louvain")
                pos_global = nx.spring_layout(G_global, k=0.8, seed=42)
                
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
                update_fig_layout(fig_net_g, "Global Co-occurrence Network", height=700)
                st.plotly_chart(fig_net_g, use_container_width=True, config={'editable': False})
                
                # --- スナップショット (全体ネットワーク) ---
                # 1. コミュニティ構造（構造化）
                hub_keywords = patiroha.get_hub_keywords(G_global, centrality="degree")
                deg_centrality = {kw: score for kw, score in hub_keywords}
                # community_map (dict[str,int]) からコミュニティ構造を再構築
                from collections import defaultdict
                _comm_groups = defaultdict(list)
                for node, cid in community_map.items():
                    _comm_groups[cid].append(node)

                communities_structured = []
                for cid in sorted(_comm_groups.keys()):
                    members = _comm_groups[cid]
                    hub_node = max(members, key=lambda n: deg_centrality.get(n, 0))
                    communities_structured.append({
                        "id": cid,
                        "size": len(members),
                        "members": members,
                        "hub": hub_node,
                        "hub_centrality": round(deg_centrality.get(hub_node, 0), 4)
                    })
                # 旧形式（スナップショットの表示互換用）
                comm_summary = []
                for cs in communities_structured:
                    comm_summary.append(f"Group {cs['id']+1}: {', '.join(cs['members'][:5])}")

                # 2. 全ノード中心性ランキング
                hubs_ranked = []
                for node, cent in sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True):
                    hubs_ranked.append({
                        "keyword": node,
                        "degree_centrality": round(cent, 4),
                        "frequency": c_all[node],
                        "community": next((cs["id"] for cs in communities_structured if node in cs["members"]), -1)
                    })

                # 3. エッジ（Jaccard係数+共起回数付き）上位100
                edges_ranked = []
                for u, v, d in sorted(G_global.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:100]:
                    edges_ranked.append({
                        "source": u, "target": v,
                        "jaccard": round(d['weight'], 4),
                        "cooccurrence_count": pair_counts_global.get(tuple(sorted([u, v])), 0)
                    })

                # 4. ブリッジエッジ（コミュニティ間を結ぶエッジ）
                bridge_edges = []
                for u, v, d in G_global.edges(data=True):
                    cu = next((cs["id"] for cs in communities_structured if u in cs["members"]), -1)
                    cv = next((cs["id"] for cs in communities_structured if v in cs["members"]), -1)
                    if cu != cv and cu >= 0 and cv >= 0:
                        bridge_edges.append({
                            "source": u, "target": v,
                            "communities": [cu, cv],
                            "jaccard": round(d['weight'], 4)
                        })
                bridge_edges.sort(key=lambda x: x['jaccard'], reverse=True)

                # リッチサマリーと統合
                # ネットワーク文脈用のキーワード中心抽出を使用
                top_kw_global = [w for w, c in c_all.most_common(20)]
                network_reps = utils.get_keyword_centric_representatives(df_main, top_kw_global, n_reps=10)

                # Format for summary
                rep_lines = ["\n**代表的特許 (ネットワーク中心性ベース):**"]
                for i, r in enumerate(network_reps):
                     rep_lines.append(f"{i+1}. 【{r['title']}】 ({r['applicant']}) - {r['abstract'][:80]}...")

                snap_data = utils.generate_rich_summary(df_main, title_col=col_map['title'], abstract_col=col_map['abstract'])
                snap_data['module'] = 'Explorer'
                snap_data['network_stats'] = {
                    "nodes": G_global.number_of_nodes(),
                    "edges": G_global.number_of_edges(),
                    "density": round(nx.density(G_global), 4),
                    "communities": communities_structured,
                    "communities_display": "; ".join(comm_summary),
                    "hubs_ranked": hubs_ranked,
                    "edges_ranked": edges_ranked,
                    "bridge_edges": bridge_edges[:20]
                }
                # キーワード中心の代表特許を注入
                if network_reps:
                    snap_data['cluster_summary'] = (snap_data.get('cluster_summary', '') + "\n".join(rep_lines))

                # AIインサイト (全体) - 上部へ移動
                insight_context_g = f"""
                **チャートタイプ**: 全体共起ネットワーク (Explorer)
                **対象データ**: 全データの共起頻度上位 {global_net_top_n} キーワード。
                **手法**: 複合名詞の共起分析 + モジュラリティ最適化によるコミュニティ検出。
                **視覚的エンコーディング**:
                - **ノード**: 技術キーワード。サイズは出現頻度。
                - **エッジ**: 共起関係（つながりの強さ）。
                - **色**: 自動検出されたコミュニティ（技術群）。
                **目的**: 技術体系の全体構造と、主要な技術テーマ群を理解すること。
                """
                insight_role = "あなたは技術トレンドの専門家です。共起ネットワークから技術体系を読み解きます。"
                insight_inst_g = """
                ネットワーク図の統計情報を元に分析してください：
                1. **主要テーマ**: どのような技術コミュニティ（グループ）が形成されていますか？
                2. **ハブ**: 中心的な役割を果たしている技術概念（ハブ）は何ですか？
                3. **関係性**: 強く結びついている技術ペアから、この分野の技術的な「常識」や「基本構成」を読み取ってください。
                """
                
                full_context_g = f"""
### AI Insight Context (Auto-Generated)
{insight_context_g}

### Analyst Instructions
{insight_inst_g}
"""
                snap_data['ai_insight_context'] = full_context_g

                utils.render_snapshot_button(
                    title="全体共起ネットワーク (技術クラスター)",
                    description="全期間を通じた技術用語の共起関係とクラスター構造。",
                    fig=fig_net_g,
                    data_summary=snap_data,
                    key="exp_global_snap"
                )

                # AI Insight Button
                insight_context_g = f"""

                **チャートタイプ**: 全体共起ネットワーク (Explorer)
                **対象データ**: 全データの共起頻度上位 {global_net_top_n} キーワード。
                **手法**: 複合名詞の共起分析 + モジュラリティ最適化によるコミュニティ検出。
                **視覚的エンコーディング**:
                - **ノード**: 技術キーワード。サイズは出現頻度。
                - **エッジ**: 共起関係（つながりの強さ）。
                - **色**: 自動検出されたコミュニティ（技術群）。
                **目的**: 技術体系の全体構造と、主要な技術テーマ群を理解すること。
                """
                insight_role = "あなたは技術トレンドの専門家です。共起ネットワークから技術体系を読み解きます。"
                insight_inst_g = """
                ネットワーク図の統計情報を元に分析してください：
                1. **主要テーマ**: どのような技術コミュニティ（グループ）が形成されていますか？
                2. **ハブ**: 中心的な役割を果たしている技術概念（ハブ）は何ですか？
                3. **関係性**: 強く結びついている技術ペアから、この分野の技術的な「常識」や「基本構成」を読み取ってください。
                """
                prompt_g = utils_ai.generate_ai_insight_prompt(insight_role, insight_context_g, snap_data, insight_inst_g)
                utils_ai.render_ai_insight_button(prompt_g, "exp_global_insight")

                # CAPCOM data/ JSON出力
                try:
                    import capcom
                    if capcom.is_active():
                        capcom_nw_data = {
                            "metadata": {
                                "module": "Explorer",
                                "mode": "global_network",
                                "n_nodes": G_global.number_of_nodes(),
                                "n_edges_total": G_global.number_of_edges(),
                                "n_edges_exported": len(edges_ranked),
                                "density": round(nx.density(G_global), 4),
                                "top_n": global_net_top_n,
                                "threshold": global_net_threshold
                            },
                            "nodes": hubs_ranked,
                            "edges": edges_ranked,
                            "communities": communities_structured,
                            "bridge_edges": bridge_edges[:20]
                        }
                        capcom.save_data("explorer_global_network.json", capcom_nw_data)
                except Exception as e:
                    pass

            else: st.warning("条件に一致する共起関係が見つかりませんでした。")

# ==================================================================
# --- 5. Trend Analysis (時系列分析) ---
# ==================================================================
elif selected_tab == "トレンド分析 (Trend Analysis)":
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
                update_fig_layout(fig_growth, "Growth Rate Top 20", height=500)
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
        with col_net1: ta_net_n = st.slider("抽出単語数", 30, 100, 70, key="ta_net_n")
        with col_net2: ta_net_th = st.slider("共起閾値", 0.01, 0.3, 0.03, 0.01, key="ta_net_th")
        
        all_target_kw = [w for sublist in df_target['explorer_keywords'] for w in sublist]
        c_all = Counter(all_target_kw)
        top_nodes = [w for w, c in c_all.most_common(ta_net_n)]
        
        # patirohaで共起グラフ構築 (トレンド)
        keyword_lists_trend = [
            [w for w in kws if w in top_nodes]
            for kws in df_target['explorer_keywords']
        ]
        G = patiroha.build_cooccurrence_graph(
            keyword_lists_trend,
            top_n=ta_net_n,
            threshold=ta_net_th,
            similarity="jaccard",
        )
        # 共起回数を保持（スナップショット用）
        pair_counts = Counter()
        for kws in df_target['explorer_keywords']:
            valid_w = [w for w in set(kws) if w in top_nodes]
            if len(valid_w) >= 2:
                for pair in combinations(sorted(valid_w), 2): pair_counts[pair] += 1
        
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
            update_fig_layout(fig_net, "Trend Network", height=700)
            st.plotly_chart(fig_net, use_container_width=True, config={'editable': False})

            # --- スナップショット (トレンドネットワーク) ---
            # コミュニティ検出
            trend_communities = patiroha.detect_communities(G, algorithm="louvain")
            hub_keywords_trend = patiroha.get_hub_keywords(G, centrality="degree")
            deg_centrality = {kw: score for kw, score in hub_keywords_trend}
            trend_community_map = trend_communities  # patirohaは既にdict[str,int]を返す

            # 全ノード構造化（成長率+中心性）
            period_past_str = f"{periods[1][0]}-{periods[1][1]}" if len(periods) > 1 else "N/A"
            period_recent_str = f"{periods[0][0]}-{periods[0][1]}"

            emerging_keywords = []
            declining_keywords = []
            for node in G.nodes():
                past_c = c_pst_net.get(node, 0)
                recent_c = c_rec_net.get(node, 0)
                gr = (recent_c - past_c) / (past_c + 1)
                entry = {
                    "keyword": node,
                    "growth_rate": round(gr, 3),
                    "past_count": past_c,
                    "recent_count": recent_c,
                    "community": trend_community_map.get(node, -1),
                    "degree_centrality": round(deg_centrality.get(node, 0), 4)
                }
                if gr >= 0:
                    emerging_keywords.append(entry)
                else:
                    declining_keywords.append(entry)
            emerging_keywords.sort(key=lambda x: x['growth_rate'], reverse=True)
            declining_keywords.sort(key=lambda x: x['growth_rate'])

            # エッジ（Jaccard係数+共起回数）上位100
            trend_edges_ranked = []
            for u, v, d in sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:100]:
                trend_edges_ranked.append({
                    "source": u, "target": v,
                    "jaccard": round(d['weight'], 4),
                    "cooccurrence_count": pair_counts.get(tuple(sorted([u, v])), 0)
                })

            # コミュニティ構造化（dict[str,int]から再構築）
            from collections import defaultdict
            _trend_groups = defaultdict(list)
            for node, cid in trend_community_map.items():
                _trend_groups[cid].append(node)
            trend_communities_structured = []
            for cid in sorted(_trend_groups.keys()):
                members = _trend_groups[cid]
                hub_node = max(members, key=lambda n: deg_centrality.get(n, 0))
                trend_communities_structured.append({
                    "id": cid, "size": len(members), "members": members,
                    "hub": hub_node, "hub_centrality": round(deg_centrality.get(hub_node, 0), 4)
                })

            # トレンドネットワーク用のキーワード中心抽出
            growth_nodes = sorted([n for n in G.nodes()], key=lambda n: c_rec_net.get(n, 0) - c_pst_net.get(n, 0), reverse=True)[:15]
            trend_reps = utils.get_keyword_centric_representatives(df_target, growth_nodes, n_reps=10)

            rep_lines_t = ["\n**代表的特許 (成長領域・キーワードベース):**"]
            for i, r in enumerate(trend_reps):
                 rep_lines_t.append(f"{i+1}. 【{r['title']}】 ({r['applicant']}) - {r['abstract'][:80]}...")

            snap_data = utils.generate_rich_summary(df_target, title_col=col_map['title'], abstract_col=col_map['abstract'])
            snap_data['module'] = 'Explorer'
            snap_data['network_stats'] = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": round(nx.density(G), 4),
                "communities": trend_communities_structured,
                "hubs_ranked": [e for e in emerging_keywords + declining_keywords],
                "edges_ranked": trend_edges_ranked,
                "notes": "Nodes colored by Growth Rate (Red=High, Blue=Low)."
            }
            # トレンド分析データ（順序バグ修正: render_snapshot_buttonの前に追加）
            snap_data['trend_analysis'] = {
                "period_past": period_past_str,
                "period_recent": period_recent_str,
                "emerging_keywords": emerging_keywords,
                "declining_keywords": declining_keywords
            }
            if trend_reps:
                snap_data['cluster_summary'] = (snap_data.get('cluster_summary', '') + "\n".join(rep_lines_t))

            # AIインサイト (トレンド)
            insight_context_t = f"""
            **チャートタイプ**: トレンド・共起ネットワーク (成長率)
            **対象データ**: 時系列比較による急上昇キーワードを含む共起ネットワーク。
            **手法**: 期間比較による成長率(Growth Rate)の算出。比較期間: [{period_past_str}] vs [{period_recent_str}]
            **視覚的エンコーディング**:
            - **赤色ノード**: 高成長（萌芽）技術。
            - **青色ノード**: 低成長/減少（陳腐化）技術。
            - **エッジ**: 共起関係。
            **目的**: 技術ネットワークのどこで新しい技術が生まれているか（萌芽領域）を特定すること。
            """
            insight_role = "あなたは技術トレンドの専門家です。共起ネットワークから技術体系を読み解きます。"
            insight_inst_t = """
            ネットワーク図の成長率（Growth）情報を元に分析してください：
            1. **萌芽技術**: 赤く表示されている（成長率が高い）キーワードは、どの技術領域（コミュニティ）に出現していますか？
            2. **技術の陳腐化**: 青く表示されている（成長率が低い）キーワードは、どのような技術ですか？
            3. **シフト**: 既存のハブ技術から、新しい成長技術へのシフトや融合の兆しは見えますか？
            """

            full_context_t = f"""
### AI Insight Context (Auto-Generated)
{insight_context_t}

### Analyst Instructions
{insight_inst_t}
"""
            snap_data['ai_insight_context'] = full_context_t

            utils.render_snapshot_button(
                title="トレンド・共起ネットワーク",
                description="技術用語の共起関係に成長率（赤=急上昇）を重ね合わせたマップ。",
                fig=fig_net,
                data_summary=snap_data,
                key="exp_trend_snap"
            )

            # AI Insight Button
            prompt_t = utils_ai.generate_ai_insight_prompt(insight_role, insight_context_t, snap_data, insight_inst_t)
            utils_ai.render_ai_insight_button(prompt_t, "exp_trend_insight")

            # CAPCOM data/ JSON出力
            try:
                import capcom
                if capcom.is_active():
                    capcom_trend_data = {
                        "metadata": {
                            "module": "Explorer",
                            "mode": "trend_network",
                            "target": target_filter,
                            "n_nodes": G.number_of_nodes(),
                            "n_edges_total": G.number_of_edges(),
                            "n_edges_exported": len(trend_edges_ranked),
                            "density": round(nx.density(G), 4),
                            "top_n": ta_net_n,
                            "threshold": ta_net_th,
                            "period_past": period_past_str,
                            "period_recent": period_recent_str
                        },
                        "emerging_keywords": emerging_keywords,
                        "declining_keywords": declining_keywords,
                        "communities": trend_communities_structured,
                        "edges": trend_edges_ranked
                    }
                    capcom.save_data("explorer_trend.json", capcom_trend_data)
            except Exception as e:
                pass


# ==================================================================
# --- 6. Comparative Strategy (競合比較) ---
# ==================================================================
elif selected_tab == "競合比較戦略 (Comparative Strategy)":
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
        
        # 1. トルネードチャート
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
            fig_tornado.add_trace(go.Bar(y=df_tornado["Keyword"], x=df_tornado["My Count"], orientation='h', name=my_comp, marker_color=utils.APOLLO_COLORS[0]))
            fig_tornado.add_trace(go.Bar(y=df_tornado["Keyword"], x=df_tornado["Competitor Count"], orientation='h', name=target_comp, marker_color=utils.APOLLO_COLORS[1]))
            
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
            update_fig_layout(fig_tornado, "Tornado Chart", height=800)
            st.plotly_chart(fig_tornado, use_container_width=True, config={'editable': False})
            st.info("左側 (青系) が自社、右側 (赤/オレンジ系) が競合の出現数を示します。")
            
            utils.render_snapshot_button(
                title=f"Tornado Chart ({my_comp} vs {target_comp})",
                description="キーワード出現頻度の直接比較。左右への突出が各社の特徴を示す。",
                fig=fig_tornado,
                data_summary=df_tornado[['Keyword', 'My Abs', 'Competitor Count']].tail(20).to_string(index=False),
                key="exp_tornado_snap"
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
        with col_cs1: cs_net_n = st.slider("抽出単語数", 30, 100, 70, key="cs_net_n")
        with col_cs2: cs_net_th = st.slider("共起閾値", 0.01, 0.3, 0.03, 0.01, key="cs_net_th")
        
        combined_keywords = words_my + words_target
        c_combined = Counter(combined_keywords)
        top_nodes = [w for w, c in c_combined.most_common(cs_net_n)]
        
        if 'applicant_main' in df_main.columns:
            mask_2 = df_main['applicant_main'].apply(lambda x: isinstance(x, list) and (my_comp in x or target_comp in x))
        else:
            mask_2 = df_main[col_map['applicant']].fillna('').str.contains(re.escape(my_comp) + "|" + re.escape(target_comp))
        df_2 = df_main[mask_2]
        
        # patirohaで共起グラフ構築 (競合比較)
        keyword_lists_comp = [
            [w for w in kws if w in top_nodes]
            for kws in df_2['explorer_keywords']
        ]
        G = patiroha.build_cooccurrence_graph(
            keyword_lists_comp,
            top_n=cs_net_n,
            threshold=cs_net_th,
            similarity="jaccard",
        )
        # 共起回数を保持（スナップショット用）
        pair_counts = Counter()
        for kws in df_2['explorer_keywords']:
            valid_w = [w for w in set(kws) if w in top_nodes]
            if len(valid_w) >= 2:
                for pair in combinations(sorted(valid_w), 2): pair_counts[pair] += 1
        
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
            update_fig_layout(fig_net, "Dominance Network", height=700)
            st.plotly_chart(fig_net, use_container_width=True, config={'editable': False})

            # --- スナップショット (支配率ネットワーク) ---
            # コミュニティ検出
            dom_communities = patiroha.detect_communities(G, algorithm="louvain")
            hub_keywords_dom = patiroha.get_hub_keywords(G, centrality="degree")
            deg_centrality = {kw: score for kw, score in hub_keywords_dom}
            dom_community_map = dom_communities  # patirohaは既にdict[str,int]を返す

            # 全ノード構造化（支配率+中心性）
            dom_keywords = []
            my_exclusive = []
            target_exclusive = []
            contested = []
            for node in G.nodes():
                m = c_my[node]; t_val = c_tgt[node]
                dom_val = m / (m + t_val) if (m + t_val) > 0 else 0.5
                entry = {
                    "keyword": node,
                    "my_count": m,
                    "target_count": t_val,
                    "dominance": round(dom_val, 3),
                    "community": dom_community_map.get(node, -1),
                    "degree_centrality": round(deg_centrality.get(node, 0), 4)
                }
                dom_keywords.append(entry)
                if t_val == 0 and m > 0:
                    my_exclusive.append(node)
                elif m == 0 and t_val > 0:
                    target_exclusive.append(node)
                elif 0.4 <= dom_val <= 0.6:
                    contested.append(node)

            # エッジ（Jaccard係数+共起回数）上位100
            dom_edges_ranked = []
            for u, v, d in sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:100]:
                dom_edges_ranked.append({
                    "source": u, "target": v,
                    "jaccard": round(d['weight'], 4),
                    "cooccurrence_count": pair_counts.get(tuple(sorted([u, v])), 0)
                })

            # コミュニティ構造化（dict[str,int]から再構築）
            _dom_groups = defaultdict(list)
            for node, cid in dom_community_map.items():
                _dom_groups[cid].append(node)
            dom_communities_structured = []
            for cid in sorted(_dom_groups.keys()):
                members = _dom_groups[cid]
                hub_node = max(members, key=lambda n: deg_centrality.get(n, 0))
                dom_communities_structured.append({
                    "id": cid, "size": len(members), "members": members,
                    "hub": hub_node, "hub_centrality": round(deg_centrality.get(hub_node, 0), 4)
                })

            # 代表特許抽出
            dom_reps = utils.get_keyword_centric_representatives(df_2, list(G.nodes()), n_reps=10)
            rep_lines_d = ["\n**代表的特許 (支配率ネットワーク・キーワードベース):**"]
            for i, r in enumerate(dom_reps):
                 rep_lines_d.append(f"{i+1}. 【{r['title']}】 ({r['applicant']}) - {r['abstract'][:80]}...")

            snap_data = utils.generate_rich_summary(df_2, title_col=col_map['title'], abstract_col=col_map['abstract'])
            snap_data['module'] = 'Explorer'
            snap_data['network_stats'] = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": round(nx.density(G), 4),
                "communities": dom_communities_structured,
                "hubs_ranked": sorted(dom_keywords, key=lambda x: x['degree_centrality'], reverse=True),
                "edges_ranked": dom_edges_ranked,
                "notes": f"Dominance: {my_comp} vs {target_comp}. Blue favorable to {my_comp}, Red favorable to {target_comp}."
            }
            # 支配率分析データ（順序バグ修正: render_snapshot_buttonの前に追加）
            snap_data['dominance_analysis'] = {
                "my_company": my_comp,
                "target_company": target_comp,
                "keywords": dom_keywords,
                "my_exclusive": my_exclusive,
                "target_exclusive": target_exclusive,
                "contested": contested
            }
            if dom_reps:
                snap_data['cluster_summary'] = (snap_data.get('cluster_summary', '') + "\n".join(rep_lines_d))

            # AI Insight (比較分析)
            insight_context_c = f"""
            **チャートタイプ**: 支配率共起ネットワーク (Dominance Network)
            **対象データ**: 2社({my_comp} vs {target_comp})の共起ネットワーク比較。
            **手法**: 出現頻度比率による支配率算出。
            **視覚的エンコーディング**:
            - **青色ノード**: {my_comp} が優勢な技術領域。
            - **赤色ノード**: {target_comp} が優勢な技術領域。
            - **白色/中間色**: 拮抗している領域（激戦区）。
            **目的**: 競合との強み・弱みの違い、技術ポートフォリオの重複と差異を可視化すること。
            """
            insight_role = "あなたは知財戦略コンサルタントです。競合分析を行い、差別化戦略を立案します。"
            insight_inst_c = """
            ネットワーク図の支配率（Dominance）情報を元に分析してください：
            1. **自社の強み**: 青く表示されている（自社優勢）領域は、具体的にどのような技術ですか？
            2. **競合の強み**: 赤く表示されている（競合優勢）領域は、どのような技術ですか？自社に欠けているものは？
            3. **激戦区**: 色が白に近い（拮抗している）領域はどこですか？そこでの主導権争いはどうなっていますか？
            """

            full_context_c = f"""
### AI Insight Context (Auto-Generated)
{insight_context_c}

### Analyst Instructions
{insight_inst_c}
"""
            snap_data['ai_insight_context'] = full_context_c

            utils.render_snapshot_button(
                title=f"Dominance Network ({my_comp} vs {target_comp})",
                description="共起ネットワーク上での両社の優劣（支配率）分布。",
                fig=fig_net,
                data_summary=snap_data,
                key="exp_dom_snap"
            )

            # AI Insight Button
            prompt_c = utils_ai.generate_ai_insight_prompt(insight_role, insight_context_c, snap_data, insight_inst_c)
            utils_ai.render_ai_insight_button(prompt_c, "exp_dom_insight")

            # CAPCOM data/ JSON出力
            try:
                import capcom
                if capcom.is_active():
                    capcom_dom_data = {
                        "metadata": {
                            "module": "Explorer",
                            "mode": "dominance_network",
                            "my_company": my_comp,
                            "target_company": target_comp,
                            "n_nodes": G.number_of_nodes(),
                            "n_edges_total": G.number_of_edges(),
                            "n_edges_exported": len(dom_edges_ranked),
                            "density": round(nx.density(G), 4),
                            "top_n": cs_net_n,
                            "threshold": cs_net_th
                        },
                        "keywords": dom_keywords,
                        "my_exclusive": my_exclusive,
                        "target_exclusive": target_exclusive,
                        "contested": contested,
                        "communities": dom_communities_structured,
                        "edges": dom_edges_ranked
                    }
                    capcom.save_data("explorer_dominance.json", capcom_dom_data)
            except Exception as e:
                pass


# ==================================================================
# --- 7. Context Search (文脈検索) ---
# ==================================================================
elif "Context Search" in selected_tab:
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