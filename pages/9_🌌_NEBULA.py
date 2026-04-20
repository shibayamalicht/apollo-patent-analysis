
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re
import utils
import utils_ai
import patiroha
import networkx as nx
from collections import Counter
from itertools import combinations
import time
import networkx.algorithms.community as community

# ==================================================================
# --- ページ設定 ---
# ==================================================================
st.set_page_config(page_title="APOLLO v8 | NEBULA", page_icon="🌌", layout="wide")
st.session_state['current_page'] = 'NEBULA'
utils.render_sidebar()

st.title("🌌 NEBULA")
st.markdown("論文・ニュース・政策文書と特許を統合した環境分析。Hype Cycle・学術ランドスケープ・成長キーワードで、技術と社会のギャップを可視化します。")

# ==================================================================
# --- データ準備 ---
# ==================================================================
if 'df_main' not in st.session_state or st.session_state.df_main is None:
    st.warning("有効な特許データがありません。Homeモジュールでデータをアップロードしてください。")
    st.stop()

# 特許データのロード (Main)
df_main = st.session_state.df_main.copy()
col_map = st.session_state.col_map

# NPLデータのロード (df_npl) - 別変数として保持
df_npl = None
if 'df_npl' in st.session_state and st.session_state.df_npl is not None:
    df_npl = st.session_state.df_npl.copy()

# ==================================================================
# --- Helper Functions ---
# ==================================================================

def update_fig_layout_local(fig, title, height=500):
    fig.update_layout(
        template=utils.APOLLO_TEMPLATE,
        title=dict(text=title, font=dict(size=18, color=utils.APOLLO_TEXT)),
        paper_bgcolor=utils.APOLLO_BG,
        plot_bgcolor=utils.APOLLO_BG,
        font=dict(color=utils.APOLLO_TEXT),
        height=height,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def render_network(df_target, title, top_n=50, threshold=0.05, height=600):
    """
    Generate and render a co-occurrence network chart.
    Returns fig, net_stats for snapshot consolidation.
    """
    # 固定キーワードを優先し、存在しない場合は旧カラムを使用
    tgt_col = 'explorer_keywords_fixed' if 'explorer_keywords_fixed' in df_target.columns else 'explorer_keywords'
    
    if tgt_col not in df_target.columns:
        st.warning(f"{title} のキーワードデータがありません。")
        return None, None

    # 空リストやNaNを除外
    keywords_series = df_target[tgt_col].dropna()
    keywords_series = keywords_series[keywords_series.apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    
    if keywords_series.empty:
        st.info(f"キーワードデータが不足しているため、{title}を表示できません。")
        return None, None

    all_kws = [w for sublist in keywords_series for w in sublist]
    c_all = Counter(all_kws)
    
    if not c_all:
        st.info("キーワードが見つかりません。")
        return None, None

    top_nodes = [w for w, c in c_all.most_common(top_n)]
    
    pair_counts = Counter()
    for kws in keywords_series:
        valid_w = [w for w in set(kws) if w in top_nodes]
        if len(valid_w) >= 2:
            for pair in combinations(sorted(valid_w), 2):
                pair_counts[pair] += 1
                
    G = nx.Graph()
    for w in top_nodes:
        G.add_node(w, size=c_all[w])
        
    for (u, v), c in pair_counts.items():
        # Jaccard係数に基づく重み付け
        weight = c / (c_all[u] + c_all[v] - c)
        if weight >= threshold:
            G.add_edge(u, v, weight=weight)
            
    # 孤立ノードを削除してグラフをクリーンに保つ
    G.remove_nodes_from(list(nx.isolates(G)))
            
    if G.number_of_nodes() == 0:
        st.warning(f"{title}: 指定された閾値以上の共起関係が見つかりませんでした。閾値を下げてみてください。")
        return None, None

    # コミュニティ抽出
    communities = community.greedy_modularity_communities(G)
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
            
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    # エッジ
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # ノード
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        node_text.append(f"{node} ({c_all[node]}件)")
        node_color.append(community_map.get(node, 0))
        node_size.append(np.log(c_all[node] + 1) * 10) # 対数スケールサイズ

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='Turbo',
            color=node_color,
            size=node_size,
            line_width=1
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    fig = update_fig_layout_local(fig, title, height=height)
    st.plotly_chart(fig, use_container_width=True)
    
    # 統合用統計データの返却
    # 1. コミュニティ構造
    comm_summary = []
    for i in range(len(communities)):
        comm_words = list(communities[i])[:5] 
        comm_summary.append(f"Group {i+1}: {', '.join(comm_words)}")
    
    # 2. ハブ (次数中心性)
    deg_centrality = nx.degree_centrality(G)
    sorted_hubs = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    hubs_str = ", ".join([f"{n}({val:.2f})" for n, val in sorted_hubs])

    # 3. 最強エッジ (Strongest Edges)
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
    edges_str = ", ".join([f"{u}-{v}" for u, v, d in sorted_edges])

    net_stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "communities": "; ".join(comm_summary),
        "hubs": hubs_str,
        "strongest_edges": edges_str,
        "top_keywords": top_nodes[:15]
    }
    
    return fig, net_stats


# ==================================================================
# --- 1. グローバルフィルタ (全体設定) ---
# ==================================================================
# 全期間の計算
min_y = int(df_main['year'].min())
max_y = int(df_main['year'].max())

if df_npl is not None and 'year' in df_npl.columns:
    npl_years = df_npl['year'].dropna().astype(int)
    if not npl_years.empty:
        min_y = min(min_y, int(npl_years.min()))
        max_y = max(max_y, int(npl_years.max()))

st.markdown("### 🛠️ Global Filters")
year_range = st.slider("分析期間 (Year)", min_y, max_y, (min_y, max_y))

# Apply Filter
df_main_filtered = df_main[(df_main['year'] >= year_range[0]) & (df_main['year'] <= year_range[1])]

df_npl_filtered = pd.DataFrame()
if df_npl is not None and not df_npl.empty:
    df_npl_filtered = df_npl[(df_npl['year'] >= year_range[0]) & (df_npl['year'] <= year_range[1])]

st.markdown("---")

# ==================================================================
# --- 2. ハイプ・サイクル分析 (トレンド) ---
# ==================================================================
st.subheader("1. 📊 ハイプ・サイクル (Trends)")
st.caption("アカデミアの研究 (Academic) と 技術の実装 (Patent) のタイムラグ、およびニュース (News) のトレンドを比較します。")

if df_main_filtered.empty and df_npl_filtered.empty:
    st.warning("選択された期間にデータがありません。")
else:
    # 1. Patent Count
    counts_p = df_main_filtered.groupby('year').size()
    
    # 2. Academic Count
    counts_a = pd.Series()
    counts_n = pd.Series()
    
    if not df_npl_filtered.empty:
        # Academic
        df_aca = df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Academic']
        if not df_aca.empty:
            counts_a = df_aca.groupby('year').size()
            
        # News
        df_news = df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Business']
        if not df_news.empty:
            # Patent/Academicに合わせてYear(int)でのグルーピングを確認
            if 'year' in df_news.columns:
                 counts_n = df_news.groupby('year').size()
            elif 'parsed_date' in df_news.columns:
                 # 年カラムがない場合のフォールバック計算
                 counts_n = df_news['parsed_date'].dt.year.value_counts().sort_index()

    # 明示的なカテゴリエリア軸を持つチャートの作成
    fig_hype = go.Figure(layout=dict(
        xaxis=dict(type='category') # 文字列カテゴリとして厳密に整列
    ))
    
    # 1. ギャップを防ぐために全期間を決定
    all_years = []
    if not counts_p.empty: all_years.extend(counts_p.index.astype(int).tolist())
    if not counts_n.empty: all_years.extend(counts_n.index.astype(int).tolist())
    if not counts_a.empty: all_years.extend(counts_a.index.astype(int).tolist())
    
    if all_years:
        min_y, max_y = min(all_years), max(all_years)
        full_range = range(min_y, max_y + 1)
        
        # 2. インデックスを再設定し、文字列(カテゴリ)に変換
        counts_p = counts_p.reindex(full_range, fill_value=0)
        counts_p.index = counts_p.index.astype(str)
        
        # 線グラフについて、0へ落ちるのを避けるか？
        # ハイプサイクルは通常0を表示するため、0埋めする。
        if not counts_n.empty:
            counts_n = counts_n.reindex(full_range, fill_value=0)
            counts_n.index = counts_n.index.astype(str)
            
        if not counts_a.empty:
            counts_a = counts_a.reindex(full_range, fill_value=0)
            counts_a.index = counts_a.index.astype(str)

    # 特許バー (主軸)
    fig_hype.add_trace(go.Bar(
        x=counts_p.index,
        y=counts_p.values,
        name='Patent (Reality)',
        marker_color='#003366',
        opacity=0.6,
        yaxis='y',
        xaxis='x'
    ))
    
    # 論文ライン (副軸)
    if not counts_a.empty:
        fig_hype.add_trace(go.Scatter(
            x=counts_a.index,
            y=counts_a.values,
            name='Academic (Research)',
            line=dict(color='#FF8C00', width=3),
            mode='lines+markers',
            yaxis='y2',
            xaxis='x'
        ))
        
    # ニュースライン (副軸) - オプション
    if not counts_n.empty:
        fig_hype.add_trace(go.Scatter(
            x=counts_n.index,
            y=counts_n.values,
            name='News (Buzz)',
            line=dict(color='#C71585', width=3), # Solid Magenta MediumVioletRed
            mode='lines+markers',
            yaxis='y2',
            xaxis='x'
        ))
    elif not df_npl_filtered.empty:
         # NPLデータは存在するが News 集計が 0 件だった場合、Date 列の解析失敗の可能性があるので警告する
         news_count = len(df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Business'])
         if news_count > 0:
             st.warning(f"⚠️ News data exists ({news_count} items) but could not be plotted. Check if 'Date' column was parsed correctly in Mission Control.")

    # ベースレイアウトを先に適用
    fig_hype = update_fig_layout_local(fig_hype, "Hype Cycle Analysis", height=450)

    # レイアウトのカスタマイズ (ベースレイアウト適用後にX軸設定を維持するため)
    fig_hype.update_layout(
        title="比較トレンド: Patent vs NPL",
        # カテゴリ軸設定を明示的に強制
        xaxis=dict(
            type='category',
            visible=True
        ),
        yaxis=dict(
            title="特許件数 (Patent Count)", 
            showgrid=False,
            rangemode='tozero', # ベースラインを0に強制
        ),
        yaxis2=dict(
            title="NPL件数 (論文・ニュース)",
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero', # 主軸に合わせて0ベースライン
            matches=None # 独立したスケーリングだが同じベースラインロジック
        ),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_hype, use_container_width=True)

    # スナップショットボタン (ハイプサイクル)
    hype_stats = {
        "patent_trend": counts_p.to_dict() if not counts_p.empty else {},
        "academic_trend": counts_a.to_dict() if not counts_a.empty else {},
        "news_trend": counts_n.to_dict() if not counts_n.empty else {}
    }
    utils.render_snapshot_button(
        title="NEBULA: Hype Cycle (Trends)",
        description="Comparative timeline of Patents, Academic, and Business News.",
        fig=fig_hype,
        data_summary={"type": "trend_chart", "stats": hype_stats, "module": "NEBULA"},
        key="nebula_hype"
    )

    # CAPCOM data/ JSON出力（ハイプサイクル）
    try:
        import capcom
        if capcom.is_active():
            # 各トレンドのキーを文字列化（JSONシリアライズ対応）
            hype_json = {
                "metadata": {"module": "NEBULA", "type": "hype_cycle"},
                "patent_trend": {str(k): int(v) for k, v in hype_stats.get("patent_trend", {}).items()},
                "academic_trend": {str(k): int(v) for k, v in hype_stats.get("academic_trend", {}).items()},
                "news_trend": {str(k): int(v) for k, v in hype_stats.get("news_trend", {}).items()}
            }
            capcom.save_data("nebula_hype_cycle.json", hype_json)
    except Exception as e:
        pass

st.markdown("---")

# ==================================================================
# --- 3. マクロ環境 (政策・市場) ---
# ==================================================================


st.subheader("2. 🏛️ マクロ環境 (Policy & Market)")
st.caption("規制動向 (Policy) や市場レポート (Market) の情報を時系列で確認します。")

df_macro = pd.DataFrame()
if not df_npl_filtered.empty:
    # マクロ環境用にはPolicyとMarketのみをフィルタ (Newsはネットワークへ移動済み)
    df_macro = df_npl_filtered[df_npl_filtered['data_sub_type'].isin(['Policy', 'Market'])]

if df_macro.empty:
    st.info("選択された期間内にマクロデータ (Policy, Market) は見つかりませんでした。")
else:
    # 時系列順にソート
    df_macro = df_macro.sort_values('year', ascending=False)
    
    # コンテナまたはリストで見やすく表示
    for year, group in df_macro.groupby('year', sort=False):
        with st.expander(f"YEAR {int(year)} ({len(group)} events)", expanded=True):
            for _, row in group.iterrows():
                sub_type = row.get('data_sub_type', 'Unknown')
                title = row.get('unified_title', 'No Title')
                content = row.get('unified_content', '')
                source = row.get('unified_source', '-')
                
                # Show specific date if available
                date_str = ""
                if 'parsed_date' in row and pd.notna(row['parsed_date']):
                    date_str = f" ({row['parsed_date'].strftime('%Y-%m-%d')})"
                
                icon = "⚖️" if sub_type == 'Policy' else "📊"
                
                st.markdown(f"**{icon} [{sub_type}] {title}** <small>{date_str} ({source})</small>", unsafe_allow_html=True)
                if content:
                    st.caption(content)
    
    # スナップショットボタン (マクロコンテキスト)
    # Voyager用構造化リスト構築
    macro_items = []
    if not df_macro.empty:
        for _, row in df_macro.head(30).iterrows(): # トークンオーバーフロー回避のため上位30件に制限
             # 互換アイテムを作成
             item = {
                 "year": int(row['year']) if pd.notna(row['year']) else 'N/A',
                 "date": row['parsed_date'].strftime('%Y-%m-%d') if 'parsed_date' in row and pd.notna(row['parsed_date']) else str(row.get('unified_date', '-')),
                 "type": row.get('data_sub_type', 'Unknown'),
                 "title": row.get('unified_title', 'No Title'),
                 "source": row.get('unified_source', '-'),
                 "content": row.get('unified_content', ''),
                 # Compatibility fields for Voyager citation logic
                 "abstract": row.get('unified_content', ''), 
                 "applicant": row.get('unified_source', '-')
             }
             macro_items.append(item)

    utils.render_snapshot_button(
        title="NEBULA: Macro Context (Policy & Market)",
        description="List of Policy Regulations and Market Reports.",
        fig=None,
        data_summary={
            "type": "macro_list",
            "items": macro_items,
            "module": "NEBULA",
            "domain": "Policy/Market",
            "representatives_raw": macro_items
        },
        key="nebula_macro"
    )

    # CAPCOM data/ JSON出力（マクロイベント）
    try:
        import capcom
        if capcom.is_active():
            # macro_itemsから不要なキーを除外してJSON用に整形
            macro_json_items = []
            for item in macro_items:
                macro_json_items.append({
                    "date": item.get("date", ""),
                    "year": item.get("year", ""),
                    "type": item.get("type", ""),
                    "title": item.get("title", ""),
                    "source": item.get("source", ""),
                    "content": item.get("content", "")[:300]  # 長文は300文字で切る
                })
            macro_json = {
                "metadata": {"module": "NEBULA", "type": "macro_events", "count": len(macro_json_items)},
                "items": macro_json_items
            }
            capcom.save_data("nebula_macro_events.json", macro_json)
    except Exception as e:
        pass


st.markdown("---")

# ==================================================================
# --- 4. イノベーショントレンド & ネットワーク (急上昇ワード) ---
# ==================================================================
st.subheader("3. 🕸️ イノベーショントレンド & ネットワーク (Emerging Keywords)")
st.caption("特許、論文、そしてニュースの**急上昇キーワード**と**共起ネットワーク**を分析し、トレンドの構造変化を捉えます。")

# --- Helper: Render Growth Ranking ---
def render_growth_ranking(df_target, title_suffix, key_suffix, keywords_col='explorer_keywords_fixed'):
    """
    直近半期 (Recent Half) と 過去半期 (Past Half) を比較した「成長率 (Growth Rate)」ランキングを描画します。
    スナップショット統合用に fig と growth_stats を返します。
    """
    if df_target.empty or 'year' not in df_target.columns or keywords_col not in df_target.columns:
        st.warning(f"{title_suffix} ランキングのデータがありません。")
        return None, None

    # 分割点 (中間点) の決定
    years = df_target['year'].dropna().astype(int)
    if years.empty: return None, None
    
    min_y, max_y = years.min(), years.max()
    if min_y == max_y:
        st.info(f"単年度データ ({min_y}) のため、成長率分析はスキップします。")
        return None, None

    mid_y = (min_y + max_y) // 2
    
    # データの分割
    df_recent = df_target[df_target['year'] > mid_y]
    df_past = df_target[df_target['year'] <= mid_y]
    
    if df_recent.empty:
        st.info("比較対象となる直近データが不足しています。")
        return None, None

    # 期間の説明
    st.markdown(f"**📈 急上昇キーワード (成長率ランキング)**")
    st.caption(f"期間比較: 過去 [{min_y}-{mid_y}] vs 直近 [{mid_y+1}-{max_y}]")

    # キーワードカウント
    from collections import Counter
    c_recent = Counter([w for sublist in df_recent[keywords_col] if isinstance(sublist, list) for w in sublist])
    c_past = Counter([w for sublist in df_past[keywords_col] if isinstance(sublist, list) for w in sublist])
    
    # 成長率計算
    growth_data = []
    min_freq = max(2, len(df_recent) * 0.005) # 最低0.5%の頻度 または 2回以上
    
    for word, count_r in c_recent.items():
        if count_r < min_freq: continue
        count_p = c_past.get(word, 0)
        # 成長スコア: 単純増加率
        growth_rate = (count_r - count_p) / (count_p + 1)
        growth_data.append({"Keyword": word, "Growth Rate": growth_rate, "Recent": count_r, "Past": count_p})
    
    if not growth_data:
        st.warning("急上昇キーワードは見つかりませんでした。")
        return None, None

    df_growth = pd.DataFrame(growth_data).sort_values("Growth Rate", ascending=False).head(15)
    
    # チャート描画
    fig_growth = px.bar(
        df_growth, 
        x="Growth Rate", 
        y="Keyword", 
        orientation='h', 
        color="Growth Rate", 
        color_continuous_scale="Reds",
        title=f"Top Emerging Keywords ({title_suffix})"
    )
    fig_growth.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_growth = update_fig_layout_local(fig_growth, f"急成長キーワード ({title_suffix})", height=400)
    st.plotly_chart(fig_growth, use_container_width=True)
    
    # 統合用統計データの返却
    growth_stats = {
        "period_past": f"{min_y}-{mid_y}",
        "period_recent": f"{mid_y+1}-{max_y}",
        "top_growing_keywords": df_growth[['Keyword', 'Growth Rate', 'Recent']].to_dict('records')
    }
    
    return fig_growth, growth_stats

# --- Helper: 代表文献の抽出 (キーワードベース) ---
def extract_reps_generic(df_source, stats_growth, stats_net, col_map, domain_label):
    if df_source.empty: return []

    try:
        # 1. ターゲットキーワード収集 (成長ワード + ネットワークハブ)
        target_keywords = set()
        if stats_growth and 'top_growing_keywords' in stats_growth:
            # 成長率上位5ワードを追加
            for k in stats_growth['top_growing_keywords'][:5]:
                target_keywords.add(k['Keyword'])
                
        if stats_net and 'top_keywords' in stats_net:
            # ネットワークハブ上位5ワードを追加
            for k in stats_net['top_keywords'][:5]:
                target_keywords.add(k)
        
        # フォールバック (特定の統計がない場合)
        if not target_keywords and 'explorer_keywords_fixed' in df_source.columns:
             # DF自体から上位ワードを取得
             all_kws = [w for sublist in df_source['explorer_keywords_fixed'].dropna() for w in sublist]
             from collections import Counter
             target_keywords.update([w for w, c in Counter(all_kws).most_common(10)])

        target_kws_list = list(target_keywords)
        
        # 2. ユーティリティ呼び出し
        # n_reps=10 (リクエスト通り)
        reps = utils.get_keyword_centric_representatives(df_source, target_kws_list, n_reps=10)
        
        # 3. 必要に応じてドメイン固有ラベルで強化
        # ユーティリティは 'title', 'applicant', 'abstract', 'year' を含むdictを返す
        # Applicantフィールドは Newsの'Source' や Academicの'Author' に流用可能
        # 基本的なマッピングは utility で処理済みだが、必要ならここで調整
        
        return reps

    except Exception as e:
        pass
        return []


tab_pat, tab_aca, tab_news = st.tabs(["🏭 特許 (Patent)", "🎓 論文 (Academic)", "📰 ニュース (News)"])

# --- Tab 1: 特許ネットワーク ---
with tab_pat:
    st.markdown("#### 技術トレンド & ネットワーク (Patent)")
    
    # キーワード未計算時の安全性確保 (統一ロジック使用)
    if 'explorer_keywords_fixed' not in df_main_filtered.columns:
        if 'explorer_keywords' in df_main_filtered.columns:
            df_main_filtered['explorer_keywords_fixed'] = df_main_filtered['explorer_keywords']
        else:
            with st.spinner("キーワードを計算中..."):
                col_map = st.session_state.get('col_map', {})
                t_col = col_map.get('title', '')
                a_col = col_map.get('abstract', '')
                if t_col and t_col in df_main_filtered.columns:
                    texts = df_main_filtered[t_col].fillna('')
                    if a_col and a_col in df_main_filtered.columns:
                        texts = texts + ' ' + df_main_filtered[a_col].fillna('')
                    sw = patiroha.get_stopwords()
                    df_main_filtered['explorer_keywords_fixed'] = texts.apply(
                        lambda x: utils.extract_keywords(x, stopwords=sw))
                else:
                    df_main_filtered['explorer_keywords_fixed'] = [[] for _ in range(len(df_main_filtered))]

    # 1. ランキングマップ
    fig_growth_p, stats_growth_p = render_growth_ranking(
        df_main_filtered, 
        "Patent", 
        "pat", 
        keywords_col='explorer_keywords_fixed'
    )
    
    st.markdown("---")
    st.markdown("#### Patent Keywords Network")
    st.write("企業の技術実装、製品化に近い用語の関係性を可視化します。")
    
    cp1, cp2 = st.columns(2)
    with cp1: top_n_p = st.slider("ノード数 (Patent)", 20, 100, 70, key="net_p_n")
    with cp2: th_p = st.slider("共起閾値 (Patent)", 0.01, 0.3, 0.03, 0.01, key="net_p_th")
    
    fig_net_p, stats_net_p = render_network(df_main_filtered, "特許ネットワーク (Patent)", top_n_p, th_p)

        # 統合スナップショット (Patent)
    if fig_net_p or fig_growth_p:
        # 手法の定義
        methodology_p = "特許の表題・要約・請求項から特許用語辞書を用いてキーワードを抽出。Jaccard係数による共起ネットワーク分析を実施。"
        
        # ヘルパーの使用
        reps_p = extract_reps_generic(df_main_filtered, stats_growth_p, stats_net_p, st.session_state.col_map, 'Patent')

        consolidated_p = {
            "type": "trend_network_consolidated",
            "domain": "Patent",
            "ranking": stats_growth_p,
            "network": stats_net_p,
            "methodology": methodology_p,
            "module": "NEBULA",
            "representatives_raw": reps_p,
            "representatives": [f"- 【{r['title']}】 (出願: {r['year']}, {r['applicant']}) {r['abstract'][:100]}..." for r in reps_p]
        }
        
        # グラフリスト
        figs_p = [f for f in [fig_growth_p, fig_net_p] if f is not None]

        # AI Insight コンテキストの準備 (上に移動)
        insight_context_p = f"""
        **チャートタイプ**: 技術トレンド構造分析 (Patent / NEBULA)
        **対象データ**: 全特許データの急上昇キーワード(Growth) と 共起ネットワーク(Network)。
        **手法**:
        - Growth: 期間比較による成長率の算出。
        - Network: 複合名詞の共起分析 (Jaccard係数)。
        **視覚的エンコーディング**:
        - **ランキング**: 技術キーワードの成長率順リスト。
        - **ネットワーク**: ノードは技術概念、エッジは意味的なつながりを表す。
        **目的**: 萌芽的な技術テーマの特定と、それらの技術的な文脈（つながり）を理解すること。
        """
        insight_role = "あなたは高度な技術トレンドアナリストです。特許データから萌芽的な技術トレンドを特定します。"
        insight_inst_p = """
        以下の点について分析してください：
        1. **急上昇トレンド**: どのような技術キーワードが急成長していますか？それらの共通点は？
        2. **技術ネットワーク**: 共起ネットワークの中心（ハブ）となる技術概念は何ですか？
        3. **トレンドの文脈**: 成長キーワードとネットワーク構造から、現在どのような技術開発競争が起きていると推測できますか？
        """

        full_context_p = f"""
### AI Insight Context (Auto-Generated)
{insight_context_p}

### Analyst Instructions
{insight_inst_p}
"""
        consolidated_p['ai_insight_context'] = full_context_p

        utils.render_snapshot_button(
            title="NEBULA: 技術トレンド構造 (特許)",
            description="[統合分析] 急上昇キーワードランキング(Growth) と 共起ネットワーク構造(Network) の包含データ。",
            fig=None, # Legacy unused
            figs=figs_p, # Multi-Image
            data_summary=consolidated_p,
            key="nebula_consol_pat"
        )



        prompt_p = utils_ai.generate_ai_insight_prompt(insight_role, insight_context_p, consolidated_p, insight_inst_p)
        utils_ai.render_ai_insight_button(prompt_p, "nebula_insight_pat")



# --- Tab 2: 論文ネットワーク (Academic) ---
with tab_aca:
    st.markdown("#### 研究トレンド & ネットワーク (Academic)")
    
    if df_npl_filtered.empty:
        st.info("Academicデータがアップロードされていません (NPLデータなし)。")
    else:
        df_aca = df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Academic'].copy()
        
        if df_aca.empty:
            st.info("Academicデータがアップロードされていません。")
        else:
            # キーワード確保
            if 'explorer_keywords_fixed' not in df_aca.columns:
                if 'explorer_keywords' in df_aca.columns:
                    df_aca['explorer_keywords_fixed'] = df_aca['explorer_keywords']
                else:
                    with st.spinner("キーワードを計算中 (Academic)..."):
                        sw = patiroha.get_stopwords("npl")
                        df_aca['temp_text'] = df_aca['unified_title'].fillna('') + ' ' + df_aca['unified_content'].fillna('')
                        df_aca['explorer_keywords_fixed'] = df_aca['temp_text'].apply(
                            lambda x: utils.extract_keywords(x, stopwords=sw, clean_html=True)
                        )

            fig_growth_a, stats_growth_a = render_growth_ranking(df_aca, "Academic", "aca", keywords_col='explorer_keywords_fixed')
            
            st.markdown("---")
            st.markdown("#### Academic Keywords Network")
            st.write("研究段階の技術用語、基礎研究のトレンド関係性を可視化します。")
    
            ca1, ca2 = st.columns(2)
            with ca1: top_n_a = st.slider("ノード数 (Academic)", 20, 100, 70, key="net_a_n")
            with ca2: th_a = st.slider("共起閾値 (Academic)", 0.01, 0.3, 0.03, 0.01, key="net_a_th")
            
            fig_net_a, stats_net_a = render_network(df_aca, "論文ネットワーク (Academic)", top_n_a, th_a)
            
            if fig_net_a or fig_growth_a:
                 # 標準NPLカラムに基づくデフォルトマップ
                 def_map_a = {'title': 'unified_title', 'abstract': 'unified_content', 'date': 'year', 'applicant': 'unified_source'}
                 if 'parsed_date' in df_aca.columns: def_map_a['date'] = 'parsed_date'
                 
                 cmap_a = st.session_state.get('col_map_academic', def_map_a)
                 reps_a = extract_reps_generic(df_aca, stats_growth_a, stats_net_a, cmap_a, 'Academic')
                 
                 consolidated_a = {
                    "type": "trend_network_consolidated",
                    "domain": "Academic",
                    "ranking": stats_growth_a,
                    "network": stats_net_a,
                    "methodology": "学術論文のAbstract分析",
                    "module": "NEBULA",
                    "representatives_raw": reps_a,
                    "representatives": [f"- 【{r['title']}】 ({r['year']}, {r['applicant']}) {r['abstract'][:100]}..." for r in reps_a]
                }
                 figs_a = [f for f in [fig_growth_a, fig_net_a] if f is not None]
                 # AI Insight コンテキストの準備 (上に移動)
                 insight_context_a = f"""
                 **チャートタイプ**: 研究トレンド構造分析 (Academic / NEBULA)
                 **対象データ**: 学術論文の急上昇キーワード(Growth) と 共起ネットワーク(Network)。
                 **手法**:
                 - Growth: 学術論文ごとの期間比較成長率。
                 - Network: 論文Abstractの共起分析。
                 **視覚的エンコーディング**:
                 - **ランキング**: 急増している研究キーワード（バズワード）。
                 - **ネットワーク**: 研究トピック間の相関関係。
                 **目的**: 特許出願に至る前の、基礎研究段階の早期トレンドを検知すること。
                 """
                 insight_role_npl = "あなたは科学技術政策・研究開発戦略のアナリストです。基礎研究のトレンドを読み解きます。"
                 insight_inst_a = """
                 以下の点について分析してください：
                 1. **研究トレンド**: 今、研究者の関心を集めている（急増している）トピックは何ですか？
                 2. **分野の融合**: ネットワーク図から、異なる研究分野の融合や、新しい学際領域の兆しは見えますか？
                 3. **産業応用**: これらの研究トレンドは、将来的にどのような産業に応用されそうですか？
                 """

                 full_context_a = f"""
### AI Insight Context (Auto-Generated)
{insight_context_a}

### Analyst Instructions
{insight_inst_a}
"""
                 consolidated_a['ai_insight_context'] = full_context_a

                 utils.render_snapshot_button(
                    title="NEBULA: 研究トレンド構造 (論文)",
                    description="[統合分析] 急上昇研究キーワード(Growth) と 研究トピックネットワーク(Network) の包含データ。",
                    fig=None,
                    figs=figs_a,
                    data_summary=consolidated_a,
                    key="nebula_consol_aca"
                )


                 prompt_a = utils_ai.generate_ai_insight_prompt(insight_role_npl, insight_context_a, consolidated_a, insight_inst_a)
                 utils_ai.render_ai_insight_button(prompt_a, "nebula_insight_aca")


# --- Tab 3: ニュースネットワーク (News) ---
with tab_news:
    st.markdown("#### 市場トレンド & ネットワーク (News/Market)")
    
    if df_npl_filtered.empty:
        st.info("News/Marketデータがアップロードされていません (NPLデータなし)。")
    else:
        # 以前のコードで確認された通り、Newsには 'Business' を使用
        df_news = df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Business'].copy()
        
        if df_news.empty:
            st.info("News/Marketデータがアップロードされていません。")
        else:
            # キーワードの存在確認
            if 'explorer_keywords_fixed' not in df_news.columns:
                if 'explorer_keywords' in df_news.columns:
                    df_news['explorer_keywords_fixed'] = df_news['explorer_keywords']
                else:
                    with st.spinner("キーワードを計算中 (News)..."):
                        sw = patiroha.get_stopwords("npl")
                        df_news['temp_text'] = df_news['unified_title'].fillna('') + ' ' + df_news['unified_content'].fillna('')
                        df_news['explorer_keywords_fixed'] = df_news['temp_text'].apply(
                            lambda x: utils.extract_keywords(x, stopwords=sw, clean_html=True)
                        )
    
            fig_growth_n, stats_growth_n = render_growth_ranking(df_news, "News", "news", keywords_col='explorer_keywords_fixed')
            
            st.markdown("---")
            st.markdown("#### News/Market Keywords Network")
            st.write("市場動向、ニュース、プレスリリース等における用語の関係性を可視化します。") 
    
            cn1, cn2 = st.columns(2)
            with cn1: top_n_n = st.slider("ノード数 (News)", 20, 100, 70, key="net_n_n")
            with cn2: th_n = st.slider("共起閾値 (News)", 0.01, 0.3, 0.03, 0.01, key="net_n_th")
            
            fig_net_n, stats_net_n = render_network(df_news, "ニュース共起ネットワーク (News)", top_n_n, th_n)
            
            if fig_net_n or fig_growth_n:
                 # デフォルトマップ設定
                 def_map_n = {'title': 'unified_title', 'abstract': 'unified_content', 'date': 'year', 'applicant': 'unified_source'}
                 if 'parsed_date' in df_news.columns: def_map_n['date'] = 'parsed_date'

                 cmap_n = st.session_state.get('col_map_news', def_map_n)
                 reps_n = extract_reps_generic(df_news, stats_growth_n, stats_net_n, cmap_n, 'News')
                 
                 consolidated_n = {
                    "type": "trend_network_consolidated",
                    "domain": "News",
                    "ranking": stats_growth_n,
                    "network": stats_net_n,
                    "methodology": "ニュース・市場レポートの分析",
                    "module": "NEBULA",
                    "representatives_raw": reps_n,
                    "representatives": [f"- 【{r['title']}】 ({r['year']}, {r['applicant']}) {r['abstract'][:100]}..." for r in reps_n]
                }
                 figs_n = [f for f in [fig_growth_n, fig_net_n] if f is not None]
                 # AI Insight コンテキストの準備（上に移動）
                 insight_context_n = f"""
                 **チャートタイプ**: 市場トレンド構造分析 (News / NEBULA)
                 **対象データ**: ビジネスニュース/レポートの急上昇キーワード と 共起ネットワーク。
                 **手法**: ニュース記事のバズワード抽出と共起分析。
                 **視覚的エンコーディング**:
                 - **ランキング**: ビジネスシーンで話題のキーワード。
                 - **ネットワーク**: 市場テーマや業界動向の関連性。
                 **目的**: 市場の注目点、消費者の関心、企業の事業戦略のシフトを把握すること。
                 """
                 insight_role_news = "あなたは市場調査・マーケティング戦略のアナリストです。ニュースからビジネストレンドを読み解きます。"
                 insight_inst_n = """
                 以下の点について分析してください：
                 1. **市場の注目点**: ビジネスニュースで話題になっている（急増している）キーワードは何ですか？
                 2. **ビジネストレンド**: どのようなビジネスモデルや業界動向が議論されていますか？
                 3. **社会実装**: 研究・特許と比較して、市場（社会実装）のフェーズでの課題や関心事は何だと読み取れますか？
                 """

                 full_context_n = f"""
### AI Insight Context (Auto-Generated)
{insight_context_n}

### Analyst Instructions
{insight_inst_n}
"""
                 consolidated_n['ai_insight_context'] = full_context_n

                 utils.render_snapshot_button(
                    title="NEBULA: 市場トレンド構造 (News)",
                    description="[統合分析] 急上昇市場キーワード(Growth) と トピックネットワーク(Network) の包含データ。",
                    fig=None,
                    figs=figs_n,
                    data_summary=consolidated_n,
                    key="nebula_consol_news"
                )


                 prompt_n = utils_ai.generate_ai_insight_prompt(insight_role_news, insight_context_n, consolidated_n, insight_inst_n)
                 utils_ai.render_ai_insight_button(prompt_n, "nebula_insight_news")


# リクエストにより "About this analysis" セクションを削除

# ==================================================================
# --- 学術ランドスケープ (Academic Landscape) ---
# ==================================================================
st.markdown("---")
st.markdown("### 🔬 学術ランドスケープ (Academic Landscape)")
st.markdown("学術論文のセマンティック・クラスタリングを行い、研究テーマの全体像を可視化します。")

# Academic論文のみ抽出
if df_npl is not None and not df_npl.empty:
    df_academic = df_npl[df_npl.get('data_sub_type', pd.Series()) == 'Academic'].copy()

    if len(df_academic) >= 10:
        # --- パラメータ設定 ---
        st.markdown("##### パラメータ設定")
        n_papers = len(df_academic)

        # データサイズに応じたデフォルト値（Saturn Vの知見を反映）
        # 小規模(〜100件): 細かいクラスタを検出 → min_cs=5, min_s=3
        # 中規模(100〜500件): バランス → min_cs=10, min_s=5
        # 大規模(500件〜): 大きなテーマを抽出 → min_cs=15, min_s=10
        if n_papers < 100:
            default_min_cs, default_min_s = 5, 3
        elif n_papers < 500:
            default_min_cs, default_min_s = 10, 5
        else:
            default_min_cs, default_min_s = 15, 10

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            acad_min_cluster = st.slider(
                "最小クラスタサイズ", 3, 50, default_min_cs, key="acad_min_cs",
                help=f"論文{n_papers}件に対する推奨値: {default_min_cs}。小さいほど細かく分割、大きいほど大テーマにまとまる")
        with col_p2:
            acad_min_samples = st.slider(
                "最小サンプル数", 2, 30, default_min_s, key="acad_min_samp",
                help="クラスタのコアポイント判定閾値。小さいほどノイズが減るが、不安定なクラスタも生まれやすい")
        with col_p3:
            acad_label_top_n = st.slider("ラベル語数", 2, 5, 3, key="acad_label_n")

        # SBERTベクトルがキャッシュ済みかどうかで処理を分岐
        has_cached_vectors = 'nebula_academic_embeddings' in st.session_state and st.session_state['nebula_academic_embeddings'] is not None

        btn_label = "🔄 クラスタリング再計算（パラメータ変更）" if has_cached_vectors else "🚀 学術ランドスケープを構築"
        if has_cached_vectors:
            st.caption("💡 SBERTベクトルはキャッシュ済み。パラメータ変更時はクラスタリングのみ再計算します。")

        if st.button(btn_label, key="nebula_academic_landscape_btn"):
            # SBERT: キャッシュがなければ計算、あればスキップ
            if has_cached_vectors:
                academic_vectors = st.session_state['nebula_academic_embeddings']
            else:
                with st.spinner("SBERTベクトル化中...（初回のみ）"):
                    embedder = patiroha.SBERTEmbedder()
                    academic_vectors = embedder.encode(
                        df_academic,
                        text_columns=['unified_title', 'unified_content'],
                        column_weights={'unified_title': 2},
                        normalize_embeddings=True,
                    )
                    st.session_state['nebula_academic_embeddings'] = academic_vectors

            with st.spinner("UMAP + HDBSCAN クラスタリング中..."):
                landscape = patiroha.build_landscape(
                    academic_vectors,
                    method='hdbscan',
                    min_cluster_size=acad_min_cluster,
                    min_samples=acad_min_samples,
                )
                st.session_state['nebula_academic_landscape'] = landscape

                df_academic['acad_cluster'] = landscape.labels
                df_academic['acad_umap_x'] = landscape.coords[:, 0]
                df_academic['acad_umap_y'] = landscape.coords[:, 1]

                texts = (df_academic['unified_title'].fillna('') + ' ' +
                        df_academic['unified_content'].fillna(''))
                labels_map = patiroha.auto_label(
                    texts, landscape.labels,
                    stopwords=patiroha.get_stopwords("npl"),
                    method='c-tfidf', top_n=acad_label_top_n)
                st.session_state['nebula_academic_labels_map'] = labels_map
                st.session_state['nebula_academic_labels_map_original'] = labels_map.copy()
                df_academic['acad_cluster_label'] = df_academic['acad_cluster'].map(labels_map)
                st.session_state['df_nebula_academic'] = df_academic
                # TF-IDFキャッシュをリセット（再構築させる）
                for k in ['nebula_academic_tfidf', 'nebula_academic_feature_names']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.success(f"✅ {landscape.n_clusters}クラスタを検出（ノイズ: {landscape.noise_count}件）")
                st.rerun()

        # --- 結果表示（Saturn V風） ---
        if 'df_nebula_academic' in st.session_state:
            df_acad = st.session_state['df_nebula_academic']
            labels_map = st.session_state.get('nebula_academic_labels_map', {})
            landscape_result = st.session_state.get('nebula_academic_landscape')

            # 表示モード切替
            acad_display_mode = st.radio(
                "表示モード:", ["Scatter", "Density", "Convex Hull"],
                horizontal=True, key="acad_display_mode"
            )

            fig_acad = go.Figure()

            df_clustered = df_acad[df_acad['acad_cluster'] != -1]
            df_noise = df_acad[df_acad['acad_cluster'] == -1]

            # Saturn V と同じ色パレット
            color_sequence = utils.APOLLO_COLORS
            sorted_unique_cids = sorted(df_clustered['acad_cluster'].unique())

            # ホバーテキストを事前に組み立て（1 trace 化のため）
            def _build_acad_hover(row):
                parts = [f"<b>{row.get('unified_title', '')}</b>"]
                src = row.get('unified_source', '')
                if src:
                    parts.append(str(src))
                yr = row.get('year', '')
                if yr:
                    parts.append(str(yr))
                lbl = row.get('acad_cluster_label', '')
                if lbl:
                    parts.append(f"クラスタ: {lbl}")
                return '<br>'.join(parts)

            if not df_clustered.empty:
                df_clustered = df_clustered.copy()
                df_clustered['hover_text'] = df_clustered.apply(_build_acad_hover, axis=1)

            if acad_display_mode == "Density":
                # 密度ヒートマップ（Saturn V と同じカスタムカラースケール）
                custom_density_colorscale = [
                    [0.0, "rgba(255, 255, 255, 0)"],
                    [0.1, "rgba(225, 245, 254, 0.3)"],
                    [0.4, "rgba(129, 212, 250, 0.6)"],
                    [1.0, "rgba(2, 119, 189, 0.9)"],
                ]
                fig_acad.add_trace(go.Histogram2dContour(
                    x=df_clustered['acad_umap_x'], y=df_clustered['acad_umap_y'],
                    colorscale=custom_density_colorscale,
                    reversescale=False,
                    showscale=False,
                    contours=dict(coloring='fill', showlines=True),
                    line=dict(width=0.5, color='rgba(0, 0, 0, 0.2)'),
                    nbinsx=40, nbinsy=40, name='密度',
                ))
                # Density モードは Focus を重ねず、ラベルのみ表示
                marker_line = dict(width=1, color='white')
            else:
                marker_line = dict(width=0)

            # Convex Hull（クラスタ領域）
            if acad_display_mode == "Convex Hull" and not df_clustered.empty:
                from scipy.spatial import ConvexHull
                for i, cid in enumerate(sorted_unique_cids):
                    points = df_clustered[df_clustered['acad_cluster'] == cid][['acad_umap_x', 'acad_umap_y']].values
                    if len(points) >= 3:
                        try:
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]
                            hull_points = np.append(hull_points, [hull_points[0]], axis=0)
                            cluster_color = color_sequence[i % len(color_sequence)]
                            fig_acad.add_trace(go.Scatter(
                                x=hull_points[:, 0], y=hull_points[:, 1],
                                mode='lines', fill='toself',
                                fillcolor=cluster_color, opacity=0.1,
                                line=dict(color=cluster_color, width=2),
                                hoverinfo='skip', showlegend=False,
                            ))
                        except Exception:
                            pass

            # Focus（全件 1 trace、colorscale でクラスタ別着色）— Saturn V 方式
            if acad_display_mode != "Density" and not df_clustered.empty:
                fig_acad.add_trace(go.Scatter(
                    x=df_clustered['acad_umap_x'], y=df_clustered['acad_umap_y'],
                    mode='markers',
                    marker=dict(
                        color=df_clustered['acad_cluster'],
                        colorscale=color_sequence,
                        showscale=False,
                        size=5,
                        opacity=0.8,
                        line=marker_line,
                    ),
                    hoverinfo='text', hovertext=df_clustered['hover_text'],
                    name='論文 (Valid)', showlegend=False,
                ))

            # ノイズ点（別 trace、Saturn V と同じスタイル）
            if len(df_noise) > 0:
                noise_hover = df_noise['unified_title'].fillna('').tolist()
                fig_acad.add_trace(go.Scatter(
                    x=df_noise['acad_umap_x'], y=df_noise['acad_umap_y'],
                    mode='markers',
                    marker=dict(color='#999999', size=3, opacity=0.3, line=dict(width=0)),
                    hoverinfo='text', hovertext=noise_hover,
                    name='Noise', showlegend=False,
                ))

            # クラスタ重心ラベル（Saturn V と同じ colored border 2pt）
            for cid, label in labels_map.items():
                if cid == -1:
                    continue
                df_c = df_clustered[df_clustered['acad_cluster'] == cid]
                if df_c.empty:
                    continue
                cx, cy = df_c['acad_umap_x'].mean(), df_c['acad_umap_y'].mean()
                try:
                    color_idx = sorted_unique_cids.index(cid)
                    border_color = color_sequence[color_idx % len(color_sequence)]
                except (ValueError, IndexError):
                    border_color = "#333333"
                fig_acad.add_annotation(
                    x=cx, y=cy, text=label, showarrow=False,
                    font=dict(size=11, color='black', family="Helvetica"),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor=border_color, borderwidth=2, borderpad=4,
                )

            # レイアウト（Saturn V と同じ APOLLO テンプレート + height=1200 + aspect 1:1）
            title_txt = f"学術論文ランドスケープ ({landscape_result.n_clusters if landscape_result else '?'}クラスタ / ノイズ {len(df_noise)}件)"
            utils.update_fig_layout(fig_acad, title_txt, height=1200, show_legend=False)

            # aspect 1:1 を強制
            if not df_clustered.empty:
                x_min, x_max = df_clustered['acad_umap_x'].min(), df_clustered['acad_umap_x'].max()
                y_min, y_max = df_clustered['acad_umap_y'].min(), df_clustered['acad_umap_y'].max()
                pad_factor = 0.02
                x_pad = (x_max - x_min) * pad_factor if x_max > x_min else 1.0
                y_pad = (y_max - y_min) * pad_factor if y_max > y_min else 1.0
                fig_acad.update_layout(
                    xaxis=dict(range=[x_min - x_pad, x_max + x_pad], autorange=False),
                    yaxis=dict(
                        range=[y_min - y_pad, y_max + y_pad],
                        autorange=False,
                        scaleanchor="x", scaleratio=1,
                    ),
                )

            st.plotly_chart(fig_acad, use_container_width=True, config={
                'editable': True,
                'edits': {
                    'annotationPosition': True,
                    'annotationText': False,
                    'axisTitleText': False,
                    'legendPosition': False,
                    'legendText': False,
                    'shapePosition': False,
                    'titleText': False,
                },
            })

            # --- CSV ダウンロード（Saturn V と同じ運用） ---
            # クラスタID・ラベル・UMAP 座標を含む学術論文データを出力
            acad_export_cols = [
                c for c in [
                    'unified_title', 'unified_content', 'unified_source',
                    'year', 'citation_count', 'doi',
                    'acad_cluster', 'acad_cluster_label',
                    'acad_umap_x', 'acad_umap_y',
                ] if c in df_acad.columns
            ]
            csv_acad = df_acad[acad_export_cols].to_csv(
                index=False, encoding='utf-8-sig'
            ).encode('utf-8-sig')
            st.download_button(
                "📥 学術ランドスケープ全データ (CSV)",
                csv_acad,
                "APOLLO_NEBULA_Academic_Landscape.csv",
                "text/csv",
                key="nebula_academic_landscape_csv_dl",
            )

            # スナップショット
            utils.render_snapshot_button(
                title="NEBULA 学術論文ランドスケープ",
                description=f"学術論文{len(df_acad)}件, {landscape_result.n_clusters if landscape_result else '?'}クラスタ, ノイズ{len(df_noise)}件",
                key="snap_nebula_academic_landscape",
                fig=fig_acad,
                data_summary={
                    'type': 'academic_landscape',
                    'total_papers': len(df_acad),
                    'n_clusters': landscape_result.n_clusters if landscape_result else 0,
                    'noise_count': len(df_noise),
                    'clusters': [{'id': int(cid), 'label': lbl, 'count': len(df_acad[df_acad['acad_cluster'] == cid])}
                                 for cid, lbl in labels_map.items() if cid != -1],
                },
            )

            # --- ラベル編集 + AIアシスト ---
            st.markdown("#### クラスタラベル編集")

            # 学術テキスト用TF-IDF（AIラベルアシスト・ラベル編集に必要）
            if 'nebula_academic_tfidf' not in st.session_state:
                acad_texts = (df_acad['unified_title'].fillna('') + ' ' +
                              df_acad['unified_content'].fillna('')).tolist()
                _tfidf_mat, _feat_names = patiroha.build_tfidf(
                    acad_texts, stopwords=patiroha.get_stopwords("npl"),
                    min_df=2, max_df=0.90)
                st.session_state['nebula_academic_tfidf'] = _tfidf_mat
                st.session_state['nebula_academic_feature_names'] = _feat_names

            acad_tfidf = st.session_state['nebula_academic_tfidf']
            acad_features = st.session_state['nebula_academic_feature_names']

            # col_map互換: 学術データ用のカラムマッピング
            acad_col_map = {
                'title': 'unified_title',
                'abstract': 'unified_content',
                'applicant': 'unified_source',
            }

            # AIラベルサジェスト
            utils.render_ai_label_assistant(
                df_acad, 'acad_cluster', 'nebula_academic_labels_map',
                acad_col_map, acad_tfidf, acad_features,
                widget_key_prefix="nebula_acad_label"
            )

            # オリジナルラベルの保存（初回のみ）
            if 'nebula_academic_labels_map_original' not in st.session_state:
                st.session_state['nebula_academic_labels_map_original'] = labels_map.copy()

            # 手動ラベル編集UI
            label_widgets = utils.create_label_editor_ui(
                st.session_state['nebula_academic_labels_map_original'],
                st.session_state['nebula_academic_labels_map'],
                "nebula_acad_label"
            )
            if label_widgets:
                for cid, val in label_widgets.items():
                    labels_map[cid] = val
                df_acad['acad_cluster_label'] = df_acad['acad_cluster'].map(labels_map)
                st.session_state['nebula_academic_labels_map'] = labels_map
                st.session_state['df_nebula_academic'] = df_acad

            # クラスタ動態マップ
            dyn_data = None
            if 'year' in df_acad.columns and df_acad['year'].notna().sum() > 0:
                dyn_data = utils.render_cluster_dynamics_section(
                    df_acad, 'acad_cluster', labels_map,
                    year_col='year', cagr_window=3,
                    unique_key='nebula_acad_dynamics',
                    module_name='NEBULA Academic',
                )

            # --- 空間分析（重心計算 + 近接関係）---
            import utils_spatial
            acad_spatial_info = utils_spatial.generate_spatial_cluster_summary(
                df_acad, 'acad_cluster', 'acad_umap_x', 'acad_umap_y',
                label_map=labels_map
            )
            if acad_spatial_info and acad_spatial_info != "空間データなし":
                with st.expander("🗺️ クラスタ空間配置（近接関係）", expanded=False):
                    st.markdown(acad_spatial_info)

            # CAPCOM出力（空間分析・重心座標・代表論文を含む）
            try:
                import capcom
                if capcom.is_active():
                    landscape_result = st.session_state.get('nebula_academic_landscape')
                    acad_embeddings = st.session_state.get('nebula_academic_embeddings')
                    cluster_export = []
                    for cid, label in labels_map.items():
                        if cid == -1:
                            continue
                        df_c = df_acad[df_acad['acad_cluster'] == cid]
                        if df_c.empty:
                            continue

                        # 重心座標
                        cx = float(df_c['acad_umap_x'].mean())
                        cy = float(df_c['acad_umap_y'].mean())

                        # 代表論文（重心に近い上位5件）
                        reps = []
                        if acad_embeddings is not None and len(df_c) > 0:
                            try:
                                c_indices = df_c.index.tolist()
                                c_vecs = acad_embeddings[c_indices]
                                reps_result = patiroha.find_representatives(c_vecs, df_c, n=min(5, len(df_c)),
                                    title_col='unified_title', abstract_col='unified_content',
                                    applicant_col='unified_source', year_col='year')
                                for r in reps_result:
                                    reps.append({
                                        'title': r.title[:150],
                                        'source': r.applicant or '',
                                        'year': r.year or '',
                                        'score': round(r.score, 4),
                                    })
                            except Exception:
                                # フォールバック: 先頭5件
                                for _, row in df_c.head(5).iterrows():
                                    reps.append({
                                        'title': str(row.get('unified_title', ''))[:150],
                                        'source': str(row.get('unified_source', '')),
                                        'year': str(row.get('year', '')),
                                    })

                        cluster_export.append({
                            'cluster_id': int(cid),
                            'label': label,
                            'count': len(df_c),
                            'centroid': [round(cx, 4), round(cy, 4)],
                            'representative_papers': reps,
                        })

                    export_data = {
                        'type': 'nebula_academic_landscape',
                        'total_papers': len(df_acad),
                        'n_clusters': landscape_result.n_clusters if landscape_result else 0,
                        'noise_count': landscape_result.noise_count if landscape_result else 0,
                        'clusters': cluster_export,
                        'spatial_context': acad_spatial_info if acad_spatial_info else "",
                    }
                    if dyn_data:
                        export_data['cluster_dynamics'] = dyn_data
                    capcom.save_data('nebula_academic_clusters', export_data)
            except Exception:
                pass
    else:
        st.info(f"学術論文が{len(df_academic)}件のみです（最低10件必要）。")
else:
    st.info("NPLデータ（Academic）が読み込まれていません。Mission ControlからNPLデータをインポートしてください。")
