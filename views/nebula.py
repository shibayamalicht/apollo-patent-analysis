# ==================================================================
# APOLLO — NEBULA（環境分析）
# 旧 APOLLO pages/9_:material/blur_on:_NEBULA.py の移植版。
# 論文・ニュース・政策文書と特許を統合した環境分析
# （ハイプ・サイクル / マクロ環境 / 急上昇キーワード&ネットワーク / 学術ランドスケープ）。
# 分析ロジック・session_state キー・CAPCOM 出力スキーマは元ページのまま維持する。
# ==================================================================

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

import apollo_ui


# ==================================================================
# --- ヘルパー関数（モジュールレベルに維持） ---
# ==================================================================

def _npl_missing_notice(need="論文・ニュース・政策のデータ"):
    """NPL（外部データ）未投入時の共通の空状態案内。

    HORIZON の外部データは任意機能なので、その旨を明記したうえで
    Mission Control の「外部データ取込」への導線を添える。
    """
    st.info(
        f"このセクションには{need}が必要です。"
        "Mission Control の「外部データ取込」から追加できます（任意機能）。",
        icon=":material/school:",
    )
    pages = st.session_state.get("ap_pages", {})
    if "mission_control" in pages:
        st.page_link(pages["mission_control"], label="データ準備を開く", icon=":material/anchor:")


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
    共起ネットワーク図を生成・描画する。
    スナップショット統合用に fig, net_stats を返す。
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
        node_size.append(np.log(c_all[node] + 1) * 10)  # 対数スケールサイズ

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

    # 3. 最強エッジ（最も強い共起ペア）
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

    # キーワード分析 の「トレンド分析」と同じ方式に統一:
    # 直近 N 年窓 と その前の N 年窓 を比較する（N は下のスライダー、既定 5 年）。
    # （全範囲の半分割ではなく、Explorer の急上昇ランキングと結果が揃う方式。）
    st.markdown("**:material/trending_up: 急上昇キーワード (成長率ランキング)**")
    interval = st.slider(
        "期間の粒度 (年)", 1, 10, 5, key=f"growth_interval_{key_suffix}",
        help="トレンドを何年ごとに区切って比較するかです。直近 N 年とその前の N 年を比べます（キーワード分析のトレンド分析と同方式）。粗く（大きく）すると大まかな傾向、細かく（小さく）すると年単位の動きが見えます。"
    )

    # 直近窓 [recent_start, max_y] と その前の窓 [past_start, past_end]
    recent_start = max(min_y, max_y - interval + 1)
    past_end = recent_start - 1
    if past_end < min_y:
        st.info("比較対象となる過去データが不足しています。期間の粒度を小さくしてください。")
        return None, None
    past_start = max(min_y, past_end - interval + 1)

    st.caption(f"期間比較: 過去 [{past_start}-{past_end}] vs 直近 [{recent_start}-{max_y}]")

    df_recent = df_target[(df_target['year'] >= recent_start) & (df_target['year'] <= max_y)]
    df_past = df_target[(df_target['year'] >= past_start) & (df_target['year'] <= past_end)]
    if df_recent.empty:
        st.info("比較対象となる直近データが不足しています。")
        return None, None

    # キーワードカウント
    from collections import Counter
    c_recent = Counter([w for sublist in df_recent[keywords_col] if isinstance(sublist, list) for w in sublist])
    c_past = Counter([w for sublist in df_past[keywords_col] if isinstance(sublist, list) for w in sublist])

    # 成長率計算（Explorer と同じ閾値 1% / 上位 20）
    growth_data = []
    min_freq = max(2, len(df_recent) * 0.01)  # 最低 1% の頻度 または 2 回以上（Explorer と一致）
    for word, count_r in c_recent.items():
        if count_r < min_freq: continue
        count_p = c_past.get(word, 0)
        # 成長スコア: 単純増加率（Explorer と同一式）
        growth_rate = (count_r - count_p) / (count_p + 1)
        growth_data.append({"Keyword": word, "Growth Rate": growth_rate, "Recent": count_r, "Past": count_p})

    if not growth_data:
        st.warning("急上昇キーワードは見つかりませんでした。")
        return None, None

    df_growth = pd.DataFrame(growth_data).sort_values("Growth Rate", ascending=False).head(20)

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
    fig_growth.update_layout(yaxis={'categoryorder': 'total ascending'})
    # バー数に応じて高さを確保する。固定 height=400 だと最大20本のとき Plotly が
    # y軸ラベルを自動間引きし、全キーワードのラベルが表示されなくなる（バーだけ残る）。
    _gh = max(360, 28 * len(df_growth) + 140)
    fig_growth = update_fig_layout_local(fig_growth, f"急成長キーワード ({title_suffix})", height=_gh)
    # 1 バー約28px を確保したうえで、長いキーワードラベルが切れないよう automargin を有効化する。
    fig_growth.update_yaxes(automargin=True)
    st.plotly_chart(fig_growth, use_container_width=True)

    # 統合用統計データの返却
    growth_stats = {
        "period_past": f"{past_start}-{past_end}",
        "period_recent": f"{recent_start}-{max_y}",
        "top_growing_keywords": df_growth[['Keyword', 'Growth Rate', 'Recent']].to_dict('records')
    }

    return fig_growth, growth_stats


def extract_reps_generic(df_source, stats_growth, stats_net, col_map, domain_label):
    """代表文献の抽出（成長ワード + ネットワークハブのキーワードベース）"""
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

        # 2. ユーティリティ呼び出し (n_reps=10)
        reps = utils.get_keyword_centric_representatives(df_source, target_kws_list, n_reps=10)

        # 3. 必要に応じてドメイン固有ラベルで強化
        # ユーティリティは 'title', 'applicant', 'abstract', 'year' を含むdictを返す
        # applicant フィールドは News の Source や Academic の Author に流用可能
        # 基本的なマッピングは utility で処理済み

        return reps

    except Exception as e:
        pass
        return []


# ==================================================================
# --- メイン描画 ---
# ==================================================================

def render():
    apollo_ui.page_header(
        "環境分析", "NEBULA",
        "論文・ニュース・政策・市場データと特許のギャップ分析、学術ランドスケープ、ハイプサイクル")
    st.caption("このモジュールは任意の外部データ（論文・ニュース・政策）を使います。"
               "特許データだけでも一部機能は動きます。")

    # データガード（前処理まで完了していることを確認）
    apollo_ui.require_data()

    # スナップショットのモジュール名フォールバック用（互換維持のため 'NEBULA' のまま）
    st.session_state['current_page'] = 'NEBULA'

    # --------------------------------------------------------------
    # データ整形
    # --------------------------------------------------------------
    # 特許データのロード (Main)
    df_main = st.session_state.df_main.copy()
    col_map = st.session_state.col_map

    # NPLデータのロード (df_npl) - 別変数として保持
    df_npl = None
    if 'df_npl' in st.session_state and st.session_state.df_npl is not None:
        df_npl = st.session_state.df_npl.copy()

    # ==============================================================
    # --- 1. 全体設定（グローバルフィルタ） ---
    # ==============================================================
    # 全期間の計算
    min_y = int(df_main['year'].min())
    max_y = int(df_main['year'].max())

    if df_npl is not None and 'year' in df_npl.columns:
        npl_years = df_npl['year'].dropna().astype(int)
        if not npl_years.empty:
            min_y = min(min_y, int(npl_years.min()))
            max_y = max(max_y, int(npl_years.max()))

    st.markdown("### :material/build: 全体設定")
    year_range = st.slider("分析期間 (Year)", min_y, max_y, (min_y, max_y),
                           help="環境分析（特許・論文・ニュースの3トレンド比較）の対象期間です。")

    # フィルタ適用
    df_main_filtered = df_main[(df_main['year'] >= year_range[0]) & (df_main['year'] <= year_range[1])]

    df_npl_filtered = pd.DataFrame()
    if df_npl is not None and not df_npl.empty:
        df_npl_filtered = df_npl[(df_npl['year'] >= year_range[0]) & (df_npl['year'] <= year_range[1])]

    st.markdown("---")

    # ==============================================================
    # --- 2. ハイプ・サイクル分析 (トレンド) ---
    # ==============================================================
    st.subheader("1. :material/bar_chart: ハイプ・サイクル（Hype Cycle）")
    st.caption("アカデミアの研究 (Academic) と 技術の実装 (Patent) のタイムラグ、およびニュース (News) のトレンドを比較します。")

    if df_main_filtered.empty and df_npl_filtered.empty:
        st.warning("選択された期間にデータがありません。")
    else:
        # 1. 特許件数
        counts_p = df_main_filtered.groupby('year').size()

        # 2. 論文・ニュース件数
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
            xaxis=dict(type='category')  # 文字列カテゴリとして厳密に整列
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
                line=dict(color='#C71585', width=3),  # マゼンタ実線 (MediumVioletRed)
                mode='lines+markers',
                yaxis='y2',
                xaxis='x'
            ))
        elif not df_npl_filtered.empty:
            # NPLデータは存在するが News 集計が 0 件だった場合、Date 列の解析失敗の可能性があるので警告する
            news_count = len(df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Business'])
            if news_count > 0:
                st.warning(f":material/warning: ニュースデータが {news_count} 件ありますが、日付を解析できずグラフに描画できませんでした。"
                           "Mission Control で日付カラム（Date）の読み取り設定を確認してください。")

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
                rangemode='tozero',  # ベースラインを0に強制
            ),
            yaxis2=dict(
                title="NPL件数 (論文・ニュース)",
                overlaying='y',
                side='right',
                showgrid=False,
                rangemode='tozero',  # 主軸に合わせて0ベースライン
                matches=None  # 独立したスケーリングだが同じベースラインロジック
            ),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_hype, use_container_width=True)

        # 読み方カード（ハイプ・サイクル）
        _howto_hype = (
            "**青い棒（特許）**は技術を実用化・権利化する動き、**折れ線（論文・ニュース）**は研究の活発さと世間の注目度を表します。"
            "典型的には**論文の山が先に来て、数年遅れて特許が増える**ため、そのタイムラグが参入・仕込みどきのヒントになります。"
            "**ニュースだけが急伸**している期間は、実体より期待が先行する「過熱」の可能性に注意してください。"
            "左右の縦軸はスケールが別なので、本数の絶対値ではなく**山の形と時期のずれ**を読み比べます。"
        )
        if df_npl is None:
            _howto_hype += "（論文・ニュースの折れ線は、外部データを取り込むと表示されます）"
        apollo_ui.howto(_howto_hype)

        # スナップショットボタン (ハイプサイクル)
        hype_stats = {
            "patent_trend": counts_p.to_dict() if not counts_p.empty else {},
            "academic_trend": counts_a.to_dict() if not counts_a.empty else {},
            "news_trend": counts_n.to_dict() if not counts_n.empty else {}
        }
        utils.render_snapshot_button(
            title="ハイプ・サイクル",
            description="Comparative timeline of Patents, Academic, and Business News.",
            fig=fig_hype,
            data_summary={"type": "trend_chart", "stats": hype_stats, "module": "NEBULA"},
            key="nebula_hype"
        )
        apollo_ui.snapshot_hint()

        # CAPCOM data/ JSON出力（ハイプサイクル）
        try:
            import capcom
            if capcom.is_active():
                # 各トレンドのキーを文字列化（JSONシリアライズ対応）
                # 件数は NaN 安全化（万一 NaN が混入しても 0 件として保存する）
                hype_json = {
                    "metadata": {"module": "NEBULA", "type": "hype_cycle"},
                    "patent_trend": {str(k): (int(v) if pd.notna(v) else 0) for k, v in hype_stats.get("patent_trend", {}).items()},
                    "academic_trend": {str(k): (int(v) if pd.notna(v) else 0) for k, v in hype_stats.get("academic_trend", {}).items()},
                    "news_trend": {str(k): (int(v) if pd.notna(v) else 0) for k, v in hype_stats.get("news_trend", {}).items()}
                }
                capcom.save_data("nebula_hype_cycle.json", hype_json)
        except Exception as e:
            st.caption(f":material/warning: ハイプサイクル の CAPCOM 保存に失敗しました（要確認）: {e}")

    st.markdown("---")

    # ==============================================================
    # --- 3. マクロ環境 (政策・市場) ---
    # ==============================================================
    st.subheader("2. :material/account_balance: マクロ環境（政策・市場）")
    st.caption("規制動向 (Policy) や市場レポート (Market) の情報を時系列で確認します。")

    df_macro = pd.DataFrame()
    if not df_npl_filtered.empty:
        # マクロ環境用にはPolicyとMarketのみをフィルタ (Newsはネットワークへ移動済み)
        df_macro = df_npl_filtered[df_npl_filtered['data_sub_type'].isin(['Policy', 'Market'])]

    # NPL自体に Policy/Market が1件でもあるか（空状態メッセージの出し分け用）
    _has_macro_any = (
        df_npl is not None and not df_npl.empty
        and 'data_sub_type' in df_npl.columns
        and bool(df_npl['data_sub_type'].isin(['Policy', 'Market']).any())
    )

    if df_macro.empty:
        if _has_macro_any:
            st.info("選択された期間内にマクロデータ (Policy, Market) は見つかりませんでした。")
        else:
            _npl_missing_notice("政策・市場（Policy / Market）のデータ")
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

                    # 具体的な日付があれば表示
                    date_str = ""
                    if 'parsed_date' in row and pd.notna(row['parsed_date']):
                        date_str = f" ({row['parsed_date'].strftime('%Y-%m-%d')})"

                    icon = ":material/balance:" if sub_type == 'Policy' else ":material/bar_chart:"

                    st.markdown(f"**{icon} [{sub_type}] {title}** <small>{date_str} ({source})</small>", unsafe_allow_html=True)
                    if content:
                        st.caption(content)

        # スナップショットボタン (マクロコンテキスト)
        # レポート生成用の構造化リスト構築
        macro_items = []
        if not df_macro.empty:
            for _, row in df_macro.head(30).iterrows():  # トークンオーバーフロー回避のため上位30件に制限
                # 互換アイテムを作成
                item = {
                    "year": int(row['year']) if pd.notna(row['year']) else 'N/A',
                    "date": row['parsed_date'].strftime('%Y-%m-%d') if 'parsed_date' in row and pd.notna(row['parsed_date']) else str(row.get('unified_date', '-')),
                    "type": row.get('data_sub_type', 'Unknown'),
                    "title": row.get('unified_title', 'No Title'),
                    "source": row.get('unified_source', '-'),
                    "content": row.get('unified_content', ''),
                    # レポート生成側の引用ロジックとの互換フィールド
                    "abstract": row.get('unified_content', ''),
                    "applicant": row.get('unified_source', '-')
                }
                macro_items.append(item)

        utils.render_snapshot_button(
            title="マクロ環境（政策・市場）",
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
        apollo_ui.snapshot_hint()

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
            st.caption(f":material/warning: マクロ環境（政策・市場） の CAPCOM 保存に失敗しました（要確認）: {e}")

    st.markdown("---")

    # ==============================================================
    # --- 4. イノベーショントレンド & ネットワーク (急上昇キーワード) ---
    # ==============================================================
    st.subheader("3. :material/hub: 急上昇キーワード & 共起ネットワーク")
    st.caption("特許、論文、そしてニュースの**急上昇キーワード**と**共起ネットワーク**を分析し、トレンドの構造変化を捉えます。")

    tab_pat, tab_aca, tab_news = st.tabs([":material/factory: 特許 (Patent)", ":material/school: 論文 (Academic)", ":material/newspaper: ニュース (News)"])

    # --- タブ1: 特許ネットワーク ---
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
        st.markdown("#### 特許キーワードの共起ネットワーク")
        st.write("企業の技術実装、製品化に近い用語の関係性を可視化します。")

        cp1, cp2 = st.columns(2)
        with cp1: top_n_p = st.slider("ノード数 (Patent)", 20, 100, 70, key="net_p_n",
                                      help="ネットワークに表示するキーワード（ノード）の数です。出現頻度の上位から何語を使うかを決めます。多いほど網羅的ですが密集して読みにくく、少ないほど主要語に絞られ見やすくなります。")
        with cp2: th_p = st.slider("共起閾値 (Patent)", 0.01, 0.3, 0.03, 0.01, key="net_p_th",
                                   help="2つのキーワードを線（エッジ）で結ぶ基準の強さです。共起の強さを0〜1で表すJaccard係数がこの値以上のペアだけを結びます。高くすると強い関係だけが残ってスッキリし、低くすると弱い関係も含め線が増えて密になります。")

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
                "representatives": [f"- 【{r['title']}】[{r.get('number','N/A')}] (出願: {r['year']}, {r['applicant']}) {r['abstract'][:100]}..." for r in reps_p]
            }

            # グラフリスト
            figs_p = [f for f in [fig_growth_p, fig_net_p] if f is not None]

            # AI Insight コンテキストの準備
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
                title="技術トレンド構造（特許）",
                description="[統合分析] 急上昇キーワードランキング(Growth) と 共起ネットワーク構造(Network) の包含データ。",
                fig=None,
                figs=figs_p,  # 複数画像
                data_summary=consolidated_p,
                key="nebula_consol_pat"
            )
            apollo_ui.snapshot_hint()

            prompt_p = utils_ai.generate_ai_insight_prompt(insight_role, insight_context_p, consolidated_p, insight_inst_p)
            utils_ai.render_ai_insight_button(prompt_p, "nebula_insight_pat")

    # --- タブ2: 論文ネットワーク (Academic) ---
    with tab_aca:
        st.markdown("#### 研究トレンド & ネットワーク (Academic)")

        if df_npl_filtered.empty:
            _npl_missing_notice("論文（Academic）のデータ")
        else:
            df_aca = df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Academic'].copy()

            if df_aca.empty:
                _npl_missing_notice("論文（Academic）のデータ")
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
                st.markdown("#### 論文キーワードの共起ネットワーク")
                st.write("研究段階の技術用語、基礎研究のトレンド関係性を可視化します。")

                ca1, ca2 = st.columns(2)
                with ca1: top_n_a = st.slider("ノード数 (Academic)", 20, 100, 70, key="net_a_n",
                                              help="ネットワークに表示するキーワード（ノード）の数です。出現頻度の上位から何語を使うかを決めます。多いほど網羅的ですが密集して読みにくく、少ないほど主要語に絞られ見やすくなります。")
                with ca2: th_a = st.slider("共起閾値 (Academic)", 0.01, 0.3, 0.03, 0.01, key="net_a_th",
                                           help="2つのキーワードを線（エッジ）で結ぶ基準の強さです。共起の強さを0〜1で表すJaccard係数がこの値以上のペアだけを結びます。高くすると強い関係だけが残ってスッキリし、低くすると弱い関係も含め線が増えて密になります。")

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
                        "representatives": [f"- 【{r['title']}】[{r.get('number','N/A')}] ({r['year']}, {r['applicant']}) {r['abstract'][:100]}..." for r in reps_a]
                    }
                    figs_a = [f for f in [fig_growth_a, fig_net_a] if f is not None]
                    # AI Insight コンテキストの準備
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
                        title="研究トレンド構造（論文）",
                        description="[統合分析] 急上昇研究キーワード(Growth) と 研究トピックネットワーク(Network) の包含データ。",
                        fig=None,
                        figs=figs_a,
                        data_summary=consolidated_a,
                        key="nebula_consol_aca"
                    )
                    apollo_ui.snapshot_hint()

                    prompt_a = utils_ai.generate_ai_insight_prompt(insight_role_npl, insight_context_a, consolidated_a, insight_inst_a)
                    utils_ai.render_ai_insight_button(prompt_a, "nebula_insight_aca")

    # --- タブ3: ニュースネットワーク (News) ---
    with tab_news:
        st.markdown("#### 市場トレンド & ネットワーク (News/Market)")

        if df_npl_filtered.empty:
            _npl_missing_notice("ニュース・市場（News / Market）のデータ")
        else:
            # News には data_sub_type = 'Business' を使用
            df_news = df_npl_filtered[df_npl_filtered['data_sub_type'] == 'Business'].copy()

            if df_news.empty:
                _npl_missing_notice("ニュース・市場（News / Market）のデータ")
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
                st.markdown("#### ニュース/市場キーワードの共起ネットワーク")
                st.write("市場動向、ニュース、プレスリリース等における用語の関係性を可視化します。")

                cn1, cn2 = st.columns(2)
                with cn1: top_n_n = st.slider("ノード数 (News)", 20, 100, 70, key="net_n_n",
                                              help="ネットワークに表示するキーワード（ノード）の数です。出現頻度の上位から何語を使うかを決めます。多いほど網羅的ですが密集して読みにくく、少ないほど主要語に絞られ見やすくなります。")
                with cn2: th_n = st.slider("共起閾値 (News)", 0.01, 0.3, 0.03, 0.01, key="net_n_th",
                                           help="2つのキーワードを線（エッジ）で結ぶ基準の強さです。共起の強さを0〜1で表すJaccard係数がこの値以上のペアだけを結びます。高くすると強い関係だけが残ってスッキリし、低くすると弱い関係も含め線が増えて密になります。")

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
                        "representatives": [f"- 【{r['title']}】[{r.get('number','N/A')}] ({r['year']}, {r['applicant']}) {r['abstract'][:100]}..." for r in reps_n]
                    }
                    figs_n = [f for f in [fig_growth_n, fig_net_n] if f is not None]
                    # AI Insight コンテキストの準備
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
                        title="市場トレンド構造（ニュース）",
                        description="[統合分析] 急上昇市場キーワード(Growth) と トピックネットワーク(Network) の包含データ。",
                        fig=None,
                        figs=figs_n,
                        data_summary=consolidated_n,
                        key="nebula_consol_news"
                    )
                    apollo_ui.snapshot_hint()

                    prompt_n = utils_ai.generate_ai_insight_prompt(insight_role_news, insight_context_n, consolidated_n, insight_inst_n)
                    utils_ai.render_ai_insight_button(prompt_n, "nebula_insight_news")

    # ==============================================================
    # --- 学術ランドスケープ (Academic Landscape) ---
    # ==============================================================
    st.markdown("---")
    st.markdown("### :material/biotech: 学術ランドスケープ（論文の俯瞰図）")
    st.markdown("学術論文を意味の近さで自動分類（セマンティック・クラスタリング）し、研究テーマの全体像を地図として可視化します。")

    # Academic論文のみ抽出
    if df_npl is not None and not df_npl.empty:
        # reset_index で位置インデックスに揃える（SBERT埋め込みは位置配列のため、
        # df_npl のラベルのまま索引すると代表論文抽出がずれる/IndexErrorになる）
        df_academic = df_npl[df_npl.get('data_sub_type', pd.Series()) == 'Academic'].reset_index(drop=True).copy()

        if len(df_academic) >= 10:
            # --- パラメータ設定 ---
            st.markdown("##### パラメータ設定")
            n_papers = len(df_academic)

            # :material/smart_toy: 自動最適化モード（俯瞰図分析と同じ: 2パラメータを掃引してクラスタ数を適正化）
            acad_auto = st.checkbox(
                ":material/smart_toy: 自動最適化（最小クラスタサイズ・最小サンプル数を掃引してクラスタ数を適正化）",
                key="acad_auto_hdbscan",
                help="論文の母集団に合わせて HDBSCAN の2パラメータを自動掃引し、クラスタ数が少なすぎ(2,3)も多すぎもしない設定を選びます。")

            # データサイズに応じたデフォルト値（俯瞰図分析の知見を反映）
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
                    "最小クラスタサイズ", 3, 50, default_min_cs, key="acad_min_cs", disabled=acad_auto,
                    help=f"論文{n_papers}件に対する推奨値: {default_min_cs}。小さいほど細かく分割、大きいほど大テーマにまとまる")
            with col_p2:
                acad_min_samples = st.slider(
                    "最小サンプル数", 2, 30, default_min_s, key="acad_min_samp", disabled=acad_auto,
                    help="クラスタのコアポイント判定閾値。小さいほどノイズが減るが、不安定なクラスタも生まれやすい")
            with col_p3:
                acad_label_top_n = st.slider("ラベル語数", 2, 5, 3, key="acad_label_n",
                    help="各クラスタの自動命名に使う特徴語の数です。多いほど内容を詳しく表せますが冗長になり、少ないほど簡潔になります。")

            acad_target_k = None
            if acad_auto:
                acad_target_k = st.number_input(
                    "目標クラスタ数（目安）", min_value=2, max_value=60,
                    value=utils.suggest_target_k(n_papers), key="acad_auto_target_k",
                    help="この数に近づくよう2パラメータを掃引します（母集団規模から自動初期化）。"
                         "結果のクラスタ数や粒度に満足できない場合は、この値を増減して「学術ランドスケープを構築（再計算）」を押し直してください。"
                         "実際の値は密度構造に依存するため、目標ちょうどにならないこともあります（品質 DBCV を優先して選びます）。")
                utils.render_dbcv_help()
                if st.session_state.get('nebula_acad_auto_result'):
                    _ar = st.session_state['nebula_acad_auto_result']
                    _rv = _ar.get('validity')
                    _rv_txt = f"・品質DBCV={_rv:.2f}" if isinstance(_rv, (int, float)) else ""
                    st.info(
                        f":material/smart_toy: 前回の自動決定: 最小クラスタサイズ=**{_ar['mcs']}** / 最小サンプル数=**{_ar['ms']}** "
                        f"→ クラスタ数 **{_ar['k']}**・ノイズ {_ar['noise'] * 100:.1f}%（目標≈{_ar['target_k']}{_rv_txt}）。"
                        f"　数が合わない/粒度が好みでない場合は上の「目標クラスタ数」を変えて再構築してください。")

            # 文埋め込みモデルは Mission Control で選択したものを使う（特許と同一モデルで整合）
            _sbert_model_name = st.session_state.get('sbert_model_name') or utils.SBERT_MODEL_NAME

            # SBERTベクトルがキャッシュ済みかどうかで処理を分岐。
            # ただしモデルを切り替えた場合（次元・空間が変わる）はキャッシュを無効化して再計算する。
            has_cached_vectors = (
                st.session_state.get('nebula_academic_embeddings') is not None
                and st.session_state.get('nebula_academic_embeddings_model') == _sbert_model_name
            )

            btn_label = ":material/autorenew: クラスタリング再計算（パラメータ変更）" if has_cached_vectors else ":material/map: 学術ランドスケープを構築"
            if has_cached_vectors:
                st.caption(f":material/lightbulb: AI意味解析（SBERT）のベクトルはキャッシュ済み（`{_sbert_model_name}`）。パラメータ変更時はクラスタリングのみ再計算します。")
            else:
                st.caption(f"使用モデル: `{_sbert_model_name}`（Mission Control の前処理で選択したモデルに追従）")

            if st.button(btn_label, key="nebula_academic_landscape_btn"):
                # SBERT: キャッシュがなければ計算、あればスキップ
                if has_cached_vectors:
                    academic_vectors = st.session_state['nebula_academic_embeddings']
                else:
                    with st.spinner("AI意味解析（SBERT）でベクトル化中...（初回のみ）"):
                        # 特許（Mission Control）と同一の文埋め込みモデルを共有（選択モデルに追従）
                        embedder = utils.load_sbert_embedder(_sbert_model_name)
                        academic_vectors = embedder.encode(
                            df_academic,
                            text_columns=['unified_title', 'unified_content'],
                            column_weights={'unified_title': 2},
                            normalize_embeddings=True,
                        )
                        st.session_state['nebula_academic_embeddings'] = academic_vectors
                        st.session_state['nebula_academic_embeddings_model'] = _sbert_model_name

                with st.spinner("クラスタリング中（UMAP + HDBSCAN）..."):
                    landscape = patiroha.build_landscape(
                        academic_vectors,
                        method='hdbscan',
                        min_cluster_size=acad_min_cluster,
                        min_samples=acad_min_samples,
                    )
                    st.session_state['nebula_academic_landscape'] = landscape
                    _coords = landscape.coords

                    # 自動最適化: build_landscape で得た 2D UMAP 座標上で HDBSCAN を掃引して
                    # クラスタ数を適正化する（UMAP は 1 回のみ。掃引は 2D 上なので軽い）
                    if acad_auto:
                        _sweep = utils.sweep_hdbscan_params(_coords, target_k=int(acad_target_k))
                        _acad_labels = _sweep['labels']
                        st.session_state['nebula_acad_auto_result'] = {
                            'mcs': _sweep['min_cluster_size'], 'ms': _sweep['min_samples'],
                            'k': _sweep['n_clusters'], 'noise': _sweep['noise_ratio'],
                            'target_k': _sweep['target_k'], 'validity': _sweep.get('validity')}
                    else:
                        _acad_labels = landscape.labels
                        st.session_state.pop('nebula_acad_auto_result', None)
                    # 実際に使うクラスタ数（タイトル表示等で landscape.n_clusters の代わりに使う）
                    st.session_state['nebula_academic_n_clusters'] = int(len(set(_acad_labels.tolist()) - {-1}))

                    df_academic['acad_cluster'] = _acad_labels
                    df_academic['acad_umap_x'] = _coords[:, 0]
                    df_academic['acad_umap_y'] = _coords[:, 1]

                    texts = (df_academic['unified_title'].fillna('') + ' ' +
                            df_academic['unified_content'].fillna(''))
                    labels_map = utils.safe_auto_label(
                        texts, _acad_labels,
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
                    st.success(f":material/check_circle: {landscape.n_clusters}クラスタを検出（ノイズ: {landscape.noise_count}件）")
                    st.rerun()

            # --- 結果表示（俯瞰図分析と同じ描画方式） ---
            if 'df_nebula_academic' in st.session_state:
                df_acad = st.session_state['df_nebula_academic']
                labels_map = st.session_state.get('nebula_academic_labels_map', {})
                landscape_result = st.session_state.get('nebula_academic_landscape')
                # 実際に使用中のクラスタ数（自動最適化で掃引した場合は landscape.n_clusters と異なるため、
                # 実データから数える方を正とする）。タイトル/スナップショット/CAPCOM 出力で共通利用。
                if 'acad_cluster' in df_acad.columns:
                    _acad_n_clusters = int(df_acad[df_acad['acad_cluster'] != -1]['acad_cluster'].nunique())
                else:
                    _acad_n_clusters = landscape_result.n_clusters if landscape_result else 0

                # 表示モード切替（俯瞰図分析と同じ表記・既定＝クラスタ領域(凸包)）
                _acad_mode_lbl = st.radio(
                    "表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"],
                    horizontal=True, key="acad_display_mode_v2",
                    help="俯瞰図の描き方を切り替えます。クラスタ領域＝各クラスタを色付きの領域(凸包)で表示、密度マップ＝混み具合をヒートマップで表示、散布図＝個々の論文を点で表示。"
                )
                acad_display_mode = {
                    "クラスタ領域 (Clusters)": "Convex Hull",
                    "密度マップ (Density)": "Density",
                    "散布図 (Scatter)": "Scatter",
                }.get(_acad_mode_lbl, "Convex Hull")

                fig_acad = go.Figure()

                df_clustered = df_acad[df_acad['acad_cluster'] != -1]
                df_noise = df_acad[df_acad['acad_cluster'] == -1]

                # 俯瞰図分析と同じ共通スタイル（クラスタ→固定色。点・勢力圏・ラベルで共通）
                _land_cmap_a = utils.landscape_color_map(df_acad['acad_cluster'])
                _is_density_a = (acad_display_mode == "Density")

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

                if _is_density_a and not df_clustered.empty:
                    # クラスタ別ソフト密度（技術の地形図スタイル・全クラスタ共通メッシュ＝俯瞰図分析と同方式・端は余白ビンで見切れ防止）
                    _abx, _aby = utils.landscape_density_bins(df_clustered['acad_umap_x'], df_clustered['acad_umap_y'], 40)
                    for _cid, _grp in df_clustered.groupby('acad_cluster'):
                        utils.add_landscape_density(
                            fig_acad, _grp['acad_umap_x'], _grp['acad_umap_y'],
                            _land_cmap_a.get(_cid, '#8C93A6'), xbins=_abx, ybins=_aby)

                # 点は常時白フチ。密度モードは地形が主役なので点を小さく薄く（俯瞰図分析と同じ規則）
                marker_line = dict(width=0.6 if _is_density_a else 0.8, color='white')
                _pt_size_a = 4 if _is_density_a else utils.LANDSCAPE_POINT_SIZE
                _pt_opacity_a = 0.65 if _is_density_a else utils.LANDSCAPE_POINT_OPACITY

                # クラスタ領域（平滑化した勢力圏・点と同じ固定色）
                if acad_display_mode == "Convex Hull" and not df_clustered.empty:
                    for _cid, _grp in df_clustered.groupby('acad_cluster'):
                        utils.add_landscape_region(
                            fig_acad, _grp[['acad_umap_x', 'acad_umap_y']].values,
                            _land_cmap_a.get(_cid, '#8C93A6'))

                # Focus（全件 1 trace・勢力圏/ラベルと同じ固定色）— 俯瞰図分析と同方式
                if not df_clustered.empty:
                    fig_acad.add_trace(go.Scatter(
                        x=df_clustered['acad_umap_x'], y=df_clustered['acad_umap_y'],
                        mode='markers',
                        marker=dict(
                            color=[_land_cmap_a.get(_c, '#8C93A6') for _c in df_clustered['acad_cluster']],
                            size=_pt_size_a,
                            opacity=_pt_opacity_a,
                            line=marker_line,
                        ),
                        hoverinfo='text', hovertext=df_clustered['hover_text'],
                        customdata=df_clustered.index,
                        name='論文 (Valid)', showlegend=False,
                    ))

                # ノイズ点（別 trace・背景に沈める＝俯瞰図分析と同じスタイル）
                if len(df_noise) > 0:
                    noise_hover = df_noise['unified_title'].fillna('').tolist()
                    fig_acad.add_trace(go.Scatter(
                        x=df_noise['acad_umap_x'], y=df_noise['acad_umap_y'],
                        mode='markers',
                        marker=dict(line=dict(width=0), **utils.LANDSCAPE_NOISE_STYLE),
                        hoverinfo='text', hovertext=noise_hover,
                        customdata=df_noise.index,
                        name='Noise', showlegend=False,
                    ))

                # クラスタ重心ラベル（ピル型・点/勢力圏と同色ドット・件数上位は強調・端は見切れ防止）
                _asizes = df_clustered['acad_cluster'].value_counts().to_dict()
                _atop = utils.landscape_top_clusters(_asizes)
                _alab_xr = (float(df_clustered['acad_umap_x'].min()), float(df_clustered['acad_umap_x'].max())) if not df_clustered.empty else None
                _alab_yr = (float(df_clustered['acad_umap_y'].min()), float(df_clustered['acad_umap_y'].max())) if not df_clustered.empty else None
                for cid, label in labels_map.items():
                    if cid == -1:
                        continue
                    df_c = df_clustered[df_clustered['acad_cluster'] == cid]
                    if df_c.empty:
                        continue
                    cx, cy = df_c['acad_umap_x'].mean(), df_c['acad_umap_y'].mean()
                    utils.add_landscape_label(
                        fig_acad, cx, cy, label, _land_cmap_a.get(cid, '#8C93A6'),
                        emphasized=(cid in _atop), x_range=_alab_xr, y_range=_alab_yr)

                # レイアウト（俯瞰図分析と同じ規則: 図内タイトルなし + height=1200 + aspect 1:1）
                st.caption(f"学術論文ランドスケープ — {_acad_n_clusters}クラスタ / 未分類 {len(df_noise)}件")
                utils.update_fig_layout(fig_acad, "", height=1200, show_legend=False)
                fig_acad.update_layout(margin=dict(l=16, r=16, t=24, b=16))

                # aspect 1:1 を強制
                if not df_clustered.empty:
                    x_min, x_max = df_clustered['acad_umap_x'].min(), df_clustered['acad_umap_x'].max()
                    y_min, y_max = df_clustered['acad_umap_y'].min(), df_clustered['acad_umap_y'].max()
                    # 密度モードは等高線のすそ野が点群の外に広がるため余白を広めに取る（俯瞰図分析と同じ）
                    pad_factor = 0.06 if _is_density_a else 0.02
                    x_pad = (x_max - x_min) * pad_factor if x_max > x_min else 1.0
                    y_pad = (y_max - y_min) * pad_factor if y_max > y_min else 1.0
                    # constrain="domain": 1:1 アスペクトをプロット領域側で調整し、
                    # ラベル(注釈)の有無で表示範囲が広がって俯瞰図が縮む現象を防ぐ（俯瞰図分析と同じ）。
                    # height=1200 を再設定し、yaxis にも constrain="domain" を付ける。
                    #   これが欠けるとアスペクト計算がコンテナ寸法に依存し、初回レンダリングで俯瞰図が小さく
                    #   表示され、ラベル編集等の再実行までは正しい大きさにならない。
                    fig_acad.update_layout(
                        height=1200,
                        xaxis=dict(range=[x_min - x_pad, x_max + x_pad], autorange=False, constrain="domain"),
                        yaxis=dict(
                            range=[y_min - y_pad, y_max + y_pad],
                            autorange=False,
                            scaleanchor="x", scaleratio=1,
                            constrain="domain",
                        ),
                    )

                @st.fragment
                def _nebula_click_acad():
                    utils.disable_selection_fade(fig_acad)
                    acad_selection = st.plotly_chart(
                        fig_acad, use_container_width=True,
                        on_select="rerun", selection_mode="points", key="nebula_acad_map",
                        config={
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
                        },
                    )
                    # 論文点クリック → 詳細ダイアログ（customdata は df_acad のインデックス）
                    utils.handle_map_click(
                        acad_selection, "nebula_acad",
                        title="クリックした論文", df=df_acad,
                        card_renderer=utils.render_paper_card,
                    )
                _nebula_click_acad()

                # 読み方カード（学術ランドスケープ）
                apollo_ui.howto(
                    "**近くにある点**は内容が似た論文で、**同じ色のまとまり**が研究テーマ（クラスタ）です。"
                    "**大きなかたまり**は研究者の関心が集中する主流テーマ、**灰色の小さな点（ノイズ）**は"
                    "どのテーマにも属さない論文で、萌芽的・学際的な研究が混ざっていることがあります。"
                    "縦軸・横軸の数値そのものに意味はなく、**点どうしの距離（近さ）だけ**を読みます。"
                    "俯瞰図分析 の特許マップと見比べると、研究が先行して特許がまだ薄い「ギャップ領域」を探せます。"
                )

                # --- CSV ダウンロード（俯瞰図分析と同じ運用） ---
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
                    ":material/download: 学術ランドスケープ全データ (CSV)",
                    csv_acad,
                    "APOLLO_NEBULA_Academic_Landscape.csv",
                    "text/csv",
                    key="nebula_academic_landscape_csv_dl",
                )

                # スナップショット
                utils.render_snapshot_button(
                    title="学術論文ランドスケープ",
                    description=f"学術論文{len(df_acad)}件, {_acad_n_clusters}クラスタ, ノイズ{len(df_noise)}件",
                    key="snap_nebula_academic_landscape",
                    fig=fig_acad,
                    data_summary={
                        'type': 'academic_landscape',
                        'total_papers': len(df_acad),
                        'n_clusters': _acad_n_clusters,
                        'noise_count': len(df_noise),
                        'clusters': [{'id': int(cid), 'label': lbl, 'count': len(df_acad[df_acad['acad_cluster'] == cid])}
                                     for cid, lbl in labels_map.items() if cid != -1],
                    },
                )
                apollo_ui.snapshot_hint()

                # :material/image: 整理版ランドスケープ（スライド/レポート用・上位クラスタのみ）
                if st.checkbox(":material/image: 整理版ランドスケープ（スライド/レポート用・上位クラスタのみ）を表示", key="nebula_curated_show"):
                    st.caption("全クラスタを密にラベルすると重なるため、件数上位の学術クラスタだけを大きく示したスライド向けの俯瞰図です。")
                    _ncl_topn = st.slider("ラベル表示するクラスタ数（件数上位）", 3, 15, 8, key="nebula_curated_topn",
                        help="整理版（スライド/レポート用）の俯瞰図で、件数が多い上位何クラスタにだけ大きなラベルを付けるかです。全クラスタに付けると重なって読めないため、主要クラスタだけを強調します。")
                    _ncl_style_lbl = st.radio("表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"],
                                              horizontal=True, key="nebula_curated_style",
                                              help="俯瞰図の描き方を切り替えます。クラスタ領域＝色付き領域で表示、密度マップ＝混み具合をヒートマップで表示、散布図＝個々の特許を点で表示。")
                    _ncl_style = {"クラスタ領域 (Clusters)": "hull", "密度マップ (Density)": "density", "散布図 (Scatter)": "points"}.get(_ncl_style_lbl, "hull")
                    try:
                        _ncl_fig = utils.build_curated_landscape(
                            df_acad, cluster_col='acad_cluster', label_col='acad_cluster_label',
                            x_col='acad_umap_x', y_col='acad_umap_y', top_n=_ncl_topn, region_style=_ncl_style)
                        st.plotly_chart(_ncl_fig, use_container_width=True, config={'editable': False})
                        _ncl_nall = int(df_acad[df_acad['acad_cluster'] != -1]['acad_cluster'].nunique()) if 'acad_cluster' in df_acad.columns else 0
                        # CAPCOMアクティブ時はクリーンPNGをZIPに同梱（スライド/レポート用の整理版を下流へ）
                        utils.save_curated_to_capcom(
                            _ncl_fig, snap_id="nebula_academic_curated_landscape",
                            cache_token=f"{_ncl_topn}_{_ncl_style}")
                        utils.render_report_png_button(
                            _ncl_fig, key="nebula_curated_report",
                            default_title="学術ランドスケープ：主要クラスタ",
                            default_subtitle=f"学術論文 {len(df_acad):,}件 / 上位{_ncl_topn}クラスタ（全{_ncl_nall}クラスタ）",
                            default_caption="",
                            label=":material/palette: 整理版をスライド/レポート用PNGに書き出す")
                    except Exception as _e:
                        st.warning(f"整理版ランドスケープの生成に失敗しました: {_e}")

                # --- ラベル編集 + AIアシスト ---
                st.markdown("#### クラスタラベル編集")

                # 学術テキスト用TF-IDF（AIラベルアシスト・ラベル編集に必要）
                # utils.safe_build_tfidf で Janome 例外耐性を確保
                if 'nebula_academic_tfidf' not in st.session_state:
                    acad_texts = (df_acad['unified_title'].fillna('') + ' ' +
                                  df_acad['unified_content'].fillna('')).tolist()
                    _tfidf_mat, _feat_names = utils.safe_build_tfidf(
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

                # AIラベルサジェスト（学術論文ドメイン: 命名プロンプトを論文用語にする）
                utils.render_ai_label_assistant(
                    df_acad, 'acad_cluster', 'nebula_academic_labels_map',
                    acad_col_map, acad_tfidf, acad_features,
                    widget_key_prefix="nebula_acad_label", domain="academic"
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
                    with st.expander(":material/map: クラスタ空間配置（近接関係）", expanded=False):
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
                                'cluster_id': int(cid) if pd.notna(cid) else -1,
                                'label': label,
                                'count': len(df_c),
                                'centroid': [round(cx, 4), round(cy, 4)],
                                'representative_papers': reps,
                            })

                        export_data = {
                            'type': 'nebula_academic_landscape',
                            'total_papers': len(df_acad),
                            'n_clusters': _acad_n_clusters,
                            'noise_count': landscape_result.noise_count if landscape_result else 0,
                            'clusters': cluster_export,
                            'spatial_context': acad_spatial_info if acad_spatial_info else "",
                        }
                        if dyn_data:
                            export_data['cluster_dynamics'] = dyn_data
                        capcom.save_data('nebula_academic_clusters', export_data)
                except Exception as e:
                    st.caption(f":material/warning: 学術ランドスケープ（クラスタ） の CAPCOM 保存に失敗しました（要確認）: {e}")

                # --- AI Insight: 学術ランドスケープ分析 ---
                _acad_clusters = sorted(
                    [{'label': lbl, 'count': int((df_acad['acad_cluster'] == cid).sum())}
                     for cid, lbl in labels_map.items() if cid != -1],
                    key=lambda d: d['count'], reverse=True)
                _acad_cluster_str = utils_ai.df_to_markdown(
                    pd.DataFrame(_acad_clusters).rename(columns={'label': '研究テーマ', 'count': '件数'}),
                    index=False) if _acad_clusters else "（クラスタなし）"
                _acad_noise_n = int(len(df_noise))
                _acad_meta = {
                    '分析単位': '学術論文 (NPL)',
                    '学術論文件数': len(df_acad),
                    'クラスタ数': len(_acad_clusters),
                    'ノイズ件数': _acad_noise_n,
                    '手法': 'SBERT + UMAP + HDBSCAN',
                }
                if 'year' in df_acad.columns and df_acad['year'].notna().any():
                    _yy = df_acad['year'].dropna()
                    _acad_meta['期間'] = f"{int(_yy.min())}年 ～ {int(_yy.max())}年"
                # クラスタ動態・ノイズ（萌芽研究）の追加コンテキストと指示
                _acad_extra, _acad_inst = utils_ai.build_map_dynamics_noise_addon(
                    dynamics_data=dyn_data, noise_count=_acad_noise_n, total_count=len(df_acad))
                _acad_prompt = utils_ai.generate_ai_insight_prompt(
                    role="学術ランドスケープ・研究動向分析の専門家として、学術論文のクラスタ構造から研究トレンドと技術機会を分析してください。",
                    context=("SBERT（意味ベクトル化）+ UMAP + HDBSCAN で学術論文（NPL）を自動クラスタリングした"
                             "『学術ランドスケープ』です。各クラスタ=研究テーマ群、点=論文、近接=意味的類似を表します。"
                             "クラスタ動態（累積件数×CAGR）と空間配置（クラスタ間の近接関係）も併せて提示します。"),
                    data_summary=(f"学術論文 全{len(df_acad)}件 / {len(_acad_clusters)}クラスタ / ノイズ{_acad_noise_n}件\n"
                                  f"[研究テーマ クラスタ別件数]\n{_acad_cluster_str}"),
                    instructions="""\
以下の観点で分析してください:
1. **研究テーマ構成**: 主要な研究クラスタ（テーマ）を名指しし、学術コミュニティの関心の重心を特定
2. **注目・主要テーマ**: 件数上位クラスタの研究的意義と、産業・技術への波及可能性
3. **萌芽研究**: ノイズ（既存テーマに属さない論文）に潜む新規・学際的研究の芽
4. **研究動態**: クラスタ動態から、新興/成長中の研究領域と、成熟・停滞領域を区別
5. **学際・融合領域**: 空間配置で近接するクラスタ間の融合・境界領域の研究機会
6. **学術と特許のギャップ**: 学術で先行するテーマと、特許化（実用化）が追いついていない領域の差を示唆

各主張には必ずクラスタ名と具体的な数値を1つ以上含めてください。""" + _acad_inst,
                    metadata=_acad_meta,
                    constraints="学術（NPL）と特許は母集団・性質が非対称（学術=新規性・基礎研究が先行、特許=実用化・権利化視点）。この非対称性を前提に論じること。",
                    output_format="Markdown形式。見出し付きの構造化された分析レポート。",
                    extra_content=f"\n# 空間配置情報 (Spatial Context)\n{acad_spatial_info}\n{_acad_extra}"
                )
                utils_ai.render_ai_insight_button(_acad_prompt, "nebula_academic_insight")
        elif len(df_academic) == 0:
            _npl_missing_notice("論文（Academic）のデータ")
        else:
            st.info(f"学術論文が{len(df_academic)}件のみです（最低10件必要）。")
    else:
        _npl_missing_notice("論文（Academic）のデータ")
