import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import re
import patiroha
import utils
import utils_ai

utils.configure_matplotlib_font()

# ==================================================================
# --- 1. 設定・ヘルパー関数 ---
# ==================================================================
warnings.filterwarnings('ignore')



@st.cache_data
def parse_ipc_atlas(ipc, level):
    if not isinstance(ipc, str):
        return ""
    ipc = ipc.strip().upper()
    
    if level == 1:
        return ipc[:4]
    elif level == 2:
        match = re.match(r'([A-H][0-9]{2}[A-Z]\s*[0-9]+)', ipc)
        return f"{match.group(1).strip()}/00" if match else ipc
    else:
        return ipc

@st.cache_data
def create_treemap_data(df_stats, start_year, end_year, mode="ipc"):
    df_target = df_stats.copy()
    
    if mode == "ipc":
        df_exploded = df_target['ipc_normalized'].explode().dropna().astype(str).str.upper()
        data = []
        for ipc in df_exploded:
            if len(ipc) >= 4:
                section = ipc[0]
                ipc_class = ipc[:3]
                subclass = ipc[:4]
                data.append([section, ipc_class, subclass])
        df_tree = pd.DataFrame(data, columns=['Section', 'Class', 'Subclass'])
        # 同一サブクラスの出現回数を集計
        df_tree = df_tree.groupby(['Section', 'Class', 'Subclass']).size().reset_index(name='count')
        return df_tree
        
    elif mode == "applicant":
        df_exploded = df_target['applicant_main'].explode().dropna()
        df_tree = df_exploded.value_counts().reset_index()
        df_tree.columns = ['Applicant', 'count']
        df_tree = df_tree.head(50)
        df_tree['Root'] = 'Total'
        return df_tree

def update_fig_layout(fig, title, height=600, show_legend=True):
    # タイトルから暗黙的/明示的なHTMLタグを除去してサニタイズ
    if isinstance(title, str):
        title = re.sub(r'<[^>]+>', '', title)

    layout_params = dict(
        template=utils.APOLLO_TEMPLATE,
        title=dict(text=title, font=dict(size=18, color=utils.APOLLO_TEXT, family="Helvetica Neue", weight="normal")),
        paper_bgcolor=utils.APOLLO_BG,
        plot_bgcolor=utils.APOLLO_BG,
        font_color=utils.APOLLO_TEXT,
        height=height
    )
    if not show_legend:
        layout_params['showlegend'] = False
        
    fig.update_layout(**layout_params)
    return fig

# ==================================================================
# --- 2. アプリケーション初期化 & UI構成 ---
# ==================================================================

st.set_page_config(
    page_title="APOLLO v8 | ATLAS",
    page_icon="🌍",
    layout="wide"
)

st.session_state['current_page'] = 'ATLAS'

utils.render_sidebar()

st.title("🌍 ATLAS")
st.markdown("出願推移・出願人ランキング・IPC分布・市場集中度（HHI/Entropy/Gini）を自動可視化し、データセットの全体像を把握します。")

# ==================================================================
# --- 3. データロード & 前処理チェック ---
# ==================================================================

if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    try:
        df_main = st.session_state.df_main
        col_map = st.session_state.col_map
        required_cols = ['year', 'applicant_main', 'ipc_normalized']
        if not all(col in df_main.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_main.columns]
            st.error(f"エラー: 必要なカラム {missing} が見つかりません。Mission Controlで前処理を再実行してください。")
            st.stop()
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        st.stop()

# ==================================================================
# --- 4. 分析アプリケーション ---
# ==================================================================

st.subheader("共通フィルタ設定")
min_year = int(df_main['year'].min())
max_year = int(df_main['year'].max())

col1, col2 = st.columns(2)
with col1:
    stats_start_year = st.number_input('集計開始年:', min_value=min_year, max_value=max_year, value=min_year, key="atlas_start_year")
with col2:
    stats_end_year = st.number_input('集計終了年:', min_value=min_year, max_value=max_year, value=max_year, key="atlas_end_year")

try:
    df_filtered = df_main[
        (df_main['year'] >= int(stats_start_year)) & 
        (df_main['year'] <= int(stats_end_year))
    ].copy()
    st.success(f"集計対象: {int(stats_start_year)}年～{int(stats_end_year)}年 (全 {len(df_filtered)} 件)")
except Exception as e:
    st.error(f"期間フィルタの適用に失敗しました: {e}")
    df_filtered = pd.DataFrame()

# CAPCOM data/ JSON出力（ATLAS基本統計）
try:
    import capcom
    if capcom.is_active() and not df_filtered.empty:
        _yearly = df_filtered['year'].value_counts().sort_index()
        _trend_dict = {str(int(k)): int(v) for k, v in _yearly.items()}

        # CAGR計算
        _cagr_val, _trend_label = utils.calculate_cagr_slope(df_filtered)

        # 出願人ランキング
        _app_ranking = {}
        if 'applicant_main' in df_filtered.columns:
            _all_apps = [a for sublist in df_filtered['applicant_main'] for a in sublist]
            _app_counts_ser = pd.Series(_all_apps).value_counts()
            _app_ranking = {k: int(v) for k, v in _app_counts_ser.head(30).items()}

        # 多様性指標（HHI / Entropy / Gini）
        _div_result = None
        if _app_ranking:
            _counts_list = list(_app_ranking.values())
            _div_result = patiroha.calculate_diversity(_counts_list)

        # IPC分布
        _ipc_ranking = {}
        ipc_col = col_map.get('ipc')
        if ipc_col and ipc_col in df_filtered.columns:
            _delim = st.session_state.get('delimiters', {}).get('ipc', ',')
            _ipc_ser = df_filtered[ipc_col].fillna('').str.split(_delim).explode().str.strip().str[:4]
            _ipc_counts = _ipc_ser[_ipc_ser != ''].value_counts()
            _ipc_ranking = {k: int(v) for k, v in _ipc_counts.head(20).items()}

        atlas_json = {
            "metadata": {
                "module": "ATLAS",
                "period": f"{int(stats_start_year)}-{int(stats_end_year)}",
                "total_patents": len(df_filtered),
                "unique_applicants": len(_app_ranking) if _app_ranking else 0
            },
            "trend": _trend_dict,
            "cagr": f"{_cagr_val:.1%}" if _cagr_val is not None else "N/A",
            "trend_direction": _trend_label if _trend_label else "N/A",
            "applicant_ranking": _app_ranking,
            "ipc_ranking": _ipc_ranking,
            "hhi": round(_div_result.hhi, 4) if _div_result else None,
            "hhi_status": _div_result.hhi_status if _div_result else None,
            "entropy": round(_div_result.entropy, 4) if _div_result else None,
            "gini": round(_div_result.gini, 4) if _div_result else None,
            "n_entities": _div_result.n_entities if _div_result else None
        }

        # ステータス分布（存在する場合のみ）
        _status_col = col_map.get('status')
        if _status_col and _status_col in df_filtered.columns:
            _status_counts = df_filtered[_status_col].value_counts()
            atlas_json["status_distribution"] = {str(k): int(v) for k, v in _status_counts.items()}
            _status_by_year = df_filtered.groupby(['year', _status_col]).size().unstack(fill_value=0)
            atlas_json["status_by_year"] = {
                str(int(year)): {str(col): int(val) for col, val in row.items()}
                for year, row in _status_by_year.iterrows()
            }

        capcom.save_data("atlas_statistics.json", atlas_json)
except Exception as e:
    pass

st.markdown("---")

tab1, tab1_line, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "件数推移", 
    "件数推移（折れ線）",
    "出願人ランキング", 
    "IPCランキング", 
    "出願人×年 バブル", 
    "IPC×出願人 バブル",
    "構成比マップ (Treemap)",
    "ライフサイクルマップ"
])

# --- ステータスの配色設定 (APOLLO公式パレット — CAPCOMレポートと統一) ---
# ステータスの意味からセマンティックに色を割り当てる。
# 全タブで色が統一されるように、ステータスごとの色を固定する。
status_color_map = {}
status_col = st.session_state.col_map.get('status')
if status_col:
    # APOLLO 公式カラーパレット (CAPCOM report_style.typ と統一)
    APOLLO_STATUS_COLORS = {
        'granted':   "#1B2A4A",  # NAVY        — 登録/有効 (主役)
        'published': "#2E5090",  # BLUE        — 公開
        'pending':   "#3B7DD8",  # ACCENT      — 出願中
        'examining': "#D4A017",  # AMBER       — 審査中 (注視)
        'rejected':  "#D64545",  # RED_ACCENT  — 拒絶 (マイナス)
        'withdrawn': "#666666",  # MEDIUM_GRAY — 取下げ
        'expired':   "#CCCCCC",  # BORDER_GRAY — 放棄/消滅/失効
    }
    # 未分類ステータス用のフォールバック循環色 (NAVYのモノクロ濃淡)
    APOLLO_STATUS_FALLBACK = ["#5B6E92", "#8A9CC0", "#A8B6D0", "#7D6B8A", "#9C8FA8"]

    def _classify_status(s):
        """ステータス文字列をセマンティックカテゴリに分類する"""
        s_lower = str(s).lower()
        # 登録/有効
        if any(k in s_lower for k in ['granted', 'registered', 'active', '登録', '有効', '権利存続', '存続']):
            return 'granted'
        # 拒絶 (← 「拒絶査定後の出願」等の誤判定を防ぐため pending より先)
        if any(k in s_lower for k in ['rejected', 'refused', 'denied', '拒絶', '却下']):
            return 'rejected'
        # 取下げ
        if any(k in s_lower for k in ['withdrawn', 'withdraw', '取下', '取り下げ']):
            return 'withdrawn'
        # 放棄/消滅/失効
        if any(k in s_lower for k in ['expired', 'lapsed', 'abandoned', 'dead', '消滅', '失効', '放棄', '満了']):
            return 'expired'
        # 審査中
        if any(k in s_lower for k in ['examining', 'examination', 'review', '審査', '審理']):
            return 'examining'
        # 公開
        if any(k in s_lower for k in ['published', 'publication', '公開', '公表']):
            return 'published'
        # 出願中
        if any(k in s_lower for k in ['pending', 'application', 'filed', 'filing', '出願', '係属']):
            return 'pending'
        return None  # 未分類

    # 全てのユニークなステータスを取得（ソートして順序を固定）
    unique_statuses_all = sorted(df_filtered[status_col].dropna().unique().astype(str))
    fallback_idx = 0
    for s in unique_statuses_all:
        category = _classify_status(s)
        if category and category in APOLLO_STATUS_COLORS:
            status_color_map[s] = APOLLO_STATUS_COLORS[category]
        else:
            status_color_map[s] = APOLLO_STATUS_FALLBACK[fallback_idx % len(APOLLO_STATUS_FALLBACK)]
            fallback_idx += 1

# 1. 件数推移
with tab1:
    st.subheader("出願件数時系列推移")
    
    # Status Breakdown Option
    use_status_breakdown = False
    status_col = st.session_state.col_map.get('status')
    if status_col:
        use_status_breakdown = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab1")

    if st.button("件数推移グラフを描画", key="atlas_run_map1"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            if use_status_breakdown and status_col:
                 # Stacked Bar Chart by Status
                plot_data = df_filtered.groupby(['year', status_col]).size().reset_index(name='count')
                # Use color_discrete_map for consistency
                fig = px.bar(plot_data, x='year', y='count', color=status_col, labels={'year': '出願年', 'count': '出願件数', status_col: 'ステータス'}, 
                             color_discrete_map=status_color_map,
                             category_orders={status_col: sorted(status_color_map.keys())} # 凡例の順序を統一
                            )
            else:
                # Standard Bar Chart
                yearly_counts = df_filtered['year'].value_counts().sort_index()
                plot_data = yearly_counts.reindex(range(int(stats_start_year), int(stats_end_year) + 1), fill_value=0)
                fig = px.bar(x=plot_data.index, y=plot_data.values, labels={'x': '出願年', 'y': '出願件数'}, color_discrete_sequence=[utils.APOLLO_COLORS[0]])
            
            update_fig_layout(fig, f'出願件数時系列推移 ({int(stats_start_year)}年～{int(stats_end_year)}年)')
            
            st.session_state['atlas_fig_trend'] = fig
            st.session_state['atlas_data_trend'] = plot_data

    # 永続表示
    if 'atlas_fig_trend' in st.session_state:
        fig = st.session_state['atlas_fig_trend']
        plot_data = st.session_state['atlas_data_trend']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # スナップショットボタン
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (Wide Format: Year | Total Only)
        if hasattr(plot_data, 'columns') and 'year' in plot_data.columns and 'count' in plot_data.columns:
             # Group by Year and Sum Count, ignoring Status (Stacked Bar)
             df_snap_safe = plot_data.groupby('year')['count'].sum().reset_index()
             df_snap_safe['year'] = df_snap_safe['year'].astype(int)
        elif hasattr(plot_data, 'reset_index'):
             # Handle Series case (Standard Bar) -> Convert to DataFrame
             df_snap_safe = plot_data.reset_index()
             if df_snap_safe.shape[1] == 2:
                 df_snap_safe.columns = ['year', 'count']
             # Ensure year is int if possible
             if 'year' in df_snap_safe.columns:
                 df_snap_safe['year'] = df_snap_safe['year'].astype(int)
        else:
            df_snap_safe = pd.DataFrame(plot_data)
            
        # Ensure we don't exceed token limits but prioritize showing full year range
        snap_data['chart_data'] = df_snap_safe.head(50).to_string(index=False)
        utils.render_snapshot_button(
            title=f"出願件数推移 ({int(stats_start_year)}-{int(stats_end_year)})",
            description="市場全体の出願動向を示すトレンドグラフ。",
            key="atlas_trend_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: 出願トレンド分析 ---
        _meta_trend = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        # CAGR計算
        if hasattr(plot_data, 'columns') and 'year' in plot_data.columns and 'count' in plot_data.columns:
            _yearly = plot_data.groupby('year')['count'].sum().sort_index()
        elif hasattr(plot_data, 'index'):
            _yearly = plot_data.sort_index()
        else:
            _yearly = pd.Series(dtype=float)
        if len(_yearly) >= 2:
            _first_val = _yearly.iloc[0] if _yearly.iloc[0] > 0 else 1
            _last_val = _yearly.iloc[-1] if _yearly.iloc[-1] > 0 else 1
            _n_years = len(_yearly) - 1
            _cagr = ((_last_val / _first_val) ** (1 / max(_n_years, 1)) - 1) * 100
            _meta_trend['CAGR(%)'] = f"{_cagr:.1f}%"
            _meta_trend['初年件数'] = int(_yearly.iloc[0])
            _meta_trend['最終年件数'] = int(_yearly.iloc[-1])
        _meta_trend['出願人数'] = int(df_filtered['applicant_main'].explode().str.strip().nunique())

        _trend_prompt = utils_ai.generate_ai_insight_prompt(
            role="特許分析の専門家として、出願件数の時系列推移データを分析してください。",
            context="棒グラフによる出願件数の年次推移を表示しています。各年の出願件数から、技術分野の成長・成熟・衰退のステージを判定します。",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **全体トレンド**: 出願件数の推移パターン（成長/停滞/衰退）を特定し、ライフサイクルステージを判定
2. **変曲点**: 出願件数が大きく変化した年とその背景仮説
3. **成長率分析**: CAGRの評価と今後3-5年の予測トレンド
4. **市場示唆**: 出願動向から読み取れる市場・技術動向への示唆

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_trend,
            constraints="データに基づく客観的分析を行い、推測は明確に区別すること。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_trend_prompt, "atlas_trend_insight")

# 1.5 件数推移（折れ線）
with tab1_line:
    st.subheader("件数推移 (折れ線グラフ)")
    
    col_line_1, col_line_2 = st.columns([2, 1])
    
    with col_line_1:
        # モード選択
        line_mode = st.radio("表示モード:", ["全体推移", "出願人比較"], horizontal=True, key="atlas_line_mode")
    
    with col_line_2:
        # ステータス内訳オプション (全体推移モードのみ)
        use_status_breakdown_line = False
        if line_mode == "全体推移" and status_col:
            use_status_breakdown_line = st.checkbox("ステータス内訳を表示", key="atlas_use_status_line")
    
    target_applicants = []
    
    if line_mode == "出願人比較":
        # Prepare applicant list with counts
        if not df_filtered.empty:
            # Explode and count
            assignees_exploded_line = df_filtered.explode('applicant_main')
            assignees_exploded_line['assignee_parsed'] = assignees_exploded_line['applicant_main'].str.strip()
            
            # Count per applicant
            app_counts = assignees_exploded_line['assignee_parsed'].value_counts()
            
            # Create formatted options: "Name (Count)"
            # Sort is implied by value_counts() which returns descending order
            app_options = [f"{name} ({count})" for name, count in app_counts.items()]
            app_map = {f"{name} ({count})": name for name, count in app_counts.items()}
            
            selected_options = st.multiselect(
                "出願人を選択 (最大5社):", 
                options=app_options,
                max_selections=5,
                key="atlas_line_applicants"
            )
            
            # Map back to raw names
            target_applicants = [app_map[opt] for opt in selected_options]
    
    if st.button("折れ線グラフを描画", key="atlas_run_map1_line"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            fig = None
            plot_data = None
            
            if line_mode == "全体推移":
                if use_status_breakdown_line and status_col:
                     # 積み上げ面グラフ (内訳)
                    plot_data = df_filtered.groupby(['year', status_col]).size().reset_index(name='count')
                    
                    fig = px.area(plot_data, x='year', y='count', color=status_col, markers=True,
                                  labels={'year': '出願年', 'count': '出願件数', status_col: 'ステータス'},
                                  color_discrete_map=status_color_map,
                                  category_orders={status_col: sorted(status_color_map.keys())}
                                 )
                    fig.update_layout(title=dict(text=f'全体件数推移・内訳 ({int(stats_start_year)}年～{int(stats_end_year)}年)', font=dict(size=18)), yaxis=dict(rangemode='tozero'))
                    
                else:
                    # 全体推移 (標準折れ線)
                    yearly_counts = df_filtered['year'].value_counts().sort_index()
                    plot_data = yearly_counts.reindex(range(int(stats_start_year), int(stats_end_year) + 1), fill_value=0).reset_index()
                    plot_data.columns = ['year', 'count']
                    
                    fig = px.line(plot_data, x='year', y='count', markers=True, 
                                  labels={'year': '出願年', 'count': '出願件数'},
                                  color_discrete_sequence=[utils.APOLLO_COLORS[0]])
                    
                    fig.update_layout(title=dict(text=f'全体件数推移 ({int(stats_start_year)}年～{int(stats_end_year)}年)', font=dict(size=18)), yaxis=dict(rangemode='tozero'))

            else: # 出願人比較
                if not target_applicants:
                    st.warning("出願人を選択してください。")
                else:
                    # Filter data for selected applicants
                    assignees_exploded_line = df_filtered.explode('applicant_main')
                    assignees_exploded_line['assignee_parsed'] = assignees_exploded_line['applicant_main'].str.strip()
                    
                    df_target = assignees_exploded_line[assignees_exploded_line['assignee_parsed'].isin(target_applicants)]
                    
                    if df_target.empty:
                        st.warning("選ばれた出願人のデータが期間内にありません。")
                    else:
                        # Ensure all years are represented for each applicant (fill 0)
                        full_years = range(int(stats_start_year), int(stats_end_year) + 1)
                        plot_data_list = []
                        
                        for app in target_applicants:
                            sub = df_target[df_target['assignee_parsed'] == app]
                            yearly = sub['year'].value_counts().sort_index()
                            yearly = yearly.reindex(full_years, fill_value=0).reset_index()
                            yearly.columns = ['year', 'count']
                            yearly['assignee_parsed'] = app
                            plot_data_list.append(yearly)
                            
                        plot_data = pd.concat(plot_data_list, ignore_index=True)
                        
                        fig = px.line(plot_data, x='year', y='count', color='assignee_parsed', markers=True,
                                      labels={'year': '出願年', 'count': '出願件数', 'assignee_parsed': '出願人'},
                                      color_discrete_sequence=utils.APOLLO_COLORS)
                        
                        fig.update_layout(title=dict(text='主要出願人の件数推移比較', font=dict(size=18)), yaxis=dict(rangemode='tozero'))
            
            if fig:
                update_fig_layout(fig, '件数推移(折れ線)')
                
                # Check for session state initialization
                if 'atlas_fig_trend_line' not in st.session_state:
                     st.session_state['atlas_fig_trend_line'] = None
                
                st.session_state['atlas_fig_trend_line'] = fig
                st.session_state['atlas_data_trend_line'] = plot_data

    # Persistent Display
    if 'atlas_fig_trend_line' in st.session_state and st.session_state['atlas_fig_trend_line'] is not None:
        fig = st.session_state['atlas_fig_trend_line']
        data = st.session_state['atlas_data_trend_line']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        # Snapshot Button
        snap_data = utils.generate_rich_summary(df_filtered if 'df_target' not in locals() else df_target, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        # Optimize Chart Data
        # Optimize Chart Data (Wide Format for Applicants)
        if data is not None and not data.empty:
             if 'assignee_parsed' in data.columns:
                 # Pivot: Year | App A | App B ...
                 df_pivot = data.pivot(index='year', columns='assignee_parsed', values='count').fillna(0).astype(int).reset_index()
                 snap_data['chart_data'] = df_pivot.head(40).to_string(index=False)
             else:
                 snap_data['chart_data'] = data.head(40).to_string(index=False)
        else:
             snap_data['chart_data'] = "No Data"
        utils.render_snapshot_button(
            title="件数推移 (折れ線)",
            description="出願件数の時系列推移（折れ線グラフ）。全体または特定出願人の比較。",
            key="atlas_trend_line_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: 折れ線トレンド分析 ---
        _meta_line = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        _meta_line['表示モード'] = line_mode
        _line_prompt = utils_ai.generate_ai_insight_prompt(
            role="特許分析の専門家として、出願件数の時系列推移（折れ線グラフ）を分析してください。",
            context="折れ線グラフによる出願件数の年次推移を表示しています。全体推移モードでは市場全体の動向を、出願人比較モードでは主要プレイヤーの出願戦略の変遷を可視化しています。",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **推移パターン**: 各出願人/全体の推移パターン（成長加速/減速/逆転）を特定
2. **競争動態**: 出願人間の順位変動やシェア逆転があった年を特定
3. **相関分析**: 複数出願人の推移に同期的な動きがあるかを評価
4. **予測**: 直近の傾向から今後2-3年の方向性を予測

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_line,
            constraints="折れ線グラフの傾きの変化に注目し、変曲点を見逃さないこと。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_line_prompt, "atlas_trend_line_insight")

# 2. 出願人ランキング
with tab2:
    st.subheader("出願人ランキング")
    col2_1, col2_2 = st.columns([2, 1])
    with col2_1:
         num_to_display_map2 = st.number_input("表示人数:", min_value=1, value=20, key="atlas_num_apps_map2")
    
    # Status Breakdown Option
    use_status_breakdown_tab2 = False
    status_col = st.session_state.col_map.get('status')
    with col2_2:
        if status_col:
            use_status_breakdown_tab2 = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab2")

    if st.button("出願人ランキングを描画", key="atlas_run_map2"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            # 1. 上位出願人の特定 (合計件数に基づく)
            assignee_counts = df_filtered['applicant_main'].explode().str.strip().value_counts().head(int(num_to_display_map2)).sort_values(ascending=True)
            top_applicants = assignee_counts.index.tolist()

            if use_status_breakdown_tab2 and status_col:
                # Stacked Bar Chart by Status for Top Applicants
                df_exploded = df_filtered.explode('applicant_main')
                df_exploded['applicant_parsed'] = df_exploded['applicant_main'].str.strip()
                df_top = df_exploded[df_exploded['applicant_parsed'].isin(top_applicants)]
                
                plot_data = df_top.groupby(['applicant_parsed', status_col]).size().reset_index(name='count')
                
                # Ensure sort order matches total count
                fig = px.bar(plot_data, x='count', y='applicant_parsed', color=status_col, orientation='h', 
                             labels={'count': '特許件数', 'applicant_parsed': '出願人', status_col: 'ステータス'}, 
                             color_discrete_map=status_color_map,
                             category_orders={'applicant_parsed': top_applicants[::-1], status_col: sorted(status_color_map.keys())})
            else:
                # Standard Bar Chart
                fig = px.bar(x=assignee_counts.values, y=assignee_counts.index, orientation='h', labels={'x': '特許件数', 'y': '出願人'}, color_discrete_sequence=[utils.APOLLO_COLORS[1]])
            
            update_fig_layout(fig, f'出願人ランキング ({int(stats_start_year)}年～{int(stats_end_year)}年)', height=max(600, len(assignee_counts)*30))
            
            st.session_state['atlas_fig_ranking'] = fig
            st.session_state['atlas_data_ranking'] = assignee_counts

    # 永続表示
    if 'atlas_fig_ranking' in st.session_state:
        fig = st.session_state['atlas_fig_ranking']
        assignee_counts = st.session_state['atlas_data_ranking']

        st.plotly_chart(fig, use_container_width=True, config={'editable': False})

        # 多様性指標メトリクス
        _applicant_counts_for_div = assignee_counts.values.tolist()
        if _applicant_counts_for_div:
            div = patiroha.calculate_diversity(_applicant_counts_for_div)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("HHI (集中度)", f"{div.hhi:.3f}", help=div.hhi_status)
            with c2:
                st.metric("Entropy (多様性)", f"{div.entropy:.2f}", help="高い=分散、低い=集中")
            with c3:
                st.metric("Gini (不平等度)", f"{div.gini:.3f}", help="0=完全平等、1=完全不平等")

        # スナップショットボタン
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data
        df_snap_safe = assignee_counts.head(30).reset_index()
        df_snap_safe.columns = ['Applicant', 'Count']
        df_snap_safe['Applicant'] = df_snap_safe['Applicant'].astype(str).str.slice(0, 50)
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)
        utils.render_snapshot_button(
            title=f"主要出願人ランキング ({int(stats_start_year)}-{int(stats_end_year)})",
            description="特許出願件数に基づく市場の主要プレイヤーランキング。",
            key="atlas_applicant_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: 競争環境・集中度分析 ---
        _meta_rank = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        # 多様性指標計算（HHI / Entropy / Gini）
        _total_patents = assignee_counts.sum()
        if _total_patents > 0:
            _div_rank = patiroha.calculate_diversity(assignee_counts.values.tolist())
            _shares = assignee_counts / _total_patents
            _meta_rank['HHI(市場集中度)'] = f"{_div_rank.hhi:.4f} ({_div_rank.hhi_status})"
            _meta_rank['Entropy(多様性)'] = f"{_div_rank.entropy:.2f}"
            _meta_rank['Gini(不平等度)'] = f"{_div_rank.gini:.3f}"
            _meta_rank['上位出願人数'] = len(assignee_counts)
            _meta_rank['上位出願人の合計件数'] = int(_total_patents)
            _meta_rank['1位シェア(%)'] = f"{_shares.iloc[-1]*100:.1f}%" if len(_shares) > 0 else "N/A"
        _meta_rank['全出願人数'] = int(df_filtered['applicant_main'].explode().str.strip().nunique())

        _rank_prompt = utils_ai.generate_ai_insight_prompt(
            role="特許分析・競争戦略の専門家として、出願人ランキングデータから競争環境を分析してください。",
            context="水平棒グラフによる出願人ランキングを表示しています。出願件数の多い順に主要プレイヤーを表示し、市場の競争構造を可視化しています。",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **市場集中度**: HHI指数に基づく市場構造の評価（独占/寡占/競争的）
2. **多様性・不平等度**: Entropy（情報エントロピー）とGini係数から見た出願人分布の特徴
3. **プレイヤー分類**: リーダー/チャレンジャー/フォロワー/ニッチャーに分類
4. **競争パターン**: 上位と下位の件数格差、参入障壁の高さの推定
5. **戦略的示唆**: 新規参入者やポートフォリオ戦略へのインプリケーション

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_rank,
            constraints="出願件数は必ずしも特許の質や事業規模と一致しない点に注意すること。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_rank_prompt, "atlas_ranking_insight")

# 3. IPCランキング
with tab3:
    st.subheader("IPCランキング")
    ipc_level_map3 = st.selectbox("IPCレベル:", [(1, "サブクラス (A01B)"), (2, "メイングループ (A01B 1/00)")], format_func=lambda x: x[1], key="atlas_ipc_level_map3")
    num_to_display_map3 = st.number_input("表示IPC数:", min_value=1, value=20, key="atlas_num_ipcs_map3")
    if st.button("IPCランキングを描画", key="atlas_run_map3"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            ipc_exploded = df_filtered['ipc_normalized'].explode().dropna()
            ipc_parsed = ipc_exploded.apply(lambda x: parse_ipc_atlas(x, ipc_level_map3[0]))
            ipc_counts = ipc_parsed.value_counts().head(int(num_to_display_map3)).sort_values(ascending=True)
            fig = px.bar(x=ipc_counts.values, y=ipc_counts.index, orientation='h', labels={'x': '特許件数', 'y': 'IPC分類'}, color_discrete_sequence=[utils.APOLLO_COLORS[2]])
            update_fig_layout(fig, f'IPCランキング ({ipc_level_map3[1]})', height=max(600, len(ipc_counts)*30))
            
            st.session_state['atlas_fig_ipc'] = fig
            st.session_state['atlas_data_ipc'] = ipc_counts

    # 永続表示
    if 'atlas_fig_ipc' in st.session_state:
        fig = st.session_state['atlas_fig_ipc']
        data = st.session_state['atlas_data_ipc']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # スナップショットボタン
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data
        df_snap_safe = data.head(30).reset_index()
        df_snap_safe.columns = ['IPC', 'Count']
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)
        utils.render_snapshot_button(
            title=f"IPCランキング ({ipc_level_map3[1]})",
            description="技術分野 (IPC) 別の上位ランキング。",
            key="atlas_ipc_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: IPC技術分野構成分析 ---
        _meta_ipc = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        _meta_ipc['IPC分類数'] = len(data)
        _meta_ipc['表示レベル'] = ipc_level_map3[1]
        _ipc_prompt = utils_ai.generate_ai_insight_prompt(
            role="特許分析・技術動向の専門家として、IPC分類ランキングから技術分野の構成を分析してください。",
            context="水平棒グラフによるIPC（国際特許分類）のランキングを表示しています。出願件数の多い順に技術分野を表示し、研究開発の注力領域を可視化しています。",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **技術集中度**: 上位IPC分類への集中度合い（上位3-5分類が全体に占める割合）
2. **技術的特徴**: 主要IPC分類が示す技術領域の解釈と相互関係
3. **技術多様性**: IPC分布の偏りから技術ポートフォリオの幅を評価
4. **戦略的示唆**: 特定IPC分類への過度な集中や空白領域の戦略的意味

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_ipc,
            constraints="IPC分類コードの意味を正確に解釈すること。不明なIPCには推測を明記すること。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_ipc_prompt, "atlas_ipc_insight")

# 4. 出願人×年 バブル
with tab4:
    st.subheader("出願人 × 年 バブルチャート")
    col4_1, col4_2 = st.columns([2, 1])
    with col4_1:
         num_to_display_map4 = st.number_input("表示人数:", min_value=1, value=10, key="atlas_num_apps_map4")
    
    # Status Breakdown Option
    use_status_breakdown_tab4 = False
    status_col = st.session_state.col_map.get('status')
    with col4_2:
        if status_col:
            use_status_breakdown_tab4 = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab4")

    if st.button("出願人×年 バブルを描画", key="atlas_run_map4"):
        assignees_exploded = df_filtered.explode('applicant_main')
        assignees_exploded['assignee_parsed'] = assignees_exploded['applicant_main'].str.strip()
        top_assignees = assignees_exploded['assignee_parsed'].value_counts().head(int(num_to_display_map4)).index.tolist()
        
        # Filter for top applicants upfront
        df_target = assignees_exploded[assignees_exploded['assignee_parsed'].isin(top_assignees)].copy()
        
        if df_target.empty:
            st.warning("データがありません。")
        else:
            if use_status_breakdown_tab4 and status_col:
                # --- グリッド状パイチャートの描画 ---
                # 1. グリッド寸法の計算
                start_y = int(stats_start_year)
                end_y = int(stats_end_year)
                
                # Let's align with the filter range for stability
                cols = list(range(start_y, end_y + 1))
                
                # Filter data to this range
                df_target = df_target[df_target['year'].isin(cols)]
                
                if df_target.empty:
                    st.warning("指定期間内のデータがありません。")
                else:
                    # Rows = Applicants, Cols = Years (Linear Sequence)
                    rows = top_assignees 
                    
                    n_rows = len(rows)
                    n_cols = len(cols)
                    
                    fig = go.Figure()
                    
                    # Group by [Applicant, Year, Status]
                    grid_data = df_target.groupby(['assignee_parsed', 'year', status_col]).size().reset_index(name='count')
                    total_counts = df_target.groupby(['assignee_parsed', 'year']).size().reset_index(name='total')
                    max_total = total_counts['total'].max()
                    
                    # Layout Calculation
                    x_margin_l = 0.20 # Increased to 0.20 to prevent label cutoff and align with Standard
                    x_margin_r = 0.02
                    y_margin_b = 0.10 
                    y_margin_t = 0.05
                    
                    plot_width = 1.0 - (x_margin_l + x_margin_r)
                    plot_height = 1.0 - (y_margin_b + y_margin_t)
                    
                    cell_w = plot_width / n_cols
                    cell_h = plot_height / n_rows
                    
                    # Prepare Legend Colors
                    
                    # Filter map to only statuses present in this view for the legend
                    statuses_in_view = sorted(df_target[status_col].dropna().unique().astype(str))
                    
                    # Add Dummy Traces for Legend (Scatter markers)
                    for status in statuses_in_view:
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(size=10, color=status_color_map.get(status, '#ccc')),
                            name=status,
                            showlegend=True
                        ))
                    
                    # Annotations for Axes
                    annotations = []
                    
                    # Y-Axis Labels (Applicants)
                    for i, applicant in enumerate(rows):
                        y_center = (1.0 - y_margin_t) - (i * cell_h) - (cell_h / 2)
                        
                        annotations.append(dict(
                            x=x_margin_l - 0.01, y=y_center,
                            xref="paper", yref="paper",
                            text="",
                            showarrow=False, xanchor="right", yanchor="middle",
                            font=dict(size=12, color=utils.APOLLO_TEXT)
                        ))
                        
                    # X-axis labels are now handled by layout.xaxis
                    annotations = []

                    # Add Pie Traces
                    for i, applicant in enumerate(rows):
                        for j, year in enumerate(cols):
                            cell_df = grid_data[(grid_data['assignee_parsed'] == applicant) & (grid_data['year'] == year)]
                            
                            if not cell_df.empty:
                                total = cell_df['count'].sum()
                                max_r = min(cell_w, cell_h) / 2 * 0.9
                                scale_factor = (total / max_total) ** 0.5
                                # Use sqrt scaling for visual size
                                y_center = (1.0 - y_margin_t) - (i * cell_h) - (cell_h / 2)
                                x_center = x_margin_l + (j * cell_w) + (cell_w / 2)
                                
                                # Domain Calc
                                d_w = cell_w * scale_factor
                                
                                x0 = x_center - (d_w / 2)
                                x1 = x_center + (d_w / 2)
                                y0 = y_center - (scale_factor * cell_h / 2) 
                                y1 = y_center + (scale_factor * cell_h / 2)
                                
                                # Map colors explicitly
                                labels = cell_df[status_col].astype(str).tolist()
                                values = cell_df['count'].tolist()
                                colors = [status_color_map.get(l, '#ccc') for l in labels]
                                
                                fig.add_trace(go.Pie(
                                    labels=labels,
                                    values=values,
                                    marker=dict(colors=colors),
                                    domain=dict(x=[x0, x1], y=[y0, y1]),
                                    showlegend=False, # Use dummy legend instead
                                    textinfo='none',
                                    hoverinfo='label+value',
                                    sort=False 
                                ))
                    
                    # Manual Grid Lines Removed (Handled by yaxis.showgrid)
                    shapes = []
                    
                    # Layout Finalization
                    fig.update_layout(
                        height=max(700, n_rows * 50),
                        annotations=annotations,
                        shapes=shapes,
                        showlegend=True,
                        xaxis=dict(
                            visible=True,
                            domain=[x_margin_l, 1.0 - x_margin_r],
                            # Range: [min_year - 0.5, max_year + 0.5]
                            range=[cols[0] - 0.5, cols[-1] + 0.5],
                            tickmode='auto', 
                            side='bottom',
                            color=utils.APOLLO_TEXT,
                            fixedrange=True, 
                            showgrid=False,
                            zeroline=False,
                            showline=False
                        ),
                        yaxis=dict(
                            visible=True,
                            domain=[y_margin_b, 1.0 - y_margin_t],
                            # Map rows (0..N-1) to Y-axis. Top-down order.
                            range=[-0.5, n_rows - 0.5],
                            tickmode='array',
                            tickvals=list(range(n_rows)),
                            ticktext=rows[::-1], # Reverse to put Top Applicant at Top
                            color=utils.APOLLO_TEXT,
                            fixedrange=True, 
                            showgrid=True,   
                            gridcolor="#eee",
                            zeroline=False,
                            showline=False
                        ),
                        margin=dict(l=0, r=0, t=40, b=0),
                        paper_bgcolor=utils.APOLLO_BG, 
                        plot_bgcolor=utils.APOLLO_BG,
                        font_color=utils.APOLLO_TEXT,
                        title=dict(text=f'出願年別 出願人動向 (内訳: {status_col})', font=dict(size=18, weight="normal"))
                    )
                    
                # Save to unified state
                st.session_state['atlas_fig_bubble_tab4'] = fig

                # Re-create grid data for state storage since it was local
                grid_data_export = df_target.groupby(['year', 'assignee_parsed', status_col]).size().reset_index(name='count')
                st.session_state['atlas_data_bubble_tab4'] = grid_data_export
            else:
                # Standard Bubble Chart
                plot_data = df_target.groupby(['year', 'assignee_parsed']).size().reset_index(name='件数')
                
                # --- Shared Layout Constants ---
                x_margin_l = 0.20 # Match Breakdown
                x_margin_r = 0.02
                y_margin_b = 0.10
                y_margin_t = 0.05
                
                fig = px.scatter(plot_data, x='year', y='assignee_parsed', size='件数', color='assignee_parsed', 
                                 labels={'year': '出願年', 'assignee_parsed': '出願人', '件数': '件数'}, 
                                 color_discrete_sequence=utils.APOLLO_COLORS, 
                                 category_orders={"assignee_parsed": top_assignees}) # px handles order
                
                # Apply Strict Layout to Match Breakdown
                update_fig_layout(fig, '出願年別 出願人動向', height=max(700, len(top_assignees)*50))
                
                fig.update_layout(
                     margin=dict(l=0, r=0, t=40, b=0),
                     xaxis=dict(
                         domain=[x_margin_l, 1.0 - x_margin_r],
                         fixedrange=True,
                         side='bottom'
                     ),
                     yaxis=dict(
                         domain=[y_margin_b, 1.0 - y_margin_t],
                         fixedrange=True,
                         showgrid=True,
                         gridcolor="#eee",
                         visible=True # Ensure visible
                     )
                )
                
                # Save to unified state
                st.session_state['atlas_fig_bubble_tab4'] = fig
                st.session_state['atlas_data_bubble_tab4'] = plot_data

    # Persistent Display (Unified)
    if 'atlas_fig_bubble_tab4' in st.session_state:
        fig = st.session_state['atlas_fig_bubble_tab4']
        data = st.session_state['atlas_data_bubble_tab4']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # Snapshot Button
        # Snapshot Button
        snap_data = utils.generate_rich_summary(data, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data
        # Optimize Chart Data
        if hasattr(data, 'head'):

             # Copy data to avoid mutating session state
             chart_df = data.copy()
             
             # Normalize column names (件数 -> count)
             if '件数' in chart_df.columns:
                 chart_df.rename(columns={'件数': 'count'}, inplace=True)
             
             # If data is 'grid_data_export' (Year, Applicant, Status, Count) or 'plot_data' (Year, Applicant, Count)
             
             # Filter only necessary columns
             target_cols = [c for c in ['year', 'assignee_parsed', 'count', status_col] if c in chart_df.columns]
             df_snap_safe = chart_df[target_cols].copy()
             
             # Format
             if 'assignee_parsed' in df_snap_safe.columns:
                 df_snap_safe['assignee_parsed'] = df_snap_safe['assignee_parsed'].astype(str).str.slice(0, 30)
             
             # Pivot for readability (Year | App | Count...) is still long.
             # Maybe Pivot: Year vs Applicant (Values = Total Count)
             if 'year' in df_snap_safe.columns and 'assignee_parsed' in df_snap_safe.columns:
                 # Aggregate to remove status if just showing bubble position
                 df_pivot = df_snap_safe.groupby(['year', 'assignee_parsed'])['count'].sum().reset_index()
                 df_pivot = df_pivot.pivot(index='year', columns='assignee_parsed', values='count').fillna(0).astype(int).reset_index()
                 snap_data['chart_data'] = df_pivot.head(40).to_string(index=False)
             else:
                 snap_data['chart_data'] = df_snap_safe.head(40).to_string(index=False)
        else:
             snap_data['chart_data'] = "Data Summary"

        utils.render_snapshot_button(
            title="出願年別 出願人バブルチャート",
            description="主要出願人の時系列活動量 (内訳含む)",
            key="atlas_bubble_tab4_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: 出願人×年バブル分析 ---
        _meta_bubble = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        _meta_bubble['表示出願人数'] = num_to_display_map4
        _bubble_prompt = utils_ai.generate_ai_insight_prompt(
            role="特許分析・競争戦略の専門家として、出願人の年別出願活動をバブルチャートから分析してください。",
            context="バブルチャートで主要出願人の年別出願件数を可視化しています。バブルの大きさは件数、位置は年×出願人を示します。ステータス内訳がある場合はパイチャートグリッドで表示されています。",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **参入・撤退パターン**: 各出願人の出願開始年・活動期間・直近の活動有無から参入・撤退を特定
2. **活動量の変遷**: 各出願人の年別出願件数の増減パターン（加速/減速/一時停止）
3. **競合分析**: 同時期に活動が集中している出願人群の特定と競争構造の推定
4. **戦略転換**: 出願量が急変した出願人とその時期の特定

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_bubble,
            constraints="バブルサイズの違いに注目し、出願人間の規模感の差を明確にすること。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_bubble_prompt, "atlas_app_year_insight")

# 5. IPC×出願人 バブル
with tab5:
    st.subheader("IPC × 出願人 バブルチャート")
    col1, col2, col3 = st.columns(3)
    with col1: ipc_level_map5 = st.selectbox("IPCレベル:", [(1, "サブクラス"), (2, "メイングループ")], format_func=lambda x: x[1], key="atlas_ipc_level_map5")
    with col2: num_ipcs_map5 = st.number_input("IPC数 (Y軸):", min_value=1, value=15, key="atlas_num_ipcs_map5")
    with col3: num_apps_map5 = st.number_input("出願人数 (X軸):", min_value=1, value=15, key="atlas_num_apps_map5")
    if st.button("IPC×出願人 バブルを描画", key="atlas_run_map5"):
        df_exploded = df_filtered.explode('applicant_main').explode('ipc_normalized')
        df_exploded.dropna(subset=['applicant_main', 'ipc_normalized'], inplace=True)
        df_exploded['assignee_parsed'] = df_exploded['applicant_main'].str.strip()
        df_exploded['ipc_parsed'] = df_exploded['ipc_normalized'].apply(lambda x: parse_ipc_atlas(x, ipc_level_map5[0]))
        top_assignees = df_exploded['assignee_parsed'].value_counts().head(int(num_apps_map5)).index.tolist()
        top_ipcs = df_exploded['ipc_parsed'].value_counts().head(int(num_ipcs_map5)).index.tolist()
        df_top = df_exploded[df_exploded['assignee_parsed'].isin(top_assignees) & df_exploded['ipc_parsed'].isin(top_ipcs)]
        plot_data = df_top.groupby(['assignee_parsed', 'ipc_parsed']).size().reset_index(name='件数')
        if plot_data.empty:
            st.warning("データがありません。")
        else:
            fig = px.scatter(plot_data, x='assignee_parsed', y='ipc_parsed', size='件数', color='ipc_parsed', labels={'assignee_parsed': '出願人', 'ipc_parsed': 'IPC分類', '件数': '件数'}, color_discrete_sequence=utils.APOLLO_COLORS, category_orders={"ipc_parsed": top_ipcs})
            update_fig_layout(fig, f'IPC ({ipc_level_map5[1]}) × 出願人 ポートフォリオ', height=800)
            
            st.session_state['atlas_fig_bubble_ipc'] = fig
            st.session_state['atlas_data_bubble_ipc'] = plot_data

    # Persistent Display
    if 'atlas_fig_bubble_ipc' in st.session_state:
        fig = st.session_state['atlas_fig_bubble_ipc']
        data = st.session_state['atlas_data_bubble_ipc']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # Snapshot Button
        # Snapshot Button
        snap_data = utils.generate_rich_summary(data, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (IPC Bubble)
        df_snap_safe = data.head(30).copy()
        if 'assignee_parsed' in df_snap_safe.columns:
             df_snap_safe['assignee_parsed'] = df_snap_safe['assignee_parsed'].astype(str).str.slice(0, 50)
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)

        utils.render_snapshot_button(
            title=f"IPC x 出願人 ポートフォリオ",
            description="主要出願人の技術分野（IPC）ごとの注力度合いを示すバブルチャート。",
            key="atlas_bubble_ipc_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: IPC×出願人ポートフォリオ分析 ---
        _meta_ipc_app = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        _ipc_app_prompt = utils_ai.generate_ai_insight_prompt(
            role="特許分析・技術ポートフォリオの専門家として、IPC×出願人のバブルチャートから技術戦略を分析してください。",
            context="バブルチャートで主要出願人(X軸)と技術分野IPC(Y軸)の交差を可視化しています。バブルの大きさは各出願人が各IPC分類に出願した件数を示します。",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **専門vs多角**: 各出願人が特定IPCに集中しているか、広範囲に分散しているかを評価
2. **技術領域の競争**: 同一IPCに集中する出願人群を特定し、競争の激しい技術領域を明らかに
3. **独自領域**: 特定出願人のみが出願しているIPC（独占的技術領域）の特定
4. **ホワイトスペース**: 主要出願人が未参入のIPC×出願人の空白領域

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_ipc_app,
            constraints="バブルが存在しない空白セルにも注目し、ホワイトスペースの戦略的意味を分析すること。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_ipc_app_prompt, "atlas_ipc_app_insight")

# 6. 構成比マップ
with tab6:
    st.subheader("構成比マップ (Treemap)")
    tree_mode = st.radio("表示モード:", ["IPC階層 (技術分野)", "出願人シェア"], horizontal=True, key="atlas_tree_mode")
    if st.button("ツリーマップを描画", key="atlas_run_treemap"):
        with st.spinner("作成中..."):
            if tree_mode == "IPC階層 (技術分野)":
                df_tree = create_treemap_data(df_filtered, stats_start_year, stats_end_year, mode="ipc")
                if df_tree.empty:
                    st.warning("IPCデータがありません。")
                else:
                    fig = px.treemap(df_tree, path=['Section', 'Class', 'Subclass'], values='count', color='Section', color_discrete_sequence=utils.APOLLO_COLORS)
                    update_fig_layout(fig, 'IPC階層構造マップ', height=700)
                    
                    st.session_state['atlas_fig_tree'] = fig
                    st.session_state['atlas_data_tree'] = df_tree

            elif tree_mode == "出願人シェア":
                df_tree = create_treemap_data(df_filtered, stats_start_year, stats_end_year, mode="applicant")
                if df_tree.empty:
                    st.warning("出願人データがありません。")
                else:
                    fig = px.treemap(df_tree, path=['Root', 'Applicant'], values='count', color='count', color_continuous_scale='Blues', labels={'Applicant': '出願人', 'count': '件数', 'Root': '全体'})
                    update_fig_layout(fig, '出願人シェアマップ', height=700)
                    
                    st.session_state['atlas_fig_tree'] = fig
                    st.session_state['atlas_data_tree'] = df_tree

    # Persistent Display
    if 'atlas_fig_tree' in st.session_state:
        fig = st.session_state['atlas_fig_tree']
        data = st.session_state['atlas_data_tree']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # Snapshot Button
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (Treemap)
        df_snap_safe = data.head(30).copy()
        if 'Applicant' in df_snap_safe.columns:
             df_snap_safe['Applicant'] = df_snap_safe['Applicant'].astype(str).str.slice(0, 50)
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)

        utils.render_snapshot_button(
            title="構成比マップ (Treemap)",
            description="技術分野または出願人のシェア構成を示すツリーマップ。",
            key="atlas_tree_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: 構成比マップ分析 ---
        _meta_tree = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        _meta_tree['表示モード'] = tree_mode
        _tree_prompt = utils_ai.generate_ai_insight_prompt(
            role="特許分析・ポートフォリオ戦略の専門家として、構成比マップ（ツリーマップ）を分析してください。",
            context=f"ツリーマップで{'技術分野（IPC階層）' if 'IPC' in tree_mode else '出願人'}の構成比を可視化しています。面積が大きいほど件数が多いことを示します。",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **構成比分析**: 最大領域とそのシェア、上位3-5項目の累積シェアを算出
2. **集中度**: 少数項目への集中か広範な分散かを評価（ロングテール構造の有無）
3. **階層パターン**: IPC階層の場合、特定セクション/クラスへの偏りを評価
4. **ニッチ領域**: 小面積だが存在する領域の戦略的意味（差別化機会）

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_tree,
            constraints="面積比から直感的に読み取れる情報と、データから算出すべき正確な比率を区別すること。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_tree_prompt, "atlas_treemap_insight")

# 7. ライフサイクルマップ
with tab7:
    st.subheader("技術ライフサイクルマップ")
    st.info("""
    **技術の発展段階（ライフサイクル）を診断します。**
    - 縦軸: 出願人数（参入企業の多さ＝競争の激しさ）
    - 横軸: 出願件数（技術活動の活発さ）
    - プロット: 出願年ごとの生データを曲線で近似して繋いでいます。
    """)
    
    if st.button("ライフサイクルマップを描画", key="atlas_run_lifecycle"):
        with st.spinner("計算中..."):
            df_lc = df_filtered.copy()
            df_lc_applicants = df_lc.explode('applicant_main')
            df_lc_applicants['applicant_main'] = df_lc_applicants['applicant_main'].str.strip()
            df_lc_applicants = df_lc_applicants[df_lc_applicants['applicant_main'] != '']
            
            apps_count = df_lc.groupby('year').size()
            inventors_count = df_lc_applicants.groupby('year')['applicant_main'].nunique()
            
            lifecycle_data = pd.DataFrame({
                'year': apps_count.index,
                'applications': apps_count.values,
                'applicants': inventors_count.reindex(apps_count.index, fill_value=0).values
            })
            
            if lifecycle_data.empty or len(lifecycle_data) < 2:
                st.warning("データ不足のためマップを描画できません（期間を広げてください）。")
            else:
                lifecycle_data['year_label'] = lifecycle_data['year'].apply(lambda y: f"'{str(int(y))[-2:]}")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=lifecycle_data['applications'],
                    y=lifecycle_data['applicants'],
                    mode='lines',
                    line=dict(shape='spline', smoothing=1.3, width=3, color='#aaaaaa'),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=lifecycle_data['applications'],
                    y=lifecycle_data['applicants'],
                    mode='markers+text',
                    text=lifecycle_data['year_label'],
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=lifecycle_data['year'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="出願年")
                    ),
                    showlegend=False,
                    hovertemplate="<b>%{text}</b><br>件数: %{x}<br>人数: %{y}<extra></extra>"
                ))
                
                update_fig_layout(fig, '技術ライフサイクル (出願人数 vs 出願件数)', height=700)
                
                fig.update_layout(
                    xaxis_title="出願件数 (技術活動量)",
                    yaxis_title="出願人数 (参入プレイヤー数)"
                )
                
                st.session_state['atlas_fig_life'] = fig
                st.session_state['atlas_data_life'] = lifecycle_data

    # Persistent Display
    if 'atlas_fig_life' in st.session_state:
        fig = st.session_state['atlas_fig_life']
        data = st.session_state['atlas_data_life']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        st.markdown("""
        ##### 💡 マップの読み方
        * **右上へ伸びる**: 多くのプレイヤーが参入し、出願も増えている「成長期」。
        * **右下へ向かう**: 出願数は多いが、プレイヤーが減っている（淘汰が進んでいる）「成熟期」。
        * **左下へ戻る**: プレイヤーも出願も減っている「衰退期」または「ニッチ化」。
        """)
        
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (Lifecycle)
        df_snap_safe = data.head(30).copy()
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)
        utils.render_snapshot_button(
            title="技術ライフサイクルマップ",
            description="出願件数と出願人数（参入企業数）の相関から、技術の成熟度を診断するマップ。",
            key="atlas_life_snap",
            fig=fig,
            data_summary=snap_data
        )

        # --- AI Insight: ライフサイクルステージ診断 ---
        _meta_life = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
            filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
        # ライフサイクル固有メタデータ
        if len(data) >= 2:
            _latest = data.iloc[-1]
            _prev = data.iloc[-2]
            _meta_life['最新年データ'] = f"{int(_latest['year'])}年: 出願{int(_latest['applications'])}件, 出願人{int(_latest['applicants'])}名"
            _meta_life['前年データ'] = f"{int(_prev['year'])}年: 出願{int(_prev['applications'])}件, 出願人{int(_prev['applicants'])}名"
            _meta_life['出願件数変化率'] = f"{((_latest['applications'] - _prev['applications']) / max(_prev['applications'], 1)) * 100:.1f}%"
            _meta_life['出願人数変化率'] = f"{((_latest['applicants'] - _prev['applicants']) / max(_prev['applicants'], 1)) * 100:.1f}%"
        _meta_life['データポイント数'] = len(data)

        _life_prompt = utils_ai.generate_ai_insight_prompt(
            role="技術経営・イノベーション分析の専門家として、技術ライフサイクルマップを分析してください。",
            context="""\
散布図（スプライン曲線で連結）による技術ライフサイクルマップを表示しています。
- X軸: 出願件数（技術活動量）
- Y軸: 出願人数（参入プレイヤー数）
- 各点: 出願年ごとのデータ（年ラベル付き）
- 軌跡の方向が技術のライフサイクルステージを示します。""",
            data_summary=snap_data.get('chart_data', ''),
            instructions="""\
以下の観点で分析してください:
1. **ライフサイクルステージ判定**: 現在のステージ（萌芽期/成長期/成熟期/衰退期）を判定し根拠を説明
2. **軌跡パターン**: 年次推移の方向性（右上=成長、右下=成熟、左下=衰退）の変化点を特定
3. **参入障壁**: 出願人数の変化から参入・退出の動向を分析
4. **将来予測**: 現在の軌跡が続いた場合の技術分野の将来像
5. **戦略提言**: ステージに応じた研究開発投資・知財戦略の方向性

各主張には必ず具体的な数値を1つ以上含めてください。""",
            metadata=_meta_life,
            constraints="ライフサイクル判定は単一の正解がないため、複数の解釈可能性に言及すること。",
            output_format="Markdown形式。見出し付きの構造化された分析レポート。"
        )
        utils_ai.render_ai_insight_button(_life_prompt, "atlas_lifecycle_insight")