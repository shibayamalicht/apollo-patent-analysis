# ==================================================================
# APOLLO — ATLAS（基本統計）
# 出願推移・出願人ランキング・IPC分布・市場集中度から母集団の全体像を掴む。
# 旧 APOLLO: pages/1_🌍_ATLAS.py の移植（分析ロジック・session_state キーは不変）。
# ==================================================================
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
import apollo_ui

utils.configure_matplotlib_font()

# ==================================================================
# --- 1. 設定・ヘルパー関数（モジュールレベルに維持） ---
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
        return f"{match.group(1).replace(' ', '')}/00" if match else ipc
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
    # 横グリッド線の視認性を確保する。plotly_white 既定の gridcolor=#e6eaf1 は淡すぎて、
    # 棒グラフの上では事実上見えず（50/100 等の線が「無い」ように見える原因）。
    # ややダークな点線にし、above traces で棒の上にも描画して全目盛りで線が見えるようにする。
    fig.update_yaxes(gridcolor="#c4c4c4", griddash="dot", layer="above traces")
    return fig


# ==================================================================
# --- 2. ビュー本体 ---
# ==================================================================
def render():
    # スナップショットのモジュール名互換のため維持（CAPCOM ZIP スキーマと互換）
    st.session_state['current_page'] = 'ATLAS'

    apollo_ui.page_header(
        "基本統計", "ATLAS",
        "出願推移・出願人構成・分類分布・集中度指標（HHI / Entropy / Gini）・権利化率で母集団の構造を把握します")

    # --- データガード ---
    apollo_ui.require_data()
    try:
        df_main = st.session_state.df_main
        col_map = st.session_state.col_map
        # 表示用の分類コード名（IPC/FI。分析に使っている方に合わせて全タブの表記を切り替える）
        _code_label = utils.classification_code_label(col_map)
        _code_desc = "FI（IPCを細展開した日本独自の特許分類）" if _code_label == 'FI' else "IPC（国際特許分類）"
        required_cols = ['year', 'applicant_main', 'ipc_normalized']
        if not all(col in df_main.columns for col in required_cols):
            _ja_names = {'year': '出願年', 'applicant_main': '出願人', 'ipc_normalized': f'{_code_label}分類'}
            missing = [_ja_names.get(col, col) for col in required_cols if col not in df_main.columns]
            st.error(f"必要な情報（{'・'.join(missing)}）が見つかりません。「Mission Control」で前処理を再実行してください。")
            _pages = st.session_state.get("ap_pages", {})
            if "mission_control" in _pages:
                st.page_link(_pages["mission_control"], label="データ準備を開く", icon=":material/anchor:")
            st.stop()
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        st.stop()

    # ==================================================================
    # --- 3. 共通フィルタ ---
    # ==================================================================
    st.subheader("共通フィルタ設定")
    min_year = int(df_main['year'].min())
    max_year = int(df_main['year'].max())

    col1, col2 = st.columns(2)
    with col1:
        stats_start_year = st.number_input('集計開始年:', min_value=min_year, max_value=max_year, value=min_year, key="atlas_start_year",
            help="統計・トレンドを集計する対象期間（出願年）です。")
    with col2:
        stats_end_year = st.number_input('集計終了年:', min_value=min_year, max_value=max_year, value=max_year, key="atlas_end_year",
            help="統計・トレンドを集計する対象期間（出願年）です。")

    try:
        df_filtered = df_main[
            (df_main['year'] >= int(stats_start_year)) &
            (df_main['year'] <= int(stats_end_year))
        ].copy()
        st.success(f"集計対象: {int(stats_start_year)}年～{int(stats_end_year)}年 (全 {len(df_filtered)} 件)")
    except Exception as e:
        st.error(f"期間フィルタの適用に失敗しました: {e}")
        df_filtered = pd.DataFrame()

    # CAPCOM data/ JSON出力（基本統計）
    try:
        import capcom
        if capcom.is_active() and not df_filtered.empty:
            _yearly = df_filtered['year'].value_counts().sort_index()
            _trend_dict = {str(int(k) if pd.notna(k) else 0): (int(v) if pd.notna(v) else 0) for k, v in _yearly.items()}

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
                # 多様性指標は全出願人から計算（上位N社に絞ると集中度を過大評価／不平等度を過小評価するため）
                _div_result = patiroha.calculate_diversity(_app_counts_ser.values.tolist())

            # IPC分布
            _ipc_ranking = {}
            ipc_col = utils.classification_code_column(col_map)
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
                    "unique_applicants": int(_app_counts_ser.size) if _app_ranking else 0
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
                # 欠損ステータスも「(ステータス不明)」として集計（落とすと合計が総件数と合わなくなる）
                _status_series = df_filtered[_status_col].fillna(utils.STATUS_UNKNOWN_LABEL)
                _status_counts = _status_series.value_counts()
                atlas_json["status_distribution"] = {str(k): int(v) for k, v in _status_counts.items()}
                _status_by_year = df_filtered.assign(**{_status_col: _status_series}).groupby(['year', _status_col]).size().unstack(fill_value=0)
                atlas_json["status_by_year"] = {
                    str(int(year) if pd.notna(year) else 0): {str(col): (int(val) if pd.notna(val) else 0) for col, val in row.items()}
                    for year, row in _status_by_year.iterrows()
                }

            capcom.save_data("atlas_statistics.json", atlas_json)
    except Exception as e:
        st.caption(f":material/warning: WARN 基本統計 の CAPCOM 保存に失敗しました（要確認）: {e}")

    st.markdown("---")

    tab1, tab1_line, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "件数推移",
        "件数推移（折れ線）",
        "出願人ランキング",
        f"{_code_label}ランキング",
        "出願人×年 バブル",
        f"{_code_label}×出願人 バブル",
        "構成比マップ (Treemap)",
        "ライフサイクルマップ",
        "権利化率マップ"
    ])

    # --- ステータスの配色設定 (公式パレット — CAPCOMレポートと統一) ---
    # ステータスの意味からセマンティックに色を割り当てる。
    # 全タブで色が統一されるように、ステータスごとの色を固定する。
    # ステータス色はパステル配色（utils.build_status_color_map に集約して全タブ統一）
    status_color_map = {}
    status_col = st.session_state.col_map.get('status')
    if status_col:
        unique_statuses_all = sorted(df_filtered[status_col].dropna().unique().astype(str))
        # ステータス欠損行（例: J-PlatPatの1997年より前の案件）も内訳に含めるため専用ラベルを追加
        if df_filtered[status_col].isna().any():
            unique_statuses_all.append(utils.STATUS_UNKNOWN_LABEL)
        status_color_map = utils.build_status_color_map(unique_statuses_all)

    # 1. 件数推移
    with tab1:
        st.subheader("出願件数時系列推移")

        # ステータス内訳オプション
        use_status_breakdown = False
        status_col = st.session_state.col_map.get('status')
        if status_col:
            use_status_breakdown = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab1",
                help="ONにすると棒を権利化状況（登録・拒絶・係属など）で色分けし、各年の権利状況の内訳が分かります。OFFは合計件数のみを表示します。")

        if st.button("件数推移グラフを描画", key="atlas_run_map1", type="primary"):
            if df_filtered.empty:
                st.warning("データがありません。")
            else:
                if use_status_breakdown and status_col:
                    # ステータス別の積み上げ棒グラフ。欠損は「(ステータス不明)」として集計する
                    # （groupby は NaN を落とすため、そのままだとステータスの無い年の棒ごと消える）
                    plot_data = (df_filtered
                                 .assign(**{status_col: df_filtered[status_col].fillna(utils.STATUS_UNKNOWN_LABEL)})
                                 .groupby(['year', status_col]).size().reset_index(name='count'))
                    # 配色は color_discrete_map で全タブ統一
                    fig = px.bar(plot_data, x='year', y='count', color=status_col, labels={'year': '出願年', 'count': '出願件数', status_col: 'ステータス'},
                                 color_discrete_map=status_color_map,
                                 category_orders={status_col: sorted(status_color_map.keys())} # 凡例の順序を統一
                                )
                else:
                    # 標準の棒グラフ
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

            # クリック → 該当特許群ポップアップ対応
            # 注: fig は後続タブの処理で別の図に再代入されるため、既定引数で定義時の値を束縛する
            # （束縛しないと、クリックによる部分再実行時にクロージャが最後のタブの図を描いてしまう）
            @st.fragment
            def _atlas_click_trend(fig=fig):
                utils.disable_selection_fade(fig)
                _trend_selection = st.plotly_chart(
                    fig, use_container_width=True, config={'editable': False},
                    on_select="rerun", selection_mode="points", key="atlas_chart_trend"
                )

                # クリック点（出願年・ステータス）から該当特許インデックスを解決
                def _resolve_trend(point):
                    # 出願年を取得（x優先、customdataにフォールバック）
                    _year_raw = point.get('x')
                    if _year_raw is None:
                        _cd = point.get('customdata')
                        if isinstance(_cd, (list, tuple)) and _cd:
                            _year_raw = _cd[0]
                    try:
                        _year = int(float(_year_raw))
                    except (TypeError, ValueError):
                        return None, None
                    _mask = (df_filtered['year'] == _year)
                    _title = f"{_year}年の出願"
                    # ステータス積み上げ表示時は凡例（legendgroup）でステータスも絞る
                    _st_col = st.session_state.col_map.get('status')
                    _status = point.get('legendgroup')
                    if not _status:
                        # px.bar の積み上げでは curve_number 経由でも取得しうるが、legendgroup を優先
                        _status = None
                    if _st_col and _st_col in df_filtered.columns and _status:
                        if str(_status) == utils.STATUS_UNKNOWN_LABEL:
                            _mask = _mask & df_filtered[_st_col].isna()
                        else:
                            _mask = _mask & (df_filtered[_st_col].astype(str) == str(_status))
                        _title = f"{_year}年・{_status} の出願"
                    _idx = df_filtered[_mask].index.tolist()
                    return _idx, f"{_title}（{len(_idx)} 件）"

                utils.handle_chart_click_to_records(_trend_selection, "atlas_chart_trend", _resolve_trend)
            _atlas_click_trend()

            # 読み方カード（出願推移）
            apollo_ui.howto(
                "この分野の出願活動が年ごとにどう変わってきたかを見る図です。"
                "<b>右肩上がり</b>なら投資が拡大する成長局面、<b>頭打ち・減少</b>なら成熟や淘汰の局面と読みます。"
                "ただし<b>直近1〜2年は未公開の出願（公開は出願から約1年半後）が集計に含まれず、実際より少なく見えます</b>。"
                "最新年の落ち込みだけで衰退と判断しないでください。棒をクリックすると該当年の特許一覧を確認できます。"
            )

            # スナップショットボタン
            snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
            snap_data['module'] = 'ATLAS'

            # チャートデータの整形（ワイド形式: 年 | 合計のみ）
            if hasattr(plot_data, 'columns') and 'year' in plot_data.columns and 'count' in plot_data.columns:
                 # 積み上げ棒: ステータスを無視して年ごとに合計
                 df_snap_safe = plot_data.groupby('year')['count'].sum().reset_index()
                 df_snap_safe['year'] = df_snap_safe['year'].astype(int)
            elif hasattr(plot_data, 'reset_index'):
                 # Series（標準棒グラフ）の場合 → DataFrame 化
                 df_snap_safe = plot_data.reset_index()
                 if df_snap_safe.shape[1] == 2:
                     df_snap_safe.columns = ['year', 'count']
                 # year を可能なら int 化
                 if 'year' in df_snap_safe.columns:
                     df_snap_safe['year'] = df_snap_safe['year'].astype(int)
            else:
                df_snap_safe = pd.DataFrame(plot_data)

            # トークン上限に配慮しつつ全期間の表示を優先
            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe.head(50), index=False)
            utils.render_snapshot_button(
                title=f"出願件数推移 ({int(stats_start_year)}-{int(stats_end_year)})",
                description="市場全体の出願動向を示すトレンドグラフ。",
                key="atlas_trend_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

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

            # ステータス内訳ON時は、ステータス×年の内訳と質の分析指示をAIインサイトに渡す
            _st_ex_t, _st_inst_t = utils_ai.status_insight_addon(
                use_status_breakdown, df_filtered, status_col, 'year', '出願年')
            _ctx_t = "棒グラフによる出願件数の年次推移を表示しています。各年の出願件数から、技術分野の成長・成熟・衰退のステージを判定します。"
            if _st_ex_t:
                _ctx_t += "さらにステータス内訳（権利状況の積み上げ）も表示しており、各年の権利状況構成は下記[ステータス内訳]に対応します。"
            _trend_prompt = utils_ai.generate_ai_insight_prompt(
                role="特許分析の専門家として、出願件数の時系列推移データを分析してください。",
                context=_ctx_t,
                data_summary=snap_data.get('chart_data', ''),
                instructions="""\
以下の観点で分析してください:
1. **全体トレンド**: 出願件数の推移パターン（成長/停滞/衰退）を特定し、ライフサイクルステージを判定
2. **変曲点**: 出願件数が大きく変化した年とその背景仮説
3. **成長率分析**: CAGRの評価と今後3-5年の予測トレンド
4. **市場示唆**: 出願動向から読み取れる市場・技術動向への示唆

各主張には必ず具体的な数値を1つ以上含めてください。""" + _st_inst_t,
                metadata=_meta_trend,
                constraints="データに基づく客観的分析を行い、推測は明確に区別すること。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。",
                extra_content=_st_ex_t
            )
            utils_ai.render_ai_insight_button(_trend_prompt, "atlas_trend_insight")

    # 1.5 件数推移（折れ線）
    with tab1_line:
        st.subheader("件数推移 (折れ線グラフ)")

        col_line_1, col_line_2 = st.columns([2, 1])

        with col_line_1:
            # モード選択
            line_mode = st.radio("表示モード:", ["全体推移", "出願人比較"], horizontal=True, key="atlas_line_mode",
                help="折れ線で何を比べるかを決めます。「全体推移」は市場全体の年次推移を1本で、「出願人比較」は選んだ出願人ごとに線を分けて出願戦略の違いを比較します。")

        with col_line_2:
            # ステータス内訳オプション (全体推移モードのみ)
            use_status_breakdown_line = False
            if line_mode == "全体推移" and status_col:
                use_status_breakdown_line = st.checkbox("ステータス内訳を表示", key="atlas_use_status_line",
                    help="ONにすると権利化状況（登録・拒絶・係属など）ごとに積み上げた面グラフになり、各年の権利状況の内訳が分かります。OFFは合計件数の折れ線のみを表示します。")

        target_applicants = []

        if line_mode == "出願人比較":
            # 出願人リストを件数付きで準備
            if not df_filtered.empty:
                # explode して件数を集計
                assignees_exploded_line = df_filtered.explode('applicant_main')
                assignees_exploded_line['assignee_parsed'] = assignees_exploded_line['applicant_main'].str.strip()

                # 出願人ごとの件数
                app_counts = assignees_exploded_line['assignee_parsed'].value_counts()

                # 「名前 (件数)」形式の選択肢を作成（value_counts() は降順）
                app_options = [f"{name} ({count})" for name, count in app_counts.items()]
                app_map = {f"{name} ({count})": name for name, count in app_counts.items()}

                selected_options = st.multiselect(
                    "出願人を選択 (最大5社):",
                    options=app_options,
                    max_selections=5,
                    key="atlas_line_applicants"
                )

                # 表示名 → 元の出願人名に戻す
                target_applicants = [app_map[opt] for opt in selected_options]

        if st.button("折れ線グラフを描画", key="atlas_run_map1_line", type="primary"):
            if df_filtered.empty:
                st.warning("データがありません。")
            else:
                fig = None
                plot_data = None

                if line_mode == "全体推移":
                    if use_status_breakdown_line and status_col:
                         # 積み上げ面グラフ (内訳)。欠損は「(ステータス不明)」として集計（NaN落ち防止）
                        plot_data = (df_filtered
                                     .assign(**{status_col: df_filtered[status_col].fillna(utils.STATUS_UNKNOWN_LABEL)})
                                     .groupby(['year', status_col]).size().reset_index(name='count'))

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
                        # 選択された出願人でフィルタ
                        assignees_exploded_line = df_filtered.explode('applicant_main')
                        assignees_exploded_line['assignee_parsed'] = assignees_exploded_line['applicant_main'].str.strip()

                        df_target = assignees_exploded_line[assignees_exploded_line['assignee_parsed'].isin(target_applicants)]

                        if df_target.empty:
                            st.warning("選ばれた出願人のデータが期間内にありません。")
                        else:
                            # 各出願人について全年を補完（0埋め）
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

                    # セッションステートの初期化チェック
                    if 'atlas_fig_trend_line' not in st.session_state:
                         st.session_state['atlas_fig_trend_line'] = None

                    st.session_state['atlas_fig_trend_line'] = fig
                    st.session_state['atlas_data_trend_line'] = plot_data

        # 永続表示
        if 'atlas_fig_trend_line' in st.session_state and st.session_state['atlas_fig_trend_line'] is not None:
            fig = st.session_state['atlas_fig_trend_line']
            data = st.session_state['atlas_data_trend_line']

            st.plotly_chart(fig, use_container_width=True, config={'editable': False})
            # スナップショットボタン
            snap_data = utils.generate_rich_summary(df_filtered if 'df_target' not in locals() else df_target, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
            snap_data['module'] = 'ATLAS'
            # チャートデータの整形（出願人比較はワイド形式）
            if data is not None and not data.empty:
                 if 'assignee_parsed' in data.columns:
                     # ピボット: 年 | 出願人A | 出願人B ...
                     df_pivot = data.pivot(index='year', columns='assignee_parsed', values='count').fillna(0).astype(int).reset_index()
                     snap_data['chart_data'] = utils_ai.df_to_markdown(df_pivot.head(40), index=False)
                 else:
                     snap_data['chart_data'] = utils_ai.df_to_markdown(data.head(40), index=False)
            else:
                 snap_data['chart_data'] = "No Data"
            utils.render_snapshot_button(
                title="件数推移 (折れ線)",
                description="出願件数の時系列推移（折れ線グラフ）。全体または特定出願人の比較。",
                key="atlas_trend_line_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: 折れ線トレンド分析 ---
            _meta_line = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
                filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
            _meta_line['表示モード'] = line_mode
            # ステータス内訳ON時（全体推移モード）は内訳と質の分析指示を渡す
            _st_ex_l, _st_inst_l = utils_ai.status_insight_addon(
                use_status_breakdown_line, df_filtered, status_col, 'year', '出願年')
            _ctx_l = "折れ線グラフによる出願件数の年次推移を表示しています。全体推移モードでは市場全体の動向を、出願人比較モードでは主要プレイヤーの出願戦略の変遷を可視化しています。"
            if _st_ex_l:
                _ctx_l += "全体推移モードではステータス内訳（権利状況の積み上げ面）も表示しており、各年の権利状況構成は下記[ステータス内訳]に対応します。"
            _line_prompt = utils_ai.generate_ai_insight_prompt(
                role="特許分析の専門家として、出願件数の時系列推移（折れ線グラフ）を分析してください。",
                context=_ctx_l,
                data_summary=snap_data.get('chart_data', ''),
                instructions="""\
以下の観点で分析してください:
1. **推移パターン**: 各出願人/全体の推移パターン（成長加速/減速/逆転）を特定
2. **競争動態**: 出願人間の順位変動やシェア逆転があった年を特定
3. **相関分析**: 複数出願人の推移に同期的な動きがあるかを評価
4. **予測**: 直近の傾向から今後2-3年の方向性を予測

各主張には必ず具体的な数値を1つ以上含めてください。""" + _st_inst_l,
                metadata=_meta_line,
                constraints="折れ線グラフの傾きの変化に注目し、変曲点を見逃さないこと。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。",
                extra_content=_st_ex_l
            )
            utils_ai.render_ai_insight_button(_line_prompt, "atlas_trend_line_insight")

    # 2. 出願人ランキング
    with tab2:
        st.subheader("出願人ランキング")
        col2_1, col2_2 = st.columns([2, 1])
        with col2_1:
             num_to_display_map2 = st.number_input("表示人数:", min_value=1, value=20, key="atlas_num_apps_map2",
                help="ランキングで上位何件まで表示するかです（表示数のみ変わり、集中度などの分析結果は変わりません）。")

        # ステータス内訳オプション
        use_status_breakdown_tab2 = False
        status_col = st.session_state.col_map.get('status')
        with col2_2:
            if status_col:
                use_status_breakdown_tab2 = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab2",
                    help="ONにすると各出願人の棒を権利化状況（登録・拒絶・係属など）で色分けし、出願人ごとの権利状況の内訳が分かります。OFFは合計件数のみを表示します。")

        if st.button("出願人ランキングを描画", key="atlas_run_map2", type="primary"):
            if df_filtered.empty:
                st.warning("データがありません。")
            else:
                # 1. 上位出願人の特定 (合計件数に基づく)
                _assignee_counts_full = df_filtered['applicant_main'].explode().str.strip().value_counts()  # 全量（多様性指標用）
                assignee_counts = _assignee_counts_full.head(int(num_to_display_map2)).sort_values(ascending=True)  # 表示用（上位N社）
                top_applicants = assignee_counts.index.tolist()

                if use_status_breakdown_tab2 and status_col:
                    # 上位出願人のステータス別積み上げ棒グラフ
                    df_exploded = df_filtered.explode('applicant_main')
                    df_exploded['applicant_parsed'] = df_exploded['applicant_main'].str.strip()
                    df_top = df_exploded[df_exploded['applicant_parsed'].isin(top_applicants)]

                    # 欠損ステータスは「(ステータス不明)」として集計（NaN落ちで件数が欠けるのを防ぐ）
                    plot_data = (df_top
                                 .assign(**{status_col: df_top[status_col].fillna(utils.STATUS_UNKNOWN_LABEL)})
                                 .groupby(['applicant_parsed', status_col]).size().reset_index(name='count'))

                    # 並び順は合計件数に合わせる
                    fig = px.bar(plot_data, x='count', y='applicant_parsed', color=status_col, orientation='h',
                                 labels={'count': '特許件数', 'applicant_parsed': '出願人', status_col: 'ステータス'},
                                 color_discrete_map=status_color_map,
                                 category_orders={'applicant_parsed': top_applicants[::-1], status_col: sorted(status_color_map.keys())})
                else:
                    # 標準の棒グラフ
                    fig = px.bar(x=assignee_counts.values, y=assignee_counts.index, orientation='h', labels={'x': '特許件数', 'y': '出願人'}, color_discrete_sequence=[utils.APOLLO_COLORS[1]])

                update_fig_layout(fig, f'出願人ランキング ({int(stats_start_year)}年～{int(stats_end_year)}年)', height=max(600, len(assignee_counts)*30))

                st.session_state['atlas_fig_ranking'] = fig
                st.session_state['atlas_data_ranking'] = assignee_counts
                st.session_state['atlas_data_ranking_full'] = _assignee_counts_full  # 全量（多様性指標用）

        # 永続表示
        if 'atlas_fig_ranking' in st.session_state:
            fig = st.session_state['atlas_fig_ranking']
            assignee_counts = st.session_state['atlas_data_ranking']
            _assignee_counts_full = st.session_state.get('atlas_data_ranking_full', assignee_counts)  # 全量（無ければ表示用で代替）

            # クリック → 該当特許群ポップアップ対応
            # 注: fig は後続タブの処理で別の図に再代入されるため、既定引数で定義時の値を束縛する
            # （束縛しないと、クリックによる部分再実行時にクロージャが最後のタブの図を描いてしまう）
            @st.fragment
            def _atlas_click_ranking(fig=fig):
                utils.disable_selection_fade(fig)
                _rank_selection = st.plotly_chart(
                    fig, use_container_width=True, config={'editable': False},
                    on_select="rerun", selection_mode="points", key="atlas_chart_ranking"
                )

                # クリック点（横棒＝出願人名）から該当特許インデックスを解決
                def _resolve_ranking(point):
                    # 出願人名は横棒の y。customdata を優先しつつ y にフォールバック
                    _appname = None
                    _cd = point.get('customdata')
                    if isinstance(_cd, (list, tuple)) and _cd and isinstance(_cd[0], str):
                        _appname = _cd[0]
                    if not _appname:
                        _appname = point.get('y')
                    if not isinstance(_appname, str):
                        return None, None
                    _appname = _appname.strip()
                    # applicant_main はリスト列（正規化済み）。strip 後一致で判定
                    def _has_app(lst):
                        if not isinstance(lst, list):
                            return False
                        return any(isinstance(a, str) and a.strip() == _appname for a in lst)
                    _mask = df_filtered['applicant_main'].apply(_has_app)
                    _title = _appname
                    # ステータス積み上げ表示時は凡例（legendgroup）でステータスも絞る
                    _st_col = st.session_state.col_map.get('status')
                    _status = point.get('legendgroup')
                    if _st_col and _st_col in df_filtered.columns and _status:
                        if str(_status) == utils.STATUS_UNKNOWN_LABEL:
                            _mask = _mask & df_filtered[_st_col].isna()
                        else:
                            _mask = _mask & (df_filtered[_st_col].astype(str) == str(_status))
                        _title = f"{_appname}・{_status}"
                    _idx = df_filtered[_mask].index.tolist()
                    return _idx, f"{_title}（{len(_idx)} 件）"

                utils.handle_chart_click_to_records(_rank_selection, "atlas_chart_ranking", _resolve_ranking)
            _atlas_click_ranking()

            # 読み方カード（出願人ランキング）
            apollo_ui.howto(
                "どの企業・組織がこの分野を主導しているかを見る図です。"
                "<b>上位数社への集中</b>が強ければ寡占的な市場、<b>なだらかな分布</b>なら多数のプレイヤーが競う市場と読めます。"
                "<b>出願件数の多さは特許の質や事業の強さと必ずしも一致しない</b>点に注意してください。"
                "棒をクリックすると該当出願人の特許一覧を確認できます。"
            )

            # 多様性指標メトリクス（全市場参加者＝全出願人から計算。上位N社に絞ると集中度を過大評価するため）
            _applicant_counts_for_div = _assignee_counts_full.values.tolist()
            if _applicant_counts_for_div:
                div = patiroha.calculate_diversity(_applicant_counts_for_div)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("市場集中度（HHI）", f"{div.hhi:.3f}", help=div.hhi_status)
                with c2:
                    st.metric("多様性（Entropy）", f"{div.entropy:.2f}", help="高い=分散、低い=集中")
                with c3:
                    st.metric("不平等度（Gini）", f"{div.gini:.3f}", help="0=完全平等、1=完全不平等")

                # 読み方カード（多様性指標）
                apollo_ui.howto(
                    "3つとも「出願人の顔ぶれの偏り」を別の角度から測る指標で、表示中の上位だけでなく<b>全出願人から計算</b>しています。"
                    "<b>市場集中度（HHI）</b>は高いほど少数の企業への集中が強く、<b>多様性（Entropy）</b>は高いほど幅広い顔ぶれに分散、"
                    "<b>不平等度（Gini）</b>は1に近いほど出願人間の件数格差が大きいことを示します。"
                    "<b>表示人数を変えても指標の値は変わりません</b>（表示は上位のみ、計算は全体のため）。",
                    title="3つの指標の見方",
                )

            # スナップショットボタン
            snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
            snap_data['module'] = 'ATLAS'

            # チャートデータの整形
            df_snap_safe = assignee_counts.head(30).reset_index()
            df_snap_safe.columns = ['Applicant', 'Count']
            df_snap_safe['Applicant'] = df_snap_safe['Applicant'].astype(str).str.slice(0, 50)
            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe, index=False)
            utils.render_snapshot_button(
                title=f"主要出願人ランキング ({int(stats_start_year)}-{int(stats_end_year)})",
                description="特許出願件数に基づく市場の主要プレイヤーランキング。",
                key="atlas_applicant_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: 競争環境・集中度分析 ---
            _meta_rank = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
                filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
            # 多様性指標計算（HHI / Entropy / Gini）
            _total_patents = assignee_counts.sum()
            if _total_patents > 0:
                _div_rank = patiroha.calculate_diversity(_assignee_counts_full.values.tolist())  # 全出願人から
                _shares = assignee_counts / _total_patents
                _meta_rank['HHI(市場集中度)'] = f"{_div_rank.hhi:.4f} ({_div_rank.hhi_status})"
                _meta_rank['Entropy(多様性)'] = f"{_div_rank.entropy:.2f}"
                _meta_rank['Gini(不平等度)'] = f"{_div_rank.gini:.3f}"
                _meta_rank['上位出願人数'] = len(assignee_counts)
                _meta_rank['上位出願人の合計件数'] = int(_total_patents)
                _meta_rank['1位シェア(%)'] = f"{_shares.iloc[-1]*100:.1f}%" if len(_shares) > 0 else "N/A"
            _meta_rank['全出願人数'] = int(df_filtered['applicant_main'].explode().str.strip().nunique())

            # ステータス内訳ON時は、出願人×ステータスの内訳を渡し権利化の質を分析させる
            _st_ex_r, _st_inst_r = "", ""
            if use_status_breakdown_tab2 and status_col:
                _df_exp_r = df_filtered.explode('applicant_main').copy()
                _df_exp_r['applicant_parsed'] = _df_exp_r['applicant_main'].astype(str).str.strip()
                _st_ex_r, _st_inst_r = utils_ai.status_insight_addon(
                    True, _df_exp_r, status_col, 'applicant_parsed', '出願人',
                    top_axis=int(num_to_display_map2))
            _ctx_r = "水平棒グラフによる出願人ランキングを表示しています。出願件数の多い順に主要プレイヤーを表示し、市場の競争構造を可視化しています。"
            if _st_ex_r:
                _ctx_r += "さらに出願人ごとのステータス内訳（権利状況の積み上げ）も表示しており、各社の権利状況構成は下記[ステータス内訳]に対応します。"
            _rank_prompt = utils_ai.generate_ai_insight_prompt(
                role="特許分析・競争戦略の専門家として、出願人ランキングデータから競争環境を分析してください。",
                context=_ctx_r,
                data_summary=snap_data.get('chart_data', ''),
                instructions="""\
以下の観点で分析してください:
1. **市場集中度**: HHI指数に基づく市場構造の評価（独占/寡占/競争的）
2. **多様性・不平等度**: Entropy（情報エントロピー）とGini係数から見た出願人分布の特徴
3. **プレイヤー分類**: リーダー/チャレンジャー/フォロワー/ニッチャーに分類
4. **競争パターン**: 上位と下位の件数格差、参入障壁の高さの推定
5. **戦略的示唆**: 新規参入者やポートフォリオ戦略へのインプリケーション

各主張には必ず具体的な数値を1つ以上含めてください。""" + _st_inst_r,
                metadata=_meta_rank,
                constraints="出願件数は必ずしも特許の質や事業規模と一致しない点に注意すること。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。",
                extra_content=_st_ex_r
            )
            utils_ai.render_ai_insight_button(_rank_prompt, "atlas_ranking_insight")

    # 3. 分類コード（IPC/FI）ランキング
    with tab3:
        st.subheader(f"{_code_label}ランキング")
        ipc_level_map3 = st.selectbox(f"{_code_label}レベル:", [(1, "サブクラス (A01B)"), (2, "メイングループ (A01B 1/00)")], format_func=lambda x: x[1], key="atlas_ipc_level_map3",
            help=f"技術分野（{_code_label}）をどの細かさで集計するかです。「サブクラス」は大まかな技術領域、「メイングループ」はより細かい区分で見られます（細かいほど分類数が増え、1分類あたりの件数は小さくなります）。")
        num_to_display_map3 = st.number_input(f"表示{_code_label}数:", min_value=1, value=20, key="atlas_num_ipcs_map3",
            help="ランキングで上位何件まで表示するかです（表示数のみ変わり、分析結果は変わりません）。")
        if st.button(f"{_code_label}ランキングを描画", key="atlas_run_map3", type="primary"):
            if df_filtered.empty:
                st.warning("データがありません。")
            else:
                ipc_exploded = df_filtered['ipc_normalized'].explode().dropna()
                ipc_parsed = ipc_exploded.apply(lambda x: parse_ipc_atlas(x, ipc_level_map3[0]))
                ipc_counts = ipc_parsed.value_counts().head(int(num_to_display_map3)).sort_values(ascending=True)
                fig = px.bar(x=ipc_counts.values, y=ipc_counts.index, orientation='h', labels={'x': '特許件数', 'y': f'{_code_label}分類'}, color_discrete_sequence=[utils.APOLLO_COLORS[2]])
                update_fig_layout(fig, f'{_code_label}ランキング ({ipc_level_map3[1]})', height=max(600, len(ipc_counts)*30))

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

            # チャートデータの整形
            df_snap_safe = data.head(30).reset_index()
            df_snap_safe.columns = [_code_label, 'Count']
            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe, index=False)
            utils.render_snapshot_button(
                title=f"{_code_label}ランキング ({ipc_level_map3[1]})",
                description=f"技術分野 ({_code_label}) 別の上位ランキング。",
                key="atlas_ipc_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: 分類コード技術分野構成分析 ---
            _meta_ipc = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
                filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
            _meta_ipc[f'{_code_label}分類数'] = len(data)
            _meta_ipc['表示レベル'] = ipc_level_map3[1]
            _ipc_prompt = utils_ai.generate_ai_insight_prompt(
                role=f"特許分析・技術動向の専門家として、{_code_label}分類ランキングから技術分野の構成を分析してください。",
                context=f"水平棒グラフによる{_code_desc}のランキングを表示しています。出願件数の多い順に技術分野を表示し、研究開発の注力領域を可視化しています。",
                data_summary=snap_data.get('chart_data', ''),
                instructions=f"""\
以下の観点で分析してください:
1. **技術集中度**: 上位{_code_label}分類への集中度合い（上位3-5分類が全体に占める割合）
2. **技術的特徴**: 主要{_code_label}分類が示す技術領域の解釈と相互関係
3. **技術多様性**: {_code_label}分布の偏りから技術ポートフォリオの幅を評価
4. **戦略的示唆**: 特定{_code_label}分類への過度な集中や空白領域の戦略的意味

各主張には必ず具体的な数値を1つ以上含めてください。""",
                metadata=_meta_ipc,
                constraints=f"{_code_label}分類コードの意味を正確に解釈すること。不明な{_code_label}には推測を明記すること。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。"
            )
            utils_ai.render_ai_insight_button(_ipc_prompt, "atlas_ipc_insight")

    # 4. 出願人×年 バブル
    with tab4:
        st.subheader("出願人 × 年 バブルチャート")

        # --- おすすめ実行バー（設定はそのままで押せる） ---
        with st.container(border=True):
            _bar4_l, _bar4_r = st.columns([3, 1])
            with _bar4_l:
                st.markdown("**おすすめ設定で描画できます** — 表示する人数はあとから調整できます")
                st.caption("設定を変えたら、もう一度ボタンを押すと描き直します。")
            with _bar4_r:
                _run_map4 = st.button("出願人×年 バブルを描画", key="atlas_run_map4", type="primary",
                                      use_container_width=True)

        # --- 詳細設定（widget key・既定値は従来のまま） ---
        with st.expander("詳細設定（表示する人数・ステータス内訳）", expanded=False):
            col4_1, col4_2 = st.columns([2, 1])
            with col4_1:
                 num_to_display_map4 = st.number_input("表示人数:", min_value=1, value=10, key="atlas_num_apps_map4",
                    help="バブルに表示する上位出願人を何件までにするかです（表示数のみ変わり、分析結果は変わりません）。")

            # ステータス内訳オプション
            use_status_breakdown_tab4 = False
            status_col = st.session_state.col_map.get('status')
            with col4_2:
                if status_col:
                    use_status_breakdown_tab4 = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab4",
                        help="ONにすると各セルが権利化状況（登録・拒絶・係属など）の円グラフになり、出願人×年ごとの権利状況の内訳が分かります。OFFは件数の大きさを示すバブルのみを表示します。")

        if _run_map4:
            assignees_exploded = df_filtered.explode('applicant_main')
            assignees_exploded['assignee_parsed'] = assignees_exploded['applicant_main'].str.strip()
            top_assignees = assignees_exploded['assignee_parsed'].value_counts().head(int(num_to_display_map4)).index.tolist()

            # 上位出願人で先にフィルタ
            df_target = assignees_exploded[assignees_exploded['assignee_parsed'].isin(top_assignees)].copy()

            if df_target.empty:
                st.warning("データがありません。")
            else:
                if use_status_breakdown_tab4 and status_col:
                    # --- グリッド状パイチャートの描画 ---
                    # 1. グリッド寸法の計算
                    start_y = int(stats_start_year)
                    end_y = int(stats_end_year)

                    # フィルタ期間に揃えて安定化
                    cols = list(range(start_y, end_y + 1))

                    # この期間にデータを絞る
                    df_target = df_target[df_target['year'].isin(cols)]

                    if df_target.empty:
                        st.warning("指定期間内のデータがありません。")
                    else:
                        # 行=出願人、列=年（線形の並び）
                        rows = top_assignees

                        n_rows = len(rows)
                        n_cols = len(cols)

                        fig = go.Figure()

                        # [出願人, 年, ステータス] で集計
                        grid_data = df_target.groupby(['assignee_parsed', 'year', status_col]).size().reset_index(name='count')
                        total_counts = df_target.groupby(['assignee_parsed', 'year']).size().reset_index(name='total')
                        max_total = total_counts['total'].max()

                        # レイアウト計算
                        x_margin_l = 0.20 # ラベル見切れ防止のため 0.20（標準表示と揃える）
                        x_margin_r = 0.02
                        y_margin_b = 0.10
                        y_margin_t = 0.05

                        plot_width = 1.0 - (x_margin_l + x_margin_r)
                        plot_height = 1.0 - (y_margin_b + y_margin_t)

                        cell_w = plot_width / n_cols
                        cell_h = plot_height / n_rows

                        # 凡例色の準備

                        # 表示中に存在するステータスのみ凡例に出す
                        statuses_in_view = sorted(df_target[status_col].dropna().unique().astype(str))

                        # 凡例用ダミートレース（散布マーカー）
                        for status in statuses_in_view:
                            fig.add_trace(go.Scatter(
                                x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=10, color=status_color_map.get(status, '#ccc')),
                                name=status,
                                showlegend=True
                            ))

                        # 軸用アノテーション
                        annotations = []

                        # Y軸ラベル（出願人）
                        for i, applicant in enumerate(rows):
                            y_center = (1.0 - y_margin_t) - (i * cell_h) - (cell_h / 2)

                            annotations.append(dict(
                                x=x_margin_l - 0.01, y=y_center,
                                xref="paper", yref="paper",
                                text="",
                                showarrow=False, xanchor="right", yanchor="middle",
                                font=dict(size=12, color=utils.APOLLO_TEXT)
                            ))

                        # X軸ラベルは layout.xaxis 側で処理
                        annotations = []

                        # パイトレースの追加
                        for i, applicant in enumerate(rows):
                            for j, year in enumerate(cols):
                                cell_df = grid_data[(grid_data['assignee_parsed'] == applicant) & (grid_data['year'] == year)]

                                if not cell_df.empty:
                                    total = cell_df['count'].sum()
                                    max_r = min(cell_w, cell_h) / 2 * 0.9
                                    scale_factor = (total / max_total) ** 0.5
                                    # 視覚サイズは平方根スケーリング
                                    y_center = (1.0 - y_margin_t) - (i * cell_h) - (cell_h / 2)
                                    x_center = x_margin_l + (j * cell_w) + (cell_w / 2)

                                    # ドメイン計算
                                    d_w = cell_w * scale_factor

                                    x0 = x_center - (d_w / 2)
                                    x1 = x_center + (d_w / 2)
                                    y0 = y_center - (scale_factor * cell_h / 2)
                                    y1 = y_center + (scale_factor * cell_h / 2)

                                    # 色を明示的にマッピング
                                    labels = cell_df[status_col].astype(str).tolist()
                                    values = cell_df['count'].tolist()
                                    colors = [status_color_map.get(l, '#ccc') for l in labels]

                                    fig.add_trace(go.Pie(
                                        labels=labels,
                                        values=values,
                                        marker=dict(colors=colors),
                                        domain=dict(x=[x0, x1], y=[y0, y1]),
                                        showlegend=False, # 凡例はダミートレースを使用
                                        textinfo='none',
                                        hoverinfo='label+value',
                                        sort=False
                                    ))

                        # 手動グリッド線は廃止（yaxis.showgrid で処理）
                        shapes = []

                        # レイアウト最終化
                        fig.update_layout(
                            height=max(700, n_rows * 50),
                            annotations=annotations,
                            shapes=shapes,
                            showlegend=True,
                            xaxis=dict(
                                visible=True,
                                domain=[x_margin_l, 1.0 - x_margin_r],
                                # 範囲: [最小年 - 0.5, 最大年 + 0.5]
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
                                # 行 (0..N-1) をY軸へ。上から下の順。
                                range=[-0.5, n_rows - 0.5],
                                tickmode='array',
                                tickvals=list(range(n_rows)),
                                ticktext=rows[::-1], # 反転して1位を最上段に
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

                    # 統一ステートへ保存
                    st.session_state['atlas_fig_bubble_tab4'] = fig

                    # ローカル変数だった grid_data をステート保存用に再作成
                    grid_data_export = df_target.groupby(['year', 'assignee_parsed', status_col]).size().reset_index(name='count')
                    st.session_state['atlas_data_bubble_tab4'] = grid_data_export
                else:
                    # 標準バブルチャート
                    plot_data = df_target.groupby(['year', 'assignee_parsed']).size().reset_index(name='件数')

                    # --- 共有レイアウト定数 ---
                    x_margin_l = 0.20 # 内訳表示と揃える
                    x_margin_r = 0.02
                    y_margin_b = 0.10
                    y_margin_t = 0.05

                    fig = px.scatter(plot_data, x='year', y='assignee_parsed', size='件数', color='assignee_parsed',
                                     labels={'year': '出願年', 'assignee_parsed': '出願人', '件数': '件数'},
                                     color_discrete_sequence=utils.APOLLO_COLORS,
                                     category_orders={"assignee_parsed": top_assignees}) # 並び順は px に委譲

                    # 内訳表示と厳密に揃えたレイアウト
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
                             visible=True # 表示を保証
                         )
                    )

                    # 統一ステートへ保存
                    st.session_state['atlas_fig_bubble_tab4'] = fig
                    st.session_state['atlas_data_bubble_tab4'] = plot_data

        # 永続表示（統一）
        if 'atlas_fig_bubble_tab4' in st.session_state:
            fig = st.session_state['atlas_fig_bubble_tab4']
            data = st.session_state['atlas_data_bubble_tab4']

            st.plotly_chart(fig, use_container_width=True, config={'editable': False})

            # スナップショットボタン
            snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
            snap_data['module'] = 'ATLAS'

            # チャートデータの整形
            if hasattr(data, 'head'):

                 # セッションステートを壊さないようコピー
                 chart_df = data.copy()

                 # カラム名の正規化 (件数 -> count)
                 if '件数' in chart_df.columns:
                     chart_df.rename(columns={'件数': 'count'}, inplace=True)

                 # data は grid_data_export (年, 出願人, ステータス, count) または plot_data (年, 出願人, count)

                 # 必要なカラムのみに絞る
                 target_cols = [c for c in ['year', 'assignee_parsed', 'count', status_col] if c in chart_df.columns]
                 df_snap_safe = chart_df[target_cols].copy()

                 # 整形
                 if 'assignee_parsed' in df_snap_safe.columns:
                     df_snap_safe['assignee_parsed'] = df_snap_safe['assignee_parsed'].astype(str).str.slice(0, 30)

                 # 可読性のためピボット（年 × 出願人、値=合計件数）
                 if 'year' in df_snap_safe.columns and 'assignee_parsed' in df_snap_safe.columns:
                     # バブル位置のみ示す場合はステータスを畳んで集計
                     df_pivot = df_snap_safe.groupby(['year', 'assignee_parsed'])['count'].sum().reset_index()
                     df_pivot = df_pivot.pivot(index='year', columns='assignee_parsed', values='count').fillna(0).astype(int).reset_index()
                     snap_data['chart_data'] = utils_ai.df_to_markdown(df_pivot.head(40), index=False)
                 else:
                     snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe.head(40), index=False)
            else:
                 snap_data['chart_data'] = "Data Summary"

            utils.render_snapshot_button(
                title="出願年別 出願人バブルチャート",
                description="主要出願人の時系列活動量 (内訳含む)",
                key="atlas_bubble_tab4_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: 出願人×年バブル分析 ---
            _meta_bubble = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
                filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
            _meta_bubble['表示出願人数'] = num_to_display_map4
            # ステータス内訳ON時のみ、出願人×ステータスの内訳を渡す（OFF時は内訳に言及しない）
            _st_ex_b, _st_inst_b = "", ""
            if use_status_breakdown_tab4 and status_col:
                _df_exp_b = df_filtered.explode('applicant_main').copy()
                _df_exp_b['applicant_parsed'] = _df_exp_b['applicant_main'].astype(str).str.strip()
                _st_ex_b, _st_inst_b = utils_ai.status_insight_addon(
                    True, _df_exp_b, status_col, 'applicant_parsed', '出願人',
                    top_axis=int(num_to_display_map4))
            _ctx_b = "バブルチャートで主要出願人の年別出願件数を可視化しています。バブルの大きさは件数、位置は年×出願人を示します。"
            if _st_ex_b:
                _ctx_b += "ステータス内訳をONにしており、各セルの権利状況はパイチャートグリッドで表示されています（各社の権利状況構成は下記[ステータス内訳]に対応）。"
            _bubble_prompt = utils_ai.generate_ai_insight_prompt(
                role="特許分析・競争戦略の専門家として、出願人の年別出願活動をバブルチャートから分析してください。",
                context=_ctx_b,
                data_summary=snap_data.get('chart_data', ''),
                instructions="""\
以下の観点で分析してください:
1. **参入・撤退パターン**: 各出願人の出願開始年・活動期間・直近の活動有無から参入・撤退を特定
2. **活動量の変遷**: 各出願人の年別出願件数の増減パターン（加速/減速/一時停止）
3. **競合分析**: 同時期に活動が集中している出願人群の特定と競争構造の推定
4. **戦略転換**: 出願量が急変した出願人とその時期の特定

各主張には必ず具体的な数値を1つ以上含めてください。""" + _st_inst_b,
                metadata=_meta_bubble,
                constraints="バブルサイズの違いに注目し、出願人間の規模感の差を明確にすること。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。",
                extra_content=_st_ex_b
            )
            utils_ai.render_ai_insight_button(_bubble_prompt, "atlas_app_year_insight")

    # 5. IPC×出願人 バブル
    with tab5:
        st.subheader(f"{_code_label} × 出願人 バブルチャート")

        # --- おすすめ実行バー（設定はそのままで押せる） ---
        with st.container(border=True):
            _bar5_l, _bar5_r = st.columns([3, 1])
            with _bar5_l:
                st.markdown("**おすすめ設定で描画できます** — 表示件数や分類の細かさはあとから調整できます")
                st.caption("設定を変えたら、もう一度ボタンを押すと描き直します。")
            with _bar5_r:
                _run_map5 = st.button(f"{_code_label}×出願人 バブルを描画", key="atlas_run_map5", type="primary",
                                      use_container_width=True)

        # --- 詳細設定（widget key・既定値は従来のまま） ---
        with st.expander("詳細設定（分類の細かさ・表示件数）", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1: ipc_level_map5 = st.selectbox(f"{_code_label}レベル:", [(1, "サブクラス"), (2, "メイングループ")], format_func=lambda x: x[1], key="atlas_ipc_level_map5",
                help=f"技術分野（{_code_label}）をどの細かさで集計するかです。「サブクラス」は大まかな技術領域、「メイングループ」はより細かい区分で見られます（細かいほど分類数が増えます）。")
            with col2: num_ipcs_map5 = st.number_input(f"{_code_label}数 (Y軸):", min_value=1, value=15, key="atlas_num_ipcs_map5",
                help=f"クロス集計（出願人×技術分野{_code_label}）で縦軸に上位何件の{_code_label}を取るかです。大きいほど広く見えますが密になります。")
            with col3: num_apps_map5 = st.number_input("出願人数 (X軸):", min_value=1, value=15, key="atlas_num_apps_map5",
                help=f"クロス集計（出願人×技術分野{_code_label}）で横軸に上位何件の出願人を取るかです。大きいほど広く見えますが密になります。")

        if _run_map5:
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
                fig = px.scatter(plot_data, x='assignee_parsed', y='ipc_parsed', size='件数', color='ipc_parsed', labels={'assignee_parsed': '出願人', 'ipc_parsed': f'{_code_label}分類', '件数': '件数'}, color_discrete_sequence=utils.APOLLO_COLORS, category_orders={"ipc_parsed": top_ipcs})
                update_fig_layout(fig, f'IPC ({ipc_level_map5[1]}) × 出願人 ポートフォリオ', height=800)

                st.session_state['atlas_fig_bubble_ipc'] = fig
                st.session_state['atlas_data_bubble_ipc'] = plot_data

        # 永続表示
        if 'atlas_fig_bubble_ipc' in st.session_state:
            fig = st.session_state['atlas_fig_bubble_ipc']
            data = st.session_state['atlas_data_bubble_ipc']

            st.plotly_chart(fig, use_container_width=True, config={'editable': False})

            # スナップショットボタン
            snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
            snap_data['module'] = 'ATLAS'

            # チャートデータの整形（IPCバブル）
            df_snap_safe = data.head(30).copy()
            if 'assignee_parsed' in df_snap_safe.columns:
                 df_snap_safe['assignee_parsed'] = df_snap_safe['assignee_parsed'].astype(str).str.slice(0, 50)
            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe, index=False)

            utils.render_snapshot_button(
                title=f"{_code_label} x 出願人 ポートフォリオ",
                description=f"主要出願人の技術分野（{_code_label}）ごとの注力度合いを示すバブルチャート。",
                key="atlas_bubble_ipc_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: IPC×出願人ポートフォリオ分析 ---
            _meta_ipc_app = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
                filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
            _ipc_app_prompt = utils_ai.generate_ai_insight_prompt(
                role=f"特許分析・技術ポートフォリオの専門家として、{_code_label}×出願人のバブルチャートから技術戦略を分析してください。",
                context=f"バブルチャートで主要出願人(X軸)と技術分野{_code_label}(Y軸)の交差を可視化しています。バブルの大きさは各出願人が各{_code_label}分類に出願した件数を示します。",
                data_summary=snap_data.get('chart_data', ''),
                instructions="""\
以下の観点で分析してください:
1. **専門vs多角**: 各出願人が特定IPCに集中しているか、広範囲に分散しているかを評価
2. **技術領域の競争**: 同一IPCに集中する出願人群を特定し、競争の激しい技術領域を明らかに
3. **独自領域**: 特定出願人のみが出願しているIPC（独占的技術領域）の特定
4. **ホワイトスペース**: 主要出願人が未参入の分類×出願人の空白領域

各主張には必ず具体的な数値を1つ以上含めてください。""",
                metadata=_meta_ipc_app,
                constraints="バブルが存在しない空白セルにも注目し、ホワイトスペースの戦略的意味を分析すること。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。"
            )
            utils_ai.render_ai_insight_button(_ipc_app_prompt, "atlas_ipc_app_insight")

    # 6. 構成比マップ
    with tab6:
        st.subheader("構成比マップ (Treemap)")
        tree_mode = st.radio("表示モード:", [f"{_code_label}階層 (技術分野)", "出願人シェア"], horizontal=True, key="atlas_tree_mode",
            help=f"面積マップで何の構成比を見るかを決めます。「{_code_label}階層」は技術分野ごと、「出願人シェア」は出願人ごとの件数比率を、面積の大きさで表します。")
        if st.button("ツリーマップを描画", key="atlas_run_treemap", type="primary"):
            with st.spinner("作成中..."):
                if tree_mode != "出願人シェア":
                    df_tree = create_treemap_data(df_filtered, stats_start_year, stats_end_year, mode="ipc")
                    if df_tree.empty:
                        st.warning(f"{_code_label}データがありません。")
                    else:
                        fig = px.treemap(df_tree, path=['Section', 'Class', 'Subclass'], values='count', color='Section', color_discrete_sequence=utils.APOLLO_COLORS)
                        update_fig_layout(fig, f'{_code_label}階層構造マップ', height=700)

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

        # 永続表示
        if 'atlas_fig_tree' in st.session_state:
            fig = st.session_state['atlas_fig_tree']
            data = st.session_state['atlas_data_tree']

            st.plotly_chart(fig, use_container_width=True, config={'editable': False})

            # スナップショットボタン
            snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
            snap_data['module'] = 'ATLAS'

            # チャートデータの整形（ツリーマップ）
            df_snap_safe = data.head(30).copy()
            if 'Applicant' in df_snap_safe.columns:
                 df_snap_safe['Applicant'] = df_snap_safe['Applicant'].astype(str).str.slice(0, 50)
            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe, index=False)

            utils.render_snapshot_button(
                title="構成比マップ (Treemap)",
                description="技術分野または出願人のシェア構成を示すツリーマップ。",
                key="atlas_tree_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: 構成比マップ分析 ---
            _meta_tree = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_filtered, col_map=col_map,
                filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
            _meta_tree['表示モード'] = tree_mode
            _tree_prompt = utils_ai.generate_ai_insight_prompt(
                role="特許分析・ポートフォリオ戦略の専門家として、構成比マップ（ツリーマップ）を分析してください。",
                context=f"ツリーマップで{f'技術分野（{_code_label}階層）' if '階層' in tree_mode else '出願人'}の構成比を可視化しています。面積が大きいほど件数が多いことを示します。",
                data_summary=snap_data.get('chart_data', ''),
                instructions="""\
以下の観点で分析してください:
1. **構成比分析**: 最大領域とそのシェア、上位3-5項目の累積シェアを算出
2. **集中度**: 少数項目への集中か広範な分散かを評価（ロングテール構造の有無）
3. **階層パターン**: 分類階層の場合、特定セクション/クラスへの偏りを評価
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

        # --- おすすめ実行バー（このタブに細かい設定はない・そのまま押せる） ---
        with st.container(border=True):
            _bar7_l, _bar7_r = st.columns([3, 1])
            with _bar7_l:
                st.markdown("**おすすめ設定で描画できます** — 集計する期間は上の共通フィルタであとから調整できます")
                st.caption("このタブに細かい設定はありません。期間を変えたら、もう一度ボタンを押すと描き直します。")
            with _bar7_r:
                _run_lifecycle = st.button("ライフサイクルマップを描画", key="atlas_run_lifecycle", type="primary",
                                           use_container_width=True)

        if _run_lifecycle:
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

        # 永続表示
        if 'atlas_fig_life' in st.session_state:
            fig = st.session_state['atlas_fig_life']
            data = st.session_state['atlas_data_life']

            st.plotly_chart(fig, use_container_width=True, config={'editable': False})

            # 読み方カード（旧 st.info 軸説明＋「:material/lightbulb:マップの読み方」を統合・内容維持）
            apollo_ui.howto(
                "技術の発展段階（ライフサイクル）を診断するマップです。"
                "<b>縦軸は出願人数</b>（参入企業の多さ＝競争の激しさ）、<b>横軸は出願件数</b>（技術活動の活発さ）で、"
                "出願年ごとの生データを曲線で近似して繋いでいます。"
                "<b>右上へ伸びる</b>＝多くのプレイヤーが参入し出願も増えている「成長期」、"
                "<b>右下へ向かう</b>＝出願数は多いがプレイヤーが減っている（淘汰が進んでいる）「成熟期」、"
                "<b>左下へ戻る</b>＝プレイヤーも出願も減っている「衰退期」または「ニッチ化」と読みます。",
                title="このマップの読み方",
            )

            snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
            snap_data['module'] = 'ATLAS'

            # チャートデータの整形（ライフサイクル）
            df_snap_safe = data.head(30).copy()
            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe, index=False)
            utils.render_snapshot_button(
                title="技術ライフサイクルマップ",
                description="出願件数と出願人数（参入企業数）の相関から、技術の成熟度を診断するマップ。",
                key="atlas_life_snap",
                fig=fig,
                data_summary=snap_data
            )
            apollo_ui.snapshot_hint()

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


    # 8. 権利化率マップ
    with tab8:
        st.subheader("権利化率マップ（出願数 × 権利化率）")
        _gr_status_col = st.session_state.col_map.get('status')
        if not _gr_status_col or _gr_status_col not in df_filtered.columns:
            st.info("ステータス（法的状態）の列が割り当てられていません。「Mission Control」でステータス列を指定すると、権利化率の分析が可能になります。")
            _pages = st.session_state.get("ap_pages", {})
            if "mission_control" in _pages:
                st.page_link(_pages["mission_control"], label="データ準備を開く", icon=":material/anchor:")
        else:
            # --- おすすめ実行バー（設定はそのままで押せる） ---
            with st.container(border=True):
                _bar8_l, _bar8_r = st.columns([3, 1])
                with _bar8_l:
                    st.markdown("**おすすめ設定で描画できます** — 表示件数や計算のしかたはあとから調整できます")
                    st.caption("設定を変えたら、もう一度ボタンを押すと描き直します。")
                with _bar8_r:
                    _run_grantrate = st.button("権利化率マップを描画", key="atlas_run_grantrate", type="primary",
                                               use_container_width=True)

            # --- ステータスの自動判定の説明（表示のみ・集計挙動は従来と同一） ---
            # 集計（utils.compute_grant_rate_stats）は utils.classify_status のキーワード判定を
            # 以前から使っており、手動割当は不要。ここでは「どの値がどう数えられるか」を
            # 描画前に確認できるように割当結果を見せるだけで、判定ロジックには手を加えない。
            try:
                _gr_cat_names = {
                    'granted': '権利化', 'expired': '失効・放棄（一度は登録）', 'rejected': '拒絶',
                    'withdrawn': '取下', 'appeal': '審判・異議係属中', 'examining': '審査中',
                    'pending': '出願・係属中', 'published': '公開', None: 'その他'}
                _gr_assign = {}
                for _v in sorted(df_filtered[_gr_status_col].dropna().astype(str).unique()):
                    _gr_assign.setdefault(utils.classify_status(_v), []).append(_v)
                _gr_parts = []
                for _cat in ['granted', 'expired', 'rejected', 'withdrawn', 'appeal', 'examining', 'pending', 'published', None]:
                    if _cat in _gr_assign:
                        _vals = _gr_assign[_cat]
                        _shown = "・".join(_vals[:5]) + (f" ほか{len(_vals) - 5}件" if len(_vals) > 5 else "")
                        _gr_parts.append(f"{_gr_cat_names[_cat]}={_shown}")
                if _gr_parts:
                    st.caption("ステータスの割り当ては言葉から自動で見分けます: " + " ／ ".join(_gr_parts))
            except Exception:
                pass

            # --- 詳細設定（widget key・既定値は従来のまま） ---
            with st.expander("詳細設定（集計単位・計算のしかた・表示件数）", expanded=False):
                _gr_c1, _gr_c2, _gr_c3 = st.columns(3)
                with _gr_c1:
                    _gr_unit = st.radio("分析単位:", ["出願人", _code_label], horizontal=True, key="atlas_gr_unit",
                        help=f"権利化率を何ごとに集計するかを決めます。「出願人」は会社・組織ごと、「{_code_label}」は技術分野ごとに、出願数と権利化率を比較します。")
                with _gr_c2:
                    _gr_denom_label = st.radio(
                        "権利化率の定義:", ["登録 / 全件", "登録 / 処分確定"], key="atlas_gr_denom",
                        help="「全件」=係属中・審査中も分母に含む（母集団全体の権利化度）。「処分確定」=登録+拒絶+取下+消滅のうち登録の割合（審査が終わった案件での権利化度）。")
                with _gr_c3:
                    _gr_topn = st.slider("表示上位（出願数順）:", 5, 50, 20, key="atlas_gr_topn",
                        help="マップに表示する上位を出願数の多い順に何件までにするかです。中央値の十字は表示中の主体で計算するため、表示数を変えると象限の区切り（中央値）も動きます（十字が常に表示点群の中央に来る設計）。")
                _gr_min = st.slider("最小出願数（ノイズ除外）:", 1, 20, 3, key="atlas_gr_min",
                    help="この件数未満の出願人・技術分野を除外します。少数出願のノイズ（権利化率が偶然極端になる等）を除き、傾向を安定させます。")
                _gr_cl1, _gr_cl2 = st.columns(2)
                with _gr_cl1:
                    _gr_logx = st.checkbox("横軸を対数スケール", value=True, key="atlas_gr_logx",
                        help="ONにすると横軸（出願数）を対数表示にします。少数出願と多数出願が桁違いに離れている場合でも、小規模な主体が重ならず見やすくなります（分析結果は変わりません）。")
                with _gr_cl2:
                    _gr_expired_granted = st.checkbox(
                        "失効（満了・放棄・消滅）も「権利化成功」に含める", value=True, key="atlas_gr_expired",
                        help="ON: 一度でも登録された出願は権利化成功とみなす（失効をマイナスにしない）。OFF: 現在有効（権利継続等）のみを成功とみなす（現存権利率）。")

            _gr_group_col = 'applicant_main' if _gr_unit == "出願人" else 'ipc_normalized'
            _gr_denom = "all" if _gr_denom_label == "登録 / 全件" else "decided"

            if _run_grantrate:
                if _gr_group_col not in df_filtered.columns:
                    st.warning(f"{_gr_unit}データ（{_gr_group_col}）がありません。")
                else:
                    _gr_stats = utils.compute_grant_rate_stats(
                        df_filtered, _gr_group_col, _gr_status_col, denom=_gr_denom, min_count=_gr_min,
                        expired_as_granted=_gr_expired_granted)
                    if not _gr_stats.empty:
                        _gr_stats = _gr_stats[_gr_stats['grant_rate'].notna()]
                    if _gr_stats.empty:
                        st.warning("該当データがありません（最小出願数を下げる／ステータス値をご確認ください）。")
                        st.session_state.pop('atlas_gr_stats', None)
                    else:
                        st.session_state['atlas_gr_stats'] = _gr_stats
                        st.session_state['atlas_gr_meta'] = {
                            'unit': _gr_unit, 'denom': _gr_denom_label, 'logx': _gr_logx,
                            'expired_as_granted': _gr_expired_granted}

            if 'atlas_gr_stats' in st.session_state:
                _gr_stats_all = st.session_state['atlas_gr_stats']
                _gr_meta = st.session_state.get('atlas_gr_meta', {})
                _gr_unit_d = _gr_meta.get('unit', '出願人')
                _gr_exp = bool(_gr_meta.get('expired_as_granted', True))
                _gr_defn = f"{_gr_meta.get('denom', '')}／失効は{'登録に含む' if _gr_exp else '登録に含まない'}"
                _gr_view = _gr_stats_all.head(int(_gr_topn)).copy()
                # 集計済みの結果に appeal 列が無いセッションでも表・JSON 出力が壊れないように補完
                if 'appeal' not in _gr_view.columns:
                    _gr_view['appeal'] = 0

                # 中央値（象限の十字）は「表示中の上位N者」で計算する。
                # 母集団全体（_gr_stats_all）の中央値だと、出願数のロングテールに引っ張られて
                # 縦の中央値線が表示点群の左外に出てしまい、十字に分かれない（上位者は全員右側に寄る）。
                # 表示中の主体で算出すれば、十字は常に表示点群の中央に来て4象限が機能する。
                _med_x = float(_gr_view['total'].median())
                _med_y = float(_gr_view['grant_rate'].median())

                _smax = max(int(_gr_view['total'].max()), 1)
                _sizes = (_gr_view['total'] / _smax * 48 + 8)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=_gr_view['total'], y=_gr_view['grant_rate'],
                    mode='markers+text',
                    text=_gr_view['group'], textposition="top center", textfont=dict(size=11),
                    marker=dict(
                        size=_sizes, sizemode='diameter',
                        color=_gr_view['grant_rate'], colorscale='RdYlGn',
                        cmin=0, cmax=100, showscale=True, colorbar=dict(title="権利化率(%)"),
                        line=dict(width=1, color='#333333'), opacity=0.85),
                    customdata=_gr_view[['granted_eff', 'total', 'rejected', 'expired', 'granted']].values,
                    hovertemplate=("<b>%{text}</b><br>出願数: %{customdata[1]}<br>権利化成功: %{customdata[0]}"
                                   "（権利継続 %{customdata[4]} / 失効 %{customdata[3]}）<br>権利化率: %{y:.1f}%"
                                   "<br>拒絶: %{customdata[2]}<extra></extra>")
                ))
                # 注: plotly は対数軸でも add_vline の x 座標を自動で log10 変換する。
                # 生値（_med_x）のまま渡すのが正しい（log10 を渡すと二重変換でズレる）。
                fig.add_vline(x=_med_x, line=dict(color='#888888', dash='dash'))
                fig.add_hline(y=_med_y, line=dict(color='#888888', dash='dash'))
                update_fig_layout(fig, f'権利化率マップ（{_gr_unit_d}別・{_gr_defn}）', height=720, show_legend=False)
                fig.update_layout(xaxis_title=f"出願数（{_gr_unit_d}別）", yaxis_title="権利化率 (%)")
                fig.update_yaxes(range=[-5, 105])
                if _gr_meta.get('logx'):
                    fig.update_xaxes(type='log')

                st.plotly_chart(fig, use_container_width=True, config={'editable': False})
                st.caption(f"中央値: 出願数={_med_x:.0f} / 権利化率={_med_y:.1f}%　｜　右上=量質両立・左上=少数精鋭・右下=量産だが権利化課題・左下=限定的")

                # 読み方カード（旧 st.info 軸説明を載せ替え・内容維持）
                apollo_ui.howto(
                    "「多数出願だが権利化率が低い」vs「少数だが権利化率が高い」を一望するマップです。"
                    "<b>横軸は出願数</b>（活動量）、<b>縦軸は権利化率</b>（登録の割合）、バブルの大きさも出願数を表します。"
                    "破線の十字（表示中の主体の中央値）で分かれる4象限は、<b>右上</b>=量・質ともに強い／<b>左上</b>=少数精鋭（質重視）／"
                    "<b>右下</b>=量産だが権利化に課題／<b>左下</b>=限定的、と読みます。"
                    "<b>審査中・係属中の案件が多い主体の権利化率は暫定値</b>である点に注意してください。",
                    title="このマップの読み方",
                )

                _gr_table = _gr_view[['group', 'total', 'granted_eff', 'granted', 'expired', 'rejected', 'appeal', 'examining', 'pending', 'grant_rate']].rename(
                    columns={'group': _gr_unit_d, 'total': '出願数', 'granted_eff': '権利化成功', 'granted': '権利継続',
                             'expired': '失効', 'rejected': '拒絶', 'appeal': '審判・異議', 'examining': '審査中', 'pending': '係属', 'grant_rate': '権利化率(%)'})
                st.dataframe(_gr_table, use_container_width=True, hide_index=True)
                st.caption(
                    f"権利化成功 = 権利継続{' + 失効' if _gr_exp else ''} + 異議・無効審判係属中"
                    f"（登録後に争われている特許は登録済みとして数えます。拒絶査定不服審判は未確定扱い。定義: {_gr_defn}）")

                # CAPCOM 保存
                try:
                    import capcom
                    if capcom.is_active():
                        capcom.save_data("atlas_grant_rate.json", {
                            "unit": _gr_unit_d,
                            "definition": _gr_defn,
                            "expired_as_granted": _gr_exp,
                            "median_total": _med_x,
                            "median_grant_rate": _med_y,
                            "groups": _gr_view[['group', 'total', 'granted_eff', 'granted', 'rejected', 'examining',
                                                'pending', 'withdrawn', 'expired', 'appeal', 'grant_rate']].to_dict('records'),
                        })
                except Exception as e:
                    st.caption(f":material/warning: WARN 権利化率マップ の CAPCOM 保存に失敗しました（要確認）: {e}")

                # スナップショット
                snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=5)
                snap_data['module'] = 'ATLAS'
                snap_data['chart_data'] = utils_ai.df_to_markdown(_gr_table, index=False)
                utils.render_snapshot_button(
                    title=f"権利化率マップ（{_gr_unit_d}別）",
                    description="出願数と権利化率の象限分析。多数出願だが権利化率が低い／少数だが権利化率が高い、を比較する。",
                    key="atlas_grantrate_snap", fig=fig, data_summary=snap_data)
                apollo_ui.snapshot_hint()

                # AI インサイト
                _meta_gr = utils_ai.build_common_metadata(
                    df_main=df_main, df_filtered=df_filtered, col_map=col_map,
                    filter_info=f"{int(stats_start_year)}年～{int(stats_end_year)}年")
                _meta_gr['分析単位'] = _gr_unit_d
                _meta_gr['権利化率の定義'] = _gr_defn
                _meta_gr['失効の扱い'] = '登録（権利化成功）に含める' if _gr_exp else '含めない（現存権利のみ）'
                _meta_gr['出願数の中央値'] = f"{_med_x:.0f}"
                _meta_gr['権利化率の中央値'] = f"{_med_y:.1f}%"
                _gr_prompt = utils_ai.generate_ai_insight_prompt(
                    role="知財戦略アナリストとして、出願数と権利化率の象限分析を行ってください。",
                    context="""\
権利化率マップ（バブルチャート）を表示しています。
- X軸: 出願数（活動量）／Y軸: 権利化率（= 権利化成功 ÷ 分母）／バブル径: 出願数
- **権利化成功 = 権利継続 ＋ 失効（満了・放棄・消滅）**。一度でも登録された出願は「権利化に成功した」とみなし、失効はマイナス要素にしない（その後の失効は維持判断の結果）。具体的な設定はメタデータ「失効の扱い」を参照。
- 破線は各軸の中央値で、4象限に分かれます。
- 右上=量質両立、左上=少数精鋭（質）、右下=量産だが権利化に課題、左下=限定的。""",
                    data_summary=snap_data.get('chart_data', ''),
                    instructions="""\
以下の観点で分析してください:
1. **象限ごとの主要プレイヤー**: 各象限に位置する出願人/技術を具体名で挙げる
2. **量産だが権利化率が低い主体**: その理由の仮説（拒絶率・取下げ率の高さ、出願戦略、技術成熟度、審査対応 等）
3. **少数精鋭（高権利化率）の主体**: 集中特許戦略の示唆
4. **権利化成功の内訳**: 権利継続（現存権利）と失効（過去に登録）の比率に触れ、「現存する権利の厚み」と「過去の権利化実績」を区別して論じる
5. **競争上の含意**: 権利化率の差が事業・参入障壁に与える影響
6. **戦略提言**: 自社・対象分野の知財戦略への示唆

各主張に具体的な数値（出願数・権利化率）を必ず1つ以上含めてください。""",
                    metadata=_meta_gr,
                    constraints="権利化率は「権利化成功＝権利継続＋失効（一度でも登録された）」で算出している（失効は権利化の失敗ではなく維持判断の結果）。この定義を踏まえて解釈し、審査係属中の比率が高い主体は暫定値である点にも触れること。",
                    output_format="Markdown形式。見出し付きの構造化された分析レポート。")
                utils_ai.render_ai_insight_button(_gr_prompt, "atlas_grantrate_insight")
