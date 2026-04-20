# ==================================================================
# --- 環境設定 ---
# ==================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['OMP_NUM_THREADS'] = '1'

# ==================================================================
# --- ライブラリ ---
# ==================================================================
import streamlit as st
import textwrap
import pandas as pd
import numpy as np
import warnings
import traceback
import unicodedata
import re
import time
import datetime

import patiroha

warnings.filterwarnings('ignore')

# ==================================================================
# --- ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO v8 | Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================================
# --- 定数とヘルパー関数 ---
# ==================================================================

import io

import utils
import capcom

# ==================================================================
# --- patiroha統合: SBERTモデル・ストップワード ---
# ==================================================================

@st.cache_resource
def load_sbert_embedder():
    """patiroha.SBERTEmbedderをキャッシュして1回だけロードする"""
    return patiroha.SBERTEmbedder()


def _get_current_stopwords():
    """session_stateのストップワードを取得（未設定ならpatirohaデフォルト）"""
    if 'stopwords' in st.session_state and st.session_state['stopwords']:
        sw = st.session_state['stopwords']
        return frozenset(sw) if not isinstance(sw, frozenset) else sw
    return patiroha.get_stopwords()


def advanced_tokenize(text):
    """TF-IDF用トークナイズ — patiroha.tokenize_for_tfidfに委譲"""
    current_stopwords = _get_current_stopwords()
    return patiroha.tokenize_for_tfidf(text, stopwords=current_stopwords)


def smart_map_index(current_value, options, keywords):
    """カラム紐付けの自動化ロジック（UIのselectbox用）"""
    if current_value is not None and current_value in options:
        return options.index(current_value)

    valid_cols = options[1:]

    for kw in keywords:
        for col in valid_cols:
            if kw == str(col):
                return options.index(col)

    for kw in keywords:
        for col in valid_cols:
            if kw in str(col):
                return options.index(col)

    return 0

# ==================================================================
# --- メイン画面描画 ---
# ==================================================================

utils.render_sidebar()

st.title("🛰️ Mission Control") 
st.markdown("ここは、全分析モジュールで共通のデータ準備を行う「ミッション・コントロール（データハブ）」です。")

# --- アプリケーション初期化 ---
def initialize_session_state():
    defaults = {
        "df_main": None,
        "df_npl": None,  # Non-Patent Literature
        "shared_df": None,
        "filename": "No File",
        "npl_filename": "No File",
        "sbert_model": None,
        "sbert_embeddings": None,
        "tfidf_matrix": None,
        "feature_names": None,
        "col_map": {},
        "delimiters": {'applicant': ';', 'inventor': ';', 'ipc': ';', 'fterm': ';', 'npl_category': ';'},
        "preprocess_done": False,
        # CAPCOM (In-Memory版: session_state['capcom_store'] にデータ全保持)
        "capcom_session_id": None,
        # CAPCOM 専用 Mission Objective (VOYAGER とは独立に保持)
        "capcom_mission_objective": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()


st.markdown("<h3 style='border: none; padding-bottom: 0;'>分析設定</h3>", unsafe_allow_html=True)

container = st.container() 

with container:
    tab1, tab2, tab3, tab4 = st.tabs([
        "フェーズ 1: データインポート", 
        "フェーズ 2: カラム紐付け", 
        "フェーズ 3: ストップワード管理",
        "フェーズ 4: 分析エンジン起動"
    ])

    # A-1. ファイルアップロード
    with tab1:
        st.markdown("##### 分析対象の特許リストをインポートしてください。")
        uploaded_file = st.file_uploader(
            "分析ファイルをアップロード (CSV または Excel)",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="main_file_uploader"
        )

        if uploaded_file is not None:
            # 前処理完了後のrerunでは再読み込みをスキップ（新ファイルの場合のみ実行）
            is_new_file = (uploaded_file.name != st.session_state.get('filename', ''))
            if is_new_file or not st.session_state.get('preprocess_done', False):
                try:
                    if uploaded_file.name.lower().endswith('.csv'):
                        try:
                            df = pd.read_csv(uploaded_file, dtype=str)
                        except UnicodeDecodeError:
                            df = pd.read_csv(uploaded_file, dtype=str, encoding='shift_jis')
                    else:
                        df = pd.read_excel(uploaded_file, dtype=str)

                    st.session_state.df_main = df
                    st.session_state.preprocess_done = False
                    st.session_state['shared_df'] = df
                    st.session_state['filename'] = uploaded_file.name

                    st.success(f"ファイル '{uploaded_file.name}' のインポート完了 ({len(df)}行)。")
                    st.dataframe(df.head())

                except Exception as e:
                    st.error(f"ファイルインポートエラー: {e}")
                    st.session_state.df_main = None
                    st.session_state.shared_df = None
            else:
                # 前処理済みデータが存在する場合はスキップし、既存データ情報を表示
                st.success(f"ファイル '{uploaded_file.name}' はインポート・前処理済みです ({len(st.session_state.df_main)}行)。")
                st.dataframe(st.session_state.df_main.head())
                
        st.markdown("---")
        st.markdown("##### (オプション) 特許以外の情報 (NPL) のインポート")
        with st.expander("論文・ニュース・政策文書などを取り込む (NPL)", expanded=False):
            


            # --- NPL蓄積ロジック ---
            if 'df_npl_accumulated' not in st.session_state:
                st.session_state.df_npl_accumulated = pd.DataFrame()

            st.write("データソースの種類を選択してアップロードしてください:")
            
            npl_tabs = st.tabs(['📚 Academic (論文)', '📰 Business News (ニュース)', '⚖️ Policy/Regulation (政策)', '📊 Market Report (市場)'])

            # ファイル読み込みの共通関数
            def read_uploaded_file(f):
                if f.name.lower().endswith('.csv'):
                    try:
                        return pd.read_csv(f, dtype=str)
                    except UnicodeDecodeError:
                        return pd.read_csv(f, dtype=str, encoding='shift_jis')
                else:
                    return pd.read_excel(f, dtype=str)

            # --- タブ 1: Academic (論文) — CSVアップロード + OpenALEX検索 ---
            with npl_tabs[0]:
                st.markdown("###### 📚 Academic (論文データ)")
                aca_mode = st.radio(
                    "取得方法を選択:",
                    ["📄 CSVアップロード", "🔍 OpenALEX検索"],
                    horizontal=True, key="aca_mode"
                )

                if aca_mode == "📄 CSVアップロード":
                    f_aca = st.file_uploader("論文データ (LENS等) をアップロード", type=["csv", "xlsx"], key="up_aca", accept_multiple_files=False)

                    if f_aca:
                        df = read_uploaded_file(f_aca)
                        st.caption(f"Preview: {f_aca.name}")
                        st.dataframe(df.head(3))

                        cols = [None] + list(df.columns)
                        c1, c2 = st.columns(2)
                        idx_t = smart_map_index(None, cols, ['Title', 'Article Title', 'タイトル', 'inventionTitle'])
                        idx_d = smart_map_index(None, cols, ['Date', 'Publication Date', '発行日', 'publicationDate'])
                        idx_c = smart_map_index(None, cols, ['Abstract', 'Summary', '要約', 'abstract'])
                        idx_s = smart_map_index(None, cols, ['Source', 'Journal', 'Publisher', '情報源', 'publisher'])

                        with c1:
                            m_title = st.selectbox("Title (必須):", cols, index=idx_t, key="map_aca_title")
                            m_date = st.selectbox("Date (必須):", cols, index=idx_d, key="map_aca_date")
                        with c2:
                            m_content = st.selectbox("Abstract (必須):", cols, index=idx_c, key="map_aca_content")
                            m_source = st.selectbox("Source (任意):", cols, index=idx_s, key="map_aca_source")

                        if st.button("➕ データセットに追加 (Academic)", key="add_aca"):
                            if m_title and m_date and m_content:
                                df_new = pd.DataFrame()
                                df_new['unified_title'] = df[m_title]
                                df_new['unified_date'] = df[m_date]
                                df_new['unified_content'] = df[m_content]
                                df_new['unified_source'] = df[m_source] if m_source else "Academic Source"
                                df_new['unified_region'] = 'Global'
                                df_new['unified_status'] = '-'
                                df_new['data_sub_type'] = 'Academic'
                                df_new['source_filename'] = f_aca.name

                                st.session_state.df_npl_accumulated = pd.concat([st.session_state.df_npl_accumulated, df_new], ignore_index=True)
                                st.success("追加しました。")
                                st.rerun()
                            else:
                                st.error("必須カラム(Title, Date, Abstract)を選択してください。")

                else:
                    # --- OpenALEX検索 ---
                    st.markdown("OpenAlex APIで学術論文を直接検索してデータセットに追加します。")

                    oalex_query = st.text_area(
                        "検索キーワード（1行1クエリ、複数行でOR検索）",
                        placeholder='"cellulose nanofiber"\n"nanocellulose"',
                        height=80,
                        key="oalex_query"
                    )

                    oalex_inst = st.text_input(
                        "所属機関フィルタ（セミコロン区切りでOR）",
                        placeholder="例: Toyota; MIT",
                        key="oalex_inst"
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        oalex_year_from = st.number_input("開始年", min_value=1900, max_value=2026, value=2020, key="oalex_year_from")
                    with c2:
                        oalex_year_to = st.number_input("終了年", min_value=1900, max_value=2026, value=2026, key="oalex_year_to")
                    with c3:
                        oalex_max = st.number_input("取得上限", min_value=50, max_value=10000, value=200, step=50, key="oalex_max")

                    # --- 年別取得モード（10,000件/クエリ制限を回避して広い年範囲を網羅） ---
                    oalex_by_year = st.checkbox(
                        "📅 年別取得モード（年ごとに最大上限まで取得、広い年範囲で大量取得したい場合）",
                        value=False,
                        key="oalex_by_year",
                        help=(
                            "OFF: 全期間で合算して『取得上限』まで取得（高速・少量向け）\n"
                            "ON:  各年ごとに『年あたりの最大件数』まで取得し重複除去（広い年範囲・大量取得向け）\n"
                            "     年数 × ページネーション回数分のAPIコールが発生するため時間がかかります"
                        ),
                    )
                    oalex_max_per_year = 10000
                    if oalex_by_year:
                        oalex_max_per_year = st.number_input(
                            "年あたりの最大件数（max_per_year）",
                            min_value=100, max_value=10000, value=10000, step=500,
                            key="oalex_max_per_year",
                            help="各年ごとに取得する上限件数。10,000 が OpenAlex の実質上限。",
                        )
                        _years_span = max(1, int(oalex_year_to) - int(oalex_year_from) + 1)
                        st.caption(
                            f"🧮 試算: {_years_span} 年 × 最大 {oalex_max_per_year:,} 件 = "
                            f"最大 {_years_span * int(oalex_max_per_year):,} 件（重複除去前）。"
                            f"所要時間: 年数とページ送りに比例（広範囲だと数分以上）"
                        )

                    # --- 論文種別フィルタ（複数選択可、未選択＝全種別） ---
                    # OpenALEX_Collector.html と同等の10種類
                    OALEX_PUB_TYPE_OPTIONS = {
                        "Article（学術論文）": "article",
                        "Review（総説）": "review",
                        "Book Chapter": "book-chapter",
                        "Book": "book",
                        "Dataset": "dataset",
                        "Preprint": "preprint",
                        "Dissertation（学位論文）": "dissertation",
                        "Editorial": "editorial",
                        "Letter": "letter",
                        "Report（技術レポート）": "report",
                    }
                    oalex_pub_type_labels = st.multiselect(
                        "論文種別（複数選択可、未選択＝全種別）",
                        options=list(OALEX_PUB_TYPE_OPTIONS.keys()),
                        default=[],
                        key="oalex_pub_type_labels",
                        help="OpenALEX の type フィルタ。未選択の場合は全種別が対象。",
                    )
                    oalex_pub_types = [OALEX_PUB_TYPE_OPTIONS[lbl] for lbl in oalex_pub_type_labels]

                    # --- 分析品質向上フィルタ ---
                    col_filt1, col_filt2 = st.columns(2)
                    with col_filt1:
                        oalex_has_abstract = st.checkbox(
                            "📄 要約ありの論文のみ取得",
                            value=True,
                            key="oalex_has_abstract",
                            help=(
                                "OpenAlex の `has_abstract:true` フィルタを適用。\n"
                                "要約（unified_content）は SBERT 埋め込み・クラスタリングで必須のため、\n"
                                "分析精度を担保したい場合は推奨（デフォルト ON）。"
                            ),
                        )
                    with col_filt2:
                        oalex_en_only = st.checkbox(
                            "🌐 英語論文のみ取得",
                            value=False,
                            key="oalex_en_only",
                            help=(
                                "OpenAlex の `language:en` フィルタを適用。\n"
                                "多言語データ（中国語・ドイツ語等）が混在すると SBERT の精度が低下するため、\n"
                                "グローバル比較が主目的なら ON を推奨。"
                            ),
                        )

                    if st.button("🔍 OpenALEX検索実行", key="oalex_search_btn"):
                        if not oalex_query.strip():
                            st.error("検索キーワードを入力してください。")
                        else:
                            try:
                                from openalex import OpenAlexCollector
                                collector = OpenAlexCollector()

                                queries = [q.strip() for q in oalex_query.strip().split('\n') if q.strip()]

                                progress_bar = st.progress(0.0)
                                status_text = st.empty()

                                # 機関解決
                                inst_ids = []
                                if oalex_inst.strip():
                                    inst_names = [s.strip() for s in oalex_inst.split(';') if s.strip()]
                                    for name in inst_names:
                                        resolved = collector.resolve_institution(name)
                                        if resolved:
                                            inst_ids.append(resolved['id'])
                                            status_text.info(f"機関解決: {name} → {resolved['display_name']}")

                                def on_progress(current, total):
                                    if total > 0:
                                        progress_bar.progress(min(current / total, 1.0))
                                    status_text.markdown(f"取得中: {current} 件...")

                                def on_year_progress(yi, total_years, year, year_count, year_total, all_count):
                                    # 全体プログレス: 年インデックス + 当年内の進捗
                                    year_frac = (year_count / year_total) if year_total else 1.0
                                    overall = (yi + min(year_frac, 1.0)) / max(total_years, 1)
                                    progress_bar.progress(min(overall, 1.0))
                                    status_text.markdown(
                                        f"📅 {year} 年 ({yi + 1}/{total_years}): "
                                        f"{year_count:,} / {year_total:,} 件 | 累計: {all_count:,} 件"
                                    )

                                # 共通フィルタ引数（全 4 パスで使用）
                                _common_kwargs = dict(
                                    pub_types=oalex_pub_types if oalex_pub_types else None,
                                    institution_ids=inst_ids if inst_ids else None,
                                    has_abstract=bool(oalex_has_abstract),
                                    language=("en" if oalex_en_only else None),
                                )

                                # 検索実行（通常モード / 年別取得モードで分岐）
                                if oalex_by_year:
                                    # --- 年別取得モード（10,000件/クエリ制限を回避） ---
                                    if len(queries) == 1:
                                        raw_papers = collector.search_by_year(
                                            queries[0],
                                            year_from=int(oalex_year_from),
                                            year_to=int(oalex_year_to),
                                            max_per_year=int(oalex_max_per_year),
                                            on_progress=on_year_progress,
                                            **_common_kwargs,
                                        )
                                    else:
                                        # 複数クエリ × 年別: 各クエリを個別に年別検索して重複除去
                                        raw_papers = []
                                        seen_ids = set()
                                        total_q = len(queries)
                                        for qi, q in enumerate(queries):
                                            def _q_year_progress(
                                                yi, total_years, year,
                                                year_count, year_total, all_count,
                                                _qi=qi, _tq=total_q,
                                            ):
                                                year_frac = (year_count / year_total) if year_total else 1.0
                                                within_q = (yi + min(year_frac, 1.0)) / max(total_years, 1)
                                                overall = (_qi + within_q) / _tq
                                                progress_bar.progress(min(overall, 1.0))
                                                status_text.markdown(
                                                    f"🔎 クエリ {_qi + 1}/{_tq} | 📅 {year} 年 "
                                                    f"({yi + 1}/{total_years}): {year_count:,} / {year_total:,} 件 | "
                                                    f"統合累計: {len(raw_papers):,} 件"
                                                )
                                            batch = collector.search_by_year(
                                                q,
                                                year_from=int(oalex_year_from),
                                                year_to=int(oalex_year_to),
                                                max_per_year=int(oalex_max_per_year),
                                                on_progress=_q_year_progress,
                                                **_common_kwargs,
                                            )
                                            for paper in batch:
                                                pid = paper.get("id", "")
                                                if pid and pid not in seen_ids:
                                                    seen_ids.add(pid)
                                                    raw_papers.append(paper)
                                elif len(queries) == 1:
                                    # --- 通常モード（単一クエリ） ---
                                    raw_papers = collector.search(
                                        queries[0],
                                        year_from=oalex_year_from, year_to=oalex_year_to,
                                        max_results=oalex_max,
                                        on_progress=on_progress,
                                        **_common_kwargs,
                                    )
                                else:
                                    # --- 通常モード（複数クエリ OR） ---
                                    raw_papers = collector.search_multi_query(
                                        queries,
                                        year_from=oalex_year_from, year_to=oalex_year_to,
                                        max_results=oalex_max,
                                        **_common_kwargs,
                                    )

                                if raw_papers:
                                    papers = [collector.transform_paper(p) for p in raw_papers]
                                    df_oalex = collector.to_npl_dataframe(papers)

                                    # 英語のみフラグ ON の場合、タイトル側も英語判定で追加フィルタ
                                    # （OpenAlex の `language:en` は abstract ベース判定のため、
                                    #   タイトルが別言語の論文が混入することがある）
                                    if oalex_en_only and not df_oalex.empty:
                                        # CJK 漢字・ひらがな・カタカナ・ハングル・キリル・アラビア・タイ・ヘブライ文字等を検出
                                        _non_en_pat = re.compile(
                                            r'[\u3040-\u309F'      # ひらがな
                                            r'\u30A0-\u30FF'       # カタカナ
                                            r'\u4E00-\u9FFF'       # CJK 統合漢字
                                            r'\u3400-\u4DBF'       # CJK 統合漢字拡張 A
                                            r'\uAC00-\uD7AF'       # ハングル音節
                                            r'\u0400-\u04FF'       # キリル文字
                                            r'\u0590-\u05FF'       # ヘブライ文字
                                            r'\u0600-\u06FF'       # アラビア文字
                                            r'\u0E00-\u0E7F'       # タイ文字
                                            r'\u0900-\u097F'       # デーヴァナーガリー
                                            r']'
                                        )
                                        _before_n = len(df_oalex)
                                        _title_series = df_oalex['unified_title'].fillna('').astype(str)
                                        df_oalex = df_oalex[~_title_series.str.contains(_non_en_pat, regex=True)].reset_index(drop=True)
                                        _removed_n = _before_n - len(df_oalex)
                                        if _removed_n > 0:
                                            status_text.info(
                                                f"🌐 タイトルが非英語の論文 **{_removed_n:,} 件** を除外しました "
                                                f"（OpenAlex の `language:en` は要約ベースの判定のため、"
                                                f"タイトルだけ日本語・中国語等の多言語ジャーナル論文が混入することがあります）"
                                            )

                                    progress_bar.progress(1.0)

                                    if df_oalex.empty:
                                        status_text.warning("フィルタ後、該当する論文が 0 件になりました。条件を緩めてください。")
                                        st.session_state.pop('oalex_last_result', None)
                                    else:
                                        status_text.success(f"✅ {len(df_oalex):,} 件の論文を取得しました。")
                                        # 検索結果を session_state に保持してページ再描画後も使えるようにする
                                        st.session_state['oalex_last_result'] = df_oalex
                                else:
                                    status_text.warning("該当する論文が見つかりませんでした。")
                                    st.session_state.pop('oalex_last_result', None)

                            except Exception as e:
                                st.error(f"OpenALEX検索エラー: {e}")

                    # --- 検索結果プレビュー + CSV ダウンロード + データセット追加 ---
                    df_oalex_cached = st.session_state.get('oalex_last_result')
                    if df_oalex_cached is not None and not df_oalex_cached.empty:
                        st.markdown("###### 🔎 検索結果プレビュー")

                        # 要約の取得成功率を表示（分析精度に直結するため明示）
                        _has_abstract = df_oalex_cached['unified_content'].fillna('').astype(str).str.strip().astype(bool)
                        _abs_ratio = _has_abstract.sum() / len(df_oalex_cached) * 100
                        _abs_color = "🟢" if _abs_ratio >= 80 else ("🟡" if _abs_ratio >= 50 else "🔴")
                        st.caption(
                            f"{_abs_color} 要約取得率: **{_abs_ratio:.1f}%** "
                            f"({_has_abstract.sum():,} / {len(df_oalex_cached):,} 件) — "
                            f"SBERT 埋め込み・クラスタリングは `unified_content`（要約）を使用します"
                        )

                        # プレビュー表示用に要約を切り詰め（全文は CSV ダウンロードで取得可）
                        _preview = df_oalex_cached[[
                            'unified_title', 'unified_content', 'unified_date',
                            'unified_source', 'citation_count',
                        ]].head(10).copy()
                        _preview['unified_content'] = _preview['unified_content'].fillna('').astype(str).apply(
                            lambda s: (s[:150] + '…') if len(s) > 150 else s
                        )
                        st.dataframe(
                            _preview,
                            column_config={
                                'unified_title': st.column_config.TextColumn('タイトル', width='medium'),
                                'unified_content': st.column_config.TextColumn('要約（先頭150字）', width='large'),
                                'unified_date': st.column_config.TextColumn('出版日', width='small'),
                                'unified_source': st.column_config.TextColumn('ジャーナル', width='medium'),
                                'citation_count': st.column_config.NumberColumn('被引用数', width='small'),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.caption(
                            f"全 {len(df_oalex_cached)} 件を取得済み。先頭10件をプレビュー表示（要約は切り詰め）。"
                            f"分析対象の全カラム: `unified_title` / `unified_content`（要約）/ `unified_date` / "
                            f"`unified_source` / `unified_region`（所属機関）/ `citation_count` / `doi` / "
                            f"`data_sub_type`（= Academic）"
                        )

                        col_dl, col_add = st.columns(2)
                        with col_dl:
                            # CSV ダウンロード（取得した全件、Excel で開けるよう UTF-8 BOM）
                            csv_bytes = df_oalex_cached.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                            st.download_button(
                                "📥 検索結果をCSVでダウンロード",
                                data=csv_bytes,
                                file_name=f"openalex_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
                                mime="text/csv",
                                key="oalex_csv_dl",
                            )
                        with col_add:
                            if st.button("➕ データセットに追加 (OpenALEX)", key="oalex_add_btn"):
                                st.session_state.df_npl_accumulated = pd.concat(
                                    [st.session_state.df_npl_accumulated, df_oalex_cached], ignore_index=True)
                                st.success(f"{len(df_oalex_cached)}件を追加しました。")
                                st.rerun()

            # --- タブ 2: News (ニュース) ---
            with npl_tabs[1]:
                st.markdown("###### 📰 Business News (ニュース)")
                f_news = st.file_uploader("ニュースデータをアップロード", type=["csv", "xlsx"], key="up_news", accept_multiple_files=False)
                
                if f_news:
                    df = read_uploaded_file(f_news)
                    st.caption(f"Preview: {f_news.name}")
                    st.dataframe(df.head(3))
                    
                    cols = [None] + list(df.columns)
                    c1, c2 = st.columns(2)
                    idx_t = smart_map_index(None, cols, ['Title', 'Headline', 'タイトル', '見出し'])
                    idx_d = smart_map_index(None, cols, ['Date', 'Published', '日付'])
                    idx_c = smart_map_index(None, cols, ['Content', 'Body', '本文', 'Project Description'])
                    idx_s = smart_map_index(None, cols, ['Source', 'Media', '媒体'])
                    
                    with c1:
                        m_title = st.selectbox("Headline (必須):", cols, index=idx_t, key="map_news_title")
                        m_date = st.selectbox("Date (必須):", cols, index=idx_d, key="map_news_date")
                    with c2:

                        m_content = st.selectbox("Content (任意):", cols, index=smart_map_index(None, cols, ['Content', 'Body', '本文', 'Project Description']), key="map_news_content")
                        m_source = st.selectbox("Source (任意):", cols, index=smart_map_index(None, cols, ['Source', 'Media', '媒体']), key="map_news_source")
                        
                    if st.button("➕ データセットに追加 (News)", key="add_news"):
                        if m_title and m_date:
                            df_new = pd.DataFrame()
                            df_new['unified_title'] = df[m_title]
                            df_new['unified_date'] = df[m_date]
                            df_new['unified_content'] = df[m_content] if m_content else "" # Optional
                            df_new['unified_source'] = df[m_source] if m_source else "News Source"
                            df_new['unified_region'] = 'Global'
                            df_new['unified_status'] = '-'
                            df_new['data_sub_type'] = 'Business'
                            df_new['source_filename'] = f_news.name
                            
                            st.session_state.df_npl_accumulated = pd.concat([st.session_state.df_npl_accumulated, df_new], ignore_index=True)
                            st.success("追加しました。")
                            st.rerun()
                        else:
                            st.error("必須カラム(Headline, Date)を選択してください。")

            # --- タブ 3: Policy (政策) ---
            with npl_tabs[2]:
                st.markdown("###### ⚖️ Policy & Regulation (政策)")
                
                # AIプロンプト
                if st.toggle("🤖 AIデータ作成プロンプト (Policy)"):
                    theme_pol = st.text_input("調査テーマ (例: 生成AIの著作権規制, ドローンの飛行禁止区域)", key="theme_pol")
                    
                    if not theme_pol:
                        st.caption("※ 上記にテーマを入力すると、プロンプトに反映されます。")
                        theme_pol = "[ここに調査テーマを入力してください]"

                    prompt_policy = textwrap.dedent(f"""
                        # Role (役割)
                        あなたは専門的な「戦略的政策アナリスト」です。以下の【調査テーマ】に関連する主要な規制・政策・政府ガイドラインを網羅的に調査し、抽出してください。

                        # Theme (調査テーマ)
                        {theme_pol}

                        # Objective (目的)
                        業界に影響を与える最も重要な規制イベントの構造化CSVデータセットを作成してください。促進的な政策（補助金、規制緩和）と、制限的な規制（禁止事項、コンプライアンス要件）の両方に焦点を当ててください。

                        # Formatting Rules (出力ルール)
                        - **CSVコードブロックのみ** を出力してください。挨拶や説明文は不要です。
                        - **Date**: YYYY (西暦4桁の年のみ)。正確な日付が不明な場合は施行年または発表年を使用してください。
                        - **Abstract**: 日本語で100〜200文字程度の簡潔な要約。何が「禁止」されているか、または「促進」されているかを具体的に明記してください。

                        # CSV Schema
                        Title, Abstract, Date, Region, Status, Source

                        - Title: 政策・規制の名称 (具体的かつ正式名称で)
                        - Abstract: 影響の要約 (規制/促進の内容)
                        - Date: YYYY (例: 2024, 2023)
                        - Region: 地域コード (EU, US, JP, CN, Global, UK 等)
                        - Status: [Draft, Enacted, Proposed, Under Review] (ステータス)
                        - Source: 発行機関またはURL
                    """)
                    st.code(prompt_policy, language="markdown")
                
                f_pol = st.file_uploader("政策データをアップロード", type=["csv", "xlsx"], key="up_pol", accept_multiple_files=False)
                
                if f_pol:
                    df = read_uploaded_file(f_pol)
                    st.caption(f"Preview: {f_pol.name}")
                    st.dataframe(df.head(3))
                    
                    cols = [None] + list(df.columns)
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        m_title = st.selectbox("Policy Name (必須):", cols, index=smart_map_index(None, cols, ['Title', 'Name', '名称']), key="map_pol_title")
                        m_date = st.selectbox("Date (必須):", cols, index=smart_map_index(None, cols, ['Date', 'Effective', '日付']), key="map_pol_date")
                        m_region = st.selectbox("Region (任意):", cols, index=smart_map_index(None, cols, ['Region', 'Country', '国']), key="map_pol_reg")
                    with c2:
                        m_content = st.selectbox("Summary (必須):", cols, index=smart_map_index(None, cols, ['Abstract', 'Summary', 'Description', '要約']), key="map_pol_cont")
                        m_source = st.selectbox("Source (任意):", cols, index=smart_map_index(None, cols, ['Source', 'Ministry', '情報源']), key="map_pol_src")
                        m_status = st.selectbox("Status (任意):", cols, index=smart_map_index(None, cols, ['Status', 'State', '状態']), key="map_pol_stat")

                    if st.button("➕ データセットに追加 (Policy)", key="add_pol"):
                        if m_title and m_date and m_content:
                            df_new = pd.DataFrame()
                            df_new['unified_title'] = df[m_title]
                            df_new['unified_date'] = df[m_date]
                            df_new['unified_content'] = df[m_content]
                            df_new['unified_source'] = df[m_source] if m_source else "Policy Source"
                            df_new['unified_region'] = df[m_region] if m_region else 'Global'
                            df_new['unified_status'] = df[m_status] if m_status else '-'
                            df_new['data_sub_type'] = 'Policy'
                            df_new['source_filename'] = f_pol.name
                            
                            st.session_state.df_npl_accumulated = pd.concat([st.session_state.df_npl_accumulated, df_new], ignore_index=True)
                            st.success("追加しました。")
                            st.rerun()
                        else:
                            st.error("必須カラムを選択してください。")

            # --- タブ 4: Market Report (市場) ---
            with npl_tabs[3]:
                st.markdown("###### 📊 Market Report (市場レポート)")
                
                # AIプロンプト
                if st.toggle("🤖 AIデータ作成プロンプト (Market)"):
                    theme_mkt = st.text_input("調査テーマ (例: 全固体電池の市場規模, 空飛ぶクルマの市場予測)", key="theme_mkt")
                    
                    if not theme_mkt:
                        st.caption("※ 上記にテーマを入力すると、プロンプトに反映されます。")
                        theme_mkt = "[ここに調査テーマを入力してください]"

                    prompt_market = textwrap.dedent(f"""
                        # Role (役割)
                        あなたは「シニア市場インテリジェンスアナリスト」です。以下の【調査テーマ】に関する市場規模データ、成長予測、および主要な競合動向を抽出してください。

                        # Theme (調査テーマ)
                        {theme_mkt}

                        # Objective (目的)
                        市場環境を表す構造化CSVデータセットを作成してください。定量的データ（米ドル換算の市場規模、CAGR/年平均成長率）および主要なM&Aや戦略的シフトを優先して抽出してください。

                        # Formatting Rules (出力ルール)
                        - **CSVコードブロックのみ** を出力してください。
                        - **Date**: YYYY-MM-DD (推奨) または YYYY。
                        - **Abstract**: **必ず具体的な数値を含めてください** (例: "市場規模500億ドル", "CAGR 15%")。主要なドライバーやトレンドを日本語で要約してください。

                        # CSV Schema
                        Title, Abstract, Date, Region, Status, Source

                        - Title: レポートタイトルまたは市場セグメント名
                        - Abstract: 市場データとトレンド (数値を必ず含むこと！)
                        - Date: YYYY-MM-DD または YYYY
                        - Region: 対象市場 (Global, North America, APAC, etc.)
                        - Status: [Growth, Mature, Emerging, Declining] (市場ステージ)
                        - Source: 調査会社またはメディア名
                    """)
                    st.code(prompt_market, language="markdown")
                
                f_mkt = st.file_uploader("市場データをアップロード", type=["csv", "xlsx"], key="up_mkt", accept_multiple_files=False)
                
                if f_mkt:
                    df = read_uploaded_file(f_mkt)
                    st.caption(f"Preview: {f_mkt.name}")
                    st.dataframe(df.head(3))
                    
                    cols = [None] + list(df.columns)
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        m_title = st.selectbox("Report Title (必須):", cols, index=smart_map_index(None, cols, ['Title', 'Segment', 'タイトル']), key="map_mkt_title")
                        m_date = st.selectbox("Date (必須):", cols, index=smart_map_index(None, cols, ['Date', 'Published', '日付']), key="map_mkt_date")
                    with c2:
                        m_content = st.selectbox("Market Summary (必須):", cols, index=smart_map_index(None, cols, ['Abstract', 'Summary', 'Description', '要約']), key="map_mkt_cont")
                        m_source = st.selectbox("Source (任意):", cols, index=smart_map_index(None, cols, ['Source', 'Firm', '出典']), key="map_mkt_src")

                    if st.button("➕ データセットに追加 (Market)", key="add_mkt"):
                        if m_title and m_date and m_content:
                            df_new = pd.DataFrame()
                            df_new['unified_title'] = df[m_title]
                            df_new['unified_date'] = df[m_date]
                            df_new['unified_content'] = df[m_content]
                            df_new['unified_source'] = df[m_source] if m_source else "Market Report"
                            df_new['unified_region'] = 'Global'
                            df_new['unified_status'] = '-'
                            df_new['data_sub_type'] = 'Market'
                            df_new['source_filename'] = f_mkt.name
                            
                            st.session_state.df_npl_accumulated = pd.concat([st.session_state.df_npl_accumulated, df_new], ignore_index=True)
                            st.success("追加しました。")
                            st.rerun()
                        else:
                            st.error("必須カラムを選択してください。")

            # --- 現在のデータセットの状態 ---
            st.markdown("---")
            if not st.session_state.df_npl_accumulated.empty:
                df_acc = st.session_state.df_npl_accumulated
                st.markdown(f"##### 📚 現在のNPLデータセット: 合計 {len(df_acc)} 件")
                
                # 内訳を表示
                stats = df_acc['data_sub_type'].value_counts()
                st.dataframe(pd.DataFrame({"Count": stats}).T)
                
                st.dataframe(df_acc.head(3))
                
                # リセットボタン
                if st.button("🗑️ データをクリア (Reset NPL)", type="secondary"):
                    st.session_state.df_npl_accumulated = pd.DataFrame()
                    if 'df_npl' in st.session_state: del st.session_state.df_npl
                    st.rerun()
            else:
                st.markdown("現在NPLデータは読み込まれていません。")
                

        if st.session_state.df_main is not None:
            # タブ2に移動
            pass
            
    with tab2:
        st.markdown("##### 特許データのカラムを分析用フィールドに割り当てます。")
        if st.session_state.df_main is not None:
            df = st.session_state.df_main
            columns_with_none = [None] + list(df.columns)
            
            kw_title = ['発明の名称', '名称', 'Title', 'Title of Invention']
            kw_abstract = ['要約', '要約(抄録)', 'Abstract']
            kw_claim = ['請求項', 'Claim']
            kw_app_num = ['出願番号', 'Application Number', 'App No']
            kw_date = ['出願日', '出願日（遡及）', 'Date', 'Filing']
            kw_applicant = ['出願人', 'Applicant', 'Assignee']
            kw_inventor = ['発明者', 'Inventor']
            kw_ipc = ['国際特許分類', '国際特許分類(IPC)', 'IPC', 'Int. Cl']
            kw_fterm = ['Fターム', 'テーマコード', 'F-Term']

            col_map = {}
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### 必須テキスト項目")
                col_map['title'] = st.selectbox("発明の名称:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('title'), columns_with_none, kw_title), key="col_title")
                col_map['abstract'] = st.selectbox("要約:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('abstract'), columns_with_none, kw_abstract), key="col_abstract")
                col_map['claim'] = st.selectbox("請求項:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('claim'), columns_with_none, kw_claim), key="col_claim")
            with col2:
                st.markdown("##### 必須メタデータ項目")
                col_map['app_num'] = st.selectbox("出願番号:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('app_num'), columns_with_none, kw_app_num), key="col_app_num")
                col_map['date'] = st.selectbox("出願日:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('date'), columns_with_none, kw_date), key="col_date")
                col_map['applicant'] = st.selectbox("出願人:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('applicant'), columns_with_none, kw_applicant), key="col_applicant")
                applicant_delimiter = st.text_input("出願人区切り文字:", value=st.session_state.delimiters.get('applicant', ';'), key="del_applicant")

                # IPC (必須)
                col_map['ipc'] = st.selectbox("国際特許分類 (IPC):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('ipc'), columns_with_none, kw_ipc), key="col_ipc")
                ipc_delimiter = st.text_input("IPC区切り文字:", value=st.session_state.delimiters.get('ipc', ';'), key="del_ipc")
                
            with col3:
                st.markdown("##### 任意メタデータ項目")

                # 公開番号 (CAPCOM用)
                kw_pub_num = ['公開番号', '公報番号', 'Publication Number', 'Pub No', 'Document Number']
                col_map['pub_number'] = st.selectbox("公開番号 (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('pub_number'), columns_with_none, kw_pub_num), key="col_pub_number")

                # 発明者
                col_map['inventor'] = st.selectbox("発明者 (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('inventor'), columns_with_none, kw_inventor), key="col_inventor")
                inventor_delimiter = st.text_input("発明者区切り文字:", value=st.session_state.delimiters.get('inventor', ';'), key="del_inventor")

                # Fターム
                col_map['fterm'] = st.selectbox("Fターム (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('fterm'), columns_with_none, kw_fterm), key="col_fterm")
                fterm_delimiter = st.text_input("Fターム区切り文字:", value=st.session_state.delimiters.get('fterm', ';'), key="del_fterm")

                # ステータス
                col_map['status'] = st.selectbox("ステータス (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('status'), columns_with_none, ['ステータス', 'Status', 'Legal Status', '法的状態']), key="col_status")
                
            st.session_state.col_map = col_map
            st.session_state.delimiters = {
                'applicant': applicant_delimiter,
                'inventor': inventor_delimiter,
                'ipc': ipc_delimiter,
                'fterm': fterm_delimiter
            }
        else:
            st.info("フェーズ1でファイルをインポートすると、カラム紐付け設定が表示されます。")

    # A-3. ストップワード管理
    with tab3:
        st.markdown("##### 分析から除外するストップワードを管理します。")
        
        # 初期化
        if 'stopwords' not in st.session_state:
            st.session_state['stopwords'] = utils.get_stopwords()
            
        if 'sw_version' not in st.session_state:
            st.session_state.sw_version = 0
        
        # 検索機能
        search_query = st.text_input("リスト内検索 (正規表現も可)", placeholder="検索したい単語を入力...", key="sw_search")
        
        # フィルタリング or 全量
        # 確実にリスト(set)として扱う
        if isinstance(st.session_state['stopwords'], list):
             st.session_state['stopwords'] = set(st.session_state['stopwords'])
             
        full_stopwords = sorted(list(st.session_state['stopwords']))
        
        if search_query:
            try:
                filtered_stopwords = [w for w in full_stopwords if re.search(search_query, w)]
            except re.error:
                filtered_stopwords = [w for w in full_stopwords if search_query in w]
            is_filtered = True
        else:
            filtered_stopwords = full_stopwords
            is_filtered = False
            
        stopwords_text = "\n".join(filtered_stopwords)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            label_suffix = f" (表示中: {len(filtered_stopwords)} / 全 {len(full_stopwords)} 語)" if is_filtered else f" (全 {len(full_stopwords)} 語)"
            if is_filtered:
                st.warning("⚠️ フィルタリング中: ここでの編集（追加・削除）は、表示されている単語に対して適用され、メインリストにマージされます。")
            
            # Keyにバージョンを含めて強制リフレッシュ
            editor_key = f"stopwords_editor_{hash(search_query)}_{st.session_state.sw_version}" 
            new_stopwords_text = st.text_area(f"ストップワードリスト{label_suffix}", value=stopwords_text, height=300, key=editor_key)
            
            if st.button("変更を適用", key="apply_stopwords"):
                edited_lines = set([line.strip() for line in new_stopwords_text.split('\n') if line.strip()])
                
                if is_filtered:
                    original_matches = set(filtered_stopwords)
                    removed_words = original_matches - edited_lines
                    added_words = edited_lines - original_matches
                    
                    current_set = st.session_state['stopwords']
                    new_set = (current_set - removed_words) | added_words
                    st.session_state['stopwords'] = new_set
                    msg = f"更新完了: {len(added_words)} 語を追加, {len(removed_words)} 語を削除しました。"
                else:
                    st.session_state['stopwords'] = edited_lines
                    msg = f"リストを全量更新しました (計 {len(edited_lines)} 語)。"
                
                st.session_state.sw_version += 1
                st.success(msg)
                st.rerun()

        with c2:
            st.markdown("**インポート / エクスポート**")
            
            # インポート
            sw_file = st.file_uploader("リストをインポート (.txt, .csv)", type=['txt', 'csv'], key="sw_uploader")
            if sw_file:
                try:
                    stringio = io.StringIO(sw_file.getvalue().decode("utf-8"))
                    imported_lines = [line.strip() for line in stringio.read().split('\n') if line.strip()]
                    if st.button(f"リストを置換してインポート ({len(imported_lines)}語)", key="import_sw"):

                        st.session_state['stopwords'] = set(imported_lines)
                        st.session_state.sw_version += 1
                        st.success("リストを置換しました。")
                        st.rerun()
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

            # エクスポート
            st.download_button(
                label="リストをエクスポート (.txt)",
                data="\n".join(sorted(list(st.session_state['stopwords']))),
                file_name="apollo_stopwords.txt",
                mime="text/plain"
            )
            
            st.markdown("---")
            if st.button("デフォルトに戻す", key="reset_stopwords"):
                st.session_state['stopwords'] = utils.get_stopwords()
                st.session_state.sw_version += 1
                st.rerun()

    # A-4. 前処理実行
    with tab4:
        st.markdown("##### 全モジュール共通の分析エンジンを起動します。")
        st.write("データ量に応じて数分かかる場合があります。")

        if st.button("分析エンジン起動 (SBERT/TF-IDF)", type="primary", key="run_preprocess"):
            required_cols = ['title', 'abstract', 'claim', 'app_num', 'date', 'applicant', 'ipc']
            
            if st.session_state.df_main is None:
                st.error("フェーズ1でファイルをアップロードしてください。")
            elif any(v is None for k, v in st.session_state.col_map.items() if k in required_cols):
                missing = [k for k, v in st.session_state.col_map.items() if v is None and k in required_cols]
                st.error(f"エラー: フェーズ2の必須カラムが選択されていません: {missing}")
            else:
                try:
                    # 分析用NPLデータの同期（存在する場合）
                    if 'df_npl_accumulated' in st.session_state and not st.session_state.df_npl_accumulated.empty:
                        st.session_state.df_npl = st.session_state.df_npl_accumulated.copy()
                    
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    start_time = time.time()
                    
                    phases = {
                        'init': 0.05,
                        'text': 0.05,
                        'sbert': 0.70,
                        'tfidf': 0.10,
                        'norm': 0.08,
                        'clean': 0.02
                    }

                    def update_progress(phase_key, phase_progress=0.0):
                        cumulative = 0.0
                        for k, w in phases.items():
                            if k == phase_key:
                                cumulative += w * phase_progress
                                break
                            else:
                                cumulative += w
                        
                        total_progress = min(0.99, cumulative)
                        
                        elapsed = time.time() - start_time
                        if total_progress > 0.01:
                            estimated_total = elapsed / total_progress
                            remaining = estimated_total - elapsed
                            eta_str = f"{int(remaining // 60):02}:{int(remaining % 60):02}"
                        else:
                            eta_str = "--:--"
                            
                        elapsed_str = f"{int(elapsed // 60):02}:{int(elapsed % 60):02}"
                        
                        progress_bar.progress(total_progress)
                        return elapsed_str, eta_str

                    # 1. モデル読み込み (初期化) — patiroha.SBERTEmbedder
                    status_text.markdown("🔄 **Phase 1/6: モデルロード中...**")
                    update_progress('init', 0.5)

                    df = st.session_state.df_main.copy()
                    col_map = st.session_state.col_map
                    delimiters = st.session_state.delimiters

                    _embedder = load_sbert_embedder()  # patiroha経由でキャッシュ
                    st.session_state.sbert_model = _embedder  # 後方互換
                    update_progress('init', 1.0)
                    # 2. 特許データの前処理
                    status_text.markdown("🔄 **Phase 2/6: 特許データの前処理中...**")
                    df['data_type'] = 'Patent'
                    df['text_for_sbert'] = (
                        df[col_map['title']].fillna('') + ' ' +
                        df[col_map['abstract']].fillna('') + ' ' +
                        df[col_map['claim']].fillna('')
                    )
                    # 3. NPLデータ処理 (NPL個別処理)
                    status_text.markdown("🔄 **Phase 3/6: 特許以外の情報(NPL)の個別処理中...**")
                    if 'df_npl' in st.session_state and st.session_state.df_npl is not None:
                        df_n = st.session_state.df_npl.copy()
                        df_n['data_type'] = 'NPL'

                        # 統合カラムマッピング適用
                        n_title = df_n['unified_title'].fillna('')
                        n_content = df_n['unified_content'].fillna('')

                        df_n[col_map['title']] = n_title
                        df_n[col_map['date']] = df_n['unified_date']
                        df_n[col_map['applicant']] = df_n['unified_source'].fillna('N/A')
                        df_n['region'] = df_n['unified_region'].fillna('Global')
                        df_n['status'] = df_n['unified_status'].fillna('Unknown')
                        df_n[col_map['abstract']] = n_content
                        df_n[col_map['app_num']] = 'NPL-' + df_n.index.astype(str)

                        # 日付パース — patiroha.parse_date
                        df_n['parsed_date'] = patiroha.parse_date(df_n[col_map['date']])
                        df_n['year'] = df_n['parsed_date'].dt.year

                        # NaTの場合は正規表現で年を抽出
                        mask_nat = df_n['parsed_date'].isna()
                        if mask_nat.any():
                            raw_dates_n = df_n[col_map['date']].astype(str)
                            extracted_years = raw_dates_n[mask_nat].str.extract(r'(\d{4})')[0]
                            df_n.loc[mask_nat, 'year'] = pd.to_numeric(extracted_years, errors='coerce')

                        # Academic/Newsのみキーワード抽出 — patiroha.extract_keywords
                        npl_sw = patiroha.get_stopwords("npl")

                        def process_npl_keywords(row):
                            sub_type = str(row.get('data_sub_type', ''))
                            if sub_type in ['Academic', 'Business', 'Academic Source', 'News Source']:
                                t_val = str(row['unified_title']) if pd.notna(row['unified_title']) else ""
                                c_val = str(row['unified_content']) if pd.notna(row['unified_content']) else ""
                                txt = t_val + " " + c_val
                                return utils.extract_keywords(txt, stopwords=npl_sw)
                            else:
                                return []

                        df_n['explorer_keywords'] = df_n.apply(process_npl_keywords, axis=1)
                        st.session_state.df_npl = df_n

                    update_progress('text', 1.0)

                    # 4. SBERTエンコード (Patent ONLY) — patiroha.SBERTEmbedder
                    status_text.markdown("🔄 **Phase 4/6: AIベクトル化 (SBERT - 特許のみ)...**")
                    embedder = load_sbert_embedder()

                    def sbert_progress(frac):
                        el_str, et_str = update_progress('sbert', frac)
                        pct = int(frac * 100)
                        status_text.markdown(
                            f"🔄 **Phase 4/6: AIベクトル化 (SBERT) 実行中...** ({pct}%)\n\n"
                            f"⏱️ 経過: {el_str} | ⏳ 残り: {et_str} (目安)")

                    sbert_embeddings = embedder.encode(
                        df,
                        text_columns=[col_map['title'], col_map['abstract'], col_map['claim']],
                        batch_size=128,
                        normalize_embeddings=True,
                        progress_callback=sbert_progress,
                    )
                    st.session_state.sbert_embeddings = sbert_embeddings

                    # 5. TF-IDF & Keyword (Patent ONLY) — patiroha
                    status_text.markdown("🔄 **Phase 5/6: キーワード抽出 (TF-IDF - 特許のみ)...**")
                    current_sw = _get_current_stopwords()

                    # Explorer用キーワードリスト
                    df['explorer_keywords'] = df['text_for_sbert'].apply(
                        lambda x: utils.extract_keywords(x, stopwords=current_sw))

                    # TF-IDF行列 — patiroha.build_tfidf
                    tfidf_matrix, feature_names = patiroha.build_tfidf(
                        df['text_for_sbert'].tolist(),
                        stopwords=current_sw,
                        min_df=5, max_df=0.80)
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.feature_names = feature_names
                    update_progress('tfidf', 1.0)

                    # 6. メタデータ正規化 — patiroha
                    status_text.markdown("🔄 **Phase 6/6: メタデータ (日付・IPC・出願人) 正規化中...**")

                    # 日付 — patiroha.parse_date
                    df['parsed_date'] = patiroha.parse_date(df[col_map['date']])
                    df['year'] = df['parsed_date'].dt.year
                    df['app_num_main'] = df[col_map['app_num']].astype(str).str.strip()

                    # IPC — patiroha.extract_ipc
                    ipc_delimiter = delimiters['ipc']
                    df['ipc_normalized'] = df[col_map['ipc']].apply(
                        lambda x: patiroha.extract_ipc(x, delimiter=ipc_delimiter) if isinstance(x, str) else [])
                    ipc_raw_list = df[col_map['ipc']].fillna('').astype(str).str.split(ipc_delimiter)
                    df['ipc_main_group'] = ipc_raw_list.apply(
                        lambda terms: list(set([t.strip().split('/')[0].strip().upper() for t in terms if t.strip()])))

                    # Fターム
                    if col_map['fterm']:
                        fterm_delimiter = delimiters['fterm']
                        fterm_raw_list = df[col_map['fterm']].fillna('').astype(str).str.split(fterm_delimiter)
                        df['fterm_main'] = fterm_raw_list.apply(
                            lambda terms: list(set([t.strip()[:5].upper() for t in terms if t.strip() and len(t) >= 5])))
                    else:
                        df['fterm_main'] = [[] for _ in range(len(df))]

                    # 出願人 — patiroha.normalize_applicant
                    applicant_delimiter = delimiters['applicant']
                    df['applicant_main'] = df[col_map['applicant']].apply(
                        lambda x: patiroha.normalize_applicant(x, delimiter=applicant_delimiter) if isinstance(x, str) else [])

                    # 発明者
                    if col_map['inventor'] and col_map['inventor'] in df.columns:
                        inventor_delimiter = delimiters['inventor']
                        def clean_inventors(val):
                            if pd.isna(val): return []
                            val = str(val).replace('▲', '').replace('▼', '').replace('　', '')
                            return list(set([n.strip() for n in val.split(inventor_delimiter) if n.strip()]))
                        df['inventor_main'] = df[col_map['inventor']].apply(clean_inventors)
                    else:
                        df['inventor_main'] = [[] for _ in range(len(df))]
                    update_progress('norm', 1.0)

                    # クリーンアップ
                    status_text.markdown("🔄 **Phase 6/6: 最終処理中...**")
                    df.drop(columns=['text_for_sbert'], errors='ignore', inplace=True)
                    st.session_state.df_main = df 
                    st.session_state.shared_df = df 
                    st.session_state.preprocess_done = True
                    update_progress('clean', 1.0)
                    
                    # 完了
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ 分析エンジン起動完了 (所要時間: {int(time.time() - start_time)}秒)")
                    st.info("サイドバーのナビゲーションから分析モジュールを選択し、ミッションを開始してください。")

                except Exception as e:
                    st.error(f"前処理中にエラーが発生しました: {e}")
                    import traceback
                    st.exception(traceback.format_exc())

        # --- CAPCOM セッション管理 ---
        st.markdown("---")

        if capcom.is_active():
            session_id = capcom.get_session_id()
            # session_stateベースのテレメトリ（ファイルI/Oなし）
            _tel = capcom.get_telemetry()
            snap_n, prompt_n, data_n = _tel['snapshots'], _tel['prompts'], _tel['data']

            st.markdown(f"""<div style="background: linear-gradient(135deg, #f0f4f8 0%, #e4ecf4 50%, #dce6f0 100%);
border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
border: 1px solid #003366;">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
<div style="display: flex; align-items: center; gap: 10px;">
<span style="font-size: 26px;">📡</span>
<div>
<div style="color: #003366; font-size: 22px; font-weight: 800; letter-spacing: 3px;">CAPCOM</div>
<div style="color: #455A64; font-size: 13px; letter-spacing: 1px; margin-top: 2px;">CAPSULE COMMUNICATOR</div>
</div>
</div>
<div style="display: flex; align-items: center; gap: 6px;
background: rgba(0,80,30,0.85); border-radius: 20px;
padding: 5px 14px; border: 1px solid rgba(0,200,83,0.6);">
<span style="display: inline-block; width: 8px; height: 8px;
background: #69F0AE; border-radius: 50%;
box-shadow: 0 0 6px #69F0AE;
animation: capcom-blink 2s ease-in-out infinite;"></span>
<span style="color: #fff; font-size: 13px; font-weight: 700;">ONLINE</span>
</div>
</div>
<div style="background: rgba(0,51,102,0.06); border-radius: 8px;
padding: 10px 14px; margin-bottom: 12px;
font-family: 'SF Mono', 'Consolas', 'Courier New', monospace;">
<div style="color: #37474F; font-size: 12px; font-weight: 600; margin-bottom: 4px;">SESSION ID</div>
<div style="color: #263238; font-size: 15px;">{session_id}</div>
</div>
<div style="display: flex; gap: 12px;">
<div style="flex: 1; background: rgba(0,51,102,0.05); border-radius: 8px;
padding: 10px 12px; text-align: center;">
<div style="color: #1565C0; font-size: 24px; font-weight: 700;">{snap_n}</div>
<div style="color: #37474F; font-size: 12px; font-weight: 600; letter-spacing: 1px;">SNAPSHOTS</div>
</div>
<div style="flex: 1; background: rgba(0,51,102,0.05); border-radius: 8px;
padding: 10px 12px; text-align: center;">
<div style="color: #E65100; font-size: 24px; font-weight: 700;">{prompt_n}</div>
<div style="color: #37474F; font-size: 12px; font-weight: 600; letter-spacing: 1px;">PROMPTS</div>
</div>
<div style="flex: 1; background: rgba(0,51,102,0.05); border-radius: 8px;
padding: 10px 12px; text-align: center;">
<div style="color: #2E7D32; font-size: 24px; font-weight: 700;">{data_n}</div>
<div style="color: #37474F; font-size: 12px; font-weight: 600; letter-spacing: 1px;">DATA</div>
</div>
</div>
</div>
<style>@keyframes capcom-blink {{
0%, 100% {{ opacity: 1; }}
50% {{ opacity: 0.3; }}
}}</style>""", unsafe_allow_html=True)
            # ZIPダウンロード方式のためパス表示は不要

        else:
            # CAPCOM 待機状態
            st.markdown("""<div style="background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
border: 1px solid #ddd;">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
<div style="display: flex; align-items: center; gap: 10px;">
<span style="font-size: 26px; opacity: 0.4;">📡</span>
<div>
<div style="color: #78909C; font-size: 22px; font-weight: 800; letter-spacing: 3px;">CAPCOM</div>
<div style="color: #90A4AE; font-size: 13px; letter-spacing: 1px; margin-top: 2px;">CAPSULE COMMUNICATOR</div>
</div>
</div>
<div style="display: flex; align-items: center; gap: 6px;
background: rgba(0,0,0,0.03); border-radius: 20px;
padding: 5px 14px; border: 1px solid #ccc;">
<span style="display: inline-block; width: 8px; height: 8px; background: #bbb; border-radius: 50%;"></span>
<span style="color: #777; font-size: 13px; font-weight: 700;">STANDBY</span>
</div>
</div>
<div style="color: #607D8B; font-size: 14px; line-height: 1.6;">
分析結果をファイル出力し、Claude Code から読み取り可能にします。<br/>
分析エンジン起動後にセッションを開始できます。
</div>
</div>""", unsafe_allow_html=True)

            if st.session_state.get('preprocess_done', False):
                if st.button("📡 CAPCOMセッション開始", type="primary", key="start_capcom"):
                    session_id, _ = capcom.init_session()
                    st.success(f"📡 CAPCOM セッション開始: `{session_id}` (In-Memory)")
                    st.rerun()