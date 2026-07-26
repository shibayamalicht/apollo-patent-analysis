# ==================================================================
# APOLLO — Mission Control（母集団設計・データ準備）
# 旧 Home.py（Mission Control）の移植。
# データ取込 → カラム紐付け → 除外ワード → 外部データ(NPL)取込（任意）→
# 前処理（6フェーズ）→ 分析の記録セッション(CAPCOM) を担う母港。
# session_state のキー名・前処理ロジックは旧版から変更しない。
# （セッション初期化 initialize_session_state は app.py に移設済み）
# ==================================================================

# --- 環境設定（import時に一度だけ実行） ---
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['OMP_NUM_THREADS'] = '1'

# --- ライブラリ ---
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
import io

import patiroha

warnings.filterwarnings('ignore')

import utils
import capcom
import apollo_ui
import flight_recorder

# ==================================================================
# --- patiroha統合: SBERTモデル・ストップワード ---
# ==================================================================

def load_sbert_embedder(model_name=None):
    """文埋め込みモデルを取得する（実体は utils 側でモデル名ごとに 1 度だけ構築・共有）。

    モデル名は「詳細設定」のラジオ選択（st.session_state['sbert_model_name']）を渡す。
    接頭辞・GPU(CUDA/MPS)選択は utils.build_sbert_embedder に集約。
    MPSが不安定な場合は環境変数 APOLLO_FORCE_CPU=1 でCPU強制可能。
    """
    return utils.load_sbert_embedder(model_name or st.session_state.get('sbert_model_name'))


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
# --- ミッション進行ステッパー ---
# ==================================================================

def _render_mission_stepper(area):
    """ミッション進行ステッパー + データの現在地。

    タブ内の処理（アップロード・前処理）を同じ実行内で反映した最新の
    session_state で描画するため、render() 冒頭で枠（area）だけ確保しておき、
    タブ処理が終わった render() 末尾からこの関数で描き込む。
    """
    df_main = st.session_state.get("df_main")
    preprocess_done = bool(st.session_state.get("preprocess_done", False))
    snapshots = st.session_state.get("snapshots", []) or []
    n_snap = len(snapshots)
    pages = st.session_state.get("ap_pages", {})

    with area:
        with st.container(border=True):
            c_step, c_now = st.columns([7, 5], gap="large")

            with c_step:
                # Step 1: データ投入
                if df_main is not None:
                    fname = st.session_state.get("filename", "No File")
                    apollo_ui.step("done", "1", "データ投入",
                                     f"{fname}（{len(df_main):,}件）読込済み")
                else:
                    apollo_ui.step("now", "1", "データ投入",
                                     "「1. データ取込」タブで特許リスト（CSV/Excel）を読み込みます")

                # Step 2: 前処理
                if preprocess_done:
                    apollo_ui.step("done", "2", "前処理", "完了 — 全モジュールが利用可能です")
                else:
                    _s2 = "now" if df_main is not None else "todo"
                    apollo_ui.step(_s2, "2", "前処理",
                                     "「5. 前処理」タブから実行します")

                # Step 3: 分析とスナップショットの収集
                if n_snap > 0:
                    apollo_ui.step("done", "3", "分析とスナップショットの収集",
                                     f"スナップショット {n_snap} 枚 保存済み — 収集を終えたら CAPCOM へ")
                else:
                    _s3 = "now" if preprocess_done else "todo"
                    apollo_ui.step(_s3, "3", "分析とスナップショットの収集",
                                     "各分析モジュールの「レポート用に保存」で図を収集します")

                # Step 4: レポートを作る（案内のみ）
                _s4 = "now" if n_snap > 0 else "todo"
                apollo_ui.step(_s4, "4", "レポートを作る",
                                 "集めたスナップショットからレポートを生成します")
                if "capcom" in pages:
                    st.page_link(pages["capcom"], label="レポート生成を開く",
                                 icon=":material/menu_book:")

            with c_now:
                st.markdown("##### データの現在地")
                n_app = None
                if df_main is not None and "applicant_main" in getattr(df_main, "columns", []):
                    try:
                        n_app = int(df_main["applicant_main"].explode().dropna().nunique())
                    except Exception:
                        n_app = None
                st.metric("読込データ（件）", f"{len(df_main):,}" if df_main is not None else "—")
                st.metric("出願人（ユニーク数）", f"{n_app:,}" if n_app is not None else "—")
                st.metric("スナップショット（枚）", n_snap)

        # 次のおすすめ: 前処理済みでまだスナップショットが1枚もないとき
        if preprocess_done and n_snap == 0 and "atlas" in pages:
            with st.container(border=True):
                st.markdown("**:material/explore: 次の推奨手順** — まず母集団の全体像を把握します")
                st.page_link(pages["atlas"], label="基本統計を開く",
                             icon=":material/query_stats:")


# ==================================================================
# --- メイン描画 ---
# ==================================================================

def _process_npl_data(col_map):
    """外部データ(NPL)を分析用に整える処理。
    「前処理」の Phase 3 と、外部データ取込タブの「いますぐ反映」ボタンの両方から呼ばれる。
    特許側の重い計算（SBERT等）は含まないため単独実行は高速。
    処理内容は旧 Home.py の Phase 3 と同一（ロジック無改変）。
    """
    if 'df_npl' not in st.session_state or st.session_state.df_npl is None:
        return
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
    if col_map.get('abstract'):  # 要約列がないデータ（J-PlatPat要約なし等）では本文の書き込み先がない
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
    # year 列の型を確定（NaT補完で object に退行するのを防ぎ、下流の年次/CAGR処理を安定化）
    df_n['year'] = pd.to_numeric(df_n['year'], errors='coerce').astype('Int64')

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


def render():
    apollo_ui.page_header(
        "母集団設計・データ準備", "Mission Control",
        "母集団の設計・取込・カラム紐付け・前処理。分析の前提条件をここで確定します。",
    )

    # ミッション進行ステッパーの枠を先に確保（描画はタブ処理後に行い、最新状態を反映する）
    _stepper_area = st.container()

    # タブ順は作業の流れ順: 母集団設計 → 取込 → 紐付け → 除外 → 外部データ（任意）→ 前処理
    # ※ コード上のブロック順（前処理 → 外部データ）は移動していない。表示順はこのタブ定義に従う。
    tab_design, tab1, tab2, tab3, tab_npl, tab_prep = st.tabs([
        "0. 母集団設計",
        "1. データ取込",
        "2. カラム紐付け",
        "3. 除外ワード（任意）",
        "4. 外部データ取込（任意）",
        "5. 前処理",
    ])

    # ==============================================================
    # タブ0: 母集団設計 — 何を、どう切り出した集合なのかを最初に記録する。
    # ここが空でも分析は動くが、レポートの前提条件と付録が薄くなる。
    # 入力値の session_state キーは ZIP スキーマ互換のため旧名のまま。
    # ==============================================================
    with tab_design:
        st.markdown("##### この母集団が何を切り出したものかを記録します（すべて任意）。")
        st.caption(
            "記録した内容はレポート生成 AI に渡り、**分析の前提条件としてレポートに反映**されます。"
            "論理式は付録に、設計意図・収録年・データベース名は本文と付録の両方に組み込まれます。")

        _q_intent = st.text_area(
            ":material/target: 母集団論理式の設計意図",
            value=st.session_state.get('capcom_query_intent', ''),
            height=100,
            placeholder=(
                "例: 太陽光発電と蓄電を組み合わせた技術の母集団。"
                "「太陽光発電」「蓄電」の概念ごとに、同じことを指すキーワードと分類コードを"
                "ひとまとめの集合として定義し（語で書かれていない出願を分類で拾うため）、"
                "その2つの集合を掛け合わせて抽出した。"
            ),
            key="capcom_query_intent_input",
            help="分析レポートの冒頭・前提条件欄・付録に反映されます。",
        )
        st.session_state['capcom_query_intent'] = _q_intent

        _q_logic = st.text_area(
            ":material/search: 母集団論理式",
            value=st.session_state.get('capcom_query_logic', ''),
            height=100,
            # 概念ブロック法をそのまま見せる。1つの概念を指すものは キーワードも分類コードも
            # まとめて + (OR) で1つの集合にし、異なる概念の集合どうしを * (AND) で掛け合わせる。
            # 分類を「別概念」として AND で足すと、語で書かれていない出願を拾えなくなる。
            # 書式は J-PlatPat の論理式入力（* = AND / + = OR、検索項目は語末の / タグ。
            # /(ab+ti+cl) のように複数項目をまとめて指定できる）。特許庁が公表している
            # GXTI（GX技術区分表）の検索式が同じ形式なので、実例はそちらを参照するとよい。
            # DB 名を冒頭に置くのは、CORE のルール分類も + と * を使うため、
            # 利用者がこの欄を APOLLO 独自文法だと誤解しないようにするため。
            placeholder=(
                "例（J-PlatPat 形式）: "
                "[(太陽光発電+太陽電池)/(ab+ti+cl)+H01L31/ip+H02S/ip]"
                "*[(蓄電池+二次電池+蓄電システム)/(ab+ti+cl)+H01M10/ip+H02J7/ip]"
            ),
            key="capcom_query_logic_input",
            help="**レポート付録に全文掲載**されます。機密情報は含めないでください。",
        )
        st.session_state['capcom_query_logic'] = _q_logic

        _c_cov, _c_db = st.columns(2)
        with _c_cov:
            _cov = st.text_input(
                ":material/calendar_month: 収録年情報",
                value=st.session_state.get('capcom_coverage_years', ''),
                placeholder="例: 収録開始〜2026-07-26（検索日）",
                key="capcom_coverage_years_input",
                help=(
                    "**検索日（データを取得した日）を必ず入れてください。** 直近の出願は公開まで最大18ヶ月かかるため、検索日が分かると「どの年から未公開で少なく見えるのか」をレポートが特定できます。開始を絞っていなければ「収録開始〜」で構いません。開始も指定した場合は「2015-01-01〜2026-07-26（出願日ベース）」のように書きます。"
                ),
            )
            st.session_state['capcom_coverage_years'] = _cov
        with _c_db:
            _dbname = st.text_input(
                ":material/database: 使用した特許データベース名",
                value=st.session_state.get('capcom_database_name', ''),
                placeholder="例: J-PlatPat / 社内特許DB / Derwent Innovation / PatSnap",
                key="capcom_database_name_input",
                help="付録および分析注記（カバレッジ制約）の記述に反映されます。",
            )
            st.session_state['capcom_database_name'] = _dbname

        # 入力は即座に飛行記録の層1（母集団の来歴）へ反映する
        flight_recorder.set_lineage(
            query_logic=_q_logic.strip() or None,
            coverage_years=_cov.strip() or None,
            database_name=_dbname.strip() or None,
            design_intent=_q_intent.strip() or None,
        )

        _n_filled = sum(1 for v in (_q_intent, _q_logic, _cov, _dbname) if str(v).strip())
        if _n_filled:
            st.success(f"母集団設計 {_n_filled}/4 項目を記録しました。", icon=":material/check_circle:")
        st.divider()
        flight_recorder.render_card(title="飛行記録 — いまの分析条件")

    # ==============================================================
    # タブ1: データ取込（特許リスト）
    # ==============================================================
    with tab1:
        st.markdown("##### 分析対象の特許リストを読み込みます。")
        st.caption("J-PlatPatの検索結果CSVはそのまま読み込めます（自動認識）。分割ダウンロードしたファイルは、まとめて選ぶと1つに結合されます。")
        uploaded_files = st.file_uploader(
            "分析ファイルをアップロード (CSV または Excel・複数選択可)",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="main_file_uploader",
            accept_multiple_files=True,
        )

        if uploaded_files:
            _display_name = (uploaded_files[0].name if len(uploaded_files) == 1
                             else f"{uploaded_files[0].name} ほか{len(uploaded_files) - 1}ファイル")
            # 前処理完了後のrerunでは再読み込みをスキップ（新ファイルの場合のみ実行）
            is_new_file = (_display_name != st.session_state.get('filename', ''))
            if is_new_file or not st.session_state.get('preprocess_done', False):
                try:
                    _dfs, _names = [], []
                    for _uf in uploaded_files:
                        if _uf.name.lower().endswith('.csv'):
                            try:
                                _d = pd.read_csv(_uf, dtype=str)
                            except UnicodeDecodeError:
                                _uf.seek(0)
                                _d = pd.read_csv(_uf, dtype=str, encoding='shift_jis')
                        else:
                            _d = pd.read_excel(_uf, dtype=str)
                        _dfs.append(_d)
                        _names.append(_uf.name)

                    # 複数ファイルは縦結合（同一カラム構成のみ・重複行は除去）
                    _merge_info = None
                    if len(_dfs) == 1:
                        df = _dfs[0]
                        df.columns = utils.clean_header_columns(df.columns)
                    else:
                        df, _merge_info = utils.merge_patent_files(_dfs, _names)

                        # 母集団の来歴を飛行記録へ（層1: レポート本文で使える情報）
                        if df is not None:
                            flight_recorder.set_lineage(
                                source_files=[f"{n}（{len(d):,}件）" for n, d in zip(_names, _dfs)],
                                rows_loaded=int(sum(len(d) for d in _dfs)),
                                rows_merged=int(len(df)),
                                rows_deduped=int(_merge_info.get('duplicates_removed', 0)
                                                 if isinstance(_merge_info, dict) else 0),
                            )

                    if df is None:
                        st.error(_merge_info['error'])
                        st.session_state.df_main = None
                        st.session_state.shared_df = None
                    else:
                        # J-PlatPat形式の自動認識 → 正規化+カラム紐付け・区切り文字のプリセット
                        _is_jplat = utils.is_jplatpat_columns(df.columns)
                        _has_abstract = True
                        if _is_jplat:
                            df = utils.prepare_jplat_df(df)
                            _has_abstract = '要約' in df.columns
                            # プリセットは新しいファイルの初回だけ行う（このブロック自体は前処理
                            # 完了まで毎ラン再実行されるため、毎回リセットすると「2. カラム紐付け」
                            # でのユーザーの手動変更が消えてしまう）
                            if is_new_file:
                                st.session_state.col_map = {
                                    **utils.JPLAT_COL_MAP,
                                    'abstract': '要約' if _has_abstract else None,
                                }
                                st.session_state.delimiters = dict(utils.JPLAT_DELIMITERS)
                                # 「2. カラム紐付け」のウィジェットに残っている前のデータの選択値を消す。
                                # プリセットは col_map / delimiters から index・value 経由で初期化させる
                                # （ウィジェットキーへ直接代入すると default との二重設定の警告が出るため）
                                for _wk in ('col_title', 'col_abstract', 'col_claim', 'col_app_num',
                                            'col_date', 'col_applicant', 'col_ipc', 'col_fi', 'code_pref',
                                            'col_pub_number', 'col_inventor', 'col_fterm', 'col_status',
                                            'col_doc_url',
                                            'del_applicant', 'del_inventor', 'del_ipc', 'del_fterm'):
                                    st.session_state.pop(_wk, None)
                        elif is_new_file and st.session_state.get('jplat_mode'):
                            # 直前がJ-PlatPatだった場合、自動プリセットの区切り文字（出願人','等）が
                            # 次の通常データに残ったまま誤分割しないよう既定へ戻す
                            st.session_state.delimiters = {
                                'applicant': ';', 'inventor': ';', 'ipc': ';', 'fterm': ';', 'npl_category': ';'}
                            for _wk in ('del_applicant', 'del_inventor', 'del_ipc', 'del_fterm'):
                                st.session_state.pop(_wk, None)
                        st.session_state['jplat_mode'] = _is_jplat

                        st.session_state.df_main = df
                        st.session_state.preprocess_done = False
                        st.session_state['shared_df'] = df
                        st.session_state['filename'] = _display_name

                        if _merge_info:
                            st.success(
                                f"{_merge_info['n_files']}ファイルを結合しました: "
                                f"計 {_merge_info['n_rows_total']:,} 行 → 重複 {_merge_info['n_dup_removed']:,} 行を除いて "
                                f"**{_merge_info['n_rows_final']:,} 行** をインポート。")
                            with st.expander("結合の内訳", expanded=False):
                                for _n, _r in _merge_info['per_file']:
                                    st.caption(f"・{_n}: {_r:,} 行")
                        else:
                            st.success(f"ファイル '{_names[0]}' のインポート完了 ({len(df)}行)。")

                        if _is_jplat and _has_abstract:
                            st.info(
                                ":material/public: **J-PlatPat形式を検出しました。** カラムの紐付けと区切り文字を自動設定済みです。"
                                "このまま「5. 前処理」に進めます。技術分類はFIを使用します。")
                        elif _is_jplat:
                            st.info(
                                ":material/public: **J-PlatPat形式（要約なし）を検出しました。** カラムの紐付けは自動設定済みです。")
                            st.warning(
                                "**このファイルには要約がありません。** 内容の分析は発明の名称だけで行うため、"
                                "俯瞰図・キーワード分析の精度は下がります。ルール分類は、名称に加えて"
                                "FI（技術分類コード）での分類がおすすめです。"
                                "要約を使いたい場合は、J-PlatPatで「要約」付きのCSV（一度に500件まで）を"
                                "分割ダウンロードし、まとめて読み込み直してください。")
                        if _is_jplat:
                            st.caption(
                                ":material/info: FIはカンマ区切りですが、FI記号の内部にもカンマが使われるため（例: `H02J3/00,170` で1つのFI）、"
                                "取込時にFI同士の区切りをセミコロン（;）へ自動整理しています。区切り文字の設定は変更不要です。")
                        st.dataframe(df.head())

                except Exception as e:
                    st.error(f"ファイルインポートエラー: {e}")
                    st.session_state.df_main = None
                    st.session_state.shared_df = None
            else:
                # 前処理済みデータが存在する場合はスキップし、既存データ情報を表示
                st.success(f"ファイル '{_display_name}' はインポート・前処理済みです ({len(st.session_state.df_main)}行)。")
                st.dataframe(st.session_state.df_main.head())

        # --- サンプルデータ（初回体験用） ---
        st.markdown("")
        with st.container(border=True):
            _c_l, _c_r = st.columns([2.4, 1], vertical_alignment="center")
            with _c_l:
                st.markdown("**:material/science: まずは試してみたい方へ** — 同梱の架空サンプルデータ（全固体電池・960件）で全機能を体験できます。")
                st.caption(
                    ":material/warning: すべて架空の合成データです。特許・出願人（蒼海自動車 等）・発明者名を含め、"
                    "実在の特許・企業・人物とは一切関係ありません。"
                    "カラム紐付けまで自動設定されるため、読み込み後は「5. 前処理」を実行します。")
            with _c_r:
                if st.button("サンプルデータで試す", key="load_sample_btn", use_container_width=True):
                    try:
                        _sample_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sample_data", "sample_patents.csv")
                        _df_sample = pd.read_csv(_sample_path, dtype=str)
                        st.session_state.df_main = _df_sample
                        st.session_state['shared_df'] = _df_sample
                        st.session_state['filename'] = "架空サンプル（全固体電池）"
                        st.session_state.preprocess_done = False
                        st.session_state.col_map = {
                            'title': '発明の名称', 'abstract': '要約', 'claim': '請求の範囲',
                            'app_num': '出願番号', 'date': '出願日', 'applicant': '出願人',
                            'ipc': 'IPC', 'inventor': '発明者', 'status': 'ステータス',
                        }
                        # 直前にJ-PlatPat等を読み込んでいた場合、その区切り文字（出願人','等）が
                        # 残るとサンプルの出願人・発明者が正しく分割されないため既定へ戻す
                        st.session_state.delimiters = {
                            'applicant': ';', 'inventor': ';', 'ipc': ';', 'fterm': ';', 'npl_category': ';'}
                        for _wk in ('col_title', 'col_abstract', 'col_claim', 'col_app_num',
                                    'col_date', 'col_applicant', 'col_ipc', 'col_fi', 'code_pref',
                                    'col_pub_number', 'col_inventor', 'col_fterm', 'col_status',
                                    'col_doc_url',
                                    'del_applicant', 'del_inventor', 'del_ipc', 'del_fterm'):
                            st.session_state.pop(_wk, None)
                        st.session_state['jplat_mode'] = False
                        st.toast("サンプルデータを読み込みました。「5. 前処理」へ進んでください", icon=":material/science:")
                        st.rerun()
                    except Exception as _e_sample:
                        st.error(f"サンプルデータの読み込みに失敗しました: {_e_sample}")

    # ==============================================================
    # タブ2: カラム紐付け
    # ==============================================================
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
            kw_applicant = ['出願人', '出願人/権利者', 'Applicant', 'Assignee']
            kw_inventor = ['発明者', 'Inventor']
            kw_ipc = ['国際特許分類', '国際特許分類(IPC)', 'IPC', 'Int. Cl']
            kw_fi = ['FI', 'ＦＩ', 'FI記号']
            kw_fterm = ['Fターム', 'テーマコード', 'F-Term']

            col_map = {}
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### テキスト項目")
                col_map['title'] = st.selectbox("発明の名称（必須）:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('title'), columns_with_none, kw_title), key="col_title",
                    help="発明のタイトル列を指定します。要約・請求項と合わせてクラスタリングやキーワード抽出など意味解析の本文として使われます。正しく割り当てると技術内容の分類精度が上がります。")
                col_map['abstract'] = st.selectbox("要約（推奨）:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('abstract'), columns_with_none, kw_abstract), key="col_abstract",
                    help="特許の要約（抄録）列を指定します。クラスタリングやキーワード抽出など意味解析に使う主要な本文列です。無くても分析はできますが、名称だけの分析になるため俯瞰図やキーワード分析の精度は下がります。")
                col_map['claim'] = st.selectbox("請求項（任意）:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('claim'), columns_with_none, kw_claim), key="col_claim",
                    help="特許の請求項（クレーム）列を指定します。タイトル・要約と合わせて意味解析に使われ、権利範囲を含む技術内容の理解に役立ちます。無いデータ（J-PlatPatのCSV等）でも分析できます。")
            with col2:
                st.markdown("##### 必須メタデータ項目")
                col_map['app_num'] = st.selectbox("出願番号:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('app_num'), columns_with_none, kw_app_num), key="col_app_num",
                    help="各特許を一意に識別する出願番号の列です。件数集計や個別特許の特定に使われます。")
                col_map['date'] = st.selectbox("出願日:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('date'), columns_with_none, kw_date), key="col_date",
                    help="出願トレンドや成長率(CAGR)・年次分析に使う日付列です。正しく割り当てると件数推移やライフサイクル分析が正確になります。")
                col_map['applicant'] = st.selectbox("出願人:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('applicant'), columns_with_none, kw_applicant), key="col_applicant",
                    help="出願人ランキングや競争構造の分析に使う列を指定します。正しく割り当てると上位出願人・集中度（HHI）等が正確に出ます。")
                applicant_delimiter = st.text_input("出願人区切り文字:", value=st.session_state.delimiters.get('applicant', ';'), key="del_applicant",
                    help="1セルに複数の出願人が入っている場合の区切り文字です。正しく指定すると共同出願の各社を個別に集計できます。")

                # 分類コード: IPC / FI（どちらか一方でも両方でも。両方あれば分析に使う方を選ぶ）
                col_map['ipc'] = st.selectbox("IPC 国際特許分類（推奨）:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('ipc'), columns_with_none, kw_ipc), key="col_ipc",
                    help="国際特許分類（IPC）の列です。技術分野別の分析・多様性指標・レポートなどに使われます。")
                col_map['fi'] = st.selectbox("FI 特許分類（任意）:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('fi'), columns_with_none, kw_fi), key="col_fi",
                    help="FI（IPCを細展開した日本独自の特許分類）の列です。J-PlatPatのCSVでは自動で割り当てられます。")
                if col_map['ipc'] and col_map['fi']:
                    st.radio("分析に使う分類コード:", ['IPC', 'FI'], key="code_pref", horizontal=True,
                             help="技術分野別の分析（基本統計・動態分析・レポート等）にどちらの分類を使うかを選びます。"
                                  "ルール分類の検索では、割り当てた両方の列を使えます。変更後は「5. 前処理」を再実行してください。")
                elif col_map['fi']:
                    st.caption("分析には FI を使用します。")
                ipc_delimiter = st.text_input("分類コード区切り文字:", value=st.session_state.delimiters.get('ipc', ';'), key="del_ipc",
                    help="1セルに複数の分類コードが入っている場合の区切り文字です（IPC・FI共通）。"
                         "※J-PlatPatのFIはカンマ区切りですが、FI記号の内部にもカンマが使われるため、"
                         "取込時にセミコロン（;）区切りへ自動変換しています。J-PlatPatデータではこの欄は「;」のままが正しい設定です。")

            with col3:
                st.markdown("##### 任意メタデータ項目")

                # 公開番号 (CAPCOM用)
                kw_pub_num = ['公開番号', '公報番号', 'Publication Number', 'Pub No', 'Document Number']
                col_map['pub_number'] = st.selectbox("公開番号 (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('pub_number'), columns_with_none, kw_pub_num), key="col_pub_number",
                    help="公開公報の番号列です。指定すると本格レポートで代表特許を公開番号付きで参照できます。")

                # 発明者
                col_map['inventor'] = st.selectbox("発明者 (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('inventor'), columns_with_none, kw_inventor), key="col_inventor",
                    help="発明者名の列です。指定すると発明者ランキングやネットワーク分析（共同発明関係）が利用できます。")
                inventor_delimiter = st.text_input("発明者区切り文字:", value=st.session_state.delimiters.get('inventor', ';'), key="del_inventor",
                    help="1セルに複数の発明者が入っている場合の区切り文字です。正しく指定すると各発明者を個別に集計できます。")

                # Fターム
                col_map['fterm'] = st.selectbox("Fターム (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('fterm'), columns_with_none, kw_fterm), key="col_fterm",
                    help="日本独自の技術分類「Fターム」の列です。指定するとIPCに加えてより細かい観点での技術分野分析が可能になります。")
                fterm_delimiter = st.text_input("Fターム区切り文字:", value=st.session_state.delimiters.get('fterm', ';'), key="del_fterm",
                    help="1セルに複数のFタームが入っている場合の区切り文字です。正しく指定すると複数の分類を個別に集計できます。")

                # ステータス
                col_map['status'] = st.selectbox("ステータス (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('status'), columns_with_none, ['ステータス', 'ステージ', 'Status', 'Legal Status', '法的状態']), key="col_status",
                    help="権利化状況（登録・拒絶・係属など）の列です。指定すると権利化率分析やステータス内訳の表示が利用できます。")

                # 文献URL（原文リンク）
                col_map['doc_url'] = st.selectbox("文献URL (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('doc_url'), columns_with_none, ['文献URL', 'URL', 'Link', 'リンク']), key="col_doc_url",
                    help="各特許の原文ページへのリンク列です。指定すると、マップの点クリックなどの詳細カードに「原文を開く」リンクが表示されます。J-PlatPatのCSVでは文献URL列が自動で割り当てられます。")

            st.session_state.col_map = col_map
            st.session_state.delimiters = {
                'applicant': applicant_delimiter,
                'inventor': inventor_delimiter,
                'ipc': ipc_delimiter,
                'fterm': fterm_delimiter
            }
        else:
            st.info("「1. データ取込」タブでファイルをインポートすると、カラム紐付け設定が表示されます。")

    # ==============================================================
    # タブ3: 除外ワード（ストップワード管理・任意）
    # ==============================================================
    with tab3:
        st.caption("通常は既定のままで問題ありません。定型語の除去が必要な場合に調整します。")
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
                st.warning(":material/warning: フィルタリング中: ここでの編集（追加・削除）は、表示されている単語に対して適用され、メインリストにマージされます。")

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

    # ==============================================================
    # タブ「5. 前処理」: 前処理の実行
    # ==============================================================
    with tab_prep:
        st.markdown("##### 特許データを分析可能な形式に変換します。")
        st.write("実行すると全分析モジュールが有効になります。所要時間はデータ量に比例し、1,000件で2分前後が目安です。")
        _n_npl_acc = len(st.session_state.get('df_npl_accumulated', pd.DataFrame()))
        if _n_npl_acc > 0:
            st.caption(f":material/download: 「4. 外部データ取込」で取り込んだ外部データ {_n_npl_acc:,} 件も、あわせて分析に反映されます。")

        # --- 詳細設定: 文埋め込み(SBERT)モデルの選択 ---
        # 高精度モデル(e5-base)は重く、環境によっては前処理が止まる/非常に遅いことがあるため、
        # 軽量モデル(MiniLM)も選べるようにする。既定は高速側（安全側）。
        # 注: expander の中身は折りたたみ中も毎回実行されるため、
        #     sbert_model_name の同期はこの位置のままで機能する。
        with st.expander("詳細設定（通常は変更不要）", expanded=False):
            _sbert_keys = list(utils.SBERT_MODELS.keys())
            _default_sbert_idx = _sbert_keys.index(utils.DEFAULT_SBERT_KEY)
            _sel_sbert_key = st.radio(
                "AI意味解析モデル（SBERT）",
                options=_sbert_keys,
                index=_default_sbert_idx,
                format_func=lambda k: utils.SBERT_MODELS[k]["label"],
                key="sbert_model_choice",
                help=(
                    "クラスタリングや代表特許抽出に使う意味ベクトル化モデルです。\n"
                    ":material/bolt:高速: 軽量で速い（前バージョンのモデル）。重くて処理が止まる場合はこちら。\n"
                    ":material/crisis_alert:高精度: 多言語E5で精度が上がりますが重く、初回はモデルのダウンロード(約440MB)が走ります。"
                ),
            )
            # 選択モデル名をセッションに保持（Mission Control=特許 / NEBULA=論文 / メタ表示で共有）
            st.session_state['sbert_model_name'] = utils.SBERT_MODELS[_sel_sbert_key]["name"]
            st.caption(f"使用モデル: `{st.session_state['sbert_model_name']}`（切り替え後は「前処理を実行」を再実行してください）")

        # 要約なしデータの注意（実行は可能。精度の期待値を先に示す）
        if st.session_state.df_main is not None and not st.session_state.col_map.get('abstract'):
            st.warning(
                "**要約カラムが割り当てられていません。** AIによる内容の分析（俯瞰図・キーワード分析など）は"
                "発明の名称だけで行うため、精度は下がります。要約列があるデータなら「2. カラム紐付け」で割り当ててください。")

        if st.button("前処理を実行", type="primary", key="run_preprocess"):
            # 要約・請求項・分類コードは「あれば使う」— J-PlatPat等の要約なしCSVでも動かす
            required_cols = ['title', 'app_num', 'date', 'applicant']
            # 内部キー名 → 日本語カラム名（エラー表示用）
            _col_jp = {
                'title': '発明の名称', 'app_num': '出願番号', 'date': '出願日', 'applicant': '出願人',
            }
            missing = [k for k in required_cols if st.session_state.col_map.get(k) is None]

            if st.session_state.df_main is None:
                st.error("「1. データ取込」タブでファイルをアップロードしてください。")
            elif missing:
                st.error(
                    "必須カラムが未選択です: **" + "・".join(_col_jp.get(k, k) for k in missing) + "** — "
                    "「2. カラム紐付け」タブで割り当ててから、もう一度実行してください。"
                )
            else:
                try:
                    # 分析用NPLデータの同期（存在する場合）
                    if 'df_npl_accumulated' in st.session_state and not st.session_state.df_npl_accumulated.empty:
                        st.session_state.df_npl = st.session_state.df_npl_accumulated.copy()

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    start_time = time.time()
                    _phase_secs = {}  # 各フェーズ実測秒（計測のみ・出力に影響なし）

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
                    status_text.markdown("**Phase 1/6: 初期化**")
                    update_progress('init', 0.5)

                    df = st.session_state.df_main.copy()
                    col_map = st.session_state.col_map
                    delimiters = st.session_state.delimiters

                    _embedder = load_sbert_embedder()  # patiroha経由でキャッシュ
                    st.session_state.sbert_model = _embedder  # 後方互換
                    # この前処理で実際に使用したモデルを確定記録（AIインサイト/CAPCOMのメタに反映）
                    st.session_state['sbert_model_used'] = (
                        st.session_state.get('sbert_model_name') or utils.SBERT_MODEL_NAME
                    )
                    update_progress('init', 1.0)
                    # 2. 特許データの前処理
                    status_text.markdown("**Phase 2/6: 特許データの正規化**")
                    df['data_type'] = 'Patent'
                    # 割り当てのあるテキスト列だけを結合（要約・請求項なしのデータでも動く）
                    df['text_for_sbert'] = utils.combine_text_columns(df, col_map)
                    # 3. NPLデータ処理 (NPL個別処理)
                    status_text.markdown("**Phase 3/6: 外部データ（NPL）の正規化**")
                    if 'df_npl' in st.session_state and st.session_state.df_npl is not None:
                        _t0_npl = time.time()
                        _process_npl_data(col_map)
                        _phase_secs['Phase3 外部データ整理'] = time.time() - _t0_npl

                    update_progress('text', 1.0)

                    # 4. SBERTエンコード (Patent ONLY) — patiroha.SBERTEmbedder
                    status_text.markdown("**Phase 4/6: 文埋め込みの生成（SBERT）— 最も時間を要します**")
                    embedder = load_sbert_embedder()

                    def sbert_progress(frac):
                        el_str, et_str = update_progress('sbert', frac)
                        pct = int(frac * 100)
                        status_text.markdown(
                            f"**Phase 4/6: 文埋め込みの生成（SBERT）** ({pct}%)\n\n"
                            f":material/timer: 経過: {el_str} | ⏳ 残り: {et_str} (目安)")

                    _t0 = time.time()
                    _sbert_cols = [col_map[k] for k in ('title', 'abstract', 'claim')
                                   if col_map.get(k) and col_map[k] in df.columns]
                    sbert_embeddings = embedder.encode(
                        df,
                        text_columns=_sbert_cols,
                        batch_size=128,
                        normalize_embeddings=True,
                        progress_callback=sbert_progress,
                    )
                    st.session_state.sbert_embeddings = sbert_embeddings
                    _phase_secs['Phase4 AI意味解析（SBERT）'] = time.time() - _t0

                    # 5. TF-IDF & Keyword (Patent ONLY) — patiroha
                    status_text.markdown("**Phase 5/6: 特徴語の抽出（TF-IDF）**")
                    current_sw = _get_current_stopwords()

                    # Explorer用キーワードリスト（マルチコアがあればプロセス並列で高速化／少コアは逐次）
                    _t0 = time.time()
                    df['explorer_keywords'] = pd.Series(
                        utils.extract_keywords_batch(df['text_for_sbert'].tolist(), stopwords=current_sw),
                        index=df.index)
                    _phase_secs['Phase5-1 特徴語抽出(複合名詞・Janome)'] = time.time() - _t0

                    # TF-IDF行列 — 上で抽出した複合名詞(explorer_keywords)を再利用して構築
                    # （Janome 二重トークナイズを回避。特徴語/ラベルは複合名詞ベースになる）
                    _t0 = time.time()
                    tfidf_matrix, feature_names = utils.build_tfidf_from_tokens(
                        df['explorer_keywords'].tolist(),
                        min_df=5, max_df=0.80)
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.feature_names = feature_names
                    _phase_secs['Phase5-2 TF-IDF行列(キーワード再利用)'] = time.time() - _t0
                    update_progress('tfidf', 1.0)

                    # 6. メタデータ正規化 — patiroha
                    status_text.markdown("**Phase 6/6: メタデータの正規化（日付・分類コード・出願人名）**")

                    # 日付 — patiroha.parse_date
                    _t0 = time.time()
                    df['parsed_date'] = patiroha.parse_date(df[col_map['date']])
                    df['year'] = df['parsed_date'].dt.year
                    df['app_num_main'] = df[col_map['app_num']].astype(str).str.strip()

                    # 分類コード — patiroha.extract_ipc（IPC/FI両方あれば「分析に使う分類コード」の
                    # 選択に従う。未割り当てなら空のまま進む）
                    _code_col = utils.classification_code_column(col_map)
                    if _code_col and _code_col in df.columns:
                        ipc_delimiter = delimiters['ipc']
                        df['ipc_normalized'] = df[_code_col].apply(
                            lambda x: patiroha.extract_ipc(x, delimiter=ipc_delimiter) if isinstance(x, str) else [])
                        ipc_raw_list = df[_code_col].fillna('').astype(str).str.split(ipc_delimiter)
                        # メイングループ抽出は「/」（主分類）に加えて「:」（FIのインデキシングコード）でも
                        # 区切る。H02J101:20 → H02J101 のように集約され、主分類と粒度が揃う
                        df['ipc_main_group'] = ipc_raw_list.apply(
                            lambda terms: list(set([re.split(r'[/:]', t.strip())[0].strip().upper() for t in terms if t.strip()])))
                    else:
                        df['ipc_normalized'] = [[] for _ in range(len(df))]
                        df['ipc_main_group'] = [[] for _ in range(len(df))]

                    # Fターム
                    if col_map['fterm']:
                        fterm_delimiter = delimiters['fterm']
                        fterm_raw_list = df[col_map['fterm']].fillna('').astype(str).str.split(fterm_delimiter)
                        df['fterm_main'] = fterm_raw_list.apply(
                            lambda terms: list(set([t.strip()[:5].upper() for t in terms if t.strip() and len(t) >= 5])))
                    else:
                        df['fterm_main'] = [[] for _ in range(len(df))]

                    # 出願人 — patiroha.normalize_applicant（外字記号 ▲▼ は正規化前に除去）
                    applicant_delimiter = delimiters['applicant']
                    df['applicant_main'] = df[col_map['applicant']].apply(
                        lambda x: patiroha.normalize_applicant(utils.strip_gaiji_marks(x), delimiter=applicant_delimiter) if isinstance(x, str) else [])

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
                    _phase_secs['Phase6 メタデータ正規化'] = time.time() - _t0
                    update_progress('norm', 1.0)

                    # クリーンアップ
                    status_text.markdown("**Phase 6/6: 後処理**")
                    df.drop(columns=['text_for_sbert'], errors='ignore', inplace=True)
                    st.session_state.df_main = df
                    st.session_state.shared_df = df
                    st.session_state.preprocess_done = True

                    # 母集団の性質と前処理の再現条件を飛行記録へ
                    try:
                        _sw_user = st.session_state.get("stopwords") or []
                        flight_recorder.set_lineage(
                            rows_final=int(len(df)),
                            stopword_count=len(_sw_user) or None,
                            abstract_available=bool(col_map.get('abstract')),
                            classification_code_label=(
                                utils.classification_code_label(col_map)
                                if utils.classification_code_column(col_map) else "未割当"),
                        )
                        flight_recorder.record(
                            "Mission Control", "前処理",
                            judgment={
                                "母集団件数": int(len(df)),
                                "要約の有無": bool(col_map.get('abstract')),
                                "除外語数": len(_sw_user),
                            },
                            repro={
                                "文埋め込みモデル": utils.current_sbert_model_info().get('name'),
                                "所要秒数": round(sum(_phase_secs.values()), 1),
                            })
                    except Exception:
                        pass  # 記録の失敗で前処理を止めない
                    # ルール分類の「分類コードも検索対象に含める」の既定をデータ内容から再評価
                    # （要約なしデータでは分類コード検索で名称だけの検索を補うため既定ON）
                    st.session_state['core_search_include_ipc'] = bool(
                        utils.classification_code_column(col_map) and not col_map.get('abstract'))
                    update_progress('clean', 1.0)

                    # 完了
                    progress_bar.progress(1.0)
                    _total_secs = time.time() - start_time
                    status_text.success(f":material/check_circle: 前処理が完了しました (所要時間: {int(_total_secs)}秒)")
                    # フェーズ別所要時間の内訳（ボトルネック特定用・計測のみでディスク永続化なし）
                    # 端末へは print せず、内訳はアプリ内の expander で確認できるようにする。
                    if _phase_secs:
                        with st.expander(":material/timer: フェーズ別所要時間（ボトルネック確認）", expanded=False):
                            for _k, _v in sorted(_phase_secs.items(), key=lambda kv: kv[1], reverse=True):
                                _pct = (_v / _total_secs * 100) if _total_secs > 0 else 0
                                st.write(f"- **{_k}**: {_v:.1f} 秒 ({_pct:.0f}%)")

                    # --- 分析の記録セッション(CAPCOM)の自動開始 ---
                    # 旧 Home.py の「:material/settings_input_antenna: CAPCOMセッション開始」ボタンと同じ処理を、
                    # 未開始（capcom_session_id が None）の場合のみ自動実行する。
                    # （手動開始は下の「分析の記録セッション（詳細）」expander に温存）
                    if st.session_state.get('capcom_session_id') is None:
                        _auto_session_id, _ = capcom.init_session()
                        st.toast("分析の記録セッションを開始しました（レポート用データを自動収集します）")

                    # --- 次のアクション誘導 ---
                    st.success("全分析モジュールが有効になりました。ATLAS（基本統計）から着手することを推奨します。")
                    _pages = st.session_state.get("ap_pages", {})
                    if "atlas" in _pages:
                        st.page_link(_pages["atlas"], label="基本統計を開く",
                                     icon=":material/query_stats:")

                except Exception as e:
                    st.error(f"前処理中にエラーが発生しました: {e}")
                    st.exception(traceback.format_exc())

        # --- 分析の記録セッション(CAPCOM)の詳細 ---
        st.markdown("---")

        with st.expander("分析の記録セッション（詳細）", expanded=False):
            st.caption("「前処理」の完了時に自動で開始されます。ここでは状態の確認と手動開始ができます。")

            if capcom.is_active():
                session_id = capcom.get_session_id()
                # session_stateベースのテレメトリ（ファイルI/Oなし）
                _tel = capcom.get_telemetry()
                snap_n, prompt_n, data_n = _tel['snapshots'], _tel['prompts'], _tel['data']

                st.markdown(f"""<div style="border: 1px solid #DEE5EC; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; background: #FBFCFE;">
<div style="font-size: 15px; font-weight: 700; color: #141C24;"><span class=msym>menu_book</span> 分析の記録セッション
<span style="color: #1A7A4A; font-size: 13px; margin-left: 8px;">● 記録中</span></div>
<div style="color: #5A6875; font-size: 12.5px; margin-top: 8px;">分析結果を自動で集めて、レポート生成の材料にしています。</div>
<div style="color: #5A6875; font-size: 12.5px; margin-top: 4px;">蓄積状況 — 画像 <b>{snap_n}</b> ・ 分析メモ <b>{prompt_n}</b> ・ データ <b>{data_n}</b> ファイル</div>
<div style="color: #57687A; font-size: 11.5px; margin-top: 6px; font-family: 'SF Mono', 'Consolas', monospace;">ID: {session_id}</div>
</div>""", unsafe_allow_html=True)
                # ZIPダウンロード方式のためパス表示は不要

            else:
                # 記録セッション 未開始状態
                st.markdown("""<div style="border: 1px solid #DEE5EC; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; background: #F7F9FB;">
<div style="font-size: 15px; font-weight: 700; color: #57687A;"><span class=msym>menu_book</span> 分析の記録セッション
<span style="font-size: 13px; margin-left: 8px;">○ 未開始</span></div>
<div style="color: #5A6875; font-size: 13px; margin-top: 8px; line-height: 1.7;">
分析結果を自動で集めて、レポート生成の材料にする仕組みです。<br/>
「前処理」が完了すると自動で始まります。</div>
</div>""", unsafe_allow_html=True)

                if st.session_state.get('preprocess_done', False):
                    if st.button(":material/menu_book: 記録セッションを開始", type="primary", key="start_capcom"):
                        session_id, _ = capcom.init_session()
                        st.success(f":material/menu_book: 記録セッションを開始しました: `{session_id}`")
                        st.rerun()

    # ==============================================================
    # タブ「4. 外部データ取込（任意）」: NPL取込
    # ==============================================================
    with tab_npl:
        st.markdown("##### 論文・ニュース・政策・市場のデータを足すと、技術の「外の動き」まで分析できます。")
        apollo_ui.howto(
            "ここで追加した外部データは<b>環境分析</b>のページで使われ、特許の動きを"
            "論文・政策・市場の動きと突き合わせられるようになります。"
            "<b>特許だけを分析するなら、このタブは飛ばして大丈夫です</b>"
            "（分析を始めたあとからでも追加できます。その場合は、追加後にこのタブ上部へ出る「↻ いますぐ反映」を押すだけ）。<br>"
            "使い方は3ステップ — <b>① 下で種類を選ぶ → ② 取り込む（ファイル読込み。論文はオンライン検索も可）→ "
            "③「データセットに追加」を押す</b>。最後に「5. 前処理」を実行すると反映されます。",
            title="このタブですること",
        )

        # --- NPL蓄積ロジック ---
        if 'df_npl_accumulated' not in st.session_state:
            st.session_state.df_npl_accumulated = pd.DataFrame()

        # --- 取込済みの外部データ（件数・内訳・反映状態を最上部に常設） ---
        _acc = st.session_state.df_npl_accumulated
        _cur = st.session_state.get('df_npl')
        _n_acc = len(_acc)
        _n_cur = 0 if _cur is None else len(_cur)
        _TYPE_JP = {'Academic': '論文', 'Business': 'ニュース', 'Policy': '政策・規制', 'Market': '市場レポート'}
        if _n_acc > 0:
            with st.container(border=True):
                _stats = _acc['data_sub_type'].value_counts()
                _breakdown = "・".join(f"{_TYPE_JP.get(str(_k), str(_k))} {_v:,}件" for _k, _v in _stats.items())
                _c_l, _c_r = st.columns([3, 1], vertical_alignment="center")
                with _c_l:
                    st.markdown(f"**:material/download: 取込済みの外部データ: 合計 {_n_acc:,} 件**（{_breakdown}）")
                    if not st.session_state.get('preprocess_done', False):
                        st.caption("「5. 前処理」を実行すると、まとめて分析に反映されます。")
                    elif _n_acc != _n_cur:
                        st.caption(
                            f":material/warning: まだ分析に反映されていないデータが {abs(_n_acc - _n_cur)} 件あります。"
                            "右の「いますぐ反映」で反映できます（特許側の重い計算はやり直さないので、すぐ終わります）。")
                    else:
                        st.caption(":material/check_circle: 分析に反映済みです。環境分析のページで使えます。")
                with _c_r:
                    if st.session_state.get('preprocess_done', False) and _n_acc != _n_cur:
                        if st.button("↻ いますぐ反映", type="primary", key="apply_npl_now",
                                     use_container_width=True):
                            with st.spinner("外部データを整理しています..."):
                                st.session_state.df_npl = st.session_state.df_npl_accumulated.copy()
                                _process_npl_data(st.session_state.col_map)
                            st.toast(f"外部データ {len(st.session_state.df_npl)} 件を分析に反映しました。環境分析で使えます", icon=":material/check_circle:")
                            st.rerun()
                    if st.button(":material/delete: すべてクリア", key="reset_npl_btn", use_container_width=True):
                        st.session_state.df_npl_accumulated = pd.DataFrame()
                        if 'df_npl' in st.session_state:
                            del st.session_state.df_npl
                        st.rerun()
                with st.expander("取込済みデータのプレビュー（先頭3件）", expanded=False):
                    st.dataframe(_acc.head(3))

        # 注: Streamlit は expander の入れ子を禁止しているため、ここは expander ではなく
        # 枠付きコンテナにする（内側の「API キー設定」「検索式の書き方」expander を有効にするため）。
        with st.container(border=True):
            st.markdown("**追加したいデータの種類を選んでください**")

            npl_tabs = st.tabs([':material/school: 論文', ':material/newspaper: ニュース', ':material/balance: 政策・規制', ':material/bar_chart: 市場レポート'])

            # ファイル読み込みの共通関数
            def read_uploaded_file(f):
                if f.name.lower().endswith('.csv'):
                    try:
                        return pd.read_csv(f, dtype=str)
                    except UnicodeDecodeError:
                        return pd.read_csv(f, dtype=str, encoding='shift_jis')
                else:
                    return pd.read_excel(f, dtype=str)

            # --- タブ 1: 論文 (Academic) — CSVアップロード + OpenALEX検索 ---
            with npl_tabs[0]:
                st.markdown("###### :material/school: 論文データの取込")
                st.caption("手元の論文リスト（LENS等からダウンロードしたCSV/Excel）を読み込むか、オンラインで直接検索して取り込みます。")
                aca_mode = st.radio(
                    "取得方法を選択:",
                    [":material/description: CSVアップロード", ":material/search: オンライン検索（OpenALEX）"],
                    horizontal=True, key="aca_mode",
                    help="論文データの入手方法を選びます。「CSVアップロード」は手元のファイルを取り込み、「オンライン検索」はキーワードから論文データベース OpenAlex を直接検索して取り込みます。"
                )

                if aca_mode == ":material/description: CSVアップロード":
                    f_aca = st.file_uploader("論文リストをアップロード (LENS等のCSV / Excel)", type=["csv", "xlsx"], key="up_aca", accept_multiple_files=False)

                    if f_aca:
                        df = read_uploaded_file(f_aca)
                        st.caption(f"プレビュー: {f_aca.name}")
                        st.dataframe(df.head(3))

                        cols = [None] + list(df.columns)
                        st.caption("ファイルのどの列を使うかを指定します（列名から自動推定済み。違っていたら直してください）:")
                        c1, c2 = st.columns(2)
                        idx_t = smart_map_index(None, cols, ['Title', 'Article Title', 'タイトル', 'inventionTitle'])
                        idx_d = smart_map_index(None, cols, ['Date', 'Publication Date', '発行日', 'publicationDate'])
                        idx_c = smart_map_index(None, cols, ['Abstract', 'Summary', '要約', 'abstract'])
                        idx_s = smart_map_index(None, cols, ['Source', 'Journal', 'Publisher', '情報源', 'publisher'])

                        with c1:
                            m_title = st.selectbox("タイトルの列 (必須):", cols, index=idx_t, key="map_aca_title")
                            m_date = st.selectbox("日付の列 (必須):", cols, index=idx_d, key="map_aca_date")
                        with c2:
                            m_content = st.selectbox("要約の列 (必須):", cols, index=idx_c, key="map_aca_content")
                            m_source = st.selectbox("出典・ジャーナルの列 (任意):", cols, index=idx_s, key="map_aca_source")

                        if st.button(":material/add: データセットに追加（論文）", key="add_aca"):
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
                                st.toast(f"論文データ {len(df_new)} 件を追加しました", icon=":material/add:")
                                st.rerun()
                            else:
                                st.error("必須の列（タイトル・日付・要約）を選択してください。")

                else:
                    # --- OpenALEX検索 ---
                    st.markdown("論文データベース OpenAlex をオンライン検索して、結果をそのままデータセットに追加します。")

                    # --- OpenAlex API キー設定（2026-02-13 より必須・無料） ---
                    with st.expander(":material/key: OpenAlex API キー設定（必須・無料）", expanded=True):
                        st.caption(
                            "OpenAlex は 2026-02-13 より API キーが必須です（従来の mailto / polite pool 方式は廃止）。"
                            "[openalex.org/settings/api](https://openalex.org/settings/api) で無料アカウントを作成し API キーを取得して貼り付けてください。"
                            "**本アプリは API キー必須です（キーを入力しないと検索できません）。**"
                        )
                        st.text_input(
                            "OpenAlex API キー",
                            type="password",
                            key="openalex_api_key",
                            help="入力したキーはこのセッション（st.session_state）にのみ保持され、OpenAlex API 以外には送信されません。共用 PC では使用後にクリアしてください。",
                        )
                        _kc1, _kc2 = st.columns([1, 3])
                        with _kc1:
                            if st.button("接続テスト", key="oalex_test_conn"):
                                from openalex import OpenAlexCollector
                                if not (st.session_state.get("openalex_api_key") or "").strip():
                                    st.warning("先に API キーを入力してください。")
                                else:
                                    _ok, _msg = OpenAlexCollector(api_key=st.session_state.get("openalex_api_key")).test_connection()
                                    (st.success if _ok else st.error)(_msg)
                        with _kc2:
                            if (st.session_state.get("openalex_api_key") or "").strip():
                                st.caption(":material/check_circle: API キー設定済み")
                            else:
                                st.caption(":material/warning: API キー未設定（検索にはキーが必要です）")

                    oalex_mode = st.radio(
                        "検索モード",
                        ["コマンドライン検索式（TI=/AB=/TA= + near/adj）", "キーワード検索（1行1クエリ・複数行でOR）"],
                        index=0,  # 既定はコマンドライン検索式（特許文献検索風）を左端に配置
                        horizontal=True,
                        key="oalex_search_mode",
                        help=(
                            "コマンドライン検索式: TI=/AB=/TA=/TX=/FT= でフィールドを指定し、AND/OR/NOT・"
                            "近傍 nearN/adjN・ワイルドカード(* ?) が使える特許文献検索風モード。"
                            "near/adj はタイトル・要旨に対し取得後にローカルで厳密照合します。\n"
                            "キーワード検索: 検索語をそのまま OpenAlex へ。1行1クエリ・複数行で OR 検索。"
                        ),
                    )
                    _is_cmd = oalex_mode.startswith("コマンドライン")

                    if _is_cmd:
                        oalex_query = st.text_area(
                            "コマンドライン検索式",
                            placeholder=(
                                '#01  TA=((sulfide OR oxide) near5 electrolyte)\n'
                                '#02  AB=((degradation AND capacity) near8 mechanism)\n'
                                '#03  TA=(review)\n'
                                'T=(#01 AND #02) AND NOT #03'
                            ),
                            height=140,
                            key="oalex_query_cmd",
                        )
                        oalex_query = oalex_query or ""
                        with st.expander(":material/menu_book: 検索式の書き方", expanded=False):
                            st.markdown(
                                "- **フィールド**: `TI=`(タイトル) / `AB=`(要旨) / `TA=`(タイトル+要旨) / "
                                "`TX=`(OpenAlex総合) / `FT=`(全文索引)\n"
                                "- **演算子**: `AND` / `OR` / `NOT`、近傍 `nearN`(順不同・間に N 語以内) / "
                                "`adjN`(順序固定・左→右で N 語以内)\n"
                                "- **ワイルドカード**: `*`(任意長) `?`(1文字)。例 `electrol*`、`wom?n`。"
                                "OpenAlex 索引は前方一致非対応のため、ワイルドカード語は**必ず具体語との AND/OR で併用**してください"
                                "（例 `TI=(battery AND electrol*)`。単独・NOT のみは不可）。"
                                "OpenAlex 検索は語幹一致するので `*` は多くの場合不要（`plastic` で `plastics` もヒット）。"
                                "**`near/adj` の片側を全てワイルドカード語にすると、その側が候補取得から丸ごと外れ、検索が過度に広がって取りこぼす**ため、"
                                "各 near/adj の左右には具体語を最低1つ入れてください（例 `(biomass) near3 (plastic OR polymer)`）\n"
                                "- **複数行**: `#01 ...` `#02 ...` と書き、最後に `T=(#01 AND #02) OR #03` で結合"
                                "（1行のみなら T 式は省略可）\n"
                                "- **単一式**でも可: 例 `TA=((solid OR all-solid) adj3 electrolyte) NOT AB=(review)`\n"
                                "- `TI`/`AB`/`TA` の near/adj・NOT・ワイルドカードは取得後にローカルで厳密照合します"
                                "（`TX`/`FT` は OpenAlex 索引による候補取得のみ）。\n"
                                "- 下の **:material/calendar_month: 年別取得モード** はこのコマンドライン検索式でも利用できます"
                                "（各候補クエリを年別取得 → ローカル厳密照合）。"
                            )

                        # --- クエリ作成補助: 外部AIへ渡すプロンプトを生成 ---
                        # 検索テーマを入力させ、文法を厳守してそのまま貼れる検索式を出力させるプロンプトを表示する。
                        with st.expander(":material/smart_toy: AIにクエリ作成を依頼するプロンプトを表示", expanded=False):
                            st.caption(
                                "検索したいテーマを記入 → 下のプロンプトを:material/content_paste:でコピー → ChatGPT / Claude / Gemini 等に貼付。"
                                "AIが出力した検索式を、そのまま上の「コマンドライン検索式」欄に貼り付けて使えます。"
                            )
                            _ai_theme = st.text_area(
                                "検索したいテーマ・技術領域・目的（具体的なほど精度が上がる）",
                                placeholder="例: 全固体電池の固体電解質と電極の界面抵抗・劣化メカニズムに関する研究。",
                                height=90,
                                key="oalex_ai_theme",
                            )
                            _theme_txt = (_ai_theme or "").strip() or "（ここに検索したいテーマ・技術領域・目的を具体的に記述してください）"
                            _ai_prompt = f"""あなたは学術文献・特許文献の検索式設計の専門家です。
以下の「検索したいテーマ」を網羅的かつ的確に検索するための検索式を、後述の「コマンドライン検索式」の文法に厳密に従って1つ作成してください。

# 検索したいテーマ・目的
{_theme_txt}

# 作業手順（必ず吟味すること）
1. テーマの中核概念を2〜4個に絞り込む。各概念の同義語・表記ゆれ（英語/カタカナ・略語・スペル違い）は、最も一般的なものだけを各概念2〜3語まで厳選する（網羅・列挙しすぎない）。
2. 各概念をどのフィールドで検索するか決める（精度重視=TI/TA、再現率重視=TX/FT。要旨に多く現れる概念はAB）。
3. 概念どうしの関係を AND / OR と近傍 nearN/adjN で表現する。語の近接が重要なら near/adj を使う。
4. ワイルドカード(*?)は乱用しない。OpenAlex 検索は語幹一致するため plastic は plastics にもヒットし、多くの場合 * は不要（むしろ候補取得を広げて API を浪費する）。**特に near/adj の片側をすべてワイルドカード語にしない**（その側が OpenAlex 候補取得から丸ごと落ち、検索が極端に広くなって取りこぼしの原因になる）。各 near/adj の左右には具体語を最低1つ入れる。* を使う場合も必ず具体語と AND/OR で併用する（単独・全ワイルドカードは不可）。
5. 再現率と精度のバランスを取り、広すぎ・狭すぎを避ける。
6. 総説・レビュー等の「論文種別」での絞り込みは検索式に含めない（アプリ側の「論文種別フィルタ」で行うため）。

# 検索式は短く保つ（API消費の節約・最重要）
- このシステムは OR の各選択肢を 1 つずつ別々の検索クエリに展開して API を呼び出す。さらに AND で結ばれた複数の OR グループは掛け算で展開される（例: (a OR b OR c) AND (d OR e) → 3×2 = 6 クエリ）。OR を増やすほど API 呼び出しが急増する。
- そのため次を守ること:
  - OR グループは式全体で 2〜3 個まで、各 OR グループの語数も 2〜3 語までに抑える。
  - 「各 OR グループの語数の掛け算（＝展開後の候補クエリ数）」が概ね 6 件以内に収まるようにする。
  - 同義語を増やすより、AND で概念を絞り込み、適切なフィールド指定で精度を上げる方を優先する。
  - 迷ったら、短く絞り込んだ式にする。

# コマンドライン検索式の文法（厳守）
- フィールド指定（各語・式の前に必須）: TI=（タイトル） / AB=（要旨） / TA=（タイトル+要旨） / TX=（OpenAlex総合索引） / FT=（全文索引）。フィールド未指定の語・式はエラーになる。
- 演算子: AND / OR / NOT。グループ化は丸括弧 ( ) を使う。
- 近傍: nearN=順不同でN語以内、adjN=左→右の順でN語以内。例: TA=((solid OR all-solid) adj3 electrolyte)
- ワイルドカード: *（任意長）/ ?（1文字）。前方一致は不可なので、必ず具体語と併用する。例: TI=(battery AND electrol*)
- 複数条件: 各条件を #01, #02 … の行で書き、最後に T=(#01 AND #02) OR #03 のように結合する。条件が1つなら T 式は省略してよい。
- 単一式でも可。例: TA=((solid OR all-solid) adj3 electrolyte) NOT AB=(review)
- 補足: TI/AB/TA の near/adj・NOT・ワイルドカードは取得後にタイトル・要旨へローカルで厳密照合される。TX/FT は索引による候補取得のみ。

# 出力形式（重要）
- 最終的な検索式だけを出力する。前置き・説明・囲み記号は付けず、そのままフォームに貼り付けて使える形にする。
- 複数条件を使う場合は #01 … #0N の各行と、最後の T= 行を出力する。
- 上記「検索式は短く保つ」を守り、OR を盛りすぎない短い式にする。

# 出力例（この形式で・短く出力する。展開後の候補クエリは 4 件程度）
#01  TA=((solid-state OR all-solid-state) adj3 electrolyte)
#02  AB=(interface near5 (resistance OR degradation))
T=(#01 AND #02)"""
                            st.code(_ai_prompt, language="text")

                        # --- 構文チェック / OpenAlex候補式プレビュー（OpenALEX Collector の「構文チェック」表示と同等）---
                        # 押すと、入力した検索式を compile_command_query で解析し、構文の可否・展開後の式・
                        # 実際にOpenAlexへ投げる候補検索式・検索範囲(scope)・ローカル厳密照合の有無を表示する。
                        if st.button(
                            ":material/search: 構文チェック / OpenAlex候補式プレビュー", key="oalex_query_preview_btn",
                            help="入力したコマンドライン検索式の構文を検証し、実際にOpenAlexへ投げる候補検索式・検索範囲・ローカル厳密照合の有無を表示します。",
                        ):
                            st.session_state['oalex_show_preview'] = True
                        if st.session_state.get('oalex_show_preview'):
                            import openalex_query as _oaq
                            if not (oalex_query or "").strip():
                                st.info("コマンドライン検索式を入力してから「構文チェック」を押すとプレビューを表示します。")
                            else:
                                _scope_labels = {
                                    "all": "OpenAlex総合（全文索引）", "title": "タイトル", "abstract": "要旨",
                                    "title_and_abstract": "タイトル＋要旨", "fulltext": "全文",
                                }
                                try:
                                    _plan = _oaq.compile_command_query(oalex_query)
                                    _cands = _plan["candidate_queries"]
                                    _scope_lbl = _scope_labels.get(_plan["scope"], _plan["scope"])
                                    st.success(":material/check_circle: 構文OK")
                                    st.markdown("**展開後の検索式**")
                                    st.code(_plan["expanded"], language="text")
                                    st.markdown(
                                        f"**OpenAlex候補検索式（{len(_cands)} 件 / 検索範囲 scope = {_scope_lbl}）**"
                                    )
                                    st.code(
                                        "\n".join(f"{_i + 1}. {_c or '(空: フィルタのみ)'}" for _i, _c in enumerate(_cands)),
                                        language="text",
                                    )
                                    if _plan["needs_local_filter"]:
                                        st.caption(":material/biotech: near/adj・NOT・ワイルドカードを含むため、OpenAlex候補取得後にタイトル・要旨へローカルで厳密照合します（内部除外あり）。")
                                    else:
                                        st.caption("通常キーワード検索です。OpenAlex候補をそのまま採用し、内部での絞り込みは行いません。")
                                    st.caption("※ コマンドラインモードでは詳細フィルタの「検索範囲」は使わず、各行の TI=/AB=/TA=/TX=/FT= から候補取得 scope を自動決定します。")
                                except _oaq.QueryError as _qe:
                                    st.error(f"構文エラー: {_qe}")
                                except Exception as _e:
                                    st.error(f"検索式の解析に失敗しました: {_e}")
                    else:
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
                        oalex_year_from = st.number_input("開始年", min_value=1900, max_value=2026, value=2020, key="oalex_year_from",
                            help="OpenALEXから学術論文を取得する対象期間（出版年）の開始年です。")
                    with c2:
                        oalex_year_to = st.number_input("終了年", min_value=1900, max_value=2026, value=2026, key="oalex_year_to",
                            help="OpenALEXから学術論文を取得する対象期間（出版年）の終了年です。")
                    with c3:
                        oalex_max = st.number_input("取得上限", min_value=50, max_value=10000, value=200, step=50, key="oalex_max",
                            help="OpenALEXから取得する論文の最大件数です。上げると多く集まりますが取得に時間がかかります。")

                    # --- 年別取得モード（10,000件/クエリ制限を回避して広い年範囲を網羅） ---
                    oalex_by_year = st.checkbox(
                        ":material/calendar_month: 年別取得モード（年ごとに最大上限まで取得、広い年範囲で大量取得したい場合）",
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
                            f":material/calculate: 試算: {_years_span} 年 × 最大 {oalex_max_per_year:,} 件 = "
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
                            ":material/description: 要約ありの論文のみ取得",
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
                            ":material/language: 英語論文のみ取得",
                            value=False,
                            key="oalex_en_only",
                            help=(
                                "OpenAlex の `language:en` フィルタを適用。\n"
                                "多言語データ（中国語・ドイツ語等）が混在すると SBERT の精度が低下するため、\n"
                                "グローバル比較が主目的なら ON を推奨。"
                            ),
                        )

                    _oalex_has_key = bool((st.session_state.get("openalex_api_key") or "").strip())
                    if not _oalex_has_key:
                        st.info(":material/key: 検索するには上の「OpenAlex API キー設定」でキーを入力してください（必須）。")
                    # 「:material/report: 検索を終了」で中断した直後の再実行でメッセージを出す（フラグは消費する）
                    if st.session_state.pop('oalex_search_stopped', False):
                        st.info(":material/report: 検索を中断しました（途中までの取得結果は破棄されました）。")
                    if st.button(":material/search: OpenALEX検索実行", key="oalex_search_btn", disabled=not _oalex_has_key):
                        if not _oalex_has_key:
                            st.error("OpenAlex API キーを入力してください（本アプリはキー必須です）。")
                        elif not oalex_query.strip():
                            st.error("検索キーワードを入力してください。")
                        else:
                            try:
                                from openalex import OpenAlexCollector
                                collector = OpenAlexCollector(api_key=st.session_state.get("openalex_api_key"))

                                queries = [q.strip() for q in oalex_query.strip().split('\n') if q.strip()]

                                progress_bar = st.progress(0.0)
                                status_text = st.empty()
                                # 検索を途中で止めるボタン。押すと再実行が要求され、次の on_progress 内の
                                # st.* 呼び出しで Streamlit が現在の実行を割り込み終了する（RerunException は
                                # BaseException 由来なので下の except Exception には捕まらない）。途中結果は破棄。
                                _stop_ph = st.empty()
                                _stop_ph.button(
                                    ":material/report: 検索を終了", key="oalex_stop_btn",
                                    on_click=lambda: st.session_state.update(oalex_search_stopped=True),
                                    help="実行中の OpenALEX 検索を中断します（途中までの取得結果は保存されません）。",
                                )

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
                                        f":material/calendar_month: {year} 年 ({yi + 1}/{total_years}): "
                                        f"{year_count:,} / {year_total:,} 件 | 累計: {all_count:,} 件"
                                    )

                                def on_cmd_year_progress(ci, n_cand, yi, total_years, year,
                                                         year_count, year_total, all_count):
                                    # コマンドライン × 年別: 候補クエリ進捗 + 当年内進捗を合算
                                    year_frac = (year_count / year_total) if year_total else 1.0
                                    within_cand = (yi + min(year_frac, 1.0)) / max(total_years, 1)
                                    overall = (ci + within_cand) / max(n_cand, 1)
                                    progress_bar.progress(min(overall, 1.0))
                                    status_text.markdown(
                                        f":material/search: 候補 {ci + 1}/{n_cand} | :material/calendar_month: {year} 年 "
                                        f"({yi + 1}/{total_years}): {year_count:,} / {year_total:,} 件 | "
                                        f"統合累計: {all_count:,} 件"
                                    )

                                def on_multi_progress(qi, total_q, current, total):
                                    # 複数クエリ OR（通常モード）の進捗。st.* を毎クエリ呼ぶので
                                    # 「:material/report: 検索を終了」での割り込みもこの経路で効くようになる。
                                    frac = (current / total) if total else 1.0
                                    overall = (qi + min(frac, 1.0)) / max(total_q, 1)
                                    progress_bar.progress(min(overall, 1.0))
                                    status_text.markdown(
                                        f":material/search: クエリ {qi + 1}/{total_q}: {current:,} / {total:,} 件"
                                    )

                                # 共通フィルタ引数（全 4 パスで使用）
                                _common_kwargs = dict(
                                    pub_types=oalex_pub_types if oalex_pub_types else None,
                                    institution_ids=inst_ids if inst_ids else None,
                                    has_abstract=bool(oalex_has_abstract),
                                    language=("en" if oalex_en_only else None),
                                )

                                # --- コマンドライン検索式モード（TI=/AB=/TA= + near/adj） ---
                                if _is_cmd:
                                    import openalex_query as _oq
                                    try:
                                        if oalex_by_year:
                                            # 年別取得モード（候補クエリごとに年別取得）
                                            raw_papers = collector.search_command_query(
                                                oalex_query,
                                                year_from=int(oalex_year_from),
                                                year_to=int(oalex_year_to),
                                                by_year=True,
                                                max_per_year=int(oalex_max_per_year),
                                                on_progress=on_cmd_year_progress,
                                                **_common_kwargs,
                                            )
                                        else:
                                            raw_papers = collector.search_command_query(
                                                oalex_query,
                                                year_from=oalex_year_from, year_to=oalex_year_to,
                                                max_results=oalex_max,
                                                on_progress=on_progress,
                                                **_common_kwargs,
                                            )
                                    except _oq.QueryError as _qe:
                                        raw_papers = []
                                        st.error(f"検索式エラー: {_qe}")
                                # 検索実行（通常モード / 年別取得モードで分岐）
                                elif oalex_by_year:
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
                                                    f":material/search: クエリ {_qi + 1}/{_tq} | :material/calendar_month: {year} 年 "
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
                                        on_progress=on_multi_progress,
                                        **_common_kwargs,
                                    )

                                _stop_ph.empty()  # 検索が最後まで走った：停止ボタンを消す
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
                                                f":material/language: タイトルが非英語の論文 **{_removed_n:,} 件** を除外しました "
                                                f"（OpenAlex の `language:en` は要約ベースの判定のため、"
                                                f"タイトルだけ日本語・中国語等の多言語ジャーナル論文が混入することがあります）"
                                            )

                                    progress_bar.progress(1.0)

                                    if df_oalex.empty:
                                        status_text.warning("フィルタ後、該当する論文が 0 件になりました。条件を緩めてください。")
                                        st.session_state.pop('oalex_last_result', None)
                                    else:
                                        status_text.success(f":material/check_circle: {len(df_oalex):,} 件の論文を取得しました。")
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
                        st.markdown("###### :material/search: 検索結果プレビュー")

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
                                ":material/download: 検索結果をCSVでダウンロード",
                                data=csv_bytes,
                                file_name=f"openalex_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
                                mime="text/csv",
                                key="oalex_csv_dl",
                            )
                        with col_add:
                            if st.button(":material/add: データセットに追加（検索結果）", key="oalex_add_btn"):
                                st.session_state.df_npl_accumulated = pd.concat(
                                    [st.session_state.df_npl_accumulated, df_oalex_cached], ignore_index=True)
                                st.toast(f"論文データ {len(df_oalex_cached)} 件を追加しました", icon=":material/add:")
                                st.rerun()

            # --- タブ 2: ニュース (News) ---
            with npl_tabs[1]:
                st.markdown("###### :material/newspaper: ニュースデータの取込")
                st.caption("業界ニュース・プレスリリース等の一覧（CSV/Excel）を取り込みます。")
                f_news = st.file_uploader("ニュース一覧をアップロード (CSV / Excel)", type=["csv", "xlsx"], key="up_news", accept_multiple_files=False)

                if f_news:
                    df = read_uploaded_file(f_news)
                    st.caption(f"プレビュー: {f_news.name}")
                    st.dataframe(df.head(3))

                    cols = [None] + list(df.columns)
                    st.caption("ファイルのどの列を使うかを指定します（列名から自動推定済み。違っていたら直してください）:")
                    c1, c2 = st.columns(2)
                    idx_t = smart_map_index(None, cols, ['Title', 'Headline', 'タイトル', '見出し'])
                    idx_d = smart_map_index(None, cols, ['Date', 'Published', '日付'])

                    with c1:
                        m_title = st.selectbox("見出しの列 (必須):", cols, index=idx_t, key="map_news_title")
                        m_date = st.selectbox("日付の列 (必須):", cols, index=idx_d, key="map_news_date")
                    with c2:

                        m_content = st.selectbox("本文の列 (任意):", cols, index=smart_map_index(None, cols, ['Content', 'Body', '本文', 'Project Description']), key="map_news_content")
                        m_source = st.selectbox("媒体・出典の列 (任意):", cols, index=smart_map_index(None, cols, ['Source', 'Media', '媒体']), key="map_news_source")

                    if st.button(":material/add: データセットに追加（ニュース）", key="add_news"):
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
                            st.toast(f"ニュースデータ {len(df_new)} 件を追加しました", icon=":material/add:")
                            st.rerun()
                        else:
                            st.error("必須の列（見出し・日付）を選択してください。")

            # --- タブ 3: 政策・規制 (Policy) ---
            with npl_tabs[2]:
                st.markdown("###### :material/balance: 政策・規制データの取込")
                st.caption(
                    "補助金・規制・ガイドライン等の一覧（CSV/Excel）を取り込みます。"
                    "手元にデータがない場合は、下のスイッチでAIに一覧を作ってもらうのが手軽です。")

                # AIプロンプト
                if st.toggle(":material/smart_toy: AIにデータ作成を依頼する（政策・規制の一覧）"):
                    st.caption(
                        "使い方: 調査テーマを入力 → 表示されるプロンプトを:material/content_paste:でコピーして ChatGPT / Claude / Gemini 等に貼付 → "
                        "AIが出力したCSVをファイルに保存 → 下のアップロード欄から取り込む。")
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

                f_pol = st.file_uploader("政策・規制の一覧をアップロード (CSV / Excel)", type=["csv", "xlsx"], key="up_pol", accept_multiple_files=False)

                if f_pol:
                    df = read_uploaded_file(f_pol)
                    st.caption(f"プレビュー: {f_pol.name}")
                    st.dataframe(df.head(3))

                    cols = [None] + list(df.columns)
                    st.caption("ファイルのどの列を使うかを指定します（列名から自動推定済み。違っていたら直してください）:")
                    c1, c2 = st.columns(2)

                    with c1:
                        m_title = st.selectbox("政策・規制名の列 (必須):", cols, index=smart_map_index(None, cols, ['Title', 'Name', '名称']), key="map_pol_title")
                        m_date = st.selectbox("日付の列 (必須):", cols, index=smart_map_index(None, cols, ['Date', 'Effective', '日付']), key="map_pol_date")
                        m_region = st.selectbox("地域の列 (任意):", cols, index=smart_map_index(None, cols, ['Region', 'Country', '国']), key="map_pol_reg")
                    with c2:
                        m_content = st.selectbox("要約の列 (必須):", cols, index=smart_map_index(None, cols, ['Abstract', 'Summary', 'Description', '要約']), key="map_pol_cont")
                        m_source = st.selectbox("発行機関・出典の列 (任意):", cols, index=smart_map_index(None, cols, ['Source', 'Ministry', '情報源']), key="map_pol_src")
                        m_status = st.selectbox("ステータスの列 (任意):", cols, index=smart_map_index(None, cols, ['Status', 'State', '状態']), key="map_pol_stat")

                    if st.button(":material/add: データセットに追加（政策・規制）", key="add_pol"):
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
                            st.toast(f"政策・規制データ {len(df_new)} 件を追加しました", icon=":material/add:")
                            st.rerun()
                        else:
                            st.error("必須の列（政策・規制名・日付・要約）を選択してください。")

            # --- タブ 4: 市場レポート (Market) ---
            with npl_tabs[3]:
                st.markdown("###### :material/bar_chart: 市場データの取込")
                st.caption(
                    "市場規模・成長予測・主要企業動向等の一覧（CSV/Excel）を取り込みます。"
                    "手元にデータがない場合は、下のスイッチでAIに一覧を作ってもらうのが手軽です。")

                # AIプロンプト
                if st.toggle(":material/smart_toy: AIにデータ作成を依頼する（市場データの一覧）"):
                    st.caption(
                        "使い方: 調査テーマを入力 → 表示されるプロンプトを:material/content_paste:でコピーして ChatGPT / Claude / Gemini 等に貼付 → "
                        "AIが出力したCSVをファイルに保存 → 下のアップロード欄から取り込む。")
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

                f_mkt = st.file_uploader("市場データの一覧をアップロード (CSV / Excel)", type=["csv", "xlsx"], key="up_mkt", accept_multiple_files=False)

                if f_mkt:
                    df = read_uploaded_file(f_mkt)
                    st.caption(f"プレビュー: {f_mkt.name}")
                    st.dataframe(df.head(3))

                    cols = [None] + list(df.columns)
                    st.caption("ファイルのどの列を使うかを指定します（列名から自動推定済み。違っていたら直してください）:")
                    c1, c2 = st.columns(2)

                    with c1:
                        m_title = st.selectbox("レポート名の列 (必須):", cols, index=smart_map_index(None, cols, ['Title', 'Segment', 'タイトル']), key="map_mkt_title")
                        m_date = st.selectbox("日付の列 (必須):", cols, index=smart_map_index(None, cols, ['Date', 'Published', '日付']), key="map_mkt_date")
                    with c2:
                        m_content = st.selectbox("市場サマリーの列 (必須):", cols, index=smart_map_index(None, cols, ['Abstract', 'Summary', 'Description', '要約']), key="map_mkt_cont")
                        m_source = st.selectbox("調査会社・出典の列 (任意):", cols, index=smart_map_index(None, cols, ['Source', 'Firm', '出典']), key="map_mkt_src")

                    if st.button(":material/add: データセットに追加（市場）", key="add_mkt"):
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
                            st.toast(f"市場データ {len(df_new)} 件を追加しました", icon=":material/add:")
                            st.rerun()
                        else:
                            st.error("必須の列（レポート名・日付・市場サマリー）を選択してください。")

            # （取込済みデータの件数・内訳・プレビュー・クリアはタブ最上部の状態カードにある）

    # --- ミッション進行ステッパーの描画（タブ内の処理を反映した最新状態で） ---
    _render_mission_stepper(_stepper_area)
