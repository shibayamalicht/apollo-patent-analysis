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

# page_icon に実写アイコンを使うため、set_page_config より前に utils を import する
import utils

# ==================================================================
# --- ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO v9 | Mission Control",
    page_icon=utils.module_icon("home"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================================
# --- 定数とヘルパー関数 ---
# ==================================================================

import io

import capcom

# ==================================================================
# --- patiroha統合: SBERTモデル・ストップワード ---
# ==================================================================

def load_sbert_embedder(model_name=None):
    """文埋め込みモデルを取得する（実体は utils 側でモデル名ごとに 1 度だけ構築・共有）。

    モデル名はフェーズ4のラジオ選択（st.session_state['sbert_model_name']）を渡す。
    接頭辞・GPU(CUDA/MPS)選択は utils.build_sbert_embedder に集約。
    MPSが不安定な場合は環境変数 APOLLO_FORCE_CPU=1 でCPU強制可能。
    """
    return utils.load_sbert_embedder(model_name or st.session_state.get('sbert_model_name'))
    return embedder


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

utils.module_header("home", "Mission Control") 
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
        # 注: Streamlit は expander の入れ子を禁止しているため、ここは expander ではなく
        # 枠付きコンテナにする（内側の「API キー設定」「検索式の書き方」expander を有効にするため）。
        with st.container(border=True):
            st.caption("論文・ニュース・政策文書などを取り込む (NPL)")

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
                    horizontal=True, key="aca_mode",
                    help="論文データの入手方法を選びます。「CSVアップロード」は手元のファイルを取り込み、「OpenALEX検索」はキーワードからオンラインで論文を直接検索して取り込みます。"
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

                    # --- OpenAlex API キー設定（2026-02-13 より必須・無料） ---
                    with st.expander("🔑 OpenAlex API キー設定（必須・無料）", expanded=True):
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
                                st.caption("✅ API キー設定済み")
                            else:
                                st.caption("⚠️ API キー未設定（検索にはキーが必要です）")

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
                        with st.expander("📖 検索式の書き方", expanded=False):
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
                                "- 下の **📅 年別取得モード** はこのコマンドライン検索式でも利用できます"
                                "（各候補クエリを年別取得 → ローカル厳密照合）。"
                            )

                        # --- クエリ作成補助: 外部AIへ渡すプロンプトを生成 ---
                        # 検索テーマを入力させ、文法を厳守してそのまま貼れる検索式を出力させるプロンプトを表示する。
                        with st.expander("🤖 AIにクエリ作成を依頼するプロンプトを表示", expanded=False):
                            st.caption(
                                "検索したいテーマを記入 → 下のプロンプトを📋でコピー → ChatGPT / Claude / Gemini 等に貼付。"
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
                            "🔎 構文チェック / OpenAlex候補式プレビュー", key="oalex_query_preview_btn",
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
                                    st.success("✅ 構文OK")
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
                                        st.caption("🔬 near/adj・NOT・ワイルドカードを含むため、OpenAlex候補取得後にタイトル・要旨へローカルで厳密照合します（内部除外あり）。")
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

                    _oalex_has_key = bool((st.session_state.get("openalex_api_key") or "").strip())
                    if not _oalex_has_key:
                        st.info("🔑 検索するには上の「OpenAlex API キー設定」でキーを入力してください（必須）。")
                    # 「🛑 検索を終了」で中断した直後の再実行でメッセージを出す（フラグは消費する）
                    if st.session_state.pop('oalex_search_stopped', False):
                        st.info("🛑 検索を中断しました（途中までの取得結果は破棄されました）。")
                    if st.button("🔍 OpenALEX検索実行", key="oalex_search_btn", disabled=not _oalex_has_key):
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
                                    "🛑 検索を終了", key="oalex_stop_btn",
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
                                        f"📅 {year} 年 ({yi + 1}/{total_years}): "
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
                                        f"🔎 候補 {ci + 1}/{n_cand} | 📅 {year} 年 "
                                        f"({yi + 1}/{total_years}): {year_count:,} / {year_total:,} 件 | "
                                        f"統合累計: {all_count:,} 件"
                                    )

                                def on_multi_progress(qi, total_q, current, total):
                                    # 複数クエリ OR（通常モード）の進捗。st.* を毎クエリ呼ぶので
                                    # 「🛑 検索を終了」での割り込みもこの経路で効くようになる。
                                    frac = (current / total) if total else 1.0
                                    overall = (qi + min(frac, 1.0)) / max(total_q, 1)
                                    progress_bar.progress(min(overall, 1.0))
                                    status_text.markdown(
                                        f"🔎 クエリ {qi + 1}/{total_q}: {current:,} / {total:,} 件"
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
                col_map['title'] = st.selectbox("発明の名称:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('title'), columns_with_none, kw_title), key="col_title",
                    help="発明のタイトル列を指定します。要約・請求項と合わせてクラスタリングやキーワード抽出など意味解析の本文として使われます。正しく割り当てると技術内容の分類精度が上がります。")
                col_map['abstract'] = st.selectbox("要約:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('abstract'), columns_with_none, kw_abstract), key="col_abstract",
                    help="特許の要約（抄録）列を指定します。クラスタリングやキーワード抽出など意味解析に使う主要な本文列です。正しく割り当てると技術マップやクラスタの精度が上がります。")
                col_map['claim'] = st.selectbox("請求項:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('claim'), columns_with_none, kw_claim), key="col_claim",
                    help="特許の請求項（クレーム）列を指定します。タイトル・要約と合わせて意味解析に使われ、権利範囲を含む技術内容の理解に役立ちます。")
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

                # IPC (必須)
                col_map['ipc'] = st.selectbox("国際特許分類 (IPC):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('ipc'), columns_with_none, kw_ipc), key="col_ipc",
                    help="技術分野の分類（IPC）に使う列です。多様性指標や技術構成・技術分野別の分析に使われます。")
                ipc_delimiter = st.text_input("IPC区切り文字:", value=st.session_state.delimiters.get('ipc', ';'), key="del_ipc",
                    help="1セルに複数のIPCが入っている場合の区切り文字です。正しく指定すると複数の技術分類を個別に集計できます。")
                
            with col3:
                st.markdown("##### 任意メタデータ項目")

                # 公開番号 (CAPCOM用)
                kw_pub_num = ['公開番号', '公報番号', 'Publication Number', 'Pub No', 'Document Number']
                col_map['pub_number'] = st.selectbox("公開番号 (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('pub_number'), columns_with_none, kw_pub_num), key="col_pub_number",
                    help="公開公報の番号列です。指定するとレポート（CAPCOM）で代表特許を公開番号付きで参照できます。")

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
                col_map['status'] = st.selectbox("ステータス (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('status'), columns_with_none, ['ステータス', 'Status', 'Legal Status', '法的状態']), key="col_status",
                    help="権利化状況（登録・拒絶・係属など）の列です。指定すると権利化率分析やステータス内訳の表示が利用できます。")
                
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

        # --- 文埋め込み(SBERT)モデルの選択 ---
        # 高精度モデル(e5-base)は重く、環境によっては前処理が止まる/非常に遅いことがあるため、
        # 軽量モデル(MiniLM)も選べるようにする。既定は高速側（安全側）。
        _sbert_keys = list(utils.SBERT_MODELS.keys())
        _default_sbert_idx = _sbert_keys.index(utils.DEFAULT_SBERT_KEY)
        _sel_sbert_key = st.radio(
            "文埋め込みモデル（SBERT）",
            options=_sbert_keys,
            index=_default_sbert_idx,
            format_func=lambda k: utils.SBERT_MODELS[k]["label"],
            key="sbert_model_choice",
            help=(
                "クラスタリングや代表特許抽出に使う意味ベクトル化モデルです。\n"
                "⚡高速: 軽量で速い（前バージョンのモデル）。重くて処理が止まる場合はこちら。\n"
                "🎯高精度: 多言語E5で精度が上がりますが重く、初回はモデルのダウンロード(約440MB)が走ります。"
            ),
        )
        # 選択モデル名をセッションに保持（Home=特許 / NEBULA=論文 / メタ表示で共有）
        st.session_state['sbert_model_name'] = utils.SBERT_MODELS[_sel_sbert_key]["name"]
        st.caption(f"使用モデル: `{st.session_state['sbert_model_name']}`（切り替え後は「分析エンジン起動」を再実行してください）")

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
                    status_text.markdown("🔄 **Phase 1/6: モデルロード中...**")
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

                        _t0_npl = time.time()
                        df_n['explorer_keywords'] = df_n.apply(process_npl_keywords, axis=1)
                        _phase_secs['Phase3 NPLキーワード抽出'] = time.time() - _t0_npl
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

                    _t0 = time.time()
                    sbert_embeddings = embedder.encode(
                        df,
                        text_columns=[col_map['title'], col_map['abstract'], col_map['claim']],
                        batch_size=128,
                        normalize_embeddings=True,
                        progress_callback=sbert_progress,
                    )
                    st.session_state.sbert_embeddings = sbert_embeddings
                    _phase_secs['Phase4 SBERTベクトル化'] = time.time() - _t0

                    # 5. TF-IDF & Keyword (Patent ONLY) — patiroha
                    status_text.markdown("🔄 **Phase 5/6: キーワード抽出 (TF-IDF - 特許のみ)...**")
                    current_sw = _get_current_stopwords()

                    # Explorer用キーワードリスト（マルチコアがあればプロセス並列で高速化／少コアは逐次）
                    _t0 = time.time()
                    df['explorer_keywords'] = pd.Series(
                        utils.extract_keywords_batch(df['text_for_sbert'].tolist(), stopwords=current_sw),
                        index=df.index)
                    _phase_secs['Phase5-1 キーワード抽出(複合名詞・Janome)'] = time.time() - _t0

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
                    status_text.markdown("🔄 **Phase 6/6: メタデータ (日付・IPC・出願人) 正規化中...**")

                    # 日付 — patiroha.parse_date
                    _t0 = time.time()
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
                    _phase_secs['Phase6 メタデータ正規化'] = time.time() - _t0
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
                    _total_secs = time.time() - start_time
                    status_text.success(f"✅ 分析エンジン起動完了 (所要時間: {int(_total_secs)}秒)")
                    # フェーズ別所要時間の内訳（ボトルネック特定用・計測のみでディスク永続化なし）
                    # 端末へは print せず、内訳はアプリ内の expander で確認できるようにする。
                    if _phase_secs:
                        with st.expander("⏱️ フェーズ別所要時間（ボトルネック確認）", expanded=False):
                            for _k, _v in sorted(_phase_secs.items(), key=lambda kv: kv[1], reverse=True):
                                _pct = (_v / _total_secs * 100) if _total_secs > 0 else 0
                                st.write(f"- **{_k}**: {_v:.1f} 秒 ({_pct:.0f}%)")
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