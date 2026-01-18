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

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from janome.tokenizer import Tokenizer

warnings.filterwarnings('ignore')

# ==================================================================
# --- ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO | Mission Control", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================================
# --- 定数とヘルパー関数 ---
# ==================================================================

import io

import utils

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_tokenizer():
    return Tokenizer()

t = load_tokenizer()

def advanced_tokenize(text):
    # ストップワードを動的に取得
    if 'stopwords' in st.session_state and st.session_state['stopwords']:
        current_stopwords = st.session_state['stopwords']
    else:
        current_stopwords = utils.get_stopwords()

    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'[\(（][\w\s]+[\)）]', ' ', text)
    text = re.sub(r'\b(図|fig|step|s)\s?\d+\b', ' ', text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)
    
    tokens = list(t.tokenize(text))
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token1 = tokens[i]
        base_form = token1.base_form if token1.base_form != '*' else token1.surface
        
        if base_form in current_stopwords or len(base_form) < 2:
            i += 1
            continue
        
        if (i + 1) < len(tokens):
            token2 = tokens[i+1]
            base_form2 = token2.base_form if token2.base_form != '*' else token2.surface
            pos1 = token1.part_of_speech.split(',')[0]
            pos2 = token2.part_of_speech.split(',')[0]
            if pos1 == '名詞' and pos2 == '名詞' and base_form2 not in current_stopwords:
                compound_word = base_form + base_form2
                processed_tokens.append(compound_word)
                i += 2
                continue
        
        pos = token1.part_of_speech.split(',')[0]
        if pos == '名詞':
            processed_tokens.append(base_form)
        i += 1
    return " ".join(processed_tokens)

def robust_parse_date(series):
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.notna().mean() > 0.5: return parsed
    
    parsed = pd.to_datetime(series, format='%Y%m%d', errors='coerce')
    if parsed.notna().mean() > 0.5: return parsed
    
    parsed = pd.to_datetime(series, format='%Y', errors='coerce')
    if parsed.notna().mean() > 0.5: return parsed
    
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.notna().sum() > 0 and numeric_series.mean() > 30000:
            parsed = pd.to_datetime(numeric_series, unit='D', origin='1899-12-30', errors='coerce')
            return parsed
    except:
        pass
    return parsed

def extract_ipc(text, delimiter=';'):
    if not isinstance(text, str): return [] 
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'[\(（][^)]*[\)）]', ' ', text)
    ipc_codes = []
    parts = text.split(delimiter)
    for part in parts:
        part = part.strip()
        if not part: continue
        match = re.search(r'([a-z]\d{2}[a-z])\s*(\d{1,4}/\d{2,})', part)
        if match:
            ipc_code = match.group(1) + match.group(2)
            ipc_codes.append(ipc_code)
        else:
            match_main = re.search(r'\b([a-z]\d{2}[a-z])\b', part)
            if match_main:
                ipc_codes.append(match_main.group(1))
    return ipc_codes 

def smart_map_index(current_value, options, keywords):
    """
    カラム紐付けの自動化ロジック
    """
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
        "preprocess_done": False
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
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
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
                
        st.markdown("---")
        st.markdown("##### (オプション) 非特許情報 (NPL) のインポート")
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

            # --- タブ 1: Academic (論文) ---
            with npl_tabs[0]:
                st.markdown("###### 📚 Academic (論文データ)")
                f_aca = st.file_uploader("論文データ (LENSなど) をアップロード", type=["csv", "xlsx"], key="up_aca", accept_multiple_files=False)
                
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
        st.markdown("##### patent_dfのカラムを分析用フィールドにマッピングします。")
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
        if 'stopwords' not in st.session_state or not st.session_state['stopwords']:
            st.session_state['stopwords'] = utils.get_stopwords()
        
        # 検索機能
        search_query = st.text_input("リスト内検索 (正規表現も可)", placeholder="検索したい単語を入力...", key="sw_search")
        
        full_stopwords = sorted(list(st.session_state['stopwords']))
        
        if search_query:
            try:
                # 正規表現検索を試みる
                filtered_stopwords = [w for w in full_stopwords if re.search(search_query, w)]
            except re.error:
                # 正規表現エラー時は単純な部分一致
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
            

            editor_key = f"stopwords_editor_{hash(search_query)}" 
            new_stopwords_text = st.text_area(f"ストップワードリスト{label_suffix}", value=stopwords_text, height=300, key=editor_key)
            
            if st.button("変更を適用", key="apply_stopwords"):
                edited_lines = set([line.strip() for line in new_stopwords_text.split('\n') if line.strip()])
                
                if is_filtered:
                    # フィルタリング時のスマートマージ
                    # 1. 検索ヒットしていたはずの元の単語群 (変更前)
                    original_matches = set(filtered_stopwords)
                    # 2. 削除された単語 = (元ヒット) - (編集後)
                    removed_words = original_matches - edited_lines
                    # 3. 追加された単語 = (編集後) - (元ヒット)
                    added_words = edited_lines - original_matches
                    
                    # 4. メインリストから削除対象を除き、追加分を足す
                    current_set = st.session_state['stopwords']
                    new_set = (current_set - removed_words) | added_words
                    st.session_state['stopwords'] = new_set
                    msg = f"更新完了: {len(added_words)} 語を追加, {len(removed_words)} 語を削除しました。"
                else:
                    # 全量置換
                    st.session_state['stopwords'] = edited_lines
                    msg = f"リストを全量更新しました (計 {len(edited_lines)} 語)。"
                
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
                    if st.button(f"追加インポート ({len(imported_lines)}語)", key="import_sw"):
                        st.session_state['stopwords'].update(imported_lines)
                        st.success("インポートしました。")
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

                    # 1. モデル読み込み (初期化)
                    status_text.markdown("🔄 **Phase 1/6: モデルロード中...**")
                    update_progress('init', 0.5)

                    df = st.session_state.df_main.copy() 
                    col_map = st.session_state.col_map
                    delimiters = st.session_state.delimiters
                    
                    sbert_model = load_sbert_model()
                    st.session_state.sbert_model = sbert_model
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
                    status_text.markdown("🔄 **Phase 3/6: 非特許情報(NPL)の個別処理中...**")
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
                        # 特許固有項目はダミーまたは空
                        df_n[col_map['app_num']] = 'NPL-' + df_n.index.astype(str)
                        
                        # 日付パース (ロバスト - NPL用直接処理)
                        raw_dates_n = df_n[col_map['date']].astype(str)
                        # 強制変換、エラーはNaTに
                        df_n['parsed_date'] = pd.to_datetime(raw_dates_n, errors='coerce')
                        
                        # 全て失敗した場合の"YYYY"形式のフォールバック
                        if df_n['parsed_date'].isna().all():
                             # 可能であれば年整数としてパース
                             df_n['parsed_date'] = pd.to_datetime(raw_dates_n, format='%Y', errors='coerce')

                        df_n['year'] = df_n['parsed_date'].dt.year
                        
                        # 'year'のNaN処理 (Nebulaはフィルタリングするため現状はNaNのまま)
                        # NaTの場合は正規表現で年を抽出
                        mask_nat = df_n['parsed_date'].isna()
                        if mask_nat.any():
                            # 4桁数字の正規表現抽出
                            extracted_years = raw_dates_n[mask_nat].str.extract(r'(\d{4})')[0]
                            # 該当行の'year'カラムを直接更新
                            df_n.loc[mask_nat, 'year'] = pd.to_numeric(extracted_years, errors='coerce')

                        # Academic/Newsのみキーワード抽出
                        npl_text_series = n_title + ' ' + n_content
                        sw_list = utils.get_npl_stopwords()
                        
                        def process_npl_keywords(row):
                            sub_type = str(row.get('data_sub_type', ''))
                            if sub_type in ['Academic', 'Business', 'Academic Source', 'News Source']:
                                t_val = str(row['unified_title']) if pd.notna(row['unified_title']) else ""
                                c_val = str(row['unified_content']) if pd.notna(row['unified_content']) else ""
                                txt = t_val + " " + c_val
                                return utils.extract_keywords(txt, t, sw_list)
                            else:
                                return []
                        
                        # Apply
                        df_n['explorer_keywords'] = df_n.apply(process_npl_keywords, axis=1)
                        
                        # セッション状態に保存
                        st.session_state.df_npl = df_n
                    
                    update_progress('text', 1.0)

                    # 4. SBERTエンコード (Patent ONLY)
                    status_text.markdown("🔄 **Phase 4/6: AIベクトル化 (SBERT - 特許のみ)...**")
                    texts_for_sbert_list = df['text_for_sbert'].tolist()
                    batch_size = 128
                    total_batches = (len(texts_for_sbert_list) + batch_size - 1) // batch_size
                    embeddings_list = []
                    
                    for i in range(total_batches):
                        batch_texts = texts_for_sbert_list[i*batch_size : (i+1)*batch_size]
                        batch_embeddings = sbert_model.encode(batch_texts, show_progress_bar=False)
                        embeddings_list.append(batch_embeddings)
                        
                        phase_prog = (i + 1) / total_batches
                        el_str, et_str = update_progress('sbert', phase_prog)
                        status_text.markdown(f"🔄 **Phase 4/6: AIベクトル化 (SBERT) 実行中...** (Batch {i+1}/{total_batches})\n\n⏱️ 経過: {el_str} | ⏳ 残り: {et_str} (目安)")
                    
                    sbert_embeddings = np.vstack(embeddings_list)
                    sbert_embeddings = normalize(sbert_embeddings, norm='l2')
                    st.session_state.sbert_embeddings = sbert_embeddings

                    # 5. TF-IDF & Keyword (Patent ONLY)
                    status_text.markdown("🔄 **Phase 5/6: キーワード抽出 (TF-IDF - 特許のみ)...**")
                    # Explorer用 (キーワードリスト)
                    sw_list = utils.get_stopwords()
                    df['explorer_keywords'] = df['text_for_sbert'].apply(lambda x: utils.extract_keywords(x, t, sw_list))
                    
                    # 検索用 (TF-IDF行列)
                    df['text_for_tfidf'] = df['text_for_sbert'].apply(advanced_tokenize)
                    vectorizer = TfidfVectorizer(max_features=None, min_df=5, max_df=0.80)
                    st.session_state.tfidf_matrix = vectorizer.fit_transform(df['text_for_tfidf'])
                    st.session_state.feature_names = np.array(vectorizer.get_feature_names_out())
                    update_progress('tfidf', 1.0)

                    # 6. 正規化 (Patent Norm)
                    status_text.markdown("🔄 **Phase 6/6: メタデータ (日付・IPC・出願人) 正規化中...**")
                    raw_dates = df[col_map['date']].astype(str)
                    df['parsed_date'] = robust_parse_date(raw_dates)
                    df['year'] = df['parsed_date'].dt.year
                    df['app_num_main'] = df[col_map['app_num']].astype(str).str.strip()

                    ipc_delimiter = delimiters['ipc']
                    df['ipc_normalized'] = df[col_map['ipc']].apply(lambda x: extract_ipc(x, ipc_delimiter))
                    ipc_raw_list = df[col_map['ipc']].fillna('').astype(str).str.split(ipc_delimiter)
                    df['ipc_main_group'] = ipc_raw_list.apply(lambda terms: list(set([t.strip().split('/')[0].strip().upper() for t in terms if t.strip()])))

                    if col_map['fterm']:
                        fterm_delimiter = delimiters['fterm']
                        fterm_raw_list = df[col_map['fterm']].fillna('').astype(str).str.split(fterm_delimiter)
                        df['fterm_main'] = fterm_raw_list.apply(lambda terms: list(set([t.strip()[:5].upper() for t in terms if t.strip() and len(t) >= 5])))
                    else:
                        df['fterm_main'] = [[] for _ in range(len(df))]

                    applicant_delimiter = delimiters['applicant']
                    applicant_raw_list = df[col_map['applicant']].fillna('').astype(str).str.split(applicant_delimiter)
                    df['applicant_main'] = applicant_raw_list.apply(lambda names: list(set([n.strip() for n in names if n.strip()])))
                    
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
                    
                    # 6. クリーンアップ (Clean)
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