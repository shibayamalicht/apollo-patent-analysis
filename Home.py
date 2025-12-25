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
        "shared_df": None,
        "filename": "No File",
        "sbert_model": None,
        "sbert_embeddings": None,
        "tfidf_matrix": None,
        "feature_names": None,
        "col_map": {},
        "delimiters": {'applicant': ';', 'inventor': ';', 'ipc': ';', 'fterm': ';'},
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
                
    # A-2. カラム紐付け
    with tab2:
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

                # IPC (Required)
                col_map['ipc'] = st.selectbox("国際特許分類 (IPC):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('ipc'), columns_with_none, kw_ipc), key="col_ipc")
                ipc_delimiter = st.text_input("IPC区切り文字:", value=st.session_state.delimiters.get('ipc', ';'), key="del_ipc")
                
            with col3:
                st.markdown("##### 任意メタデータ項目")
                
                # Inventor
                col_map['inventor'] = st.selectbox("発明者 (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('inventor'), columns_with_none, kw_inventor), key="col_inventor")
                inventor_delimiter = st.text_input("発明者区切り文字:", value=st.session_state.delimiters.get('inventor', ';'), key="del_inventor")
                

                
                # F-term
                col_map['fterm'] = st.selectbox("Fターム (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('fterm'), columns_with_none, kw_fterm), key="col_fterm")
                fterm_delimiter = st.text_input("Fターム区切り文字:", value=st.session_state.delimiters.get('fterm', ';'), key="del_fterm") 
                
                # Status
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

                    # 1. モデルロード (Init)
                    status_text.markdown("🔄 **Phase 1/6: モデルロード中...**")
                    update_progress('init', 0.5)
                    
                    df = st.session_state.df_main.copy() 
                    col_map = st.session_state.col_map
                    delimiters = st.session_state.delimiters
                    
                    sbert_model = load_sbert_model()
                    st.session_state.sbert_model = sbert_model
                    update_progress('init', 1.0)

                    # 2. テキスト結合 (Text)
                    status_text.markdown("🔄 **Phase 2/6: テキストデータを結合中...**")
                    df['text_for_sbert'] = (
                        df[col_map['title']].fillna('') + ' ' +
                        df[col_map['abstract']].fillna('') + ' ' +
                        df[col_map['claim']].fillna('')
                    )
                    update_progress('text', 1.0)

                    # 3. SBERTエンコード (SBERT)
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
                        status_text.markdown(f"🔄 **Phase 3/6: AIベクトル化 (SBERT) 実行中...** (Batch {i+1}/{total_batches})\n\n⏱️ 経過: {el_str} | ⏳ 残り: {et_str} (目安)")
                    
                    sbert_embeddings = np.vstack(embeddings_list)
                    sbert_embeddings = normalize(sbert_embeddings, norm='l2')
                    st.session_state.sbert_embeddings = sbert_embeddings

                    # 4. TF-IDF (TF-IDF)
                    status_text.markdown("🔄 **Phase 4/6: キーワード抽出 (TF-IDF) 計算中...**")
                    df['text_for_tfidf'] = df['text_for_sbert'].apply(advanced_tokenize)
                    vectorizer = TfidfVectorizer(max_features=None, min_df=5, max_df=0.80)
                    st.session_state.tfidf_matrix = vectorizer.fit_transform(df['text_for_tfidf'])
                    st.session_state.feature_names = np.array(vectorizer.get_feature_names_out())
                    update_progress('tfidf', 1.0)

                    # 5. 正規化 (Norm)
                    status_text.markdown("🔄 **Phase 5/6: メタデータ (日付・IPC・出願人) 正規化中...**")
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