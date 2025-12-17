# ==================================================================
# --- 0. 環境変数の設定 (最優先) ---
# ==================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['OMP_NUM_THREADS'] = '1'

# ==================================================================
# --- 1. ライブラリのインポート ---
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
# --- 2. ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO | Mission Control", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================================
# --- 3. 定数・共通関数定義 ---
# ==================================================================

STOP_WORDS = {
    "する","ある","なる","ため","こと","よう","もの","これ","それ","あれ","ここ","そこ","どれ","どの",
    "この","その","当該","該","および","及び","または","また","例えば","例えばは","において","により",
    "に対して","に関して","について","として","としては","場合","一方","他方","さらに","そして","ただし",
    "なお","等","など","等々","いわゆる","所謂","同様","同時","前記","本","同","各","各種","所定","所望",
    "一例","他","一部","一つ","複数","少なくとも","少なくとも一つ","上記","下記","前述","後述","既述",
    "関する","基づく","用いる","使用","利用","有する","含む","備える","設ける","すなわち","従って",
    "しかしながら","次に","特に","具体的に","詳細に","いずれ","うち","それぞれ","とき",
    "かかる","かような","かかる場合","本件","本願","本出願","本明細書","これら","それら","各々","随時","適宜",
    "任意","必ずしも","通常","一般に","典型的","代表的",
    "本発明","発明","実施例","実施形態","変形例","請求","請求項","図","図面","符号","符号の説明",
    "図面の簡単な説明","発明の詳細な説明","技術分野","背景技術","従来技術","発明が解決しようとする課題","課題",
    "解決手段","効果","要約","発明の効果","目的","手段","構成","構造","工程","処理","方法","手法","方式",
    "システム","プログラム","記憶媒体","特徴","特徴とする","特徴部","ステップ","フロー","シーケンス","定義",
    "関係","対応","整合","実施の形態","実施の態様","態様","変形","修正例","図示","図示例","図示しない",
    "参照","参照符号","段落","詳細説明","要旨","一実施形態","他の実施形態","一実施例","別の側面","付記",
    "適用例","用語の定義","開示","本開示","開示内容","記載","記述","掲載","言及","内容","詳細","説明","表記","表現","箇条書き","以下の","以上の","全ての","任意の","特定の",
    "上部","下部","内部","外部","内側","外側","表面","裏面","側面","上面","下面","端面","先端","基端","後端","一端","他端","中心","中央","周縁","周辺",
    "近傍","方向","位置","空間","領域","範囲","間隔","距離","形状","形態","状態","種類","層","膜","部",
    "部材","部位","部品","機構","装置","容器","組成","材料","用途","適用","適用例","片側","両側","左側",
    "右側","前方","後方","上流","下流","隣接","近接","離間","間置","介在","重畳","概ね","略","略中央",
    "固定側","可動側","伸長","収縮","係合","嵌合","取付","連結部","支持体","支持部","ガイド部",
    "データ","情報","信号","出力","入力","制御","演算","取得","送信","受信","表示","通知","設定","変更",
    "更新","保存","削除","追加","実行","開始","終了","継続","停止","判定","判断","決定","選択","特定",
    "抽出","検出","検知","測定","計測","移動","回転","変位","変形","固定","配置","生成","付与","供給",
    "適用","照合","比較","算出","解析","同定","初期化","読出","書込","登録","記録","配信","連携","切替",
    "起動","復帰","監視","通知処理","取得処理","演算処理","良好","容易","簡便","適切","有利","有用","有効",
    "効果的","高い","低い","大きい","小さい","新規","改良","改善","抑制","向上","低減","削減","増加",
    "減少","可能","好適","好ましい","望ましい","優れる","優れた","高性能","高効率","低コスト","コスト",
    "簡易","安定","安定性","耐久","耐久性","信頼性","簡素","簡略","単純","最適","最適化","汎用","汎用性",
    "実現","達成","確保","維持","防止","回避","促進","不要","必要","高精度","省電力","省資源","高信頼",
    "低負荷","高純度","高密度","高感度","迅速","円滑","簡略化","低価格","実効的","可能化","有効化",
    "非必須","適合","互換","出願","出願人","出願番号","出願日","出願書","出願公開","公開","公開番号",
    "公開公報","公報","公報番号","特許","特許番号","特許文献","非特許文献","引用","引用文献","先行技術",
    "審査","審査官","拒絶","意見書","補正書","優先","優先日","分割出願","継続出願","国内移行","国際出願",
    "国際公開","PCT","登録","公開日","審査請求","拒絶理由","補正","訂正","無効審判","異議","取消","取下げ",
    "事件番号","代理人","弁理士","係属","経過",
    "第","第一","第二","第三","第1","第２","第３","第１","第２","第３","１","２","３","４","５","６","７","８","９","０",
    "一","二","三","四","五","六","七","八","九","零","数","複合","多数","少数","図1","図2","図3","図4","図5","図6","図7","図8","図9",
    "表1","表2","表3","式1","式2","式3","０","１","２","３","４","５","６","７","８","９","%","％","wt%","vol%","質量%","重量%","容量%","mol","mol%","mol/L","M","mm","cm","m","nm","μm","μ","rpm",
    "Pa","kPa","MPa","GPa","N","W","V","A","mA","Hz","kHz","MHz","GHz","℃","°C","K","mL","L","g","kg","mg","wt","vol",
    "h","hr","hrs","min","s","sec","ppm","ppb","bar","Ω","ohm","J","kJ","Wh","kWh",
    "株式会社","有限会社","合資会社","合名会社","合同会社","Inc","Inc.","Ltd","Ltd.","Co","Co.","Corp","Corp.","LLC",
    "GmbH","AG","BV","B.V.","S.A.","S.p.A.","（株）","㈱","（有）",
    "溶液","溶媒","触媒","反応","生成物","原料","成分","含有","含有量","配合","混合","混合物","濃度","温度","時間",
    "割合","比率","基","官能基","化合物","組成物","樹脂","ポリマー","モノマー","基板","基材","フィルム","シート",
    "粒子","粉末","比較例","参考例","試験","試料","評価","条件","実験","実験例","反応条件","反応時間","反応温度",
    "処理装置","端末","ユニット","モジュール","回路","素子","電源","電圧","電流","信号線","配線","端子","端部","接続",
    "接続部","演算部","記憶部","記憶装置","記録媒体","ユーザ","利用者","クライアント","サーバ","画面","UI","GUI",
    "インターフェース","データベース","DB","ネットワーク","通信","要求","応答","リクエスト","レスポンス","パラメータ",
    "引数","属性","プロパティ","フラグ","ID","ファイル","データ構造","テーブル","レコード",
    "軸","シャフト","ギア","モータ","エンジン","アクチュエータ","センサ","バルブ","ポンプ","筐体","ハウジング","フレーム",
    "シャーシ","駆動","伝達","支持","連結","解決", "準備", "提供", "発生", "以上", "十分",
    "できる", "いる", "明細書"
}

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_tokenizer():
    return Tokenizer()

t = load_tokenizer()

def advanced_tokenize(text):
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
        
        if base_form in STOP_WORDS or len(base_form) < 2:
            i += 1
            continue
        
        if (i + 1) < len(tokens):
            token2 = tokens[i+1]
            base_form2 = token2.base_form if token2.base_form != '*' else token2.surface
            pos1 = token1.part_of_speech.split(',')[0]
            pos2 = token2.part_of_speech.split(',')[0]
            if pos1 == '名詞' and pos2 == '名詞' and base_form2 not in STOP_WORDS:
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
    カラム紐付けの自動化ロジック (Fix: 初期状態Noneでも検索を実行)
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
# --- 4. デザイン設定 ---
# ==================================================================

st.markdown("""
<style>
    html, body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    [data-testid="stSidebar"] h1 { color: #003366; font-weight: 900 !important; font-size: 2.5rem !important; }
    h1 { color: #003366; font-weight: 700; }
    h2, h3 { color: #333333; font-weight: 500; border-bottom: 2px solid #f0f0f0; padding-bottom: 5px; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #003366; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("APOLLO") 
    st.markdown("Advanced Patent & Overall Landscape-analytics Logic Orbiter")
    st.markdown("**v.3**")
    st.markdown("---")
    st.subheader("Home"); st.page_link("Home.py", label="Mission Control", icon="🛰️")
    st.subheader("Modules")
    st.page_link("pages/1_🌍_ATLAS.py", label="ATLAS", icon="🌍")
    st.page_link("pages/2_💡_CORE.py", label="CORE", icon="💡")
    st.page_link("pages/3_🚀_Saturn_V.py", label="Saturn V", icon="🚀")
    st.page_link("pages/4_📈_MEGA.py", label="MEGA", icon="📈")
    st.page_link("pages/5_🧭_Explorer.py", label="Explorer", icon="🧭")
    st.page_link("pages/6_🔗_CREW.py", label="CREW", icon="🔗")
    st.markdown("---")
    st.caption("ナビゲーション:\n1. Mission Control でデータをアップロードし、前処理を実行します。\n2. 上のリストから分析モジュールを選択します。")
    st.markdown("---")
    st.caption("© 2025 しばやま")

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

st.markdown("---")
st.subheader("分析設定")

container = st.container() 

with container:
    tab1, tab2, tab3 = st.tabs([
        "フェーズ 1: データインポート", 
        "フェーズ 2: カラム紐付け", 
        "フェーズ 3: 分析エンジン起動"
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
                
                col_map['inventor'] = st.selectbox("発明者 (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('inventor'), columns_with_none, kw_inventor), key="col_inventor")
                inventor_delimiter = st.text_input("発明者区切り文字:", value=st.session_state.delimiters.get('inventor', ';'), key="del_inventor")

            with col3:
                st.markdown("##### 分析軸項目")
                col_map['ipc'] = st.selectbox("IPC:", columns_with_none, index=smart_map_index(st.session_state.col_map.get('ipc'), columns_with_none, kw_ipc), key="col_ipc")
                ipc_delimiter = st.text_input("IPC区切り文字:", value=st.session_state.delimiters.get('ipc', ';'), key="del_ipc")
                col_map['fterm'] = st.selectbox("Fターム (任意):", columns_with_none, index=smart_map_index(st.session_state.col_map.get('fterm'), columns_with_none, kw_fterm), key="col_fterm")
                fterm_delimiter = st.text_input("Fターム区切り文字:", value=st.session_state.delimiters.get('fterm', ';'), key="del_fterm") 
                
            st.session_state.col_map = col_map
            st.session_state.delimiters = {
                'applicant': applicant_delimiter,
                'inventor': inventor_delimiter,
                'ipc': ipc_delimiter,
                'fterm': fterm_delimiter
            }
        else:
            st.info("フェーズ1でファイルをインポートすると、カラム紐付け設定が表示されます。")

    # A-3. 前処理実行
    with tab3:
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