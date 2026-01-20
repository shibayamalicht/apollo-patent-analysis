import streamlit as st
import platform
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import string
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from janome.tokenizer import Tokenizer

# ==================================================================
# --- 1. フォント設定 (共通) ---
# ==================================================================
def get_japanese_font_path():
    """OSを判定して適切な日本語フォントパスを返す"""
    system = platform.system()
    font_paths = []
    
    if system == "Darwin": # Mac
        font_paths = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans W3.ttc",
            "/System/Library/Fonts/Hiragino Kaku Gothic ProN.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc" 
        ]
    elif system == "Windows": # Windows
        font_paths = [
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/yugothr.ttc",
            "C:/Windows/Fonts/YuGothR.ttc"
        ]
    else: # Linux (Streamlit Cloudなど)
        font_paths = [
            "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto/NotoSansCJKjp-Regular.otf"
        ]
        
    for path in font_paths:
        if os.path.exists(path): return path
    return None

def configure_matplotlib_font():
    """Matplotlibのフォント設定を適用する"""
    font_path = get_japanese_font_path()
    if font_path:
        try:
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            return font_path
        except:
            pass
    return None

# ==================================================================
# --- 2. ストップワード (共通) ---
# ==================================================================
# 基本ストップワードリスト

# 1. 一般的な日本語ストップワード（接続詞・代名詞・形式名詞など）
_sw_general = [
    "する","ある","なる","ため","こと","よう","もの","これ","それ","あれ","ここ","そこ","どれ","どの",
    "この","その","当該","該","および","及び","または","また","例えば","例えばは","において","により",
    "に対して","に関して","について","として","としては","場合","一方","他方","さらに","そして","ただし",
    "なお","等","など","等々","いわゆる","所謂","同様","同時","前記","本","同","各","各種","所定","所望",
    "一例","他","一部","一つ","複数","少なくとも","少なくとも一つ","上記","下記","前述","後述","既述",
    "関する","基づく","用いる","使用","利用","有する","含む","備える","設ける","すなわち","従って",
    "しかしながら","次に","特に","具体的に","詳細に","いずれ","うち","それぞれ","とき",
    "かかる","かような","かかる場合","本件","本願","本出願","本明細書","これら","それら","各々","随時","適宜",
    "任意","必ずしも","通常","一般に","典型的","代表的","並びに","若しくは","又は","且つ","即ち","何ら","一切",
    "係る","関わる","介して","沿って","伴う","基づいて","更なる","単数","全体","全部","大半","約","概して","ほぼ",
    "できる", "いる", "明細書", "解決", "準備", "提供", "発生", "未満", "超", "際", "十分"
]

# 2. 特許特有の専門用語・定型句・区分
_sw_patent_terms = [
    "本発明","発明","実施例","実施形態","変形例","請求","請求項","図","図面","符号","符号の説明",
    "図面の簡単な説明","発明の詳細な説明","技術分野","背景技術","従来技術","発明が解決しようとする課題","課題",
    "解決手段","効果","要約","発明の効果","目的","手段","構成","構造","工程","処理","方法","手法","方式",
    "特徴","特徴とする","特徴部","ステップ","フロー","シーケンス","定義",
    "関係","対応","整合","実施の形態","実施の態様","態様","変形","修正例","図示","図示例","図示しない",
    "参照","参照符号","段落","詳細説明","要旨","一実施形態","他の実施形態","一実施例","別の側面","付記",
    "適用例","用語の定義","開示","本開示","開示内容","記載","記述","掲載","言及","内容","詳細","説明","表記","表現","箇条書き","以下の","以上の","全ての","任意の","特定の",
    "出願","出願人","出願番号","出願日","出願書","出願公開","公開","公開番号",
    "公開公報","公報","公報番号","特許","特許番号","特許文献","非特許文献","引用","引用文献","先行技術",
    "審査","審査官","拒絶","意見書","補正書","優先","優先日","分割出願","継続出願","国内移行","国際出願",
    "国際公開","PCT","登録","公開日","審査請求","拒絶理由","補正","訂正","無効審判","異議","取消","取下げ",
    "公知","周知","慣用","既知","市販","容易","困難","不可能","重要","問題","結果","作用",
    "事件番号","代理人","弁理士","係属","経過", "比較例","参考例","試験","試料","評価","条件","実験","実験例"
]

# 3. 構造・位置・方向・形状（一般名詞）
_sw_structure = [
    "上部","下部","内部","外部","内側","外側","表面","裏面","側面","上面","下面","端面","先端","基端","後端","一端","他端","中心","中央","周縁","周辺",
    "近傍","方向","位置","空間","領域","範囲","間隔","距離","形状","形態","状態","種類","層","膜","部",
    "部材","部位","部品","機構","装置","容器","組成","材料","用途","適用","適用例","片側","両側","左側",
    "右側","前方","後方","上流","下流","隣接","近接","離間","間置","介在","重畳","概ね","略","略中央",
    "固定側","可動側","伸長","収縮","係合","嵌合","取付","連結部","支持体","支持部","ガイド部",
    "軸","シャフト","ギア","モータ","エンジン","アクチュエータ","センサ","バルブ","ポンプ","筐体","ハウジング","フレーム",
    "シャーシ","駆動","伝達","支持","連結", "処理装置","端末","ユニット","モジュール","回路","素子"
]

# 4. IT・データ・制御関連
_sw_it_control = [
    "システム","プログラム","記憶媒体","データ","情報","信号","出力","入力","制御","演算","取得","送信","受信","表示","通知","設定","変更",
    "更新","保存","削除","追加","実行","開始","終了","継続","停止","判定","判断","決定","選択","特定",
    "抽出","検出","検知","測定","計測","移動","回転","変位","変形","固定","配置","生成","付与","供給",
    "適用","照合","比較","算出","解析","同定","初期化","読出","書込","登録","記録","配信","連携","切替",
    "起動","復帰","監視","通知処理","取得処理","演算処理",
    "電源","電圧","電流","信号線","配線","端子","端部","接続","接続部","演算部","記憶部","記憶装置","記録媒体",
    "ユーザ","利用者","クライアント","サーバ","画面","UI","GUI",
    "インターフェース","データベース","DB","ネットワーク","通信","要求","応答","リクエスト","レスポンス","パラメータ",
    "引数","属性","プロパティ","フラグ","ID","ファイル","データ構造","テーブル","レコード"
]

# 5. 化学・材料・実験条件
_sw_chemistry = [
    "溶液","溶媒","触媒","反応","生成物","原料","成分","含有","含有量","配合","混合","混合物","濃度","温度","時間",
    "割合","比率","基","官能基","化合物","組成物","樹脂","ポリマー","モノマー","基板","基材","フィルム","シート",
    "粒子","粉末","反応条件","反応時間","反応温度",
    "良好","容易","簡便","適切","有利","有用","有効",
    "効果的","高い","低い","大きい","小さい","新規","改良","改善","抑制","向上","低減","削減","増加",
    "減少","可能","好適","好ましい","望ましい","優れる","優れた","高性能","高効率","低コスト","コスト",
    "簡易","安定","安定性","耐久","耐久性","信頼性","簡素","簡略","単純","最適","最適化","汎用","汎用性",
    "実現","達成","確保","維持","防止","回避","促進","不要","必要","高精度","省電力","省資源","高信頼",
    "低負荷","高純度","高密度","高感度","迅速","円滑","簡略化","低価格","実効的","可能化","有効化",
    "非必須","適合","互換"
]

# 6. 数字・単位・特殊記号・法人格
_sw_misc = [
    "第","第一","第二","第三","第1","第２","第３","第１","第２","第３","１","２","３","４","５","６","７","８","９","０",
    "一","二","三","四","五","六","七","八","九","零","数","複合","多数","少数","図1","図2","図3","図4","図5","図6","図7","図8","図9",
    "表1","表2","表3","式1","式2","式3","０","１","２","３","４","５","６","７","８","９","%","％","wt%","vol%","質量%","重量%","容量%","mol","mol%","mol/L","M","mm","cm","m","nm","μm","μ","rpm",
    "Pa","kPa","MPa","GPa","N","W","V","A","mA","Hz","kHz","MHz","GHz","℃","°C","K","mL","L","g","kg","mg","wt","vol",
    "h","hr","hrs","min","s","sec","ppm","ppb","bar","Ω","ohm","J","kJ","Wh","kWh",
    "株式会社","有限会社","合資会社","合名会社","合同会社","Inc","Inc.","Ltd","Ltd.","Co","Co.","Corp","Corp.","LLC",
    "GmbH","AG","BV","B.V.","S.A.","S.p.A.","（株）","㈱","（有）",
    "以上", "以下"
]

# 7. 非特許文献 (NPL: Academic, News)
_sw_npl = [
    "論文","研究","報告","学会","発表","大学","研究所","教授","博士","著者","Abstract","Introduction","Conclusion","Fig","Table","et al",
    "ニュース","報道","市場","企業","会社","本日","現地時間","増","減","前年比","見通し","予測","アナリスト","価格","円","ドル",
    "発売","開始","終了","買収","提携","合意","設立","方針","戦略","計画","事業","売上","利益","業績","決算",
    "株価","株式","投資","ファンド","調達","資金","規模","シェア","拡大","縮小","成長","減少","推移","動向", "概況", "見込み", "予想", "分析", "調査",
    
    # 英単語ストップワード (一般)
    "the", "of", "and", "a", "an", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", 
    "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", 
    "there", "use", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", 
    "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number", 
    "no", "way", "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now", "find", "day", "did", "get",
    "come", "made", "may", "part", "over", "new", "sound", "take", "only", "little", "work", "know", "place", "year", "live", "me", "back",
    "give", "most", "very", "after", "thing", "our", "just", "name", "good", "sentence", "man", "think", "say", "great", "where", "help",
    "through", "much", "before", "line", "right", "too", "mean", "old", "any", "same", "tell", "boy", "follow", "came", "want", "show",
    "also", "around", "form", "three", "small", "set", "put", "end", "does", "another", "well", "large", "must", "big", "even", "such",
    "because", "turn", "here", "why", "ask", "went", "men", "read", "need", "land", "different", "home", "us", "move", "try", "kind",
    "hand", "picture", "again", "change", "off", "play", "spell", "air", "away", "animal", "house", "point", "page", "letter", "mother",
    "answer", "found", "should", "america", "world", "high",
    
    # 前置詞・接続詞
    "above", "across", "against", "along", "among", "behind", "below", "beneath", "beside", "between", "down",
    "during", "except", "inside", "near", "onto", "outside", "past",
    "since", "throughout", "toward", "under", "underneath", "until", "upon", "within", "without",
    "although", "unless", "while", "whenever", "wherever", "whether", "though", "even though",

    # 代名詞
    "mine", "myself", "yours", "yourself", "himself", "hers", "herself",
    "itself", "ours", "ourselves", "theirs", "themselves",
    "whom", "whose", "those",

    # 学術・研究用語
    "abstract", "introduction", "method", "methodology", "results", "discussion", "conclusion", "references", "appendix", "figure", "table",
    "study", "research", "analysis", "investigation", "examination", "experiment", "survey", "review",
    "data", "dataset", "information", "evidence", "findings", "outcome", "performance", "evaluation",
    "proposed", "presented", "developed", "designed", "implemented", "demonstrated", "shown", "observed", "measured",
    "significant", "important", "major", "key", "main", "primary", "secondary", "novel", "existing", "previous", "prior", "related",
    "approach", "technique", "algorithm", "system", "model", "framework", "architecture", "mechanism", "process", "procedure",
    "based on", "using", "via", "utilizing", "employing", "applying",
    "however", "furthermore", "moreover", "therefore", "thus", "hence", "consequently", "accordingly", "additionally",
    "in addition", "in contrast", "on the other hand", "for example", "for instance", "such as",
    "et al", "et al.", "ibid", "op cit", "vs", "versus"
]

# 一般名詞 / ノイズ単語 (日本語) - NPLノイズ削減用
_sw_npl_noise = [
    "新た", "新規", "よう", "ため", "もの", "こと", "とき", "場合", "際", "点", "ほう", "どこ", "その他", "それら", "これら",
    "一覧", "概要", "詳細", "解説", "説明", "紹介", "ページ", "サイト", "記事", "関連", "リンク"
]
_sw_npl.extend(_sw_npl_noise)

# 統合リストの作成
# 統合リストの定義 (分離のため構成を変更)
def _get_expanded_set(word_list):
    """全角半角展開したセットを返すヘルパー"""
    expanded = set(word_list)
    hankaku = string.ascii_letters + string.digits
    zenkaku = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９"
    trans = str.maketrans(hankaku, zenkaku)
    for w in word_list:
        if any(c in hankaku for c in w): expanded.add(w.translate(trans))
    return expanded

def get_patent_stopwords():
    """特許分析用のストップワード (NPL用語を含まない)"""
    base_list = (
        _sw_general + 
        _sw_patent_terms + 
        _sw_structure + 
        _sw_it_control + 
        _sw_chemistry + 
        _sw_misc
    )
    return sorted(list(_get_expanded_set(base_list)))

def get_npl_stopwords():
    """NPL分析用のストップワード (特許用 + NPL用語)"""
    base_list = (
        _sw_general + 
        _sw_patent_terms + 
        _sw_structure + 
        _sw_it_control + 
        _sw_chemistry + 
        _sw_misc + 
        _sw_npl
    )
    return sorted(list(_get_expanded_set(base_list)))

def get_stopwords(mode="patent"):
    """
    互換性のためのラッパー
    mode: "patent" (Default) or "npl"
    """
    if mode == "npl":
        return get_npl_stopwords()
    else:
        return get_patent_stopwords()

# ==================================================================
# --- 3. サイドバー設定 (共通) ---
# ==================================================================
def render_sidebar():
    """共通サイドバーを描画する"""

    
    # 共通CSSの適用
    st.markdown("""
    <style>
        html, body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
        html { overflow-y: scroll !important; overflow-x: hidden !important; }
        body { width: 100% !important; min-width: 100% !important; overflow-x: hidden !important; }
        .stApp { overflow-x: hidden !important; }
        
        /* H1 Title Spacing */
        [data-testid="stSidebar"] h1 { 
            color: #003366; 
            font-weight: 900 !important; 
            font-size: 2.5rem !important; 
            margin-top: 0 !important; 
            padding-top: 0 !important; 
            margin-bottom: 0 !important;
        }
        h1 { color: #003366; font-weight: 700; }
        h2, h3 { color: #333333; font-weight: 500; border-bottom: 2px solid #f0f0f0; padding-bottom: 5px; }
        
        /* Hide default nav */
        [data-testid="stSidebarNav"] { display: none !important; }
        
        /* Remove Top Whitespace (Robust Selectors) */
        section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }
        [data-testid="stSidebarUserContent"] { padding-top: 0rem; }
        [data-testid="stSidebar"] .block-container { padding-top: 0rem; padding-bottom: 1rem; }
        
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton>button { font-weight: 600; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 15px; }
        .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #003366; }
        [data-testid="stSidebar"] h3 { border-bottom: none !important; padding-bottom: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("APOLLO") 
        st.markdown("Advanced Patent & Overall Landscape-analytics Logic Orbiter")
        st.markdown("**v6.0.1**")
        st.markdown("---")
        st.subheader("Home"); st.page_link("Home.py", label="Mission Control", icon="🛰️")
        st.subheader("Modules")
        st.page_link("pages/1_🌍_ATLAS.py", label="ATLAS", icon="🌍")
        st.page_link("pages/2_💡_CORE.py", label="CORE", icon="💡")
        st.page_link("pages/3_🚀_Saturn_V.py", label="Saturn V", icon="🚀")
        st.page_link("pages/7_🦅_EAGLE.py", label="EAGLE", icon="🦅")
        st.page_link("pages/4_📈_MEGA.py", label="MEGA", icon="📈")
        st.page_link("pages/5_🧭_Explorer.py", label="Explorer", icon="🧭")
        st.page_link("pages/6_🔗_CREW.py", label="CREW", icon="🔗")
        st.page_link("pages/9_🌌_NEBULA.py", label="NEBULA", icon="🌌")
        st.page_link("pages/8_📝_VOYAGER.py", label="VOYAGER", icon="📝")
        st.markdown("---")
        st.caption("ナビゲーション:\n1. Mission Control でデータをアップロードし、前処理を実行します。\n2. 上のリストから分析モジュールを選択します。")
        st.markdown("---")
        st.caption("© 2025-2026 しばやま")

# ==================================================================
# --- 4. テーマ設定 (共通) ---
# ==================================================================
def get_theme_config(theme_name):
    """テーマに応じたカラー設定を返す"""
    import plotly.express as px
    
    themes = {
        "APOLLO Standard": {
            "bg_color": "#ffffff",
            "text_color": "#333333",
            "sidebar_bg": "#f8f9fa",
            "plotly_template": "plotly_white",
            "color_sequence": px.colors.qualitative.G10,
            "accent_color": "#003366",
            "density_scale": "Blues",
            "css": """
                html, body { background-color: #ffffff; color: #333333; }
                [data-testid="stSidebar"] { background-color: #f8f9fa; }
                [data-testid="stHeader"] { background-color: #ffffff; }
                h1, h2, h3 { color: #003366; }
            """
        },
        "Modern Presentation": {
            "bg_color": "#fdfdfd",
            "text_color": "#2c3e50",
            "sidebar_bg": "#eaeaea",
            "plotly_template": "plotly_white",
            "color_sequence": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8ab17d"],
            "accent_color": "#264653",
            "density_scale": "Teal",
            "css": """
                html, body { background-color: #fdfdfd; color: #2c3e50; font-family: "Helvetica Neue", Arial, sans-serif; }
                [data-testid="stSidebar"] { background-color: #eaeaea; }
                [data-testid="stHeader"] { background-color: #fdfdfd; }
                h1, h2, h3 { color: #264653; font-family: "Georgia", serif; }
                .stButton>button { background-color: #264653; color: white; border-radius: 0px; }
            """
        }
    }
    return themes.get(theme_name, themes["APOLLO Standard"])

# ==================================================================
# --- 5. Snapshot (VOYAGER連携) ---
# ==================================================================
def calculate_hhi(counts):
    """ヘルフィンダール・ハーシュマン指数 (HHI) を計算し、公取委基準で判定する"""
    if not counts or sum(counts) == 0: return 0.0, "データ不足"
    
    total = sum(counts)
    shares = [c / total for c in counts]
    hhi = sum([s ** 2 for s in shares])
    
    # 公正取引委員会の基準 (0-1スケール)
    if hhi < 0.10: status = "競争的 (分散)"
    elif hhi < 0.18: status = "中程度の集中"
    else: status = "寡占的 (高集中)"
    
    return hhi, status

def calculate_cagr_slope(df_subset, year_col='year'):
    """年平均成長率(CAGR)とトレンド(Slope)を計算する"""
    if year_col not in df_subset.columns: return None, None
    
    years = df_subset[year_col].dropna().astype(int)
    if years.empty: return None, None
    
    counts = years.value_counts().sort_index()
    if len(counts) < 2: return 0.0, "Stable"
    
    # 直近3-5年のトレンドを見る
    y_vals = counts.index.values
    c_vals = counts.values
    
    # Slope (線形回帰)
    try:
        slope, _ = np.polyfit(y_vals, c_vals, 1)
        if slope > 0.5: trend = "急上昇 📈"
        elif slope > 0: trend = "増加傾向 ↗️"
        elif slope > -0.5: trend = "減少傾向 ↘️"
        else: trend = "失速 📉"
    except:
        trend = "不明"
        slope = 0
        
    # CAGR (最初と最後)
    try:
        start_val = c_vals[0] if c_vals[0] > 0 else 1
        end_val = c_vals[-1]
        n_years = max(1, y_vals[-1] - y_vals[0])
        cagr = (end_val / start_val) ** (1/n_years) - 1
    except:
        cagr = 0.0
        
    return cagr, trend

@st.cache_data(show_spinner=False)
def generate_rich_summary(df_target, title_col='title', abstract_col='abstract', n_representatives=5):
    """
    VOYAGER v5.1用の高解像度サマリを生成する (Cached)
    - 統計情報 (HHI, CAGR, Trend)
    - 代表特許 (Centroid Distance)
    """
    summary = {
        "stats": {},
        "representatives": []
    }
    
    # 1. 統計情報の計算 (年次推移がある場合)
    if 'year' in df_target.columns:
        cagr, trend = calculate_cagr_slope(df_target)
        summary['stats']['cagr'] = f"{cagr:.1%}" if cagr is not None else "N/A"
        summary['stats']['trend'] = trend if trend else "N/A"

    # 2. HHI (市場集中度) の計算
    try:
        # 出願人情報 ('applicant_main') を利用して市場集中度を算出
        if 'applicant_main' in df_target.columns:
            all_apps = [a for sublist in df_target['applicant_main'] for a in sublist]
            counts = pd.Series(all_apps).value_counts().tolist()
            hhi, hhi_status = calculate_hhi(counts)
            summary['stats']['hhi'] = hhi
            summary['stats']['hhi_status'] = hhi_status
    except: pass
        
    # 3. 代表特許の抽出 (Centroid Distance)
    if 'sbert_embeddings' in st.session_state and not df_target.empty:
        try:
            # df_targetのindexを使ってembeddingsを抽出
            # 前提: df_mainのindexがresetされておらず、embeddingsと1対1対応していること
            valid_indices = [i for i in df_target.index if i < len(st.session_state.sbert_embeddings)]
            
            if valid_indices:
                vectors = st.session_state.sbert_embeddings[valid_indices]
                centroid = np.mean(vectors, axis=0)
                
                # 重心との距離計算 (Cosine Similarity相当)
                dots = np.dot(vectors, centroid)
                
                # 上位N件のインデックスを取得
                top_n_local_indices = np.argsort(dots)[::-1][:n_representatives]
                top_global_indices = [valid_indices[i] for i in top_n_local_indices]
                
                # データ取得
                reps = []
                invalid_count = 0
                
                # Column mapping for enhanced info
                col_map = st.session_state.get('col_map', {})
                app_col = col_map.get('applicant', 'applicant')
                
                for idx in top_global_indices:
                    try:
                        row = st.session_state.df_main.loc[idx]
                        t_val = str(row.get(title_col, ''))
                        a_val = str(row.get(abstract_col, ''))
                        
                        # Enhanced Info
                        y_val = str(row.get('year', 'N/A'))
                        app_val = "N/A"
                        if app_col and app_col in row:
                            val = row[app_col]
                            if isinstance(val, list):
                                # Clean join: Filter out None/nan/invalid
                                clean_vals = [str(x).strip() for x in val if x and str(x).lower() != 'nan']
                                app_val = ", ".join(clean_vals)
                            else: app_val = str(val)
                        
                        # Check validity
                        if (not t_val or t_val == 'nan') and (not a_val or a_val == 'nan'):
                             invalid_count += 1
                             title = "No Title"
                             abstract = "No Abstract"
                        else:
                             title = t_val if t_val and t_val != 'nan' else "No Title"
                             abstract = a_val if a_val and a_val != 'nan' else "No Abstract"
                        
                        title = title.replace('\n', ' ')
                        abstract = abstract.replace('\n', ' ')[:200] + "..." 
                        
                        # Clean up Applicant (truncate if too long)
                        if len(app_val) > 30: app_val = app_val[:30] + "..."
                        
                        reps.append(f"- 【{title}】 (出願: {y_val}, {app_val}) {abstract}")
                        
                        summary.setdefault('representatives_raw', []).append({
                            'title': title,
                            'abstract': abstract,
                            'year': y_val,
                            'applicant': app_val
                        })
                    except: pass
                
                # If mostly invalid, don't show
                if len(reps) > 0 and (invalid_count / len(reps)) > 0.5:
                     summary['representatives'] = [] # Suppress
                     summary['representatives_raw'] = []
                else:
                     summary['representatives'] = reps

        except Exception as e:
            summary['error'] = str(e)

    return summary

def get_cluster_representatives(df_subset, cluster_col='cluster', n_reps=3):
    """
    データフレーム内の各クラスタについて、重心に近い代表特許を抽出する。
    Returns:
        dict: {cluster_id: ["- 【Title】(Applicant): Abstract...", ...]}
    """
    reps_dict = {}
    
    unique_clusters = sorted(df_subset[cluster_col].unique())
    embeddings = st.session_state.get('sbert_embeddings')
    
    col_map = st.session_state.get('col_map', {})
    title_col = col_map.get('title', 'title')
    abs_col = col_map.get('abstract', 'abstract')
    app_col = col_map.get('applicant', 'applicant')

    for cid in unique_clusters:
        if cid == -1: continue # Skip noise
        
        # Cluster subset
        df_c = df_subset[df_subset[cluster_col] == cid]
        if df_c.empty: continue
        
        try:
            target_indices = []
            # Plan A: Centroid-based (if embeddings exist)
            if embeddings is not None and len(embeddings) >= df_subset.index.max():
                 valid_indices = [i for i in df_c.index if i < len(embeddings)]
                 if valid_indices:
                     vectors = embeddings[valid_indices]
                     centroid = np.mean(vectors, axis=0)
                     dots = np.dot(vectors, centroid)
                     top_indices_local = np.argsort(dots)[::-1][:n_reps]
                     target_indices = [valid_indices[i] for i in top_indices_local]

            # Plan B: Random/Head (Fallback)
            if not target_indices:
                 # Shuffle and take top N to avoid bias of file order, or just take head for stability
                 # Let's take head for stability, or user specified sort?
                 # Actually, usually there is no sort. Random might be better for "Variety"?
                 # But head is safer.
                 target_indices = df_c.index[:n_reps].tolist()

            cluster_reps = []
            for idx in target_indices:
                row = st.session_state.df_main.loc[idx]
                
                # Title
                t_val = str(row.get(title_col, 'No Title')).replace('\n', ' ')
                # Abstract
                a_val = str(row.get(abs_col, 'No Abstract')).replace('\n', ' ')[:80] + "..."
                # Applicant
                app_val = "N/A"
                if app_col in row:
                    val = row[app_col]
                    if isinstance(val, list):
                         app_val = ",".join([str(x) for x in val if x])[:20]
                    else:
                         app_val = str(val)[:20]
                
                cluster_reps.append(f"  * 【{t_val}】({app_val}): {a_val}")
            
            reps_dict[cid] = cluster_reps
            
        except Exception as e:
            print(f"Error in getting reps for cluster {cid}: {e}")
            continue
            
    return reps_dict

def get_keyword_centric_representatives(df_target, top_keywords, n_reps=10):
    """
    ネットワークの主要キーワードを多く含む特許を抽出する (Keyword-Centric)。
    Args:
        df_target: 対象DataFrame
        top_keywords: ネットワークの中心性/頻度が高いキーワードリスト (list of str)
        n_reps: 抽出件数 (User request: significantly increase)
    Returns:
        list of dict: [{'title':..., 'applicant':..., 'abstract':..., 'score':...}]
    """
    if df_target.empty or not top_keywords:
        return []

    try:
        # Avoid side effects
        df = df_target.copy()
        
        col_map = st.session_state.get('col_map', {})
        t_col = col_map.get('title', 'title')
        a_col = col_map.get('abstract', 'abstract')
        app_col = col_map.get('applicant', 'applicant')
        y_col = 'year'

        # 1. Scoring Logic: Count occurrences of top keywords in Title, Abstract, Claims(if any)
        # 正規表現の最適化: 高速化のため上位キーワードの「OR」全体一致パターンを構築
        # パフォーマンスのため上位30キーワードに制限
        target_kws = top_keywords[:30]
        # 念のためキーワードをエスケープ
        import re
        safe_kws = [re.escape(k) for k in target_kws if k and isinstance(k, str)]
        if not safe_kws: return []
        
        pattern = "|".join(safe_kws)
        
        # スコア計算
        # タイトル重み: x2, 要約: x1
        score_series = pd.Series(0, index=df.index)
        
        if t_col in df.columns:
            score_series += df[t_col].astype(str).str.count(pattern) * 2
        if a_col in df.columns:
            score_series += df[a_col].astype(str).str.count(pattern)
            
        df['_kw_score'] = score_series
        
        # 2. Sort and Extract
        # 抽出要件: スコア > 0
        df_sorted = df[df['_kw_score'] > 0].sort_values(by='_kw_score', ascending=False).head(n_reps)
        
        reps = []
        for _, row in df_sorted.iterrows():
            title = str(row.get(t_col, 'No Title')).replace('\n', ' ')
            abstract = str(row.get(a_col, 'No Abstract')).replace('\n', ' ')[:100] + "..."
            year = str(row.get(y_col, '-'))
            
            app_val = "N/A"
            if app_col in row:
                val = row[app_col]
                if isinstance(val, list): app_val = ",".join([str(x) for x in val if x])
                else: app_val = str(val)
            if len(app_val) > 20: app_val = app_val[:20] + "..."
            
            reps.append({
                'title': title,
                'abstract': abstract,
                'year': year,
                'applicant': app_val,
                'score': row['_kw_score'],
                'matched_keywords_count': row['_kw_score'] # Simplified
            })
            
        return reps

    except Exception as e:
        print(f"Error in keyword-centric extraction: {e}")
        return []

def render_snapshot_button(title, description, key, fig=None, data_summary=None, figs=None):
    """
    グラフやデータをVOYAGER用に保存するボタンを表示する
    Args:
        title (str): スナップショットのタイトル
        description (str): スナップショットの説明
        key (str): 一意の識別キー
        fig (Figure): 単一のFigure (レガシーサポート)
        data_summary (dict): 分析データのサマリー
        figs (list): Figureのリスト (複数画像スナップショット用)
    """
    if 'snapshots' not in st.session_state:
        st.session_state['snapshots'] = []

    # 保存済みかチェック
    is_saved = any(s['id'] == key for s in st.session_state['snapshots'])
    
    btn_label = "📸 Snapshot Saved" if is_saved else "📸 Save Snapshot"
    btn_type = "primary" if not is_saved else "secondary"
    
    if st.button(btn_label, key=f"snap_btn_{key}", type=btn_type, disabled=is_saved):
        # モジュール名の決定 (data_summary['module']があれば優先)
        module_name = st.session_state.get('current_page', 'Unknown')
        if data_summary and isinstance(data_summary, dict) and 'module' in data_summary:
            module_name = data_summary['module']

        snapshot_data = {
            'id': key,
            'title': title,
            'description': description,
            'data_summary': data_summary,
            'module': module_name,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'images': [] # 新機能: 画像リストを保存
        }
        
        # 対象フィギュアの統合
        target_figs = []
        if figs:
            target_figs = figs
        elif fig:
            target_figs = [fig]
            
        import io
        
        # --- ヘルパー: 1つのfigをバイト列に変換 ---
        def convert_fig_to_bytes(target_fig):
            if target_fig is None: return None, None
            img_bytes = None
            error_msg = None
            
            try:
                # Plotly
                if hasattr(target_fig, 'to_image'):
                    try:
                        # --- スマート解像度 & アスペクト比 ---
                        base_width = 1600
                        use_width = base_width
                        use_height = 1000 # デフォルトフォールバック
                        
                        is_saturn_v = module_name == 'Saturn V'
                        
                        try:
                            if is_saturn_v:
                                # SATURN V: アスペクト比を計算
                                xaxis = target_fig.layout.xaxis
                                yaxis = target_fig.layout.yaxis
                                if xaxis.range and yaxis.range:
                                    x_range = xaxis.range[1] - xaxis.range[0]
                                    y_range = yaxis.range[1] - yaxis.range[0]
                                    if x_range > 0 and y_range > 0:
                                        ratio = x_range / y_range
                                        calc_height = base_width / ratio
                                        calc_height = max(600, min(calc_height, 2400))
                                        use_height = int(calc_height)
                                    else:
                                        use_height = int(base_width * 0.618)
                                else:
                                    use_height = 1000
                            else:
                                # 標準ワイドフォーマット (16:9)
                                use_height = int(base_width * 9 / 16)
                        except:
                            use_height = 1000

                        img_bytes = target_fig.to_image(format="png", width=use_width, height=use_height, scale=3.0)
                    except Exception as e:
                        error_msg = f"Plotly Image Error (Kaleido): {str(e)}"
                
                # Matplotlib
                elif hasattr(target_fig, 'savefig'):
                    try:
                        buf = io.BytesIO()
                        target_fig.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        img_bytes = buf.getvalue()
                    except Exception as e:
                        error_msg = f"Matplotlib Image Error: {str(e)}"
                        
            except Exception as e:
                error_msg = f"General Image Error: {str(e)}"
                
            return img_bytes, error_msg

        # Process all figures
        for f in target_figs:
            ib, err = convert_fig_to_bytes(f)
            if ib:
                snapshot_data['images'].append(ib)
            if err:
                snapshot_data.setdefault('image_errors', []).append(err)
                st.warning(f"Image conversion warning: {err}")

        # レガシー互換性: メインの'image'を最初の有効な画像に設定
        if snapshot_data['images']:
            snapshot_data['image'] = snapshot_data['images'][0]
        else:
            snapshot_data['image'] = None

        st.session_state['snapshots'].append(snapshot_data)
        st.rerun()

    if is_saved:
        st.success(f"'{title}' をVOYAGERポケットに保存しました")

# ==================================================================
# --- 5. AI アシスタント (共通) ---
# ==================================================================
def generate_ai_cluster_prompt(df_source, cluster_col, target_cols, tfidf_matrix, feature_names, n_samples=5):
    """クラスタごとの代表文献を抽出し、命名用プロンプトを生成する"""
    if df_source.empty: return "データがありません。"
    
    unique_clusters = sorted([c for c in df_source[cluster_col].unique() if c != -1])
    if not unique_clusters: return "有効なクラスタがありません。"

    # embeddingカラムの特定
    if 'umap_x' in df_source.columns and 'umap_y' in df_source.columns:
        embedding_cols = ['umap_x', 'umap_y']
    elif 'drill_x' in df_source.columns and 'drill_y' in df_source.columns:
        embedding_cols = ['drill_x', 'drill_y']
    elif 'x' in df_source.columns and 'y' in df_source.columns: # MEGA対応
        embedding_cols = ['x', 'y']
    else:
        return "埋め込み座標が見つかりません。"

    sampled_docs = []
    
    for cid in unique_clusters:
        c_df = df_source[df_source[cluster_col] == cid]
        if c_df.empty: continue
        
        # キーワード抽出 (TF-IDF)
        keywords_str = ""
        try:
            valid_indices = [i for i in c_df.index if i < tfidf_matrix.shape[0]]
            if valid_indices:
                sub_matrix = tfidf_matrix[valid_indices]
                mean_vec = np.array(sub_matrix.mean(axis=0)).flatten()
                top_idx = np.argsort(mean_vec)[::-1][:10] # Top 10 words
                keywords = [feature_names[i] for i in top_idx]
                keywords_str = ", ".join(keywords)
        except Exception as e:
            keywords_str = f"(抽出エラー: {e})"

        # 重心計算
        coords = c_df[embedding_cols].values
        centroid = coords.mean(axis=0)
        
        # 重心に近い順にソート
        dists = euclidean_distances(coords, centroid.reshape(1, -1)).flatten()
        top_indices = np.argsort(dists)[:n_samples]
        
        docs = []
        for idx in top_indices:
            row = c_df.iloc[idx]
            text_parts = []
            for col in target_cols:
                if col and col in row and pd.notna(row[col]):
                    val = str(row[col]).replace('\n', ' ')
                    text_parts.append(val)
            docs.append(f"  - {' '.join(text_parts)}")
        
        sampled_docs.append(f"Cluster {cid}:\n[特徴語] {keywords_str}\n[代表特許]\n" + "\n".join(docs))

    sampled_docs_str = "\n\n".join(sampled_docs)

    prompt = f"""
あなたは熟練した特許情報アナリストです。
以下の「クラスタごとの特徴語と代表的特許リスト」を分析し、各クラスタの内容を端的に表す**「短い説明ラベル（日本語）」**を提案してください。

# 制約事項
- ラベルは**20文字以内**の日本語で記述してください。
- 専門用語を適切に使用し、技術的特徴や解決課題を反映させてください。
- 出力は **JSON形式のみ** としてください。解説は不要です。

# 出力例
{{
  "0": "全固体電池の固体電解質",
  "1": "画像認識による異常検知",
  "2": "カーボンニュートラル燃料製造"
}}

# 出力フォーマット (JSON)
{{
  "クラスタID (整数)": "提案ラベル",
  ...
}}

# クラスタデータ
{sampled_docs_str}
"""
    return prompt

def render_ai_label_assistant(df_source, cluster_col, label_map_key, col_map, tfidf_matrix, feature_names, widget_key_prefix=None):
    """AIラベルサジェストUI (共通部品)"""
    with st.expander("AIによるラベルサジェスト (オプション)"):
        st.markdown("LLM (ChatGPT等) にプロンプトを投げ、結果のJSONを取り込むことでラベルを自動設定します。")
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            n_samples_ai = st.number_input("1クラスタあたりのサンプル数", min_value=1, value=5, key=f"ai_n_samples_{label_map_key}")
        
        if st.button("プロンプトを生成", key=f"ai_gen_btn_{label_map_key}"):
            target_cols = [col_map.get('title'), col_map.get('abstract')]
            prompt = generate_ai_cluster_prompt(df_source, cluster_col, target_cols, tfidf_matrix, feature_names, n_samples=n_samples_ai)
            st.session_state[f"ai_prompt_{label_map_key}"] = prompt
        
        if f"ai_prompt_{label_map_key}" in st.session_state:
            st.code(st.session_state[f"ai_prompt_{label_map_key}"], language="markdown")
            st.info("👆 右上のコピーボタンでコピーし、LLMに入力してください。")

        st.markdown("---")
        st.markdown("**結果の取り込み (JSON)**")
        json_input = st.text_area("LLMの出力JSONを貼り付け:", height=150, key=f"ai_json_input_{label_map_key}")
        
        if st.button("サジェストを適用", key=f"ai_apply_btn_{label_map_key}"):
            try:
                # JSONのクリーニング (Markdownコードブロック除去)
                cleaned_json = re.sub(r'^```json\s*|\s*```$', '', json_input.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned_json)
                
                # key変換 (str -> int) & 適用
                current_map = st.session_state[label_map_key]
                count = 0
                for cid_str, label in data.items():
                    try:
                        cid = int(cid_str)
                        # df_sourceのクラスタカラムに存在するIDか確認
                        unique_cids = df_source[cluster_col].unique()
                        
                        if cid in current_map or cid in unique_cids: # 存在するクラスタのみ
                            new_val = f"[{cid}] {label}"
                            current_map[cid] = new_val
                            
                            # ウィジェットのステートも強制更新して、UI上の表示を同期させる
                            if widget_key_prefix:
                                w_key = f"{widget_key_prefix}_{cid}"
                                if w_key in st.session_state:
                                    st.session_state[w_key] = new_val
                            count += 1
                    except: pass
                
                # 反映 (session_stateのマップは参照渡しされている前提だが、念のため再代入)
                st.session_state[label_map_key] = current_map
                
                # [Saturn V] ラベルカラムの更新
                if label_map_key == "saturnv_labels_map" and 'df_main' in st.session_state:
                   # ラベル更新を反映
                   st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(current_map)
                elif label_map_key == "drill_labels_map" and 'df_drilldown_result' in st.session_state:
                   st.session_state.df_drilldown_result['drill_cluster_label'] = st.session_state.df_drilldown_result['drill_cluster'].map(current_map)

                # [MEGA] ラベルカラムの更新
                elif label_map_key == "mega_drill_labels_map" and 'df_drilldown' in st.session_state:
                   st.session_state.df_drilldown['label'] = st.session_state.df_drilldown['cluster_id'].map(current_map)
                   st.session_state.sbert_sub_cluster_map_auto = current_map

                st.success(f"{count} 件のラベルを更新しました！")
                st.rerun()
                
            except Exception as e:
                st.error(f"JSONパースエラー: {e}")

def create_label_editor_ui(original_map, current_map, key_prefix):
    """手動ラベル編集UI機能 (共通)"""
    widgets_dict = {}
    sorted_ids = sorted([cid for cid in original_map.keys() if cid != -1])
    for cluster_id in sorted_ids:
        orig_label = original_map.get(cluster_id, "")
        curr_label = current_map.get(cluster_id, orig_label)
        if orig_label == "(該当なし)": continue
        col1, col2 = st.columns([2, 3])
        with col1: st.markdown(f":green[{orig_label}]")
        with col2:
            key = f"{key_prefix}_{cluster_id}"
            if key not in st.session_state:
                st.session_state[key] = curr_label
            # value引数を指定せず、key経由でsession_stateの値を使用させる
            new_label = st.text_input(f"Edit {cluster_id}", label_visibility="collapsed", key=key)
            widgets_dict[cluster_id] = new_label
    if -1 in original_map:
        orig_noise = original_map[-1]
        curr_noise = current_map.get(-1, orig_noise)
        col1, col2 = st.columns([2, 3])
        with col1: st.markdown(f":green[{orig_noise}]")
        with col2:
            st.text_input(f"noise_label", value=curr_noise, disabled=True, key=f"{key_prefix}_noise")
            widgets_dict[-1] = curr_noise
    return widgets_dict

def update_fig_layout(fig, title, height=1000, width=800, theme_config=None, show_axes=False, show_legend=True):
    """Plotlyのレイアウトを統一的に更新する"""
    if theme_config is None:
        return fig
    
    # Sanitize title to remove implicit/explicit HTML tags (e.g. <b>)
    if isinstance(title, str):
        title = re.sub(r'<[^>]+>', '', title)

    layout_params = dict(
        template=theme_config["plotly_template"],
        title=dict(text=title, font=dict(size=18, color=theme_config["text_color"], family="Helvetica Neue", weight="normal")),
        paper_bgcolor=theme_config["bg_color"],
        plot_bgcolor=theme_config["bg_color"],
        font=dict(color=theme_config["text_color"], family="Helvetica Neue"),
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#eee", borderwidth=1
        )
    )

    if not show_legend:
        layout_params['showlegend'] = False

    if not show_axes:
        layout_params['xaxis'] = dict(visible=False, showgrid=False, zeroline=False, showticklabels=False)
        layout_params['yaxis'] = dict(
            visible=False, showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=1
        )
    else:
        if "width" in layout_params:
            del layout_params["width"]

        layout_params['xaxis'] = dict(
            visible=True, showgrid=False, zeroline=False, showline=False, showticklabels=True
        )
        layout_params['yaxis'] = dict(
            visible=True, showgrid=True, gridcolor='#eee', zeroline=False, showline=False, showticklabels=True
        )

    fig.update_layout(**layout_params)
    return fig

# ==================================================================
# --- 5. 高度なテキスト処理 (Explorer logic port) ---
# ==================================================================
import unicodedata

# 正規表現フィルター定義 (Explorerと同等)
_ngram_rows = [
    ("参照符号付き要素", r"[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[（(]\s*[0-9０-９A-Za-z]+[A-Za-z]?\s*[）)]", "regex", 1),
    ("参照符号付き要素", r"(?:上記|前記)?[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[0-9０-９A-Za-z]+[A-Za-z]?", "regex", 1),
    ("参照符号付き要素", r"[A-Z]+[0-9]+", "regex", 1),
    ("見出し・章句","一実施形態において","literal",1), ("見出し・章句","他の実施形態において","literal",1), ("見出し・章句","別の実施形態において","literal",1),
    ("見出し・章句","本明細書において","literal",1), ("見出し・章句","本明細書では","literal",1), ("見出し・章句","本発明の一側面","literal",1),
    ("見出し・章句","一実施例において","literal",1), ("見出し・章句","他の実施例において","literal",1), ("見出し・章句","好ましい態様として","literal",2),
    ("見出し・章句","好適には","literal",2), ("見出し・章句","用語の定義","literal",2), ("見出し・章句","図示しない","literal",2),
    ("図表参照", r"図[ 　]*[０-９0-9]+に示す", "regex", 1), ("図表参照", r"表[ 　]*[０-９0-9]+に示す", "regex", 1),
    ("図表参照", r"式[ 　]*[０-９0-9]+に示す", "regex", 1), ("図表参照", r"請求項[ 　]*[０-９0-9]+", "regex", 1),
    ("図表参照", r"(?:【|\[)\s*[０-９0-9]{4,5}\s*(?:】|\])", "regex", 1), ("図表参照", r"[（(][０-９0-9]+[）)]", "regex", 2),
    ("図表参照", r"第\s*[０-９0-9]+の?実施形態", "regex", 2), ("図表参照", r"段落\s*[０-９0-9]+", "regex", 2),
    ("図表参照", r"図[ 　]*[０-９0-9]+[A-Za-z]?", "regex", 2), ("定義導入", r"以下、[^、。]+を[^、。]+と称する", "regex", 1),
    ("定義導入", r"以下、[^、。]+を[^、。]+という", "regex", 1), ("機能句","してもよい","literal",1), ("機能句","であってもよい","literal",1),
    ("機能句","することができる","literal",1), ("機能句","行うことができる","literal",1), ("機能句","に限定されない","literal",1),
    ("機能句","に限られない","literal",1), ("機能句","一例として","literal",2), ("機能句","例示的には","literal",2),
    ("参照句","前述のとおり","literal",2), ("参照句","前述の通り","literal",2), ("参照句","後述するように","literal",2),
    ("参照句","後述のとおり","literal",2), ("範囲表現", r"少なくとも(?:一|１)つ", "regex", 2), ("範囲表現", "少なくとも一部", "literal", 2),
    ("範囲表現", r"複数の(?:実施形態|構成|要素)", "regex", 3), ("課題句", r"(?:上記|前記)の?課題", "regex", 1),
    ("接続・論理","一方で","literal",3), ("接続・論理","他方で","literal",3), ("接続・論理","すなわち","literal",3)
]
_ngram_compiled = []
for cat, pat, ptype, pri in _ngram_rows:
    if ptype == "regex":
        _ngram_compiled.append((cat, re.compile(pat), ptype, pri))
    else:
        _ngram_compiled.append((cat, pat, ptype, pri))

def normalize_text(text):
    if not isinstance(text, str): text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("µ", "μ")
    text = re.sub(r"\s+", " ", text)
    return text

def apply_ngram_filters(text):
    for cat, pat, ptype, pri in _ngram_compiled:
        if ptype == "literal":
            if pat in text: text = text.replace(pat, "")
        else:
            text = pat.sub("", text)
    return text

@st.cache_resource
def load_tokenizer():
    return Tokenizer()

def extract_keywords(text, tokenizer=None, stopwords=None, top_n=None, clean_html=False):
    """
    テキストから特徴語（名詞・複合名詞）を抽出する実用版関数 (Explorerロジック準拠)
    Args:
        text (str): 対象テキスト
        tokenizer: Janome Tokenizerインスタンス (Noneの場合は内部でキャッシュされたものを使用)
        stopwords (list/set): ストップワードのリスト
        top_n: 未使用 (互換性のため維持)
        clean_html (bool): Trueの場合、HTML/XMLタグを除去する (NPL推奨)
    Returns:
        list: 抽出されたキーワードのリスト
    """
    if not text: return []
    if stopwords is None: stopwords = get_patent_stopwords() # Default to patent
    if tokenizer is None: tokenizer = load_tokenizer() # Use cached tokenizer
    
    # Pre-processing
    if clean_html:
        text = re.sub(r'<[^>]+>', ' ', text) # Strip HTML tags

    text = normalize_text(text)
    text = apply_ngram_filters(text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)

    words = []
    
    # 1. Japanese Morphological Analysis (Compound Nouns)
    if tokenizer:
        tokens = tokenizer.tokenize(text)
        compound_word = ''
        
        for token in tokens:
            pos = token.part_of_speech.split(',')[0]
            if pos == '名詞':
                compound_word += token.surface
            else:
                if (len(compound_word) > 1 and
                    compound_word not in stopwords and
                    not re.fullmatch(r'[\d０-９]+', compound_word) and
                    not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
                    not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
                    not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
                    not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word) and
                    # ひらがな1文字+記号のようなゴミを除去
                    not re.fullmatch(r'[ぁ-ん]', compound_word)):
                    words.append(compound_word)
                compound_word = ''
        
        # Last word
        if (len(compound_word) > 1 and
            compound_word not in stopwords and
            not re.fullmatch(r'[\d０-９]+', compound_word) and
            not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
            not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
            not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
            not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
            words.append(compound_word)

    # 2. English Fallback (Regex based) - Only if no Japanese keywords found or text is English-heavy
    # 日本語特許でも英単語（"AI", "IoT"）は上記ロジックで名詞として拾われるケースが多いが、
    # 念のため、キーワードがゼロの場合や、明らかに英語テキストの場合の補完
    if not words and re.search(r'[a-zA-Z]', text):
         # Extract 3+ letter words, ensure lowercase check against stopwords
         skips = set([w.lower() for w in stopwords])
         candidates = re.findall(r'\b[a-zA-Z]{3,}\b', text)
         for w in candidates:
             if w.lower() not in skips:
                 words.append(w)

    return words
