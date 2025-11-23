import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import warnings
import io
import unicodedata
import re
import platform
import os
import string
from collections import Counter
from itertools import combinations

# 機械学習・自然言語処理
from umap import UMAP 
import hdbscan 
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import networkx as nx

# 描画設定
import matplotlib.pyplot as plt
import matplotlib.font_manager
import japanize_matplotlib

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 1. デザインテーマ管理 ---
# ==================================================================

def get_theme_config(theme_name):
    """選択されたテーマに応じたCSSとPlotlyテンプレート設定を返す"""
    themes = {
        "APOLLO Standard": {
            "bg_color": "#ffffff",
            "text_color": "#333333",
            "sidebar_bg": "#f8f9fa",
            "plotly_template": "plotly_white",
            "color_sequence": px.colors.qualitative.G10,
            "density_scale": "Blues",
            "accent_color": "#003366",
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
            "density_scale": "Teal",
            "accent_color": "#264653",
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

def update_fig_layout(fig, title, height=800, theme_config=None):
    """Plotlyグラフの共通レイアウト設定"""
    if theme_config is None:
        return fig
    fig.update_layout(
        template=theme_config["plotly_template"],
        title=title,
        paper_bgcolor=theme_config["bg_color"],
        plot_bgcolor=theme_config["bg_color"],
        font_color=theme_config["text_color"],
        height=height
    )
    return fig

# ==================================================================
# --- 2. テキスト処理関数 ---
# ==================================================================

@st.cache_resource
def load_tokenizer_saturn():
    return Tokenizer()

t = load_tokenizer_saturn()

# ストップワード定義
_stopwords_original_list = [
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
    "適用例","用語の定義","開示","本開示","開示内容","上部","下部","内部","外部","内側","外側","表面",
    "裏面","側面","上面","下面","端面","先端","基端","後端","一端","他端","中心","中央","周縁","周辺",
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
    "シャーシ","駆動","伝達","支持","連結","解決", "準備", "提供", "発生", "以上", "十分"
]

@st.cache_data
def expand_stopwords_to_full_width(words):
    """半角文字を全角文字に変換したセットを追加する"""
    expanded = set(words)
    hankaku_chars = string.ascii_letters + string.digits
    zenkaku_chars = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９"
    trans_table = str.maketrans(hankaku_chars, zenkaku_chars)
    for word in words:
        if any(c in hankaku_chars for c in word):
            expanded.add(word.translate(trans_table))
    return sorted(list(expanded))

stopwords = set(expand_stopwords_to_full_width(_stopwords_original_list))

# n-gramフィルタ定義
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
    ("接続・論理","一方で","literal",3), ("接続・論理","他方で","literal",3), ("接続・論理","すなわち","literal",3),
    ("接続・論理","したがって","literal",3), ("接続・論理","しかしながら","literal",3), ("接続・論理","例えば","literal",3),
    ("接続・論理","具体的には","literal",3), ("補助句","以下に説明する","literal",3), ("補助句","前記のとおり","literal",3),
    ("補助句","これにより","literal",3), ("補助句","このように","literal",3)
]

_ngram_compiled = sorted(_ngram_rows, key=lambda x: (x[3], -len(x[1]) if x[2]=="literal" else -50))
_ngram_compiled = [(cat, re.compile(pat) if ptype == "regex" else pat, ptype, pri) for cat, pat, ptype, pri in _ngram_compiled]

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

@st.cache_data
def extract_compound_nouns(text):
    text = normalize_text(text)
    text = apply_ngram_filters(text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)

    tokens = t.tokenize(text)
    words, compound_word = [], ''
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
                not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
                words.append(compound_word)
            compound_word = ''
            
    if (len(compound_word) > 1 and
        compound_word not in stopwords and
        not re.fullmatch(r'[\d０-９]+', compound_word) and
        not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
        not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
        not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
        not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
        words.append(compound_word)
    return words

def get_font_path_for_wordcloud():
    system_name = platform.system()
    candidates = []
    if system_name == 'Linux':
        candidates = ['/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf', '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf']
    elif system_name == 'Darwin': 
        candidates = ['/System/Library/Fonts/Hiragino Sans W3.ttc', '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc']
    elif system_name == 'Windows':
        candidates = ['C:\\Windows\\Fonts\\meiryo.ttc', 'C:\\Windows\\Fonts\\msgothic.ttc']
    
    for path in candidates:
        if os.path.exists(path): return path
    return None

font_path = get_font_path_for_wordcloud()

def generate_wordcloud_and_list(words, title, top_n=20, font_path=None):
    if not words:
        st.subheader(title)
        st.warning("キーワードが見つからなかったため、表示をスキップしました。")
        return None

    word_freq = Counter(words)

    try:
        wc = WordCloud(
            width=800, height=400, background_color='white',
            font_path=font_path, collocations=False,
            max_words=100
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=20)
        ax.axis('off')
        st.pyplot(fig)
        
        st.markdown(f"**上位キーワード (Top {top_n})**")
        list_data = { "キーワード": [], "出現頻度": [] }
        for word, freq in word_freq.most_common(top_n):
            list_data["キーワード"].append(word)
            list_data["出現頻度"].append(freq)
        st.dataframe(pd.DataFrame(list_data), height=200)
        
    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")
        if font_path is None:
            st.warning("日本語フォントが見つかりませんでした。")

@st.cache_data
def get_top_tfidf_words(_row_vector, feature_names, top_n=5):
    scores = _row_vector.toarray().flatten() 
    indices = np.argsort(scores)[::-1]
    non_zero_indices = [i for i in indices if scores[i] > 0]
    top_indices = non_zero_indices[:top_n]
    top_words = [feature_names[i] for i in top_indices]
    return ", ".join(top_words)

def update_hover_text(df, col_map):
    hover_texts = []
    for index, row in df.iterrows():
        text = ""
        if col_map['title'] and pd.notna(row[col_map['title']]): text += f"<b>名称:</b> {str(row[col_map['title']])[:50]}...<br>"
        if col_map['app_num'] and pd.notna(row[col_map['app_num']]): text += f"<b>番号:</b> {row[col_map['app_num']]}<br>"
        if col_map['applicant'] and pd.notna(row[col_map['applicant']]): text += f"<b>出願人:</b> {str(row[col_map['applicant']])[:50]}...<br>"
        if 'characteristic_words' in row: text += f"<b>特徴語:</b> {row['characteristic_words']}<br>"
        if 'cluster_label' in row: text += f"<b>クラスタ:</b> {row['cluster_label']}"
        hover_texts.append(text)
    df['hover_text'] = hover_texts
    return df

def update_drill_hover_text(df_subset):
    df_subset['drill_hover_text'] = df_subset.apply(
        lambda row: f"{row['hover_text']}<br><b>サブクラスタ:</b> {row['drill_cluster_label']}", axis=1
    )
    return df_subset

def _create_label_editor_ui(original_map, current_map, key_prefix):
    widgets_dict = {}
    sorted_ids = sorted([cid for cid in original_map.keys() if cid != -1])
    
    for cluster_id in sorted_ids:
        orig_label = original_map.get(cluster_id, "")
        curr_label = current_map.get(cluster_id, orig_label)
        if orig_label == "(該当なし)": continue
        col1, col2 = st.columns([2, 3])
        with col1: st.markdown(f":green[{orig_label}]")
        with col2:
            new_label = st.text_input(f"Edit {cluster_id}", value=curr_label, label_visibility="collapsed", key=f"{key_prefix}_{cluster_id}")
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

def get_date_bin_options(df_filtered, interval_years, year_column='year'):
    if df_filtered is None or df_filtered.empty: return [f"(データなし)"]
    if year_column not in df_filtered.columns: return [f"(全期間) ({len(df_filtered)}件)"]
    
    df_filtered = df_filtered.copy()
    df_filtered[year_column] = pd.to_numeric(df_filtered[year_column], errors='coerce')
    if df_filtered[year_column].isnull().all(): return [f"(全期間) ({len(df_filtered)}件)"]

    try:
        min_year = int(df_filtered[year_column].min())
        max_year = int(df_filtered[year_column].max())
        if min_year == max_year: return [f"{min_year} ({len(df_filtered)}件)"]
        
        bins = list(range(min_year, max_year + interval_years, interval_years))
        if not bins: bins = [min_year]
        if bins[-1] <= max_year: bins.append(bins[-1] + interval_years)

        labels = [f"{bins[i]}-{bins[i+1] - 1}" for i in range(len(bins)-1)]
        df_filtered['temp_date_bin'] = pd.cut(df_filtered[year_column], bins=bins, labels=labels, right=False, include_lowest=True)
        date_bin_counts = df_filtered['temp_date_bin'].value_counts()
        
        options = [f"(全期間) ({len(df_filtered)}件)"] + [f"{label} ({date_bin_counts.get(label, 0)}件)" for label in labels if date_bin_counts.get(label, 0) > 0]
        return options
    except Exception as e:
        return [f"Error: {str(e)}"]

# ==================================================================
# --- 3. Streamlit UI構成 ---
# ==================================================================

st.set_page_config(
    page_title="APOLLO | Saturn V", 
    page_icon="🚀", 
    layout="wide"
)

# CSS注入
st.markdown("""
<style>
    html, body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    [data-testid="stSidebar"] h1 { color: #003366; font-weight: 900 !important; font-size: 2.5rem !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #003366; }
</style>
""", unsafe_allow_html=True)

# --- サイドバー ---
with st.sidebar:
    st.title("APOLLO") 
    st.markdown("Advanced Patent & Overall Landscape-analytics Logic Orbiter")
    st.markdown("---")
    
    st.subheader("Home")
    st.page_link("Home.py", label="Mission Control", icon="🛰️")
    
    st.subheader("Modules")
    st.page_link("pages/1_🌍_ATLAS.py", label="ATLAS", icon="🌍")
    st.page_link("pages/2_💡_CORE.py", label="CORE", icon="💡")
    st.page_link("pages/3_🚀_Saturn_V.py", label="Saturn V", icon="🚀")
    st.page_link("pages/4_📈_MEGA.py", label="MEGA", icon="📈")
    st.page_link("pages/5_🧭_Explorer.py", label="Explorer", icon="🧭")
    
    st.markdown("---")
    
    st.caption("ナビゲーション:")
    st.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
    st.caption("2. 上のリストから分析モジュールを選択します。")
    
    st.markdown("---")
    st.caption("© 2025 しばやま")

# --- メインコンテンツ ---
st.title("🚀 Saturn V")
st.markdown("SBERT（文脈・意味）に基づき、インタラクティブな技術マップ分析モジュールです。")

# テーマ選択
col_theme, _ = st.columns([1, 3])
with col_theme:
    selected_theme = st.selectbox("表示テーマ:", ["APOLLO Standard", "Modern Presentation"], key="saturn_theme_selector")
theme_config = get_theme_config(selected_theme)
st.markdown(f"<style>{theme_config['css']}</style>", unsafe_allow_html=True)

# ==================================================================
# --- 4. データロード & 初期化 ---
# ==================================================================
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    df_main = st.session_state.df_main
    col_map = st.session_state.col_map
    delimiters = st.session_state.delimiters
    sbert_embeddings = st.session_state.sbert_embeddings
    tfidf_matrix = st.session_state.tfidf_matrix
    feature_names = st.session_state.feature_names
    
if "saturnv_sbert_umap_done" not in st.session_state: st.session_state.saturnv_sbert_umap_done = False
if "saturnv_cluster_done" not in st.session_state: st.session_state.saturnv_cluster_done = False
if "saturnv_labels_map" not in st.session_state: st.session_state.saturnv_labels_map = {}
if "main_cluster_running" not in st.session_state: st.session_state.main_cluster_running = False
if "saturnv_global_zmax" not in st.session_state: st.session_state.saturnv_global_zmax = None

# ==================================================================
# --- 5. Saturn V アプリケーション ---
# ==================================================================

# --- 初回UMAP計算 ---
if not st.session_state.saturnv_sbert_umap_done:
    with st.spinner("Saturn V モジュール初回起動中: UMAPによる次元削減 (SBERTベース) を実行しています..."):
        try:
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding = reducer.fit_transform(sbert_embeddings) 
            st.session_state.df_main['umap_x'] = embedding[:, 0]
            st.session_state.df_main['umap_y'] = embedding[:, 1]
            st.session_state.df_main['characteristic_words'] = [get_top_tfidf_words(tfidf_matrix[i], feature_names) for i in range(tfidf_matrix.shape[0])]
            
            try:
                H, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=50)
                st.session_state.saturnv_global_zmax = H.max()
            except:
                st.session_state.saturnv_global_zmax = None
            
            st.session_state.saturnv_sbert_umap_done = True
            st.success("UMAPの初期計算が完了しました。")
            st.rerun()
        except Exception as e:
            st.error(f"UMAPの初期計算中にエラーが発生しました: {e}")
            st.stop()

# --- メインUI ---
tab_main, tab_drill, tab_stats, tab_export = st.tabs([
    "Landscape Map (TELESCOPE)", 
    "Drilldown (PROBE)", 
    "特許マップ (統計分析)", 
    "Data Export"
])

# --- TELESCOPE ---
with tab_main:
    st.subheader("クラスタリング実行")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: min_cluster_size_w = st.number_input("最小クラスタサイズ:", min_value=2, value=15, key="main_min_cluster_size")
    with col2: min_samples_w = st.number_input("最小サンプル数:", min_value=1, value=10, key="main_min_samples")
    with col3: label_top_n_w = st.number_input("ラベル単語数:", min_value=1, value=3, key="main_label_top_n")
    
    if st.button("描画 (再計算)", type="primary", key="main_run_cluster", disabled=st.session_state.main_cluster_running):
        st.session_state.main_cluster_running = True
        with st.spinner("HDBSCANクラスタリングを実行中..."):
            try:
                embedding = st.session_state.df_main[['umap_x', 'umap_y']].values
                clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size_w), min_samples=int(min_samples_w), metric='euclidean', cluster_selection_method='eom')
                clustering = clusterer.fit(embedding)
                st.session_state.df_main['cluster'] = clustering.labels_
                
                labels_map = {}
                label_top_n = int(label_top_n_w)
                unique_clusters = sorted(st.session_state.df_main['cluster'].unique())
                
                for cluster_id in unique_clusters:
                    if cluster_id == -1:
                        labels_map[cluster_id] = "ノイズ / 小クラスタ"
                        continue
                    indices = st.session_state.df_main[st.session_state.df_main['cluster'] == cluster_id].index
                    if len(indices) == 0:
                        labels_map[cluster_id] = "(該当なし)"
                        continue
                    cluster_vectors = tfidf_matrix[indices]
                    mean_vector = np.array(cluster_vectors.mean(axis=0)).flatten()
                    top_indices = np.argsort(mean_vector)[::-1][:label_top_n]
                    label = ", ".join([feature_names[i] for i in top_indices])
                    labels_map[cluster_id] = f"[{cluster_id}] {label}"
                
                st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(labels_map)
                st.session_state.saturnv_labels_map = labels_map.copy()
                st.session_state.saturnv_labels_map_original = labels_map.copy()
                st.session_state.df_main = update_hover_text(st.session_state.df_main, col_map)
                st.session_state.saturnv_cluster_done = True
                st.success("クラスタリング完了")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")
            finally:
                st.session_state.main_cluster_running = False

    st.markdown("---")
    
    if st.session_state.saturnv_cluster_done:
        st.subheader("フィルタリング設定 (メイン用)")
        def on_main_interval_change():
            if "main_date_filter" in st.session_state: del st.session_state.main_date_filter

        col1, col2 = st.columns(2)
        with col1:
            if 'year' in df_main.columns and df_main['year'].notna().any():
                bin_interval_w_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="main_bin_interval", on_change=on_main_interval_change)
                date_bin_options = get_date_bin_options(df_main, int(bin_interval_w_val), 'year')
                date_bin_filter_w = st.selectbox("表示期間:", date_bin_options, key="main_date_filter")
            else:
                date_bin_filter_w = "(全期間)"
        
        with col2:
            if col_map['applicant'] and col_map['applicant'] in st.session_state.df_main.columns:
                applicants = st.session_state.df_main[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
                unique_applicants = sorted([app for app in applicants.unique() if app])
                applicant_options = [(f"(全出願人) ({len(st.session_state.df_main)}件)", "ALL")] + [(f"{app}", app) for app in unique_applicants]
                applicant_filter_w = st.multiselect("出願人:", applicant_options, default=[applicant_options[0]], format_func=lambda x: x[0], key="main_applicant_filter")
            else:
                applicant_filter_w = [(f"(全出願人) ({len(st.session_state.df_main)}件)", "ALL")]

        cluster_counts = st.session_state.df_main['cluster_label'].value_counts()
        cluster_options = [(f"(全クラスタ) ({len(st.session_state.df_main)}件)", "ALL")] + [
            (f"{st.session_state.saturnv_labels_map.get(cid)} ({cluster_counts.get(st.session_state.saturnv_labels_map.get(cid), 0)}件)", cid)
            for cid in sorted(st.session_state.df_main['cluster'].unique())
        ]
        cluster_filter_w = st.multiselect("マップ表示クラスタ:", cluster_options, default=[cluster_options[0]], format_func=lambda x: x[0], key="main_cluster_filter")

        st.subheader("分析結果 (TELESCOPE メインマップ)")
        col_mode, col_norm = st.columns([1, 1])
        with col_mode: map_mode = st.radio("表示モード:", ["散布図 (Scatter)", "密度マップ (Density)"], horizontal=True)
        use_abs_scale = False
        if map_mode == "密度マップ (Density)":
            with col_norm: use_abs_scale = st.checkbox("密度スケールを固定 (絶対評価)", value=False)
        
        show_labels_chk = st.checkbox("マップにラベルを表示する", value=True, key="main_show_labels")
        
        df_visible = st.session_state.df_main.copy()
        # フィルタ適用
        if not date_bin_filter_w.startswith("(全期間)"):
            try:
                date_bin_label = date_bin_filter_w.split(' (')[0].strip()
                start_year, end_year = map(int, date_bin_label.split('-'))
                df_visible = df_visible[(df_visible['year'] >= start_year) & (df_visible['year'] <= end_year)]
            except: pass

        applicant_values = [val[1] for val in applicant_filter_w]
        is_applicant_color_mode = True
        if "ALL" in applicant_values:
             is_applicant_color_mode = False
        else:
             mask_list = [df_visible[col_map['applicant']].fillna('').str.contains(re.escape(app)) for app in applicant_values]
             if mask_list: df_visible = df_visible[pd.concat(mask_list, axis=1).any(axis=1)]
        
        cluster_values = [val[1] for val in cluster_filter_w]
        if "ALL" not in cluster_values:
            df_visible = df_visible[df_visible['cluster'].isin(cluster_values)]

        # 描画
        fig_main = go.Figure()
        if map_mode == "密度マップ (Density)":
             contour_params = dict(x=df_visible['umap_x'], y=df_visible['umap_y'], colorscale=theme_config["density_scale"], reversescale=False, xaxis='x', yaxis='y', showscale=True, name="密度", nbinsx=50, nbinsy=50)
             if use_abs_scale and st.session_state.saturnv_global_zmax:
                 contour_params.update(dict(zauto=False, zmin=0, zmax=st.session_state.saturnv_global_zmax))
             else: contour_params.update(dict(zauto=True))
             fig_main.add_trace(go.Histogram2dContour(**contour_params))
             fig_main.add_trace(go.Scatter(x=df_visible['umap_x'], y=df_visible['umap_y'], mode='markers', marker=dict(color='white', size=3, opacity=0.3), hoverinfo='text', hovertext=df_visible['hover_text'], name='特許'))
        else:
            # 散布図
            fig_main.add_trace(go.Scatter(x=df_visible['umap_x'], y=df_visible['umap_y'], mode='markers', marker=dict(color=df_visible['cluster'], colorscale=theme_config["color_sequence"], showscale=False, size=4, opacity=0.6), hoverinfo='text', hovertext=df_visible['hover_text'], name='表示対象'))

        # ラベル追加
        if show_labels_chk:
            for cid, grp in df_visible[df_visible['cluster'].isin(df_visible['cluster'].unique())].groupby('cluster'):
                if cid == -1: continue
                mean_pos = grp[['umap_x', 'umap_y']].mean()
                if not grp.empty:
                    label_txt = grp['cluster_label'].iloc[0]
                    fig_main.add_annotation(x=mean_pos['umap_x'], y=mean_pos['umap_y'], text=label_txt, showarrow=False, font=dict(size=11, color='black' if map_mode=="散布図 (Scatter)" else 'white'), bgcolor='rgba(255,255,255,0.7)' if map_mode=="散布図 (Scatter)" else 'rgba(0,0,0,0.5)')

        norm_msg = " (絶対評価)" if use_abs_scale and map_mode == "密度マップ (Density)" else ""
        update_fig_layout(fig_main, f"Saturn V - メインマップ (SBERT UMAP){norm_msg}", height=800, theme_config=theme_config)
        fig_main.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig_main.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        st.plotly_chart(fig_main, use_container_width=True)

        st.subheader("ラベル編集")
        if "saturnv_labels_map_original" not in st.session_state: st.session_state.saturnv_labels_map_original = st.session_state.saturnv_labels_map.copy()
        st.session_state.saturnv_labels_map_custom = _create_label_editor_ui(st.session_state.saturnv_labels_map_original, st.session_state.saturnv_labels_map, "main_label")
        if st.button("ラベルを更新", key="main_update_labels"):
            st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(st.session_state.saturnv_labels_map_custom)
            st.session_state.df_main = update_hover_text(st.session_state.df_main, col_map)
            st.session_state.saturnv_labels_map = st.session_state.saturnv_labels_map_custom
            st.rerun()

    # --- PROBE (ドリルダウン) ---
    with tab_drill:
        st.subheader("分析対象クラスタの選択")
        drilldown_options = [('(選択してください)', 'NONE')]
        if "saturnv_labels_map" in st.session_state:
            drilldown_options += [(f"{label} ({count}件)", cid) for cid, label in st.session_state.saturnv_labels_map.items() if cid != -1 for count in [st.session_state.df_main['cluster'].value_counts().get(cid, 0)]]
        
        selected_drilldown_target_drill = st.selectbox("分析対象クラスタ:", options=drilldown_options, format_func=lambda x: x[0], key="drill_target_select")
        drilldown_target_id = selected_drilldown_target_drill[1] 

        st.subheader("フィルタリング設定 (ドリルダウン用)")
        if drilldown_target_id == "NONE":
            df_subset_filter = pd.DataFrame(columns=df_main.columns)
            st.info("👆 上のメニューで「分析対象クラスタ」を選択すると、フィルタメニューが表示されます。")
        else:
            df_subset_filter = df_main[df_main['cluster'] == drilldown_target_id].copy()
            
        def on_drill_interval_change():
            if "drill_date_filter_w" in st.session_state: del st.session_state.drill_date_filter_w
            
        if drilldown_target_id != "NONE":
            col1, col2 = st.columns(2)
            with col1:
                if 'year' in df_subset_filter.columns and df_subset_filter['year'].notna().any():
                    drill_bin_interval_w_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="drill_interval_w", on_change=on_drill_interval_change)
                    drill_date_bin_options = get_date_bin_options(df_subset_filter, int(drill_bin_interval_w_val), 'year')
                    drill_date_bin_filter_w = st.selectbox("表示期間:", drill_date_bin_options, key="drill_date_filter_w")
                else:
                    drill_date_bin_filter_w = "(全期間)"
            with col2:
                drill_applicant_options = [(f"(全出願人) ({len(df_subset_filter)}件)", "ALL")]
                if col_map['applicant'] and col_map['applicant'] in df_subset_filter.columns:
                    applicants_drill = df_subset_filter[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
                    applicant_counts_drill = applicants_drill.value_counts()
                    unique_applicants_drill = sorted([app for app in applicants_drill.unique() if app])
                    drill_applicant_options += [(f"{app_name} ({applicant_counts_drill.get(app_name, 0)}件)", app_name) for app_name in unique_applicants_drill if applicant_counts_drill.get(app_name, 0) > 0]
                drill_applicant_filter_w = st.multiselect("出願人:", drill_applicant_options, default=[drill_applicant_options[0]], format_func=lambda x: x[0], key="drill_applicant_filter_w")
        else:
            drill_date_bin_filter_w = "(全期間)"
            drill_applicant_filter_w = []

        st.subheader("クラスタリング設定 (ドリルダウン用)")
        col1, col2, col3 = st.columns(3)
        with col1: drill_min_cluster_size_w = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="drill_min_cluster_size_w")
        with col2: drill_min_samples_w = st.number_input('最小サンプル数:', min_value=1, value=5, key="drill_min_samples_w")
        with col3: drill_label_top_n_w = st.number_input('ラベル単語数:', min_value=1, value=3, key="drill_label_top_n_w")
        drill_show_labels_chk = st.checkbox('マップにラベルを表示する', value=True, key="drill_show_labels_chk")

        if st.button("選択クラスタで再マップ", type="primary", key="drill_run_button"):
            if drilldown_target_id == "NONE":
                st.error("エラー: 分析対象クラスタを選択してください。")
            else:
                with st.spinner(f"クラスタ {drilldown_target_id} のドリルダウンを実行中..."):
                    try:
                        df_subset = df_main[df_main['cluster'] == drilldown_target_id].copy()
                        base_label = df_subset['cluster_label'].iloc[0]
                        
                        if not drill_date_bin_filter_w.startswith("(全期間)"):
                            try:
                                date_bin_label = drill_date_bin_filter_w.split(' (')[0].strip() 
                                start_year, end_year = map(int, date_bin_label.split('-'))
                                df_subset = df_subset[(df_subset['year'] >= start_year) & (df_subset['year'] <= end_year)]
                            except: pass 

                        drill_app_values = [val[1] for val in drill_applicant_filter_w]
                        if "ALL" not in drill_app_values:
                            mask_list_drill = [df_subset[col_map['applicant']].fillna('').str.contains(re.escape(app)) for app in drill_app_values]
                            df_subset = df_subset[pd.concat(mask_list_drill, axis=1).any(axis=1)]
                        
                        if len(df_subset) < 10:
                            st.warning(f"データが少なすぎます ({len(df_subset)}件)。")
                        else:
                            subset_indices = df_subset.index
                            subset_tfidf = tfidf_matrix[subset_indices]
                            subset_sbert = sbert_embeddings[subset_indices]
                            subset_indices_pd = pd.Index(subset_indices)

                            n_neighbors = min(10, len(df_subset) - 1)
                            if n_neighbors < 2: n_neighbors = 2
                            
                            reducer_drill = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
                            embedding_drill = reducer_drill.fit_transform(subset_sbert) 
                            df_subset['drill_x'] = embedding_drill[:, 0]
                            df_subset['drill_y'] = embedding_drill[:, 1]
                            
                            clusterer_drill = hdbscan.HDBSCAN(min_cluster_size=int(drill_min_cluster_size_w), min_samples=int(drill_min_samples_w), metric='euclidean', cluster_selection_method='eom')
                            df_subset['drill_cluster'] = clusterer_drill.fit_predict(embedding_drill)
                            
                            drill_labels_map = {}
                            for cid in sorted(df_subset['drill_cluster'].unique()):
                                if cid == -1:
                                    drill_labels_map[cid] = "ノイズ"
                                    continue
                                idxs = df_subset[df_subset['drill_cluster'] == cid].index
                                tfidf_pos = [subset_indices_pd.get_loc(i) for i in idxs if i in subset_indices_pd]
                                if tfidf_pos:
                                    mean_vec = np.array(subset_tfidf[tfidf_pos].mean(axis=0)).flatten()
                                    top_idx = np.argsort(mean_vec)[::-1][:int(drill_label_top_n_w)]
                                    label = ", ".join([feature_names[i] for i in top_idx])
                                    drill_labels_map[cid] = f"[{cid}] {label}"
                            
                            df_subset['drill_cluster_label'] = df_subset['drill_cluster'].map(drill_labels_map)
                            df_subset = update_drill_hover_text(df_subset)
                            st.session_state.df_drilldown_result = df_subset.copy()
                            st.session_state.drill_labels_map = drill_labels_map.copy()
                            st.session_state.drill_labels_map_original = drill_labels_map.copy()
                            st.session_state.drill_base_label = base_label
                            st.success("完了しました。")
                            st.rerun()

                    except Exception as e:
                        st.error(f"エラー: {e}")

        if "df_drilldown_result" in st.session_state:
            df_drill = st.session_state.df_drilldown_result.copy()
            drill_labels_map = st.session_state.drill_labels_map
            
            st.subheader("サブクラスタ・ラベル編集")
            if "drill_labels_map_original" not in st.session_state:
                 st.session_state.drill_labels_map_original = drill_labels_map.copy()
            drill_label_widgets = _create_label_editor_ui(st.session_state.drill_labels_map_original, st.session_state.drill_labels_map, "drill_label")
            if st.button("サブクラスタ・ラベルを更新", key="drill_update_labels"):
                for cid, val in drill_label_widgets.items(): drill_labels_map[cid] = val
                df_drill['drill_cluster_label'] = df_drill['drill_cluster'].map(drill_labels_map)
                st.session_state.df_drilldown_result = update_drill_hover_text(df_drill)
                st.session_state.drill_labels_map = drill_labels_map
                st.rerun()

            st.subheader("ドリルダウンマップ")
            fig_drill = go.Figure()
            fig_drill.add_trace(go.Scatter(
                x=df_drill['drill_x'], y=df_drill['drill_y'], mode='markers',
                marker=dict(color=df_drill['drill_cluster'], colorscale=theme_config["color_sequence"] if isinstance(theme_config["color_sequence"], str) else 'turbo', showscale=False, size=4, opacity=0.5),
                hoverinfo='text', hovertext=df_drill['drill_hover_text'], name='表示対象'
            ))
            annotations_drill = []
            if drill_show_labels_chk:
                for cid, grp in df_drill[df_drill['drill_cluster'] != -1].groupby('drill_cluster'):
                    mean_pos = grp[['drill_x', 'drill_y']].mean()
                    annotations_drill.append(go.layout.Annotation(
                        x=mean_pos['drill_x'], y=mean_pos['drill_y'], text=drill_labels_map.get(cid, ""), showarrow=False, font=dict(size=10, color=theme_config["text_color"])
                    ))
            fig_drill.update_layout(annotations=annotations_drill)
            update_fig_layout(fig_drill, f'Saturn V ドリルダウン: {st.session_state.drill_base_label}', theme_config=theme_config)
            fig_drill.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig_drill.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            st.plotly_chart(fig_drill, use_container_width=True)
            
            st.markdown("---")
            st.subheader("クラスタ・テキスト分析 (Text Mining)")
            col_tm1, col_tm2 = st.columns(2)
            with col_tm1:
                cooc_top_n = st.slider("共起: 上位単語数", 30, 100, 50, key="cooc_top_n")
                cooc_threshold = st.slider("共起: Jaccard係数 閾値", 0.01, 0.3, 0.05, 0.01, key="cooc_threshold")
            
            if st.button("テキスト分析を実行", key="run_text_mining"):
                with st.spinner("分析中..."):
                    all_text = ""
                    for _, row in df_drill.iterrows():
                        if col_map['title'] and pd.notna(row[col_map['title']]): all_text += row[col_map['title']] + " "
                        if col_map['abstract'] and pd.notna(row[col_map['abstract']]): all_text += row[col_map['abstract']] + " "
                    words = extract_compound_nouns(all_text)
                    
                    if not words: st.warning("有効なキーワードなし")
                    else:
                        st.markdown("##### 1. ワードクラウド")
                        generate_wordcloud_and_list(words, f"クラスタ: {st.session_state.drill_base_label}", 30, font_path)
                        
                        st.markdown("##### 2. 共起ネットワーク")
                        word_freq = Counter(words)
                        top_words = [w for w, c in word_freq.most_common(cooc_top_n)]
                        pair_counts = Counter()
                        for _, row in df_drill.iterrows():
                            dt = ""
                            if col_map['title']: dt += str(row[col_map['title']]) + " "
                            if col_map['abstract']: dt += str(row[col_map['abstract']]) + " "
                            dw = set(extract_compound_nouns(dt))
                            dw = {w for w in dw if w in top_words}
                            if len(dw) >= 2:
                                for pair in combinations(sorted(list(dw)), 2): pair_counts[pair] += 1
                        
                        G = nx.Graph()
                        for w in top_words: G.add_node(w, count=word_freq[w])
                        for (w1, w2), c in pair_counts.items():
                            jac = c / (word_freq[w1] + word_freq[w2] - c)
                            if jac >= cooc_threshold: G.add_edge(w1, w2, weight=jac)
                        
                        G.remove_nodes_from(list(nx.isolates(G)))
                        if G.number_of_nodes() == 0: st.warning("共起ペアなし")
                        else:
                            pos = nx.spring_layout(G, k=0.5, seed=42)
                            edge_x, edge_y = [], []
                            for edge in G.edges():
                                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
                            
                            node_x, node_y, node_text, node_size = [], [], [], []
                            for node in G.nodes():
                                x, y = pos[node]; node_x.append(x); node_y.append(y)
                                c = G.nodes[node]['count']
                                node_text.append(f"{node} ({c})")
                                node_size.append(np.log(c+1)*10)
                            
                            node_trace = go.Scatter(
                                x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=list(G.nodes()), textposition="top center",
                                marker=dict(showscale=True, colorscale='YlGnBu', size=node_size, color=node_size, line_width=2)
                            )
                            fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='共起ネットワーク', showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                            update_fig_layout(fig_net, '共起ネットワーク', theme_config=theme_config)
                            st.plotly_chart(fig_net, use_container_width=True)

    # --- C. 特許マップ (統計分析) ---
    with tab_stats:
        st.subheader("特許マップ（統計分析）")
        if st.session_state.saturnv_cluster_done:
            cluster_counts_stats = st.session_state.df_main['cluster_label'].value_counts()
            cluster_options_stats = [(f"(全クラスタ) ({len(st.session_state.df_main)}件)", "ALL")] + [
                (f"{st.session_state.saturnv_labels_map.get(cid)} ({cluster_counts_stats.get(st.session_state.saturnv_labels_map.get(cid), 0)}件)", cid)
                for cid in sorted(st.session_state.df_main['cluster'].unique())
            ]
            stats_cluster_filter_w = st.multiselect("集計対象クラスタ:", cluster_options_stats, default=[cluster_options_stats[0]], format_func=lambda x: x[0], key="stats_cluster_filter")
            
            c1, c2 = st.columns(2)
            with c1:
                if 'stats_start_year' not in st.session_state: st.session_state.stats_start_year = 2010
                if 'stats_end_year' not in st.session_state: st.session_state.stats_end_year = 2024
                s_year = st.number_input('開始年:', key="stats_start_year", step=1)
                e_year = st.number_input('終了年:', key="stats_end_year", step=1)
            with c2:
                n_apps = st.number_input('表示人数:', min_value=1, value=15, key="stats_num_assignees")
            
            if st.button("特許マップを描画", key="stats_run_button"):
                df_s = st.session_state.df_main.copy()
                # フィルタ適用
                vals = [v[1] for v in stats_cluster_filter_w]
                if "ALL" not in vals: df_s = df_s[df_s['cluster'].isin(vals)]
                df_s = df_s[(df_s['year'] >= s_year) & (df_s['year'] <= e_year)]
                
                if df_s.empty: st.warning("データなし")
                else:
                    # 1. 時系列
                    yc = df_s['year'].value_counts().sort_index().reindex(range(s_year, e_year+1), fill_value=0)
                    fig1 = px.bar(x=yc.index, y=yc.values, labels={'x':'年', 'y':'件数'}, color_discrete_sequence=[theme_config["color_sequence"][0]])
                    update_fig_layout(fig1, '出願推移', theme_config=theme_config)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # 2. ランキング
                    ac = df_s[col_map['applicant']].astype(str).str.split(delimiters['applicant']).explode().str.strip().value_counts().head(n_apps).sort_values(ascending=True)
                    fig2 = px.bar(x=ac.values, y=ac.index, orientation='h', labels={'x':'件数', 'y':'権利者'}, color_discrete_sequence=[theme_config["color_sequence"][1]])
                    update_fig_layout(fig2, '権利者ランキング', height=max(600, len(ac)*30), theme_config=theme_config)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 3. バブル
                    ae = df_s.explode(col_map['applicant'])
                    ae['ap'] = ae[col_map['applicant']].astype(str).str.strip()
                    top_a = ae['ap'].value_counts().head(n_apps).index.tolist()
                    pd_plot = ae[ae['ap'].isin(top_a)].groupby(['year', 'ap']).size().reset_index(name='count')
                    if not pd_plot.empty:
                        fig3 = px.scatter(pd_plot, x='year', y='ap', size='count', color='ap', labels={'year':'年', 'ap':'権利者', 'count':'件数'}, category_orders={'ap': top_a})
                        update_fig_layout(fig3, '出願年別動向', height=700, theme_config=theme_config)
                        st.plotly_chart(fig3, use_container_width=True)

    # --- D. エクスポート ---
    with tab_export:
        st.subheader("データエクスポート")
        if st.session_state.saturnv_cluster_done:
            cols_drop = ['hover_text', 'parsed_date', 'drill_cluster', 'drill_cluster_label', 'drill_hover_text', 'drill_x', 'drill_y', 'temp_date_bin']
            csv = df_main.drop(columns=cols_drop, errors='ignore').to_csv(encoding='utf-8-sig', index=False).encode('utf-8-sig')
            st.download_button("メインマップ全データ (CSV)", csv, "APOLLO_SaturnV_Main.csv", "text/csv")
        
        if "df_drilldown_result" in st.session_state:
            cols_drop_d = ['hover_text', 'parsed_date', 'date_bin', 'drill_hover_text', 'drill_date_bin', 'temp_date_bin']
            csv_d = st.session_state.df_drilldown_result.drop(columns=cols_drop_d, errors='ignore').to_csv(encoding='utf-8-sig', index=False).encode('utf-8-sig')
            st.download_button("ドリルダウン結果 (CSV)", csv_d, "APOLLO_SaturnV_Drill.csv", "text/csv")

# --- 共通サイドバーフッター ---
st.sidebar.markdown("---") 
st.sidebar.caption("ナビゲーション:")
st.sidebar.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
st.sidebar.caption("2. 左のリストから分析モジュールを選択します。")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 しばやま")