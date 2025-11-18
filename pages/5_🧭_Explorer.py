# ==================================================================
# --- 1. ライブラリのインポート ---
# ==================================================================
import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import warnings
import unicodedata
import re
import string
from collections import Counter
import os

# WordCloud / Janome
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer

# 統計マップ用
import matplotlib.pyplot as plt
import matplotlib.font_manager
import japanize_matplotlib # 日本語化

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. Explorer専用ヘルパー関数 ---
# ==================================================================

# Explorerは独自のTokenizerとStopWords, n-gramを持つ
@st.cache_resource
def load_tokenizer_explorer():
    print("... Explorer: Janome Tokenizerをロード中 ...")
    return Tokenizer()

t = load_tokenizer_explorer()

# Explorer専用のストップワード
stopwords = [
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
    "関係","対応","整合",
    "実施の形態","実施の態様","態様","変形","修正例","図示","図示例","図示しない","参照","参照符号","段落",
    "詳細説明","要旨","一実施形態","他の実施形態","一実施例","別の側面","付記","適用例","用語の定義","開示","本開示","開示内容",
    "上部","下部","内部","外部","内側","外側","表面","裏面","側面","上面","下面","端面","先端","基端","後端",
    "一端","他端","中心","中央","周縁","周辺","近傍","方向","位置","空間","領域","範囲","間隔","距離","形状",
    "形態","状態","種類","層","膜","部","部材","部位","部品","機構","装置","容器","組成","材料","用途","適用",
    "適用例",
    "片側","両側","左側","右側","前方","後方","上流","下流","隣接","近接","離間","間置","介在","重畳","概ね",
    "略","略中央","固定側","可動側","伸長","収縮","係合","嵌合","取付","連結部","支持体","支持部","ガイド部",
    "データ","情報","信号","出力","入力","制御","演算","取得","送信","受信","表示","通知","設定","変更","更新",
    "保存","削除","追加","実行","開始","終了","継続","停止","判定","判断","決定","選択","特定","抽出","検出",
    "検知","測定","計測","移動","回転","変位","変形","固定","配置",
    "生成","付与","供給","適用","照合","比較","算出","解析","同定","初期化","読出","書込","登録","記録","配信",
    "連携","切替","起動","復帰","監視","通知処理","取得処理","演算処理",
    "良好","容易","簡便","適切","有利","有用","有効","効果的","高い","低い","大きい","小さい","新規","改良","改善",
    "抑制","向上","低減","削減","増加","減少","可能","好適","好ましい","望ましい","優れる","優れた","高性能",
    "高効率","低コスト","コスト","簡易","安定","安定性","耐久","耐久性","信頼性","簡素","簡略","単純","最適",
    "最適化","汎用","汎用性","実現","達成","確保","維持","防止","回避","促進","不要","必要",
    "高精度","省電力","省資源","高信頼","低負荷","高純度","高密度","高感度","迅速","円滑","簡略化","低価格","実効的",
    "可能化","有効化","非必須","適合","互換",
    "出願","出願人","出願番号","出願日","出願書","出願公開","公開","公開番号","公開公報","公報","公報番号",
    "特許","特許番号","特許文献","非特許文献","引用","引用文献","先行技術","審査","審査官","拒絶","意見書","補正書",
    "優先","優先日","分割出願","継続出願","国内移行","国際出願","国際公開","PCT","登録","公開日","審査請求",
    "拒絶理由","補正","訂正","無効審判","異議","取消","取下げ","事件番号","代理人","弁理士","係属","経過",
    "第","第一","第二","第三","第1","第２","第３","第１","第２","第３","１","２","３","４","５","６","７","８","９","０",
    "一","二","三","四","五","六","七","八","九","零","数","複合","多数","少数",
    "図1","図2","図3","図4","図5","図6","図7","図8","図9","表1","表2","表3","式1","式2","式3",
    "０","１","２","３","４","５","６","７","８","９",
    "%","％","wt%","vol%","質量%","重量%","容量%","mol","mol%","mol/L","M","mm","cm","m","nm","μm","μ","rpm",
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
    "シャーシ","駆動","伝達","支持","連結",
    "解決", "準備", "提供", "発生", "以上", "十分"
]

@st.cache_data
def expand_stopwords_to_full_width(words):
    expanded = set(words)
    hankaku_chars = string.ascii_letters + string.digits
    zenkaku_chars = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９"
    trans_table = str.maketrans(hankaku_chars, zenkaku_chars)
    for word in words:
        if any(c in hankaku_chars for c in word):
            expanded.add(word.translate(trans_table))
    return sorted(list(expanded))

stopwords = expand_stopwords_to_full_width(stopwords)
stopwords = sorted(list(set(stopwords)))

# n-gram定型句リスト (正規表現フィルタ)
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

_ngram_rows = sorted(_ngram_rows, key=lambda x: (x[3], -len(x[1]) if x[2]=="literal" else -50))
_ngram_compiled = []
for cat, pat, ptype, pri in _ngram_rows:
    if ptype == "regex":
        _ngram_compiled.append((cat, re.compile(pat), ptype, pri))
    else:
        _ngram_compiled.append((cat, pat, ptype, pri))

# テキスト処理関数
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
                not re.fullmatch(r'(図|表|式)[\d０-９]+', compound_word) and
                not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
                not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
                not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
                words.append(compound_word)
            compound_word = ''

    if (len(compound_word) > 1 and
        compound_word not in stopwords and
        not re.fullmatch(r'[\d０-９]+', compound_word) and
        not re.fullmatch(r'(図|表|式)[\d０-９]+', compound_word) and
        not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
        not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
        not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
        words.append(compound_word)

    return words

# ワードクラウド生成・表示
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
    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")
        st.error("Colab環境以外（Streamlit Cloudなど）で実行する場合、フォントパス（font_path）の指定が必要です。")

    st.subheader(f"--- {title} (上位{top_n}件) ---")
    if not word_freq:
        st.write("（キーワードなし）")
    else:
        list_data = { "キーワード": [], "出現頻度": [] }
        for word, freq in word_freq.most_common(top_n):
            list_data["キーワード"].append(word)
            list_data["出現頻度"].append(freq)
        st.dataframe(pd.DataFrame(list_data))

    return word_freq

# 特徴語抽出
@st.cache_data
def get_characteristic_words(target_counter, other_counter1, other_counter2):
    char_words = {}
    total_target = sum(target_counter.values()) + 1
    total_other1 = sum(other_counter1.values()) + 1
    total_other2 = sum(other_counter2.values()) + 1

    for word, freq in target_counter.items():
        tf_target = freq / total_target
        tf_other1 = other_counter1.get(word, 0) / total_other1
        tf_other2 = other_counter2.get(word, 0) / total_other2
        score = tf_target / (tf_other1 + tf_other2 + 1e-9)
        if score > 1: char_words[word] = freq
    return list(Counter(char_words).elements())

# ==================================================================
# --- 3. Streamlit UI ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO | Explorer",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Explorer")
st.markdown("特許テキストからキーワード（複合名詞）を抽出し、全体・競合比較・時系列でのトレンドを可視化します。")

# ==================================================================
# --- 4. セッション状態の確認と初期化 ---
# ==================================================================
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    df_main = st.session_state.df_main
    col_map = st.session_state.col_map
    delimiters = st.session_state.delimiters

# ==================================================================
# --- 5. Explorer アプリケーション ---
# ==================================================================

# --- フォントパスの設定 (Colab / Streamlit Cloud) ---
font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'
if not os.path.exists(font_path):
    system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    jp_fonts = [f for f in system_fonts if 'ipagp' in f or 'notosanscjkjp' in f.lower() or 'hiragino' in f.lower()]
    if jp_fonts:
        font_path = jp_fonts[0]
        st.info(f"日本語フォントとして {font_path} を使用します。")
    else:
        st.warning("日本語フォントが見つかりません。ワードクラウドが文字化けする可能性があります。")
        font_path = None


# --- UI設定 ---
st.subheader("分析パラメータ設定")

with st.container(border=True):
    st.markdown("##### 企業比較分析の設定")

    applicant_list = ["(指定なし)"]
    if col_map['applicant'] and col_map['applicant'] in df_main.columns:
        try:
            applicants = df_main[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
            applicants = applicants[applicants != '']
            applicant_list = ["(指定なし)"] + sorted(applicants.unique())
        except Exception as e:
            st.warning(f"出願人リストの生成に失敗: {e}")

    col1, col2, col3 = st.columns(3)
    with col1:
        my_company = st.selectbox("自社名 (MY_COMPANY):", applicant_list, key="exp_my_company")
    with col2:
        company_a = st.selectbox("競合A (COMPANY_A):", applicant_list, key="exp_comp_a")
    with col3:
        company_b = st.selectbox("競合B (COMPANY_B):", applicant_list, key="exp_comp_b")

    st.markdown("##### 時系列分析の設定")
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_time_series = st.checkbox("時系列分析を有効にする", value=True, key="exp_enable_time")
    with col2:
        date_column = col_map.get('date', None)
        st.text_input("日付カラム (自動選択):", value=date_column, disabled=True)
    with col3:
        time_slice_years = st.number_input("何年ごとに区切るか:", min_value=1, value=5, key="exp_time_slice")

    st.markdown("##### 出力設定")
    top_n_keywords = st.number_input("各キーワードリストで上位何件まで表示するか:", min_value=5, value=20, key="exp_top_n")


# --- 分析実行 ---
st.markdown("---")
if st.button("Explorer キーワード分析を実行", type="primary", key="exp_run_analysis"):

    if not (col_map['title'] and col_map['abstract'] and col_map['applicant']):
        st.error("エラー: Mission Controlで「発明の名称」「要約」「出願人」のカラムを正しく紐付けてください。")
        st.stop()
    if enable_time_series and not date_column:
        st.error("エラー: 時系列分析を有効にする場合、Mission Controlで「出願日」カラムを紐付けてください。")
        st.stop()

    try:
        df_main['text'] = df_main[col_map['title']].fillna('') + ' ' + df_main[col_map['abstract']].fillna('')
        df_main['権利者'] = df_main[col_map['applicant']].astype(str).str.split(delimiters['applicant'])
        df_exploded = df_main.explode('権利者')
        df_exploded['権利者'] = df_exploded['権利者'].str.strip()

        st.success("データの前処理が完了しました。")

        # --- 分析1: 全体のワードクラウド ---
        with st.container(border=True):
            st.header("データセット全体の技術キーワード")
            with st.spinner("データセット全体のキーワードを抽出中..."):
                all_words = []
                for text in df_main['text']:
                    all_words.extend(extract_compound_nouns(text))
                generate_wordcloud_and_list(all_words, "データセット全体の技術キーワード", top_n_keywords, font_path)

        # --- 分析2: 企業比較分析 ---
        target_companies = [c for c in [my_company, company_a, company_b] if c != "(指定なし)"]

        if target_companies:
            with st.container(border=True):
                st.header(f"企業比較: {', '.join(target_companies)}")
                company_words = {}

                with st.spinner("各企業のキーワードを抽出中..."):
                    for company in target_companies:
                        company_df = df_exploded[df_exploded['権利者'] == company]
                        if company_df.empty:
                            st.warning(f"警告: 企業 '{company}' のデータが見つかりませんでした。スキップします。")
                            company_words[company] = []
                            continue
                        words = []
                        for text in company_df['text']:
                            words.extend(extract_compound_nouns(text))
                        company_words[company] = words

                for company, words in company_words.items():
                    generate_wordcloud_and_list(words, f"'{company}'の技術キーワード", top_n_keywords, font_path)

                my_counter = Counter(company_words.get(my_company, []))
                a_counter = Counter(company_words.get(company_a, []))
                b_counter = Counter(company_words.get(company_b, []))

                if my_company != "(指定なし)" and company_a != "(指定なし)":
                    common_vs_a = list((my_counter & a_counter).elements())
                    generate_wordcloud_and_list(common_vs_a, f"'{my_company}' と '{company_a}' の共通キーワード", top_n_keywords, font_path)

                if my_company != "(指定なし)" and company_b != "(指定なし)":
                    common_vs_b = list((my_counter & b_counter).elements())
                    generate_wordcloud_and_list(common_vs_b, f"'{my_company}' と '{company_b}' の共通キーワード", top_n_keywords, font_path)

                st.markdown("---")
                st.subheader("特徴/独自キーワード")

                my_char_words = get_characteristic_words(my_counter, a_counter, b_counter)
                generate_wordcloud_and_list(my_char_words, f"'{my_company}' の特徴的キーワード", top_n_keywords, font_path)

                a_char_words = get_characteristic_words(a_counter, my_counter, b_counter)
                generate_wordcloud_and_list(a_char_words, f"'{company_a}' の特徴的キーワード", top_n_keywords, font_path)

                b_char_words = get_characteristic_words(b_counter, my_counter, a_counter)
                generate_wordcloud_and_list(b_char_words, f"'{company_b}' の特徴的キーワード", top_n_keywords, font_path)

                my_set = set(word for word, freq in my_counter.most_common(500))
                a_set = set(word for word, freq in a_counter.most_common(500))
                b_set = set(word for word, freq in b_counter.most_common(500))

                my_unique = {k: my_counter[k] for k in my_set - a_set - b_set}
                generate_wordcloud_and_list(list(Counter(my_unique).elements()), f"'{my_company}' の独自キーワード", top_n_keywords, font_path)

                a_unique = {k: a_counter[k] for k in a_set - my_set - b_set}
                generate_wordcloud_and_list(list(Counter(a_unique).elements()), f"'{company_a}' の独自キーワード", top_n_keywords, font_path)

                b_unique = {k: b_counter[k] for k in b_set - my_set - a_set}
                generate_wordcloud_and_list(list(Counter(b_unique).elements()), f"'{company_b}' の独自キーワード", top_n_keywords, font_path)

        # --- 分析3: 時系列分析 ---
        if enable_time_series:
            with st.container(border=True):
                st.header(f"技術キーワードの時系列分析 ({time_slice_years}年ごと)")
                try:
                    df_time = df_main.copy()
                    # 'year' カラムは app.py で 'parsed_date' から既に作成済み
                    df_time.dropna(subset=['year'], inplace=True)

                    min_year = int(df_time['year'].min())
                    max_year = int(df_time['year'].max())

                    with st.spinner("時系列キーワードを抽出中..."):
                        for start_year in range(min_year, max_year + 1, time_slice_years):
                            end_year = start_year + time_slice_years - 1
                            period_df = df_time[(df_time['year'] >= start_year) & (df_time['year'] <= end_year)]

                            if period_df.empty: continue

                            period_words = []
                            for text in period_df['text']:
                                period_words.extend(extract_compound_nouns(text))

                            title = f"技術キーワードの変遷 ({start_year} - {end_year})"
                            generate_wordcloud_and_list(period_words, title, top_n_keywords, font_path)

                except Exception as e:
                    st.error(f"時系列分析中にエラーが発生しました: {e}")
                    st.warning(f"ヒント: '{date_column}'列の日付形式が認識できない可能性があります。'YYYY-MM-DD'形式などを確認してください。")

        st.success("全ての分析が完了しました。")

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        import traceback
        st.exception(traceback.format_exc())

# --- 共通サイドバーフッター ---
st.sidebar.markdown("---") 
st.sidebar.caption("ナビゲーション:")
st.sidebar.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
st.sidebar.caption("2. 左のリストから分析モジュールを選択します。")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 しばやま")
