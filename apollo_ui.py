# ==================================================================
# APOLLO 共通UIシェル
# 全ビュー共通のテーマCSS・ブランド・ページヘッダー・データガード・
# 読み方カード・ステッパー等。分析ロジックは一切持たない。
#
# デザイン方針: シェル（サイドバー）はグラファイトの計器盤、
# 操作するコンテンツ領域は白基調。図の描画領域を白に保つのは、
# 俯瞰図の12色パレットが白地で色覚検証されているため。
# ==================================================================
import streamlit as st

# 正式名称（バクロニム）。サイドバーのロックアップで使う。
FULL_NAME = "Advanced Patent & Overall Landscape-analytics Logic Orbiter"

# --- シェル（サイドバー）: 暗い計器盤 ---
SHELL = "#141B23"
SHELL_RAISED = "#1D2732"
SHELL_LINE = "#2B3743"
SHELL_HOVER = "#26313D"
SHELL_INK = "#E4EBF2"
SHELL_INK2 = "#93A4B5"
SHELL_INK3 = "#5E7086"

# --- コンテンツ（白基調） ---
INK = "#141C24"
INK2 = "#57687A"
LINE = "#DFE6EC"
PAGE_BG = "#F7F9FB"

# --- アクセント: イグニッション ---
IGNITION = "#C2461A"          # 白地の上で使う
IGNITION_ON_DARK = "#E2582A"  # 暗いシェルの上で使う

GOOD = "#12734B"
GOOD_BG = "#E4F1EA"
WARN = "#9E5B08"
WARN_BG = "#FBF0DF"

# 既存ビューが参照している色名との互換
NAVY = INK
GOLD = IGNITION

# カテゴリ配色（図の系列色。文字には載せない）
PALETTE = ["#2A78D6", "#1BAF7A", "#8A63D2", "#EDA100",
           "#D9636C", "#3AA6B9", "#7A8B3D", "#C25588"]
NOISE_COLOR = "#AEB8C2"

# 内部モジュール名（スキーマ・スナップショット用）→ 画面表示名。
# APOLLO は画面もコードネーム主のため、ほぼ恒等写像になる。
# VOYAGER は Gemini API の廃止とともに退役し、レポート生成は CAPCOM に統合された。
MODULE_DISPLAY = {
    "Mission Control": "Mission Control",
    "ATLAS": "ATLAS",
    "CORE": "CORE",
    "Saturn V": "Saturn V",
    "MEGA": "MEGA",
    "Explorer": "Explorer",
    "CREW": "CREW",
    "EAGLE": "EAGLE",
    "NEBULA": "NEBULA",
    "VOYAGER": "CAPCOM",
    "CAPCOM": "CAPCOM",
}

# 内部モジュール名 → 機能名（レポート本文で使う呼称・画面の副題）
MODULE_FUNCTION = {
    "Mission Control": "母集団設計・データ準備",
    "ATLAS": "基本統計",
    "CORE": "ルール分類",
    "Saturn V": "俯瞰図分析",
    "MEGA": "動態分析",
    "Explorer": "キーワード分析",
    "CREW": "出願人・発明者ネットワーク",
    "EAGLE": "探索的クラスタリング",
    "NEBULA": "環境分析",
    "VOYAGER": "レポート生成",
    "CAPCOM": "レポート生成",
}


def module_display(internal_name: str) -> str:
    """内部モジュール名を画面表示名に変換する（未知の名前はそのまま返す）"""
    return MODULE_DISPLAY.get(internal_name, internal_name)


def module_function(internal_name: str) -> str:
    """内部モジュール名を機能名に変換する（未知の名前はそのまま返す）"""
    return MODULE_FUNCTION.get(internal_name, internal_name)


# ==================================================================
# ロゴマーク（軌道）— 円盤を軌道リングが横切り、探査機が1点。
# 小さいサイズほど線を太くして 20px でも輪郭が残るようにする。
# ==================================================================
def orbit_mark(size: int = 30, on_dark: bool = True) -> str:
    """APOLLO のロゴマークを SVG 文字列で返す。on_dark でシェル用／白地用を切り替える。"""
    ring = IGNITION_ON_DARK if on_dark else IGNITION
    disc = SHELL_INK if on_dark else INK
    sw = 3.4 if size <= 22 else (2.8 if size <= 32 else 2.2)
    craft = "" if size <= 22 else (
        f'<circle cx="37.5" cy="17.1" r="{2.8 if size <= 32 else 2.6}" fill="{ring}"/>')
    return (
        f'<svg viewBox="0 0 48 48" width="{size}" height="{size}" '
        f'style="flex:0 0 auto;display:block" aria-hidden="true">'
        f'<g transform="rotate(-28 24 24)">'
        f'<ellipse cx="24" cy="24" rx="21" ry="9" fill="none" stroke="{ring}" stroke-width="{sw}"/></g>'
        f'<circle cx="24" cy="24" r="{11 if size <= 22 else 10.5}" fill="{disc}"/>'
        f'<g transform="rotate(-28 24 24)">'
        f'<path d="M3 24A21 9 0 0 0 45 24" fill="none" stroke="{ring}" '
        f'stroke-width="{sw}" stroke-linecap="round"/>{craft}</g></svg>'
    )


_CSS = f"""
<style>
/* ---- タイポグラフィと余白 ---- */
/* padding-top はStreamlitの固定ヘッダー(stHeader)にh1が隠れない高さを確保する */
.block-container {{ padding-top: 3.4rem; padding-bottom: 4rem;
  max-width: 1720px; padding-left: 2.2rem; padding-right: 2.2rem; }}
[data-testid="stHeader"] {{ background: transparent; height: 2.75rem; }}
.stApp {{ background: {PAGE_BG}; }}
h1 {{ font-weight: 800 !important; letter-spacing: .01em; color: {INK} !important;
  font-size: 1.65rem !important; }}
h2 {{ font-weight: 700 !important; letter-spacing: .01em; font-size: 1.5rem !important; }}
h3 {{ font-weight: 700 !important; letter-spacing: .01em; font-size: 1.25rem !important; }}

/* ---- カード（枠線囲みコンテナ・expander共通）: 白+淡い影+ゆとりのある余白 ---- */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFFFFF;
  border-radius: 12px; box-shadow: 0 1px 3px rgba(16, 24, 40, .06);
  padding: .55rem .65rem; }}
[data-testid="stExpander"] {{ background: #FFFFFF; border-radius: 12px;
  box-shadow: 0 1px 3px rgba(16, 24, 40, .06); }}
[data-testid="stExpander"] details {{ border-radius: 12px; }}

/* ---- 要素間の縦の間隔を少し広げて呼吸させる ---- */
[data-testid="stVerticalBlock"] {{ gap: 1.15rem; }}

/* ---- Streamlit標準の装飾を隠す ---- */
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stToolbar"] {{ visibility: hidden; }}

/* ==================================================================
   サイドバー = 計器盤。ここだけ暗く、操作するコンテンツは白のまま。
   ================================================================== */
[data-testid="stSidebar"] {{ background: {SHELL}; border-right: 1px solid #0A0F14;
  min-width: 292px; max-width: 292px; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .42rem; }}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {{ color: {SHELL_INK2}; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: {SHELL_INK3}; }}
[data-testid="stSidebar"] hr {{ border-color: {SHELL_LINE}; }}
/* コンテンツ側の白カード化はシェルに持ち込まない（暗い地の上で白箱になってしまう） */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
  background: transparent; box-shadow: none; padding: 0; }}

/* ページリンク: 既定は控えめ、現在地はイグニッションの縦罫で示す。
   Streamlit は現在ページのリンクだけを href="" で描画するため、これを現在地の目印に使う。 */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
  border-radius: 7px; padding: 7px 11px; position: relative;
  transition: background .12s ease; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] p {{
  color: {SHELL_INK}; font-size: 14px; font-weight: 600; margin: 0;
  letter-spacing: .01em; white-space: normal; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] p strong {{
  color: {SHELL_INK}; font-size: 14px; font-weight: 700; letter-spacing: .04em;
  margin-right: 4px; }}
[data-testid="stSidebar"] [data-testid="stIconMaterial"] {{
  color: {SHELL_INK2} !important; font-size: 19px !important; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{ background: {SHELL_HOVER}; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href=""] {{ background: {SHELL_HOVER}; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href=""]::before {{
  content: ""; position: absolute; left: 0; top: 7px; bottom: 7px; width: 2.5px;
  border-radius: 2px; background: {IGNITION_ON_DARK}; }}
[data-testid="stSidebar"] a[href=""] p strong {{ color: #FFFFFF; }}
[data-testid="stSidebar"] a[href=""] [data-testid="stIconMaterial"] {{
  color: {IGNITION_ON_DARK} !important; }}

/* ---- サイドバー: ブランドロックアップ ---- */
/* 余白は内側の .inner に置く。最外側 div の padding/margin は Streamlit の
   高さ実測に含まれず、次の要素と重なってしまうため。 */
.ap-brand .inner {{ padding: 2px 0 6px; }}
.ap-brand .brandrow {{ display: flex; align-items: center; gap: 10px; }}
.ap-brand .word {{ font-size: 32px; font-weight: 800; letter-spacing: .13em;
  color: {SHELL_INK}; line-height: 1; }}
.ap-brand .tag {{ font-size: 11.5px; color: {SHELL_INK3}; letter-spacing: .03em;
  line-height: 1.6; margin-top: 8px; }}

/* Streamlit はサイドバーの自前HTMLの要素コンテナを実描画より16px低く確保するため、
   ブロック末尾に同じ高さの詰め物を置いて次の要素との重なりを防ぐ。 */
.ap-spacer {{ height: 16px; }}

/* ---- サイドバー: データ状態カード ---- */
.ap-status {{ background: {SHELL_RAISED}; border: 1px solid {SHELL_LINE}; border-radius: 9px;
  font-size: 12.5px; color: {SHELL_INK2}; line-height: 1.85; }}
.ap-status .inner {{ padding: 11px 13px; }}
.ap-status .f {{ color: {SHELL_INK}; font-weight: 600; }}
.ap-status .n {{ color: {SHELL_INK}; font-weight: 700; }}
.ap-status .ok {{ color: #48C48C; font-weight: 700; }}
.ap-status .ng {{ color: #E0A24A; font-weight: 700; }}
.ap-status .rec {{ color: {IGNITION_ON_DARK}; font-weight: 700; }}

/* ---- サイドバーグループ見出し ---- */
.ap-group {{ font-size: 11.5px; font-weight: 700; letter-spacing: .14em; color: {SHELL_INK2};
  padding: 13px 0 0 4px; line-height: 1.4; }}

/* ---- フッター ---- */
.ap-foot {{ font-size: 11px; color: {SHELL_INK3}; letter-spacing: .02em;
  line-height: 1.7; padding-top: 16px; }}

/* ---- ページヘッダー: ロゴ + 機能名（主）+ コードネーム（従）---- */
.ap-head {{ display: flex; align-items: center; gap: 12px; margin-bottom: 2px; }}
.ap-head .titles {{ display: flex; align-items: baseline; gap: 11px; min-width: 0; }}
.ap-head .fn {{ font-size: 1.55rem; font-weight: 800; color: {INK}; letter-spacing: .01em; }}
.ap-head .cn {{ font-size: .9rem; font-weight: 700; color: {IGNITION}; letter-spacing: .18em; }}
.ap-sub {{ color: {INK2}; font-size: .92rem; margin: .2rem 0 1.1rem; }}

/* ---- HTML内に埋め込むMaterial Symbolsアイコン（:material/x: が使えない文脈用）---- */
.msym {{ font-family: "Material Symbols Rounded", "Material Symbols Outlined";
  font-weight: normal; font-style: normal; font-size: 1.05em; line-height: 1;
  vertical-align: -.15em; letter-spacing: normal; text-transform: none;
  white-space: nowrap; direction: ltr; -webkit-font-feature-settings: "liga";
  -webkit-font-smoothing: antialiased; margin-right: .25em; }}

/* ---- チップ ---- */
.ap-chip {{ display: inline-block; font-size: 11.5px; font-weight: 700; padding: 1px 10px;
  border-radius: 99px; margin-right: 6px; }}

/* ---- ステッパー ---- */
.ap-step {{ display: flex; gap: 12px; padding: 10px 2px; border-bottom: 1px dashed {LINE}; }}
.ap-step:last-child {{ border-bottom: none; }}
.ap-step .dot {{ width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; }}
.ap-step .dot.done {{ background: {GOOD_BG}; color: {GOOD}; }}
.ap-step .dot.now  {{ background: {IGNITION}; color: #fff; box-shadow: 0 0 0 4px #C2461A22; }}
.ap-step .dot.todo {{ border: 2px dashed #B9C4CE; color: #B9C4CE; }}
.ap-step .t {{ font-size: 14px; font-weight: 700; color: {INK}; }}
.ap-step .d {{ font-size: 12.5px; color: {INK2}; }}

/* ---- st.metric をカード化 ---- */
[data-testid="stMetric"] {{ border: 1px solid {LINE}; border-radius: 12px;
  padding: 16px 20px; background: #fff;
  box-shadow: 0 1px 3px rgba(16, 24, 40, .06); }}
[data-testid="stMetricLabel"] p {{ font-size: 12px !important; color: {INK2} !important; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem !important; font-weight: 700;
  color: {INK}; font-variant-numeric: tabular-nums; }}

/* ---- ボタン ---- */
.stButton button[kind="primary"] {{ font-weight: 700; }}

/* ---- 読み方カード（既定は畳む。熟練者の視界を塞がない）---- */
.ap-howto {{ border-left: 3px solid {IGNITION}; background: #FDF3EF; border-radius: 0 10px 10px 0;
  padding: 2px 16px; font-size: 13px; line-height: 1.9; margin: 6px 0 10px; }}
.ap-howto > summary {{ cursor: pointer; list-style: none; padding: 8px 0;
  font-weight: 700; color: {INK}; font-size: 12.5px; }}
.ap-howto > summary::-webkit-details-marker {{ display: none; }}
.ap-howto > summary::before {{ content: "▸ "; color: {IGNITION}; font-weight: 700; }}
.ap-howto[open] > summary::before {{ content: "▾ "; }}
.ap-howto .body {{ padding: 0 0 10px 14px; color: {INK}; }}
.ap-howto b {{ color: {IGNITION}; }}

/* ---- 飛行記録（フライトレコーダー）カード ---- */
.ap-rec {{ border: 1px solid {LINE}; border-radius: 10px; background: #fff; padding: 13px 16px; }}
.ap-rec .hd {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }}
.ap-rec .pill {{ font-size: 9px; font-weight: 700; letter-spacing: .1em; color: {IGNITION};
  border: 1px solid {IGNITION}; border-radius: 3px; padding: 1px 5px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.ap-rec .ttl {{ font-size: 12.5px; font-weight: 700; color: {INK}; }}
.ap-rec table {{ width: 100%; border-collapse: collapse; font-size: 11.8px; }}
.ap-rec td {{ padding: 5px 0; border-bottom: 1px dashed {LINE}; line-height: 1.65; vertical-align: top; }}
.ap-rec tr:last-child td {{ border-bottom: none; }}
.ap-rec td.k {{ color: {INK2}; width: 132px; }}
.ap-rec td.v {{ font-weight: 600; color: {INK}; font-variant-numeric: tabular-nums; }}
</style>
"""


def inject_css():
    """全ページ共通CSS。app.py が毎回呼ぶ（views からは呼ばない）"""
    st.markdown(_CSS, unsafe_allow_html=True)


def brand_block():
    """サイドバー最上部のロックアップ（ロゴマーク + ワードマーク + 正式名称）"""
    st.markdown(
        '<div class="ap-brand"><div class="inner">'
        f'<div class="brandrow">{orbit_mark(34, on_dark=True)}'
        '<div class="word">APOLLO</div></div>'
        f'<div class="tag">{FULL_NAME}</div>'
        "</div></div><div class=ap-spacer></div>",
        unsafe_allow_html=True,
    )


def status_card():
    """サイドバー常駐のデータ状態カード（実 session_state を参照）"""
    df = st.session_state.get("df_main")
    done = st.session_state.get("preprocess_done", False)
    fname = st.session_state.get("filename", "No File")
    snaps = len(st.session_state.get("snapshots", []) or [])
    capcom_on = bool(st.session_state.get("capcom_session_id"))
    rec_n = len((st.session_state.get("flight_recorder") or {}).get("runs", []) or [])

    if df is None:
        st.markdown(
            '<div class="ap-status"><div class="inner">データ未読込 — '
            'まず <span class="f">Mission Control</span> から始めてください</div></div>'
            "<div class=ap-spacer></div>",
            unsafe_allow_html=True,
        )
        return

    pp = '<span class="ok">前処理済</span>' if done else '<span class="ng">前処理 未実行</span>'
    rec = (f'<span class="rec">● REC</span> 飛行記録 <span class="n">{rec_n}</span>件'
           if capcom_on else "記録セッション OFF")
    st.markdown(
        f'<div class="ap-status"><div class="inner"><span class="f">{fname}</span><br>'
        f'{len(df):,}件 ・ {pp}<br>'
        f'スナップショット <span class="n">{snaps}</span>枚<br>{rec}</div></div>'
        "<div class=ap-spacer></div>",
        unsafe_allow_html=True,
    )


def page_header(func_name: str, code_name: str, subtitle: str):
    """ページ見出し: ロゴマーク + 機能名（主）+ コードネーム（従）+ 1行説明"""
    st.markdown(
        f'<div class="ap-head">{orbit_mark(30, on_dark=False)}'
        f'<div class="titles"><span class="fn">{func_name}</span>'
        f'<span class="cn">{code_name}</span></div></div>'
        f'<p class="ap-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def require_data():
    """全分析ビュー共通のデータガード。未準備なら案内を出して停止する。"""
    if st.session_state.get("preprocess_done"):
        return
    st.warning("分析データが準備できていません。", icon=":material/warning:")
    with st.container(border=True):
        st.markdown(
            "**必要な手順は2つです**\n\n"
            "1. Mission Control で特許リスト（CSV / Excel）を読み込む（サンプルデータもあります）\n"
            "2. 同じ画面で「前処理を実行」を押す（完了後に全モジュールが有効になります）"
        )
        pages = st.session_state.get("ap_pages", {})
        if "mission_control" in pages:
            st.page_link(pages["mission_control"], label="Mission Control を開く",
                         icon=":material/rocket_launch:")
    st.stop()


def howto(text_html: str, title: str = "この図の読み方"):
    """図の直下に置く読み方カード。

    APOLLO は本格利用者向けのため既定で畳む。st.expander ではなく <details> を
    使うのは、expander のネスト禁止（折りたたみの入れ子でレイアウトが壊れる）に
    触れずに、どの文脈へ置いても安全にするため。
    """
    st.markdown(
        f'<details class="ap-howto"><summary>{title}</summary>'
        f'<div class="body">{text_html}</div></details>',
        unsafe_allow_html=True,
    )


def step(state: str, num: str, title: str, desc: str):
    """ステッパー1行（state: done / now / todo）"""
    mark = "✓" if state == "done" else num
    st.markdown(
        f'<div class="ap-step"><div class="dot {state}">{mark}</div>'
        f'<div><div class="t">{title}</div><div class="d">{desc}</div></div></div>',
        unsafe_allow_html=True,
    )


def chip(text: str, bg: str = "#FDF3EF", fg: str = IGNITION) -> str:
    return f'<span class="ap-chip" style="background:{bg};color:{fg}">{text}</span>'


def snapshot_hint():
    """スナップショット保存ボタンの直後に置く共通の行き先案内"""
    n = len(st.session_state.get("snapshots", []) or [])
    st.caption(
        f"現在 {n} 枚保存済み。保存した図はレポートの素材になります。"
        "ブラウザを閉じると失われるため、収集を終えたら CAPCOM で書き出してください。"
    )


def analysis_target_selector(df_main, key_prefix: str):
    """分析対象（母集団のサブセット）を選ぶ共通カード。(df_target, scope_label) を返す。

    - 全体（既定）
    - 俯瞰図分析のクラスタで絞る（Saturn V 実行後に選択可・複数選択）
    - ルール分類のカテゴリで絞る（CORE の分類実行後に選択可・軸→カテゴリ複数選択）

    返す df はフィルタ済みのビュー（index は元の df_main のまま）。呼び出し側は
    scope_label を図のキャプション等に添えて、スナップショットの読み手が
    対象範囲を誤解しないようにする。0件になる選択は全体へフォールバックする。
    """
    has_cluster = ("cluster" in getattr(df_main, "columns", [])) and bool(
        st.session_state.get("saturnv_labels_map"))
    _dfc = st.session_state.get("core_df_classified")
    _axes = [a for a in (st.session_state.get("core_classification_rules") or {}).keys()
             if _dfc is not None and a in _dfc.columns]
    has_rules = len(_axes) > 0

    _OPT_ALL = "全体"
    _OPT_CLU = "俯瞰図のクラスタで絞る"
    _OPT_CAT = "ルール分類のカテゴリで絞る"
    options = [_OPT_ALL] + ([_OPT_CLU] if has_cluster else []) + ([_OPT_CAT] if has_rules else [])

    df_t = df_main
    label = f"全体（{len(df_main):,}件）"

    with st.container(border=True):
        mode = st.selectbox(
            "分析対象:", options, key=f"{key_prefix}_scope_mode",
            help="このページの分析を、母集団全体で行うか、特定の技術テーマ（俯瞰図のクラスタ）や"
                 "分類カテゴリに絞って行うかを選びます。Saturn V・CORE を実行すると選択肢が増えます。")

        if mode == _OPT_CLU and has_cluster:
            labels_map = st.session_state["saturnv_labels_map"]
            counts = df_main["cluster"].value_counts()
            copts = [(f"{labels_map.get(cid, cid)}（{int(counts.get(cid, 0)):,}件）", cid)
                     for cid in sorted(k for k in labels_map.keys() if k != -1)]
            sel = st.multiselect(
                "対象クラスタ（複数選択可）:", copts, format_func=lambda x: x[0],
                key=f"{key_prefix}_scope_clusters",
                help="俯瞰図分析で検出された技術テーマから、分析対象にするものを選びます。")
            if sel:
                cids = [s[1] for s in sel]
                df_t = df_main[df_main["cluster"].isin(cids)]
                _names = "・".join(str(labels_map.get(c, c)) for c in cids)
                if len(_names) > 40:
                    _names = _names[:40] + f"…（{len(cids)}クラスタ）"
                label = f"クラスタ: {_names}（{len(df_t):,}件）"
            else:
                st.caption("クラスタが未選択のため、全体を対象にします。")
        elif mode == _OPT_CAT and has_rules:
            ax = st.selectbox("分類軸:", _axes, key=f"{key_prefix}_scope_axis")
            if ax:
                cat_counts = _dfc[ax].fillna("").str.split(";").explode().value_counts()
                catopts = [(f"{c}（{int(n):,}件）", c) for c, n in cat_counts.items() if c]
                sel = st.multiselect(
                    "対象カテゴリ（複数選択可）:", catopts, format_func=lambda x: x[0],
                    key=f"{key_prefix}_scope_cats_{ax}",
                    help="ルール分類の結果カテゴリから、分析対象にするものを選びます（複数該当の特許はいずれかに一致すれば対象）。")
                if sel:
                    vals = [s[1] for s in sel]
                    mask = _dfc[ax].fillna("").str.split(";").apply(
                        lambda lst: any(v in lst for v in vals))
                    idx = _dfc.index[mask]
                    df_t = df_main.loc[df_main.index.intersection(idx)]
                    _names = "・".join(vals)
                    if len(_names) > 40:
                        _names = _names[:40] + f"…（{len(vals)}カテゴリ）"
                    label = f"{ax}: {_names}（{len(df_t):,}件）"
                else:
                    st.caption("カテゴリが未選択のため、全体を対象にします。")

        if len(options) == 1:
            st.caption("Saturn V（クラスタ）や CORE（カテゴリ）を実行すると、その結果で対象を絞れるようになります。")

        if len(df_t) == 0:
            st.warning("選択した対象が0件のため、全体を対象にします。選択を見直してください。",
                       icon=":material/warning:")
            df_t = df_main
            label = f"全体（{len(df_main):,}件）"
        elif len(df_t) < len(df_main):
            _note = "（件数が少ないと分析が不安定になることがあります）" if len(df_t) < 30 else ""
            st.caption(f"現在の対象: **{label}**{_note}")

    return df_t, label
