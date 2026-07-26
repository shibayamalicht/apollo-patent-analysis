# ==================================================================
# APOLLO — 特許情報分析プラットフォーム エントリポイント
# st.navigation によるグループナビ + 共通サイドバー。
# 各ビューは views/*.py の render() に実装する。
# ==================================================================
import streamlit as st

st.set_page_config(
    page_title="APOLLO — 特許情報分析プラットフォーム",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

import apollo_ui


# --- セッション初期化（キー名は CAPCOM スキーマ互換のため不変） ---
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
        "capcom_mission_objective": "",
        "snapshots": [],
        # フライトレコーダー（分析条件の自動記録・APOLLO v10 で新設）
        "flight_recorder": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()
apollo_ui.inject_css()

# --- ビュー読込（遅延import: 開いたページのモジュールだけを読み込み、初回起動を軽くする） ---
import importlib


def _view(name: str):
    def _run():
        importlib.import_module(f"views.{name}").render()
    return _run


# --- ページ定義（コードネーム主・機能名従） ---
pg_mission = st.Page(_view("mission_control"), title="Mission Control",
                     icon=":material/rocket_launch:", url_path="mission_control", default=True)
pg_atlas = st.Page(_view("atlas"), title="ATLAS", icon=":material/query_stats:", url_path="atlas")
pg_core = st.Page(_view("core"), title="CORE", icon=":material/rule:", url_path="core")
pg_saturnv = st.Page(_view("saturnv"), title="Saturn V", icon=":material/scatter_plot:", url_path="saturnv")
pg_eagle = st.Page(_view("eagle"), title="EAGLE", icon=":material/highlight_alt:", url_path="eagle")
pg_mega = st.Page(_view("mega"), title="MEGA", icon=":material/trending_up:", url_path="mega")
pg_explorer = st.Page(_view("explorer"), title="Explorer", icon=":material/share:", url_path="explorer")
pg_crew = st.Page(_view("crew"), title="CREW", icon=":material/hub:", url_path="crew")
pg_nebula = st.Page(_view("nebula"), title="NEBULA", icon=":material/blur_on:", url_path="nebula")
pg_capcom = st.Page(_view("capcom_page"), title="CAPCOM",
                    icon=":material/settings_input_antenna:", url_path="capcom")

# ビュー間リンク用レジストリ（st.page_link で参照）
st.session_state["ap_pages"] = {
    "mission_control": pg_mission, "atlas": pg_atlas, "core": pg_core, "saturnv": pg_saturnv,
    "eagle": pg_eagle, "mega": pg_mega, "explorer": pg_explorer, "crew": pg_crew,
    "nebula": pg_nebula, "capcom": pg_capcom,
}

# EAGLE は Saturn V の結果を手で引き直すモジュールなので、俯瞰図分析の直下に置く
nav = st.navigation({
    "① 母集団": [pg_mission],
    "② 分析": [pg_atlas, pg_core, pg_saturnv, pg_eagle, pg_mega, pg_explorer, pg_crew, pg_nebula],
    "③ レポート": [pg_capcom],
}, position="hidden")

# --- カスタムサイドバー（ブランド → データ状態 → ナビ → フッター） ---
# ラベルは機能名のみ。何をする画面かが一読で分かることを優先し、コードネームは
# ページヘッダー側に小さく従えて示す（サイドバーの横幅も抑えられる）。
# タプルの第2要素（コードネーム）はページヘッダーとの対応を保つため残す。
_NAV = [
    ("① 母集団", [(pg_mission, "Mission Control", "母集団設計・データ準備", ":material/rocket_launch:")]),
    ("② 分析", [
        (pg_atlas, "ATLAS", "基本統計", ":material/query_stats:"),
        (pg_core, "CORE", "ルール分類", ":material/rule:"),
        (pg_saturnv, "Saturn V", "俯瞰図分析", ":material/scatter_plot:"),
        (pg_eagle, "EAGLE", "探索的クラスタリング", ":material/highlight_alt:"),
        (pg_mega, "MEGA", "動態分析", ":material/trending_up:"),
        (pg_explorer, "Explorer", "キーワード分析", ":material/share:"),
        (pg_crew, "CREW", "出願人ネットワーク", ":material/hub:"),
        (pg_nebula, "NEBULA", "環境分析", ":material/blur_on:"),
    ]),
    ("③ レポート", [(pg_capcom, "CAPCOM", "レポート生成", ":material/settings_input_antenna:")]),
]

with st.sidebar:
    apollo_ui.brand_block()
    apollo_ui.status_card()

    # 見出しの末尾に詰め物を付けるのは、Streamlit がサイドバーの自前HTMLの
    # 要素コンテナを実描画より16px低く確保し、次のリンクと重なるため（apollo_ui 参照）。
    for _group, _items in _NAV:
        st.markdown(f'<div class="ap-group">{_group}</div><div class=ap-spacer></div>',
                    unsafe_allow_html=True)
        for _pg, _code, _func, _icon in _items:
            st.page_link(_pg, label=_func, icon=_icon)

    st.markdown('<div class="ap-foot">APOLLO v10.0.0<br>© 2026 しばやま</div>',
                unsafe_allow_html=True)

nav.run()
