# ==================================================================
# 10_📡_CAPCOM.py — APOLLO v9.0.0 CAPCOM (Capsule Communicator)
# セッション管理・ZIPエクスポート・マルチツール連携
# （Claude Code / Codex CLI / Antigravity IDE を複数選択可能）
# ==================================================================

import streamlit as st
import os
import json
import datetime
# page_icon に実写アイコンを使うため、set_page_config より前に utils を import する
import utils

st.set_page_config(page_title="APOLLO v9 | CAPCOM", page_icon=utils.module_icon("capcom"), layout="wide")
st.session_state['current_page'] = 'CAPCOM'

import capcom

# CAPCOM モジュール（AIエージェント）選択肢: 表示ラベル → 内部キー
# ツール選択 UI と context.json 構築の両方で使う単一ソース
CAPCOM_TOOL_OPTIONS = {
    "Claude Code（Anthropic）": "claude_code",
    "Codex CLI（OpenAI）": "codex",
    "Antigravity IDE（Google）": "antigravity",
}

# レポート生成の進め方: 表示ラベル → 内部キー
# （context.json の report_mode・ZIP キャッシュキー・export_session_zip 引数で使う）
CAPCOM_REPORT_MODES = {
    "🤖 自律生成モード（従来・4フェーズ自動進行）": "autonomous",
    "💬 対話型レポート作成モード（KATHERINE）": "interactive",
}

utils.render_sidebar()

# ==================================================================
# --- ヘッダー ---
# ==================================================================
utils.module_header("capcom", "CAPCOM — Capsule Communicator")
st.markdown(
    "全モジュールの分析結果をセッションZIPにパッケージングし、AI エージェント（Claude Code 等）による"
    "本格レポート生成に橋渡しします。進め方は **自律生成モード（おまかせ）** と "
    "**対話型レポート作成モード（KATHERINE）** の2つから選べます。"
)

# ==================================================================
# --- セッションステータス ---
# ==================================================================

# In-Memory版の警告(セッションはブラウザ閉じると消失)
st.warning(
    "⚠️ **CAPCOMセッションはブラウザを閉じると失われます。**\n"
    "完了後は必ず下部の「ZIPダウンロード」ボタンでセッション一式を保存してください。\n"
    "ZIPをClaude Codeに渡すことでレポート生成が可能です。"
)

if capcom.is_active():
    session_id = capcom.get_session_id()
    _tel = capcom.get_telemetry()
    snap_n = _tel['snapshots']
    prompt_n = _tel['prompts']
    data_n = _tel['data']

    # --- ステータスパネル ---
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

    # ==================================================================
    # --- CAPCOM 専用 Mission Objective ---
    # VOYAGER を使わず CAPCOM のみ利用するユーザー向け。
    # CAPCOM 側で入力した値が CAPCOM Export 時に優先される。
    # 未入力なら VOYAGER 側 (`voyager_objective`) をフォールバックとして使用。
    # ==================================================================
    st.markdown("---")
    st.markdown("### 🎯 CAPCOM Mission Objective")
    st.markdown(
        "CAPCOM レポートの **問い (Mission Objective)** を設定してください。"
        "VOYAGER で既に入力されていればその値が初期表示されますが、"
        "**この欄で上書きすれば CAPCOM 専用の Mission Objective として優先利用されます**。"
    )

    _voyager_obj_fallback = st.session_state.get('voyager_objective', '')
    _capcom_obj_default = st.session_state.get('capcom_mission_objective', _voyager_obj_fallback)

    capcom_mission_objective = st.text_area(
        "CAPCOM レポートの問い (Mission Objective):",
        value=_capcom_obj_default,
        height=120,
        placeholder=(
            "例: 競合A社の直近3年の出願傾向から、彼らが注力している新規事業領域を特定し、"
            "自社の対抗策を提案してください。"
        ),
        key="capcom_mission_objective_input",
        help=(
            "VOYAGER でも同じ Mission Objective を入力済みなら自動でコピーされます。"
            "CAPCOM だけで使う場合はここで直接入力してください。"
        ),
    )
    # 入力値を session_state に保存 (VOYAGER 側とは独立)
    st.session_state['capcom_mission_objective'] = capcom_mission_objective

    if _voyager_obj_fallback and capcom_mission_objective != _voyager_obj_fallback:
        st.caption(
            f"ℹ️ VOYAGER 側の Mission Objective とは異なる内容で CAPCOM Export されます。"
        )

    # ==================================================================
    # --- 母集団メタ情報（全項目任意） ---
    # 設計意図 / 論理式 / 収録年情報 / 特許データベース名 を CAPCOM に渡し、
    # 分析レポート本文とレポート付録に反映させる。
    # ==================================================================
    st.markdown("---")
    st.markdown("### 🗂️ 母集団メタ情報（任意・全項目任意）")
    st.markdown(
        "分析対象の特許母集団について任意情報を記録します。入力内容は"
        "CAPCOM に送信され、**分析の前提条件としてレポートに反映**されます。"
        "論理式は付録に、設計意図・収録年情報・データベース名は分析本文と付録の両方に"
        "適切な形で組み込まれます。"
    )

    capcom_query_intent = st.text_area(
        "🎯 母集団論理式の設計意図（任意）",
        value=st.session_state.get('capcom_query_intent', ''),
        height=100,
        placeholder=(
            "例: 本母集団はCNF（セルロースナノファイバー）関連技術のうち、"
            "食品・化粧品用途を除外し、構造材料・複合材料用途に焦点を絞って抽出した。"
            "IPC C08L, C08K を主軸に、B29, B32 のコーティング関連を補助的に含めている。"
        ),
        key="capcom_query_intent_input",
        help="分析レポートの冒頭・前提条件欄・付録に反映されます。",
    )
    st.session_state['capcom_query_intent'] = capcom_query_intent

    capcom_query_logic = st.text_area(
        "🔎 母集団論理式（任意）",
        value=st.session_state.get('capcom_query_logic', ''),
        height=100,
        placeholder=(
            "例: (TI=(CNF OR セルロースナノファイバー OR nanocellulose) "
            "AND IPC=(C08L* OR C08K* OR B29*)) "
            "AND PD=(20150101:20260131)"
        ),
        key="capcom_query_logic_input",
        help="**レポート付録に全文掲載**されます。機密情報は含めないでください。",
    )
    st.session_state['capcom_query_logic'] = capcom_query_logic

    col_cov, col_db = st.columns(2)
    with col_cov:
        capcom_coverage_years = st.text_input(
            "📅 収録年情報（任意）",
            value=st.session_state.get('capcom_coverage_years', ''),
            placeholder="例: 2015-01-01〜2026-01-31（出願日ベース）",
            key="capcom_coverage_years_input",
            help="分析の時系列解釈に反映されます。",
        )
        st.session_state['capcom_coverage_years'] = capcom_coverage_years
    with col_db:
        capcom_database_name = st.text_input(
            "🗄️ 使用した特許データベース名（任意）",
            value=st.session_state.get('capcom_database_name', ''),
            placeholder="例: 社内特許DB / Derwent Innovation / PatSnap",
            key="capcom_database_name_input",
            help="付録および分析注記（カバレッジ制約）の記述に反映されます。",
        )
        st.session_state['capcom_database_name'] = capcom_database_name

    # --- 画像・スライドに関する指示（任意） ---
    capcom_image_directive = st.text_area(
        "🖼️ 画像・スライドに関する指示（任意）",
        value=st.session_state.get('capcom_image_directive', ''),
        placeholder="例: 表紙にクラスタ動態マップを使う / 権利化率マップはスライド必須 / CREWは媒介中心性の図を優先 など。",
        key="capcom_image_directive_input",
        height=90,
        help="どの画像をどこで使うか等の指示をAIに渡せます。context.json 経由でレポート/スライド生成に反映されます。レポート本文の図キャプションはAIが執筆するため要点記入は不要、スライドの要点はAIが本指示に沿って作成します。",
    )
    st.session_state['capcom_image_directive'] = capcom_image_directive
    # 収集済みスナップショット一覧（指示でタイトルを参照できるよう提示）
    _snaps_for_list = st.session_state.get('snapshots', [])
    if _snaps_for_list:
        with st.expander(f"🖼️ 収集済みスナップショット {len(_snaps_for_list)} 枚（上の指示で名前を参照できます）"):
            for _s in _snaps_for_list:
                st.caption(f"・{_s.get('module', '?')}: {_s.get('title', '(無題)')}")

    # ==================================================================
    # --- レポート生成の進め方（自律生成 / 対話型 KATHERINE） ---
    # 選択は context.json の report_mode と ZIP キャッシュキーに反映される。
    # ZIP の同梱内容はモードで変わらない（対話型スキーマは常時同梱）。
    # ==================================================================
    st.markdown("---")
    st.markdown("### 🧭 レポート生成の進め方")
    _mode_labels = list(CAPCOM_REPORT_MODES.keys())
    _mode_current = st.session_state.get('capcom_report_mode', 'autonomous')
    selected_mode_label = st.radio(
        "レポート生成の進め方",
        options=_mode_labels,
        index=1 if _mode_current == 'interactive' else 0,
        key="capcom_report_mode_input",
        captions=[
            "AI が4フェーズ（ミッション理解 → クロス分析 → Deep Dive → 統合・品質検証）を自動進行。要所の確認のみ対話。",
            "AI が案を詳しい根拠・ロジック付きで提示し、分析者が選ぶ・直す・確定しながら作る。品質ゲートは自律生成と同一。",
        ],
        label_visibility="collapsed",
    )
    report_mode = CAPCOM_REPORT_MODES[selected_mode_label]
    st.session_state['capcom_report_mode'] = report_mode
    if report_mode == 'interactive':
        st.caption(
            "💬 対話型（KATHERINE）では、母集団タイプの判定・クロス分析の仮説と検証・結論の確定などの"
            "判断ポイントで AI が「提案・根拠・別の見方」を提示し、分析者が確定しながら進みます。"
            "各ポイントで「おまかせ」も選べます。所要時間は自律生成より長くなります（複数セッション推奨）。"
        )

    # ==================================================================
    # --- CAPCOM モジュール選択（複数選択可） ---
    # 選択されたツールに対応する capcom_schema_patches/ 配下の資材が
    # ZIP 出力時に同梱される。
    # ==================================================================
    st.markdown("---")
    st.markdown("### 🤝 CAPCOM モジュール選択（複数選択可）")
    st.markdown(
        "レポート生成に使用する AI エージェントを選択してください。"
        "**ZIP に各エージェント用のパッチが自動同梱**され、手動での apply_patch.sh 実行は不要です。"
    )

    default_tools = st.session_state.get('capcom_tools_selected', ["Claude Code（Anthropic）"])
    selected_tool_labels = st.multiselect(
        "使用する CAPCOM モジュール",
        options=list(CAPCOM_TOOL_OPTIONS.keys()),
        default=default_tools,
        key="capcom_tools_selected_input",
        help=(
            "Claude Code は `capcom_schema/` 一式で動作（追加資材なし）。"
            "Codex / Antigravity を選択した場合、対応するオーバーレイ資材が ZIP 直下に展開済みで同梱されます。"
        ),
    )
    st.session_state['capcom_tools_selected'] = selected_tool_labels
    selected_tool_keys = [CAPCOM_TOOL_OPTIONS[lbl] for lbl in selected_tool_labels]

    if not selected_tool_keys:
        st.warning("⚠️ 最低1つの CAPCOM モジュールを選択してください（デフォルト: Claude Code）。")
        # 最低限 Claude Code は動作するので、後段はフォールバック
        selected_tool_keys = ["claude_code"]

    if report_mode == 'interactive' and any(k != 'claude_code' for k in selected_tool_keys):
        st.info(
            "ℹ️ 対話型レポート作成モード（KATHERINE）は **Claude Code で検証済み**です。"
            "Codex CLI / Antigravity IDE 向けの対話型対応（Codex 用補遺・Antigravity 用対話 Artifact 雛形）"
            "も ZIP に同梱されますが、両ツールでの実機検証は未了です。"
        )

    with st.expander("ℹ️ 選択ツールごとの起動方法"):
        if "claude_code" in selected_tool_keys:
            if report_mode == 'interactive':
                st.markdown(
                    "**Claude Code（対話型 KATHERINE）**: ZIPを展開 → `claude` 起動 → "
                    "「`capcom_schema/interactive/SKILL_INTERACTIVE.md` を読んで対話型でレポートを作りましょう」"
                )
            else:
                st.markdown(
                    "**Claude Code**: ZIPを展開 → `claude` 起動 → "
                    "「`capcom_schema/SKILL.md` を読んでレポートを書いて」"
                )
        if "codex" in selected_tool_keys:
            _codex_note = (
                "（対話型: `report_mode` を自動判別。詳細は同梱の `interactive_codex_addendum.md`・"
                "実機検証待ち）" if report_mode == 'interactive' else ""
            )
            st.markdown(
                "**Codex CLI**: ZIPを展開 → `codex` 起動 → "
                f"チャットで `$apollo-capcom` または `/skills` から選択{_codex_note}"
            )
        if "antigravity" in selected_tool_keys:
            _ag_note = (
                "（対話型: Review Policy=Request Review 必須・対話用 Artifact 雛形同梱・実機検証待ち）"
                if report_mode == 'interactive' else ""
            )
            st.markdown(
                "**Antigravity IDE**: ZIPを展開 → Antigravity IDE でフォルダを開く → "
                "Review Policy を「Request Review」に設定 → "
                f"チャットで「apollo-capcom スキルでレポート生成」{_ag_note}"
            )

    # ==================================================================
    # --- CAPCOM Export（voyager/にMission情報を書き出す） ---
    # ==================================================================
    st.markdown("---")
    st.markdown("### 📡 CAPCOM Export")
    st.markdown("Mission ObjectiveとスナップショットをClaude Code向けにJSON出力します。")

    snapshots = st.session_state.get('snapshots', [])
    # CAPCOM 専用 Mission Objective を優先、未入力なら VOYAGER 側をフォールバック
    mission_objective = (
        st.session_state.get('capcom_mission_objective', '').strip()
        or st.session_state.get('voyager_objective', '').strip()
    )

    export_ready = len(snapshots) > 0 and len(mission_objective) > 5
    if not export_ready:
        if len(snapshots) == 0:
            st.warning("⚠️ スナップショットが収集されていません。各分析モジュールでスナップショットを保存してください。")
        elif len(mission_objective) <= 5:
            st.warning("⚠️ Mission Objective を入力してください(上記の入力欄、または VOYAGER ページ)。最低6文字以上必要です。")

    if st.button("📡 CAPCOM Export 実行", type="primary", key="capcom_export_btn",
                 disabled=not export_ready):
        try:
            import pandas as pd

            def _get_period_str():
                df = st.session_state.get('df_main')
                if df is None or 'year' not in df.columns:
                    return "不明"
                try:
                    years = df['year'].dropna().astype(int)
                    return f"{years.min()}-{years.max()}" if len(years) > 0 else "不明"
                except Exception:
                    return "不明"

            def _describe_data_file(filename):
                desc_map = {
                    "patents.csv": "全特許データ（タイトル・要約・出願人・クラスタ情報等）",
                    "atlas_statistics.json": "ATLAS マクロ統計（時系列・ランキング・HHI/Entropy/Gini）",
                    "atlas_grant_rate.json": "ATLAS 権利化率マップ（出願数×権利化率の象限分析）",
                    "core_classification.json": "CORE ルールベース分類結果",
                    "saturnv_clusters.json": "Saturn V AIクラスタリング結果（ノイズ分析・動態マップ含む）",
                    "saturnv_drilldown.json": "Saturn V ドリルダウン分析（PROBE）",
                    "mega_momentum.json": "MEGA 動態分析（CAGR×活動量4象限）",
                    "mega_drilldown.json": "MEGA ポートフォリオ詳細",
                    "explorer_global_network.json": "Explorer グローバル共起ネットワーク",
                    "explorer_trend.json": "Explorer トレンドネットワーク",
                    "explorer_dominance.json": "Explorer ドミナンスネットワーク",
                    "eagle_clusters.json": "EAGLE 探索的クラスタリング結果",
                    "eagle_cluster_dynamics.json": "EAGLE クラスタ動態マップ",
                    "nebula_hype_cycle.json": "NEBULA ハイプサイクル分析",
                    "nebula_macro_events.json": "NEBULA マクロイベント分析",
                    "nebula_academic_clusters.json": "NEBULA 学術ランドスケープ",
                    "crew_network.json": "CREW ネットワーク分析（媒介中心性・コミュニティ）",
                }
                return desc_map.get(filename, filename)

            # 各Evidenceを個別ファイルで保存。
            # Export は現在の snapshots から evidence を作り直すので、まず古い evidence を一掃する
            # （スナップショットを削除/追加して再 Export すると順序・id が変わり、ev{N}_{module}.json が
            #  上書きされず累積→id 重複・mission.json と evidence/ の件数不整合になるのを防ぐ）。
            capcom.clear_voyager_evidence()
            evidence_list = []
            for i, snap in enumerate(snapshots):
                ev_id = i + 1
                module_name = snap.get('module', 'Unknown').lower().replace(' ', '_')

                # 画像は撮影時に render_snapshot_button が「本来名」（snap['id']＝
                # atlas_* / saturn_* / core_* など）で snapshots/ に保存済み。ここで
                # voyager_ev* として再保存すると同一画像が二重に ZIP へ入り、レポートの
                # マップ重複の原因になる。そのため再保存はせず、evidence の画像参照を
                # 撮影済みの本来名スナップショットへ向ける（snapshot_logical_path）。
                image_paths = []
                try:
                    imgs = snap.get('images') or ([snap['image']] if snap.get('image') else [])
                    snap_key = snap.get('id')
                    for j in range(len(imgs)):
                        suffix = j if len(imgs) > 1 else None
                        p = capcom.snapshot_logical_path(snap_key, index=suffix)
                        if p:
                            image_paths.append(p)
                except Exception as e:
                    st.caption(f"⚠️ WARN Evidence画像パスの解決に失敗しました（要確認）: {e}")

                ev_filename = f"ev{ev_id}_{module_name}"
                ev_data = {
                    "id": ev_id,
                    "module": snap.get('module', 'Unknown'),
                    "title": snap.get('title', ''),
                    "description": snap.get('description', ''),
                    "images": image_paths,
                    "data_summary": snap.get('data_summary', {})
                }
                capcom.save_voyager_evidence(ev_filename, ev_data)

                evidence_list.append({
                    "id": ev_id,
                    "module": snap.get('module', 'Unknown'),
                    "title": snap.get('title', ''),
                    "file": f"evidence/{ev_filename}.json",
                    "images": image_paths
                })

            # mission.json
            mission = {
                "mission_objective": mission_objective,
                "timestamp": pd.Timestamp.now().isoformat(),
                "evidence_count": len(evidence_list),
                "evidence_list": evidence_list,
                "available_data_files": capcom.list_data_files()
            }
            capcom.save_voyager_mission(mission)

            # context.json
            df_main = st.session_state.get('df_main')
            col_map = st.session_state.get('col_map', {})
            feature_names = st.session_state.get('feature_names', [])
            stopwords_set = st.session_state.get('stopwords', set())
            modules_used = sorted(set(snap.get('module', 'Unknown') for snap in snapshots))
            data_files_desc = {fname: _describe_data_file(fname) for fname in capcom.list_data_files()}

            context = {
                "session_id": st.session_state.get('capcom_session_id', ''),
                "dataset": {
                    "total_patents": len(df_main) if df_main is not None else 0,
                    "period": _get_period_str(),
                    "column_mapping": col_map,
                    "preprocessing": "patiroha + SBERT(paraphrase-multilingual-MiniLM-L12-v2)",
                    "tfidf_vocab_size": len(feature_names),
                    "stopwords_count": len(stopwords_set)
                },
                "modules_used": modules_used,
                "available_data_files": data_files_desc,
                # --- 母集団メタ情報（全項目任意） ---
                # レポート本文・付録の記述に反映される。
                # 空文字の場合は「未指定」扱いで SKILL.md のフォールバック記述が使われる。
                "population_meta": {
                    "query_intent": st.session_state.get('capcom_query_intent', '').strip(),
                    "query_logic": st.session_state.get('capcom_query_logic', '').strip(),
                    "coverage_years": st.session_state.get('capcom_coverage_years', '').strip(),
                    "database_name": st.session_state.get('capcom_database_name', '').strip(),
                },
                # --- 画像・スライドの指示（任意・レポート/スライド生成で考慮） ---
                "report_directives": {
                    "image_slide_instruction": st.session_state.get('capcom_image_directive', '').strip(),
                },
                # --- CAPCOM モジュール選択 ---
                "capcom_tools": {
                    "selected": st.session_state.get('capcom_tools_selected', ["Claude Code（Anthropic）"]),
                    "selected_keys": [
                        CAPCOM_TOOL_OPTIONS[lbl]
                        for lbl in st.session_state.get('capcom_tools_selected', ["Claude Code（Anthropic）"])
                        if lbl in CAPCOM_TOOL_OPTIONS
                    ],
                },
                # --- レポート生成の進め方（autonomous | interactive） ---
                # AI セッションはこの値でモードを自動判別する（正本は ZIP 構築時に
                # capcom.export_session_zip が最新の UI 選択で上書きした値）。
                "report_mode": st.session_state.get('capcom_report_mode', 'autonomous'),
            }
            capcom.save_voyager_context(context)

            st.success("📡 CAPCOM Export 完了: `voyager/mission.json` + `voyager/evidence/` + `voyager/context.json`")
        except Exception as e:
            st.error(f"CAPCOM Export エラー: {e}")

    # ==================================================================
    # --- ZIPダウンロード ---
    # ==================================================================
    st.markdown("---")
    st.markdown("### 📦 セッションZIPダウンロード")
    st.markdown("""
    CAPCOM Export 実行後、セッションフォルダ一式を ZIP にまとめてダウンロードします。
    展開して、選択した CAPCOM モジュール（Claude Code / Codex CLI / Antigravity IDE）で開いてください。
    """)

    _instruction_line = (
        "4. 「capcom_schema/interactive/SKILL_INTERACTIVE.md を読んで対話型でレポートを作りましょう」と指示"
        if report_mode == 'interactive'
        else "4. 「capcom_schema/SKILL.md を読んでレポートを書いて」と指示"
    )
    st.markdown(f"""
    ```
    共通手順:
    1. 下のボタンでZIPをダウンロード
    2. ZIPを任意の場所に展開
    3. 選択したツールでそのフォルダを開く
    {_instruction_line}
       （Codex/Antigravity では ZIP 直下の AGENTS.md / GEMINI.md が優先される）
    ```
    """)

    # 選択ツール分のパッチ資材を ZIP に同梱する
    # ZIP をキャッシュ（セッションID＋ストア更新カウンタ＋telemetry＋ツール選択＋レポート生成モードをキー
    # に、ページ再描画のたびのフル構築を回避）
    # - revision（capcom.get_store_revision）: save_*/clear_* のたびに +1。件数が同じままの内容更新
    #   （patents.csv の更新・スナップショット差し替え・Export のやり直し等）でも新 ZIP を構築させる
    # - session_id: 新セッションが旧セッションと同カウントでも旧 ZIP を返さない
    # - report_mode / ツール選択: 切替後も旧 context.json・旧パッチ構成の ZIP が返るのを防ぐ
    # voyager の Export 状態もキャッシュキーに含める（export_session_zip の書き出し条件と一致させる）。
    _voy_store = st.session_state.get('capcom_store', {}).get('voyager', {})
    _voy_gen = (bool(_voy_store.get('mission')), bool(_voy_store.get('context')), len(_voy_store.get('evidence', {})))
    _zip_cache_key = (
        capcom.get_session_id(),
        capcom.get_store_revision(),
        str(capcom.get_telemetry()),
        _voy_gen,
        tuple(sorted(selected_tool_keys)),
        report_mode,
    )
    if st.session_state.get('_capcom_zip_key') != _zip_cache_key:
        # フォールバック時（ツール未選択）もラベルとキーの対応を保つため、キーから逆引きする
        _labels_for_zip = [lbl for lbl, k in CAPCOM_TOOL_OPTIONS.items() if k in selected_tool_keys]
        zip_bytes, zip_filename = capcom.export_session_zip(
            selected_tools=selected_tool_keys,
            report_mode=report_mode,
            selected_tool_labels=_labels_for_zip,
        )
        st.session_state['_capcom_zip_key'] = _zip_cache_key
        st.session_state['_capcom_zip_cache'] = (zip_bytes, zip_filename)
    else:
        zip_bytes, zip_filename = st.session_state['_capcom_zip_cache']
    if zip_bytes:
        file_size_mb = len(zip_bytes) / (1024 * 1024)
        st.download_button(
            f"📦 ZIPダウンロード ({file_size_mb:.1f} MB)",
            data=zip_bytes,
            file_name=zip_filename,
            mime="application/zip",
            type="primary",
            key="capcom_page_zip_download",
        )
        st.caption(
            f"同梱ツール: {', '.join(selected_tool_labels) if selected_tool_labels else 'Claude Code（フォールバック）'}"
        )
    else:
        st.warning("セッションフォルダが見つかりません。分析モジュールを実行してデータを蓄積してください。")

    # ==================================================================
    # --- セッション内ファイル一覧 (session_state 上の In-Memory store) ---
    # ==================================================================
    st.markdown("---")
    st.markdown("### 📁 セッション内ファイル一覧 (In-Memory)")

    _store = st.session_state.get('capcom_store', {})

    # data/
    data_files = sorted(_store.get('data', {}).keys())
    if data_files:
        with st.expander(f"📂 data/ ({len(data_files)} ファイル)", expanded=True):
            for fname in data_files:
                content = _store['data'][fname]
                if isinstance(content, bytes):
                    size_kb = len(content) / 1024
                else:
                    # dict はおおよそのサイズ
                    size_kb = len(json.dumps(content, ensure_ascii=False, default=str).encode('utf-8')) / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                st.caption(f"`data/{fname}` — {size_str}")

    # snapshots/
    snap_keys = sorted(_store.get('snapshots', {}).keys())
    if snap_keys:
        with st.expander(f"📂 snapshots/ ({len(snap_keys)} ファイル)"):
            for skey in snap_keys:
                size_kb = len(_store['snapshots'][skey]['image']) / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                st.caption(f"`snapshots/{skey}.png` — {size_str}")

    # prompts/
    prompt_keys = sorted(_store.get('prompts', {}).keys())
    if prompt_keys:
        with st.expander(f"📂 prompts/ ({len(prompt_keys)} ファイル)"):
            for pkey in prompt_keys:
                size_kb = len(_store['prompts'][pkey].encode('utf-8')) / 1024
                st.caption(f"`prompts/{pkey}` — {size_kb:.1f} KB")

    # voyager/
    _voy = _store.get('voyager', {})
    voy_count = (1 if _voy.get('mission') else 0) + (1 if _voy.get('context') else 0) + len(_voy.get('evidence', {}))
    if voy_count > 0:
        with st.expander(f"📂 voyager/ ({voy_count} ファイル)"):
            if _voy.get('mission'):
                st.caption("`voyager/mission.json`")
            if _voy.get('context'):
                st.caption("`voyager/context.json`")
            for ev_fname in sorted(_voy.get('evidence', {}).keys()):
                st.caption(f"`voyager/evidence/{ev_fname}`")

    # capcom_schema (リポジトリから ZIP に同梱される、参考表示)
    _tool_assets_note = []
    if "codex" in selected_tool_keys:
        _tool_assets_note.append("Codex用 (`AGENTS.md` + `.codex/skills/apollo-capcom/`)")
    if "antigravity" in selected_tool_keys:
        _tool_assets_note.append("Antigravity用 (`GEMINI.md` + `AGENTS.md` + `.agent/`)")
    _tool_assets_str = "、加えて " + " と ".join(_tool_assets_note) if _tool_assets_note else ""
    st.caption(
        "ℹ️ ZIP には `capcom_schema/`、`CLAUDE.md`、`.claude/skills/` も同梱されます "
        f"(リポジトリ資産){_tool_assets_str}"
    )

    # ==================================================================
    # --- ツァーリ・ボンバ対策ガイド ---
    # ==================================================================
    st.markdown("---")
    with st.expander("💡 Claude Code でのトークン節約ガイド（ツァーリ・ボンバ対策）"):
        st.markdown("""
        CAPCOMレポート生成時のClaude Codeトークン消費を最小化するためのガイドです。

        #### 問題の本質
        Claude Codeはメッセージを送るたびにコンテキスト全体をAPIに再送信します。
        SKILL.md + スキーマ + exemplar + 会話履歴が毎回「再印刷」されるため、
        **トークン消費の90%がキャッシュ読み取り**になりえます。

        #### APOLLO で適用済みの対策
        1. **4フェーズ構成**: 旧6フェーズから統合し、API呼び出し回数を削減
        2. **サブエージェント禁止**: SKILL.mdで明示。フォーク（コンテキストコピー）を防止
        3. **バッチ処理**: Deep Diveは1回のやり取りで複数モジュールを処理
        4. **ファイル読み込み最小化**: 一度読んだ内容は再読み込みしない

        #### 推奨ワークフロー
        ```
        Claude Code でZIP展開フォルダを開く
        → 「capcom_schema/SKILL.md を読んでレポートを書いて」
        → 4フェーズで自動進行（Phase A → B → C → D）
        ```

        #### 対話型レポート作成モード（KATHERINE）の場合
        ```
        Claude Code でZIP展開フォルダを開く
        → 「capcom_schema/interactive/SKILL_INTERACTIVE.md を読んで対話型でレポートを作りましょう」
        → 判断ポイントごとに AI が提案+根拠を提示 → 分析者が確定しながら進行
        ```
        対話型は往復が増えるためコンテキスト消費が大きくなります。
        **1スレッド=1フェーズの分割が標準**です（中断・再開は引き継ぎ日誌で安全に行えます）。
        """)

else:
    # --- セッション未開始 ---
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
CAPCOMセッションが開始されていません。<br/>
Mission Control（Home）で分析エンジンを起動し、CAPCOMセッションを開始してください。
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
    ### CAPCOMとは？

    **CAPCOM** (Capsule Communicator) は APOLLO と AI レポート執筆エージェント
    （Claude Code / Codex CLI / Antigravity IDE）を繋ぐ通信モジュールです。

    ```
    APOLLO（分析・可視化）
        ↓ CAPCOM がデータを構造化してエクスポート
    ZIP ダウンロード（選択ツール用パッチが同梱済み）
        ↓ ユーザーが展開
    Claude Code / Codex CLI / Antigravity IDE（レポート執筆）
        ↓ 自律生成（4フェーズ自動進行）または対話型（KATHERINE）
    PDF 完成 🎉
    ```

    #### ワークフロー
    1. **Mission Control** で CAPCOM セッションを開始
    2. 各分析モジュール（ATLAS, Saturn V, MEGA, ...）を実行 → データが自動蓄積
    3. **VOYAGER** で Mission Objective 設定
    4. **このページ** で母集団メタ情報（任意）+ レポート生成の進め方（自律生成 / 対話型 KATHERINE）+ 使用ツール（複数可）を選択 → CAPCOM Export → ZIP ダウンロード
    5. ZIP を展開して選択ツールで開く → レポート生成
    """)
