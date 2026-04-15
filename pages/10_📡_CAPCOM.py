# ==================================================================
# 10_📡_CAPCOM.py — APOLLO v7.0.0 CAPCOM (Capsule Communicator)
# セッション管理・ZIPエクスポート・Claude Code連携
# ==================================================================

import streamlit as st
import os
import json
import datetime

st.set_page_config(page_title="APOLLO v7 | CAPCOM", page_icon="📡", layout="wide")
st.session_state['current_page'] = 'CAPCOM'

import utils
import capcom

utils.render_sidebar()

# ==================================================================
# --- ヘッダー ---
# ==================================================================
st.title("📡 CAPCOM — Capsule Communicator")
st.markdown("全モジュールの分析結果をセッションZIPにパッケージングし、Claude Codeによる本格レポート生成に橋渡しします。")

# ==================================================================
# --- セッションステータス ---
# ==================================================================

# セッション揮発に関する警告(ブラウザを閉じると消失)
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
                    "core_classification.json": "CORE ルールベース分類結果",
                    "saturnv_clusters.json": "Saturn V AIクラスタリング結果（ノイズ分析・動態マップ含む）",
                    "saturnv_drilldown.json": "Saturn V ドリルダウン分析（PROBE）",
                    "mega_momentum.json": "MEGA 動態分析（CAGR×活動量4象限）",
                    "mega_drilldown.json": "MEGA ポートフォリオ詳細",
                    "explorer_global_network.json": "Explorer グローバル共起ネットワーク",
                    "explorer_trend_network.json": "Explorer トレンドネットワーク",
                    "explorer_dominance_network.json": "Explorer ドミナンスネットワーク",
                    "eagle_clusters.json": "EAGLE 探索的クラスタリング結果",
                    "eagle_cluster_dynamics.json": "EAGLE クラスタ動態マップ",
                    "nebula_hype_cycle.json": "NEBULA ハイプサイクル分析",
                    "nebula_macro_events.json": "NEBULA マクロイベント分析",
                    "nebula_academic_clusters.json": "NEBULA 学術ランドスケープ",
                }
                return desc_map.get(filename, filename)

            # 各Evidenceを個別ファイルで保存
            evidence_list = []
            for i, snap in enumerate(snapshots):
                ev_id = i + 1
                module_name = snap.get('module', 'Unknown').lower().replace(' ', '_')

                image_paths = []
                try:
                    if snap.get('image'):
                        img_path = capcom.save_snapshot_image(f"voyager_ev{ev_id}", snap['image'], index=0)
                        if img_path:
                            image_paths.append(img_path)
                    elif snap.get('images'):
                        for j, img in enumerate(snap['images']):
                            img_path = capcom.save_snapshot_image(f"voyager_ev{ev_id}", img, index=j)
                            if img_path:
                                image_paths.append(img_path)
                    capcom.increment_snap_count()
                except Exception:
                    pass

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
                "available_data_files": data_files_desc
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
    CAPCOM Export実行後、セッションフォルダ一式をZIPにまとめてダウンロードします。
    展開してClaude Codeで開いてください。
    """)

    st.markdown("""
    ```
    使い方:
    1. 下のボタンでZIPをダウンロード
    2. ZIPを任意の場所に展開
    3. Claude Code でそのフォルダを開く
    4. 「capcom_schema/SKILL.md を読んでレポートを書いて」と指示
    ```
    """)

    zip_bytes, zip_filename = capcom.export_session_zip()
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
    else:
        st.warning("セッションフォルダが見つかりません。分析モジュールを実行してデータを蓄積してください。")

    # ==================================================================
    # --- セッション内ファイル一覧 ---
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
    st.caption("ℹ️ ZIP には `capcom_schema/`、`CLAUDE.md`、`.claude/skills/` も同梱されます (リポジトリ資産)")

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

        #### 対策（APOLLO v7で適用済み）
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

    **CAPCOM** (Capsule Communicator) は APOLLO と Claude Code を繋ぐ通信モジュールです。

    ```
    APOLLO（分析・可視化）
        ↓ CAPCOM がデータを構造化してエクスポート
    ZIP ダウンロード
        ↓ ユーザーが展開
    Claude Code（レポート執筆）
        ↓ 4フェーズで自動進行
    PDF 完成 🎉
    ```

    #### ワークフロー
    1. **Mission Control** でCAPCOMセッションを開始
    2. 各分析モジュール（ATLAS, Saturn V, MEGA, ...）を実行 → データが自動蓄積
    3. **VOYAGER** でMission Objective設定 + CAPCOM Export
    4. **このページ** でZIPダウンロード
    5. ZIPを展開してClaude Codeで開く → レポート生成
    """)
