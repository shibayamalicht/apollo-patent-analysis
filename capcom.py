# ==================================================================
# capcom.py — APOLLO CAPCOM (Capsule Communicator)
# In-Memory 統一版 — Hugging Face Spaces / Streamlit Cloud 対応
#
# 全データを st.session_state['capcom_store'] に保持し、
# ZIP ダウンロード時のみメモリ上で動的に組み立てる。
# ファイルシステムへの書き込みは行わない (ephemeral環境対応)。
#
# export_session_zip() は selected_tools 引数で複数 AI エージェントを選択可能。
# 選択された CAPCOM モジュール用の資材 (capcom_schema_patches/{codex,antigravity}/)
# を ZIP 直下に展開済みで同梱する。
# ==================================================================

import os
import io
import json
import uuid
import zipfile
import datetime
import streamlit as st


# ==================================================================
# --- セッション管理 ---
# ==================================================================

def _init_store(session_id):
    """session_state['capcom_store'] の初期構造を生成する"""
    return {
        'session_id': session_id,
        'created_at': datetime.datetime.now().isoformat(),
        'snapshots': {},   # snap_id (filename without ext) -> {'image': bytes, 'index': int|None}
        'data': {},        # filename (with ext) -> dict (JSON) or bytes (CSV)
        'prompts': {},     # filename (with .md) -> str
        'voyager': {       # voyager/ subdirectory
            'mission': None,    # dict or None
            'context': None,    # dict or None
            'evidence': {},     # filename (with .json) -> dict
        },
        'metadata': {
            'session_id': session_id,
            'created_at': datetime.datetime.now().isoformat(),
            'snapshots': [],
        },
        'telemetry': {'snapshots': 0, 'prompts': 0, 'data': 0},
    }


def init_session():
    """
    In-Memory CAPCOM セッションを初期化する。

    セッション ID は秒精度タイムスタンプ + UUID 短縮版で構成し、
    マルチユーザー(複数ブラウザ)同時アクセスでも衝突しないようにする。

    Returns:
        tuple: (session_id, session_id) — 後方互換のため2要素タプル
               (旧 API は (session_id, session_path) を返していた)
    """
    now = datetime.datetime.now()
    session_id = f"session_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    st.session_state['capcom_session_id'] = session_id
    st.session_state['capcom_store'] = _init_store(session_id)

    # patents.csv の初期出力 (df_main があれば)
    save_patents_csv()

    # 後方互換: 旧 API は (session_id, session_path) を返していた
    return session_id, session_id


def is_active():
    """CAPCOM セッションがアクティブかどうかを判定"""
    return 'capcom_store' in st.session_state and st.session_state['capcom_store'] is not None


def get_session_id():
    """現在のセッション ID を取得。非アクティブなら None"""
    if is_active():
        return st.session_state['capcom_store']['session_id']
    return None


def _get_store():
    """内部ヘルパ: capcom_store を取得。非アクティブなら None"""
    return st.session_state.get('capcom_store') if is_active() else None


# ==================================================================
# --- スナップショット出力 (画像) ---
# ==================================================================

def save_snapshot_image(snap_id, image_bytes, index=None):
    """
    スナップショット画像 (PNG bytes) を session_state に格納する。

    Args:
        snap_id: スナップショット ID (ファイル名のベース、拡張子不要)
        image_bytes: PNG 画像のバイトデータ
        index: 連番 (複数画像の場合)

    Returns:
        str: ZIP 内の論理パス (例: 'snapshots/voyager_ev1_0.png') または None
    """
    store = _get_store()
    if store is None or image_bytes is None:
        return None

    safe_id = _sanitize_filename(snap_id)
    if index is not None:
        key = f"{safe_id}_{index}"
    else:
        key = safe_id

    store['snapshots'][key] = {
        'image': image_bytes,
        'index': index,
    }
    return os.path.join("snapshots", f"{key}.png")


def save_metadata(snapshots_list):
    """
    metadata 情報 (スナップショットリスト全体) を更新する。

    Args:
        snapshots_list: スナップショットのメタデータリスト
    """
    store = _get_store()
    if store is None:
        return

    serializable_snapshots = []
    for snap in snapshots_list:
        s = {}
        for k, v in snap.items():
            # 画像バイトデータや figオブジェクトは除外
            if k in ('image', 'images', 'fig'):
                continue
            try:
                json.dumps(v)
                s[k] = v
            except (TypeError, ValueError):
                s[k] = str(v)
        serializable_snapshots.append(s)

    store['metadata']['snapshots'] = serializable_snapshots
    store['metadata']['updated_at'] = datetime.datetime.now().isoformat()
    store['metadata']['snapshot_count'] = len(serializable_snapshots)


# ==================================================================
# --- プロンプト出力 ---
# ==================================================================

def save_prompt(filename, prompt_text):
    """
    プロンプトを Markdown として session_state に格納する。

    Args:
        filename: ファイル名 (拡張子なし or .md付き)
        prompt_text: プロンプトのテキスト内容
    """
    store = _get_store()
    if store is None or not prompt_text:
        return

    if not filename.endswith('.md'):
        filename = filename + '.md'

    safe_name = _sanitize_filename(filename)
    store['prompts'][safe_name] = prompt_text
    _increment_counter('prompt')


# ==================================================================
# --- 生データ出力 ---
# ==================================================================

def save_data(filename, data):
    """
    分析結果を構造化 JSON として session_state に格納する。

    Args:
        filename: ファイル名 (拡張子なし or .json付き)
        data: dict 型のデータ
    """
    store = _get_store()
    if store is None or data is None:
        return

    if not filename.endswith('.json'):
        filename = filename + '.json'

    safe_name = _sanitize_filename(filename)
    store['data'][safe_name] = data
    _increment_counter('data')


def get_data(filename):
    """
    保存されている分析データを読み出す (VOYAGER 等が利用)。

    Args:
        filename: ファイル名 (拡張子付き)

    Returns:
        dict or bytes or None
    """
    store = _get_store()
    if store is None:
        return None
    return store['data'].get(filename)


def list_data_files():
    """data に格納されている全ファイル名リストを返す (JSON/CSV両方)"""
    store = _get_store()
    if store is None:
        return []
    # JSON と CSV の両方を拡張子順でソート
    return sorted(store['data'].keys())


# ==================================================================
# --- VOYAGER 専用出力 ---
# ==================================================================

def save_voyager_mission(mission_data):
    """voyager/mission.json を保存"""
    store = _get_store()
    if store is None or mission_data is None:
        return
    store['voyager']['mission'] = mission_data


def save_voyager_evidence(filename, evidence_data):
    """voyager/evidence/{filename} を保存"""
    store = _get_store()
    if store is None or evidence_data is None:
        return
    safe_name = _sanitize_filename(filename)
    if not safe_name.endswith('.json'):
        safe_name += '.json'
    store['voyager']['evidence'][safe_name] = evidence_data


def save_voyager_context(context_data):
    """voyager/context.json を保存"""
    store = _get_store()
    if store is None or context_data is None:
        return
    store['voyager']['context'] = context_data


def get_voyager_mission():
    """voyager/mission.json を取得"""
    store = _get_store()
    return store['voyager']['mission'] if store else None


def get_voyager_context():
    """voyager/context.json を取得"""
    store = _get_store()
    return store['voyager']['context'] if store else None


# ==================================================================
# --- patents.csv 出力 ---
# ==================================================================

def save_patents_csv():
    """
    df_main を CSV bytes として session_state に格納する。
    init_session 時に基本版を出力し、Saturn V/EAGLE 実行後に随時更新される。
    クラスタ列は存在する場合のみ含まれる。
    """
    import pandas as pd

    store = _get_store()
    if store is None:
        return

    df = st.session_state.get('df_main')
    if df is None or df.empty:
        return

    col_map = st.session_state.get('col_map', {})

    # 基本カラム (col_map の value=実カラム名を使用)
    base_keys = ['title', 'abstract', 'app_num', 'pub_number']
    derived_cols = ['applicant_main', 'inventor_main', 'year', 'ipc_main_group']
    cluster_cols = ['cluster', 'cluster_label', 'umap_x', 'umap_y']

    export_cols = []

    for key in base_keys:
        actual_col = col_map.get(key)
        if actual_col and actual_col in df.columns:
            export_cols.append(actual_col)

    for c in derived_cols:
        if c in df.columns:
            export_cols.append(c)

    for c in cluster_cols:
        if c in df.columns:
            export_cols.append(c)

    # 重複除去 (順序維持)
    seen = set()
    unique_cols = []
    for c in export_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    df_export = df[unique_cols].copy()

    # EAGLE クラスタ列
    df_eagle = st.session_state.get('df_eagle')
    if df_eagle is not None and 'eagle_cluster' in df_eagle.columns:
        df_export['eagle_cluster'] = df_eagle['eagle_cluster'].values
        eagle_map = st.session_state.get('eagle_labels_map', {})
        if eagle_map:
            df_export['eagle_cluster_label'] = df_eagle['eagle_cluster'].map(
                lambda x: eagle_map.get(x, "") if x != -1 else ""
            )

    # Saturn V ドリルダウン列
    df_drill = st.session_state.get('df_drilldown_result')
    if df_drill is not None and 'drill_cluster' in df_drill.columns:
        for c in ['drill_cluster', 'drill_cluster_label']:
            if c in df_drill.columns:
                df_export[c] = pd.Series(dtype='object', index=df_export.index)
                common_idx = df_export.index.intersection(df_drill.index)
                df_export.loc[common_idx, c] = df_drill.loc[common_idx, c]

    # MEGA PULSE 4象限ラベル
    df_mega_pulse = st.session_state.get('df_momentum_result')
    mega_axis_col = st.session_state.get('mega_axis_col')
    if df_mega_pulse is not None and 'Group_Auto' in df_mega_pulse.columns and mega_axis_col:
        entity_to_group = df_mega_pulse['Group_Auto'].to_dict()

        def _map_mega_group(row):
            val = row.get(mega_axis_col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            if isinstance(val, list):
                groups = [entity_to_group.get(v) for v in val if v in entity_to_group]
                return groups[0] if groups else None
            return entity_to_group.get(val)
        df_export['mega_pulse_group'] = df.apply(_map_mega_group, axis=1)

    # MEGA TELESCOPE ドリルダウン
    df_mega_drill = st.session_state.get('df_drilldown')
    if df_mega_drill is not None and 'cluster_id' in df_mega_drill.columns:
        df_export['mega_drill_cluster'] = pd.Series(dtype='object', index=df_export.index)
        df_export['mega_drill_label'] = pd.Series(dtype='object', index=df_export.index)
        common_idx = df_export.index.intersection(df_mega_drill.index)
        df_export.loc[common_idx, 'mega_drill_cluster'] = df_mega_drill.loc[common_idx, 'cluster_id']
        if 'label' in df_mega_drill.columns:
            df_export.loc[common_idx, 'mega_drill_label'] = df_mega_drill.loc[common_idx, 'label']

    # CORE 分類列
    df_core = st.session_state.get('core_df_classified')
    if df_core is not None:
        core_rules = st.session_state.get('core_classification_rules', {})
        for axis_name in core_rules.keys():
            if axis_name in df_core.columns:
                col_name = f'core_{axis_name}'
                df_export[col_name] = pd.Series(dtype='object', index=df_export.index)
                common_idx = df_export.index.intersection(df_core.index)
                df_export.loc[common_idx, col_name] = df_core.loc[common_idx, axis_name]

    # CSV bytes 化して store に格納
    csv_bytes = df_export.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    store['data']['patents.csv'] = csv_bytes


# ==================================================================
# --- テレメトリ ---
# ==================================================================

def increment_snap_count():
    """スナップショット操作カウンターを +1 する (呼び出し側が制御)"""
    _increment_counter('snap')


def get_telemetry():
    """CAPCOM テレメトリ"""
    store = _get_store()
    if store is None:
        return {'snapshots': 0, 'prompts': 0, 'data': 0}
    # 実カウント (テレメトリと実体の両方を見せる)
    return {
        'snapshots': len(store['snapshots']),
        'prompts': len(store['prompts']),
        'data': len(store['data']),
    }


def _increment_counter(key):
    """内部テレメトリカウンター更新"""
    store = _get_store()
    if store is None:
        return
    store['telemetry'][key] = store['telemetry'].get(key, 0) + 1


# ==================================================================
# --- ZIP エクスポート ---
# ==================================================================

def export_session_zip(selected_tools=None):
    """
    session_state['capcom_store'] から ZIP を動的に組み立ててバイト列で返す。
    リポジトリの不変アセット (capcom_schema/, CLAUDE.md, .claude/skills/) も同梱する。

    Args:
        selected_tools: list[str] or None
            CAPCOM モジュール選択肢。"claude_code" / "codex" / "antigravity" のいずれか1つ以上。
            None または空リストの場合は "claude_code" のみが有効。
            "codex" を含む場合: capcom_schema_patches/codex/ の資材を ZIP 直下に展開する
                               （AGENTS.md, .codex/skills/, exec_mode_addendum.md 等）
            "antigravity" を含む場合: capcom_schema_patches/antigravity/ の資材を展開する
                                     （GEMINI.md, AGENTS.md, .agent/, artifacts_templates/ 等）

    Returns:
        tuple: (zip_bytes, zip_filename) または (None, None)
    """
    store = _get_store()
    if store is None:
        return None, None

    # デフォルトは Claude Code のみ
    if not selected_tools:
        selected_tools = ["claude_code"]

    sid = store['session_id']
    project_root = os.path.dirname(os.path.abspath(__file__))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. snapshots
        for snap_id, snap in store['snapshots'].items():
            zf.writestr(f'{sid}/snapshots/{snap_id}.png', snap['image'])

        # 2. data (JSON は dump、CSV は bytes そのまま)
        for fname, content in store['data'].items():
            arcname = f'{sid}/data/{fname}'
            if isinstance(content, bytes):
                zf.writestr(arcname, content)
            else:
                zf.writestr(arcname, json.dumps(content, ensure_ascii=False, indent=2, default=str))

        # 3. prompts
        for fname, text in store['prompts'].items():
            zf.writestr(f'{sid}/prompts/{fname}', text)

        # 4. voyager/
        if store['voyager'].get('mission'):
            zf.writestr(
                f'{sid}/voyager/mission.json',
                json.dumps(store['voyager']['mission'], ensure_ascii=False, indent=2, default=str)
            )
        if store['voyager'].get('context'):
            zf.writestr(
                f'{sid}/voyager/context.json',
                json.dumps(store['voyager']['context'], ensure_ascii=False, indent=2, default=str)
            )
        for ev_fname, ev_data in store['voyager']['evidence'].items():
            zf.writestr(
                f'{sid}/voyager/evidence/{ev_fname}',
                json.dumps(ev_data, ensure_ascii=False, indent=2, default=str)
            )

        # 5. metadata.json
        zf.writestr(
            f'{sid}/metadata.json',
            json.dumps(store['metadata'], ensure_ascii=False, indent=2, default=str)
        )

        # 6. リポジトリの不変アセット（Claude Code 用の基本資材）
        _add_repo_assets_to_zip(zf, sid, project_root)

        # 7. 選択された CAPCOM モジュール用オーバーレイ資材を展開同梱
        if "codex" in selected_tools:
            _add_tool_patch_to_zip(zf, sid, project_root, "codex")
        if "antigravity" in selected_tools:
            _add_tool_patch_to_zip(zf, sid, project_root, "antigravity")

    return buf.getvalue(), f'{sid}.zip'


# ZIP 同梱から除外するファイル/ディレクトリ名
_EXCLUDE_NAMES = {'.DS_Store'}


def _is_excluded(name):
    """ZIP 同梱から除外するか判定する（隠しファイル・OSゴミ・__pycache__）"""
    if name in _EXCLUDE_NAMES:
        return True
    if name.startswith('.') and name not in ('.codex', '.agent'):
        # .codex / .agent はツール配置規約上必要なので残す
        return True
    return False


def _add_repo_assets_to_zip(zf, sid, project_root):
    """リポジトリ内の不変ファイル (スキーマ・CLAUDE.md・スキル) を ZIP に追加"""
    # CLAUDE.md
    src = os.path.join(project_root, 'CLAUDE.md')
    if os.path.exists(src):
        zf.write(src, f'{sid}/CLAUDE.md')

    # requirements-session.txt（CAPCOM スライド生成用の最小依存）
    # ZIP 展開後、ユーザーは `pip install -r requirements-session.txt` で
    # python-pptx / Pillow を一括インストールできる。
    req_src = os.path.join(project_root, 'requirements-session.txt')
    if os.path.exists(req_src):
        zf.write(req_src, f'{sid}/requirements-session.txt')

    # capcom_schema/ 配下 (再帰)
    schema_root = os.path.join(project_root, 'capcom_schema')
    if os.path.exists(schema_root):
        for root, _, files in os.walk(schema_root):
            for fname in files:
                if _is_excluded(fname) or '__pycache__' in root:
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, project_root)
                zf.write(full, f'{sid}/{rel}')

    # .claude/skills/ 配下 (再帰)
    skills_root = os.path.join(project_root, '.claude', 'skills')
    if os.path.exists(skills_root):
        for root, _, files in os.walk(skills_root):
            for fname in files:
                if _is_excluded(fname):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, project_root)
                zf.write(full, f'{sid}/{rel}')


def _add_tool_patch_to_zip(zf, sid, project_root, tool_key):
    """
    capcom_schema_patches/{tool_key}/ 配下の資材を ZIP 内セッション直下に展開同梱する。

    これにより、ユーザーは apply_patch.sh を手動実行しなくても、
    ZIP を展開するだけで Codex / Antigravity 向けのオーバーレイ資材が配置された状態になる。

    展開ルール:
      - README.md, apply_patch.sh は同梱から除外（開発者向け・実行不要）
      - それ以外のファイル/ディレクトリ（AGENTS.md, GEMINI.md, .codex/, .agent/, etc.）は
        `capcom_schema_patches/{tool_key}/<path>` → `{sid}/<path>` に転記
      - AGENTS.md のように Codex と Antigravity で同じ arcname になるファイルは、
        既に ZIP に追加済みならスキップ（両者は共通内容のため内容は同じ）
    """
    patch_root = os.path.join(project_root, 'capcom_schema_patches', tool_key)
    if not os.path.isdir(patch_root):
        return

    # 同梱除外: パッチ管理用の開発者向けファイル
    PATCH_MGMT_FILES = {'README.md', 'apply_patch.sh'}

    # ZIP 内に既に書き込まれた arcname 集合を取得（重複書込み防止）
    existing = set(zf.namelist())

    for root, dirs, files in os.walk(patch_root):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fname in files:
            if _is_excluded(fname):
                continue
            if root == patch_root and fname in PATCH_MGMT_FILES:
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, patch_root)
            arcname = f'{sid}/{rel}'
            if arcname in existing:
                # Codex と Antigravity が共通の AGENTS.md を持つケース等。既存を優先する
                continue
            zf.write(full, arcname)
            existing.add(arcname)


# ==================================================================
# --- 内部ユーティリティ ---
# ==================================================================

def _sanitize_filename(name):
    """ファイル名に使えない文字を除去する"""
    base = os.path.basename(name)
    dir_part = os.path.dirname(name)

    safe = "".join(c for c in base if c.isalnum() or c in ('_', '-', '.'))
    if not safe:
        safe = "unnamed"

    if dir_part:
        return os.path.join(dir_part, safe)
    return safe
