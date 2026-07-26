
import streamlit as st
import pandas as pd
import textwrap
import json


def df_to_markdown(df, index=False, index_label=None, max_rows=None):
    """DataFrame / Series を依存追加なし（tabulate 不要）で markdown テーブル文字列へ変換する。
    AIインサイトやスナップショットの表を、AIが解釈しやすい `| ... |` 形式で渡すために使う。

    Args:
        df: DataFrame または Series
        index: True なら行インデックスを先頭列として含める（出願人名・年など意味がある場合）
        index_label: index=True 時の先頭列ヘッダ名（省略時は df.index.name）
        max_rows: 出力する最大行数（トークン抑制）

    Returns:
        str: markdown テーブル。空・変換不能なら空文字（Seriesは1列表として扱う）。
    """
    if df is None:
        return ""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not hasattr(df, 'columns'):
        return str(df)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    def _cell(x):
        s = "" if x is None else str(x)
        # markdown テーブルのセル区切り '|' とセル内改行を無害化
        return s.replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()

    cols = [_cell(c) for c in df.columns]
    if index:
        idx_head = _cell(index_label if index_label is not None else (df.index.name or ""))
        header = [idx_head] + cols
    else:
        header = cols
    if not header:
        return ""

    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for idx, row in df.iterrows():
        vals = [_cell(v) for v in row.tolist()]
        if index:
            vals = [_cell(idx)] + vals
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_ai_insight_prompt(role, context, data_summary, instructions,
                                extra_content="",
                                constraints=None,
                                output_format=None,
                                metadata=None):
    """
    外部AI / Claude Code 用のプロンプトを生成する（6部構成 + メタデータ）

    Args:
        role: 分析者の役割定義
        context: チャートタイプ・手法・視覚エンコーディングの説明
        data_summary: 分析データのサマリー (str or dict)
        instructions: 具体的な分析指示
        extra_content: 追加コンテキスト（空間情報など）
        constraints: 制約条件（分析上の注意点）
        output_format: 期待する出力形式
        metadata: メタデータdict（母集団件数・期間・パラメータ等）
    """

    # Dedent and clean inputs
    role = textwrap.dedent(str(role)).strip()
    context = textwrap.dedent(str(context)).strip()
    instructions = textwrap.dedent(str(instructions)).strip()

    # Metadata Section
    metadata_text = ""
    if metadata and isinstance(metadata, dict):
        metadata_lines = []
        for k, v in metadata.items():
            if isinstance(v, (dict, list)):
                try:
                    formatted_v = json.dumps(v, ensure_ascii=False, indent=2)
                    metadata_lines.append(f"- **{k}**: {formatted_v}")
                except (TypeError, ValueError):
                    metadata_lines.append(f"- **{k}**: {v}")
            else:
                metadata_lines.append(f"- **{k}**: {v}")
        metadata_text = "\n".join(metadata_lines)

    # Data Section Construction
    data_text = ""
    if data_summary:
        if isinstance(data_summary, str):
            data_text = data_summary
        elif isinstance(data_summary, dict):
            for k, v in data_summary.items():
                # メタデータ系・大きなダンプ・重複キーをスキップ
                if k in ['module', 'chart_data', 'ai_insight_context']:
                    continue
                if isinstance(v, list) and k == 'representatives_raw':
                    continue

                if isinstance(v, (dict, list)):
                    try:
                        formatted_v = json.dumps(v, ensure_ascii=False, indent=2)
                        data_text += f"{k}:\n{formatted_v}\n"
                    except (TypeError, ValueError):
                        data_text += f"{k}: {v}\n"
                else:
                    data_text += f"{k}: {v}\n"

            # フォーマット済み代表特許
            if 'representatives' in data_summary:
                data_text += "\n[Representative Documents]\n"
                for r in data_summary['representatives']:
                    data_text += f"{r}\n"

    # Constraints Section
    constraints_text = ""
    if constraints:
        constraints_text = textwrap.dedent(str(constraints)).strip()

    # Output Format Section
    output_format_text = ""
    if output_format:
        output_format_text = textwrap.dedent(str(output_format)).strip()

    # プロンプト組み立て（6部構成 + メタデータ）
    sections = []

    sections.append(f"# 役割 (Role)\n{role}")

    if metadata_text:
        sections.append(f"# メタデータ (Metadata)\n{metadata_text}")

    sections.append(f"# コンテキスト (Context)\n{context}")

    data_section = f"# データ (Data)\n{data_text}"
    if extra_content:
        data_section += f"\n{extra_content}"
    sections.append(data_section)

    if constraints_text:
        sections.append(f"# 制約条件 (Constraints)\n{constraints_text}")

    if output_format_text:
        sections.append(f"# 出力形式 (Output Format)\n{output_format_text}")

    sections.append(f"# 指示 (Instructions)\n{instructions}")

    prompt = "\n\n".join(sections)
    return prompt.strip()


def render_ai_insight_button(prompt_text, unique_key):
    """
    プロンプトを表示・コピーするためのUIレンダラー。
    CAPCOMアクティブ時はファイル出力も行う。
    """
    with st.expander("AI Insight (プロンプト生成)", expanded=False):
        st.markdown("以下のプロンプトをコピーして、ChatGPT / Claude 等に貼り付けてください。\n(可能であればマップ画像も添付するとより正確な分析が可能です)")
        st.code(prompt_text, language="markdown")

        # 記録セッション出力フック（内部ファイル名は画面に出さない）
        try:
            import capcom
            if capcom.is_active():
                capcom.save_prompt(unique_key, prompt_text)
                st.caption("記録セッションに分析メモとして保存済み — レポート生成の材料に自動で使われます")
        except Exception as e:
            pass


def build_common_metadata(df_main=None, df_filtered=None, col_map=None, filter_info=None):
    """
    全プロンプト共通のメタデータを構築するヘルパー。
    各モジュールから呼び出して使う。

    Args:
        df_main: メインDataFrame
        df_filtered: フィルタ適用後のDataFrame（なければdf_mainと同じ）
        col_map: カラムマッピングdict
        filter_info: フィルタ条件の説明文字列（任意）

    Returns:
        dict: 共通メタデータ
    """
    meta = {}

    if df_main is not None:
        meta['母集団件数'] = len(df_main)

    if df_filtered is not None and df_main is not None and len(df_filtered) != len(df_main):
        meta['フィルタ後件数'] = len(df_filtered)

    if filter_info:
        meta['フィルタ条件'] = filter_info

    # 期間
    if df_main is not None and col_map and 'date' in col_map and col_map['date']:
        if 'year' in df_main.columns:
            years = df_main['year'].dropna()
            if len(years) > 0:
                meta['期間'] = f"{int(years.min())}年 ～ {int(years.max())}年"

    # カラムマッピング
    if col_map:
        mapped = {k: v for k, v in col_map.items() if v is not None}
        meta['カラムマッピング'] = mapped

    # テキスト前処理情報
    meta['テキスト前処理'] = "Janome形態素解析 + 複合名詞結合"
    try:
        import utils as _utils
        meta['SBERTモデル'] = _utils.current_sbert_model_name()
    except Exception:
        meta['SBERTモデル'] = st.session_state.get('sbert_model_name', 'paraphrase-multilingual-MiniLM-L12-v2')

    # TF-IDF語彙数
    if 'feature_names' in st.session_state and st.session_state['feature_names'] is not None:
        meta['TF-IDF語彙数'] = len(st.session_state['feature_names'])

    # ストップワード数
    if 'stopwords' in st.session_state and st.session_state['stopwords']:
        meta['ストップワード数'] = len(st.session_state['stopwords'])

    # ステータス分布（col_mapにstatusが設定されている場合）
    if df_filtered is not None and col_map and col_map.get('status'):
        _status_col = col_map['status']
        if _status_col in df_filtered.columns:
            _status_counts = df_filtered[_status_col].value_counts()
            if len(_status_counts) > 0:
                meta['ステータス分布'] = {str(k): int(v) for k, v in _status_counts.items()}
                meta['ステータスカラム'] = _status_col

    return meta


# ステータス内訳をAIに分析させる共通指示文（出願量だけでなく権利化の質に言及させる）
STATUS_INSIGHT_INSTRUCTION = (
    "\n6. **権利状況（質）の分析**: 下記[ステータス内訳]をもとに、登録/審査中/拒絶/失効等の"
    "権利状況の構成比と推移を読み解き、出願量だけでなく『権利化の成否＝ポートフォリオの質』"
    "に言及すること（量が多くても権利化率が低い／高い、といった量×質の差を明示）。"
)


def build_status_breakdown_summary(df, status_col, axis_col, axis_label,
                                   top_axis=None, max_rows=40):
    """ステータス×軸（年・出願人等）のクロス集計を、AIインサイトの extra_content 用に
    コンパクトな文字列で返す。チャートで「ステータス内訳を表示」した内容と一致させるための関数。

    Args:
        df: 集計対象 DataFrame（フィルタ適用後）
        status_col: ステータス列名
        axis_col: 軸となる列名（'year' や 出願人列など。df に存在する列）
        axis_label: 軸の日本語ラベル（'出願年' '出願人' 等）
        top_axis: 軸値を上位N件に絞る場合の件数（出願人など多数の場合に使用）
        max_rows: 出力する最大行数（トークン抑制）

    Returns:
        str: "[ステータス内訳] ..." 形式の文字列。集計不能なら空文字。
    """
    try:
        import pandas as pd
        import utils
        if df is None or status_col not in df.columns or axis_col not in df.columns:
            return ""
        # 欠損ステータスは「(ステータス不明)」として集計に含める（チャート側の表示と一致させる）
        sub = df[[axis_col, status_col]].copy()
        sub[status_col] = sub[status_col].fillna(utils.STATUS_UNKNOWN_LABEL)
        sub = sub.dropna()
        if sub.empty:
            return ""
        if top_axis:
            _keep = sub[axis_col].value_counts().head(top_axis).index
            sub = sub[sub[axis_col].isin(_keep)]
        pivot = (sub.assign(_n=1)
                    .pivot_table(index=axis_col, columns=status_col, values='_n',
                                 aggfunc='sum', fill_value=0))
        if pivot.empty:
            return ""
        # 全体のステータス別合計（構成比の母数）
        totals = pivot.sum(axis=0).sort_values(ascending=False)
        pivot = pivot[totals.index]  # 件数の多いステータス順に列を並べる
        if axis_col == 'year':
            try:
                pivot.index = pivot.index.astype(int)
            except Exception:
                pass
            pivot = pivot.sort_index()
        body = df_to_markdown(pivot.head(max_rows), index=True, index_label=axis_label)
        total_line = " / ".join(f"{k}: {int(v)}件" for k, v in totals.items())
        return (f"[ステータス内訳] {axis_label}×権利状況のクロス集計（チャートの内訳表示と同一）\n"
                f"全体内訳: {total_line}\n{body}\n")
    except Exception:
        return ""


def build_map_dynamics_noise_addon(dynamics_data=None, noise_count=None,
                                   total_count=None, noise_keywords=None):
    """技術ランドスケープ系（Saturn V / EAGLE）のメインマップ insight に、
    クラスタ動態（新興/成長/成熟/衰退）とノイズ（萌芽技術候補）の要約と分析指示を追加する。
    (extra_content, instruction_suffix) を返す。データが無ければ空文字ペア。

    Args:
        dynamics_data: render_cluster_dynamics_section の返り値 dict
                       （'quadrant_summary' と 'clusters' を持つ）
        noise_count: ノイズ点（cluster == -1）の件数
        total_count: マップ全点数（ノイズ率の母数）
        noise_keywords: ノイズ特許から抽出した萌芽キーワード（任意）
    """
    parts = []
    # --- ノイズ（萌芽技術候補） ---
    if noise_count is not None and total_count:
        rate = (noise_count / total_count * 100) if total_count else 0
        line = (f"[ノイズ（萌芽技術候補）] ノイズ点 {int(noise_count)}件 / 全{int(total_count)}件"
                f"（{rate:.1f}%）。ノイズ=どのクラスタにも属さない外れ値で、"
                f"既存技術体系から外れた萌芽・ニッチ技術の候補。")
        if noise_keywords:
            line += " 萌芽キーワード上位: " + ", ".join(str(k) for k in noise_keywords[:15])
        parts.append(line)
    # --- クラスタ動態 ---
    if dynamics_data and isinstance(dynamics_data, dict):
        qs = dynamics_data.get('quadrant_summary', {})
        if qs:
            ql = " / ".join(
                f"{q}: {v.get('count', 0)}クラスタ({v.get('total_patents', 0)}件)"
                for q, v in qs.items())
            parts.append(f"[クラスタ動態] 累積件数(X)×CAGR(Y)で4象限分類 — {ql}")
        clusters = dynamics_data.get('clusters', [])
        if clusters:
            try:
                top = sorted(clusters, key=lambda d: d.get('cagr', 0) or 0, reverse=True)[:8]
                lines = [
                    f"  {d.get('label', '?')}: 累積{d.get('cumulative_count', '?')}件, "
                    f"CAGR{(d.get('cagr', 0) or 0) * 100:.0f}%, {d.get('quadrant', '?')}"
                    for d in top]
                parts.append("[主要クラスタ動態(CAGR上位)]\n" + "\n".join(lines))
            except Exception:
                pass
    if not parts:
        return "", ""
    extra = "\n".join(parts) + "\n"
    inst = (
        "\n\n# 追加分析（必須・v8新機能）\n"
        "- **クラスタ動態**: 上記[クラスタ動態]をもとに、新興/成長リーダー/成熟/ニッチ・衰退の"
        "各クラスタを名指しし、技術領域のライフサイクル局面と注力・撤退の戦略示唆を述べること。\n"
        "- **萌芽技術（ノイズ）**: [ノイズ（萌芽技術候補）]に触れ、ノイズ率の高低が示す"
        "技術分野の成熟度（高い=萌芽/黎明、低い=体系化）を解釈し、外れ値に潜む萌芽・ニッチ技術の"
        "可能性に言及すること。")
    return extra, inst


def status_insight_addon(enabled, df, status_col, axis_col, axis_label,
                         top_axis=None):
    """ステータス内訳トグルがONのとき、AIインサイトに渡す
    (extra_content, instruction_suffix) を返す。OFF・status未設定時は空文字ペア。

    各タブで:
        _ex, _inst = utils_ai.status_insight_addon(use_breakdown, df, status_col, 'year', '出願年')
        ... generate_ai_insight_prompt(..., instructions=base + _inst, extra_content=_ex)
    のように使う。"""
    if not enabled or not status_col:
        return "", ""
    extra = build_status_breakdown_summary(df, status_col, axis_col, axis_label,
                                           top_axis=top_axis)
    if not extra:
        return "", ""
    return extra, STATUS_INSIGHT_INSTRUCTION
