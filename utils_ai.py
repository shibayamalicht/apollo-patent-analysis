
import streamlit as st
import pandas as pd
import textwrap
import json


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
    with st.expander("✨ AI Insight (プロンプト生成)", expanded=False):
        st.markdown("以下のプロンプトをコピーして、ChatGPT / Claude 等に貼り付けてください。\n(可能であればマップ画像も添付するとより正確な分析が可能です)")
        st.code(prompt_text, language="markdown")

        # CAPCOM出力フック
        try:
            import capcom
            if capcom.is_active():
                capcom.save_prompt(unique_key, prompt_text)
                st.caption(f"📡 CAPCOM: `prompts/{unique_key}.md` に出力済み")
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
    meta['SBERTモデル'] = "paraphrase-multilingual-MiniLM-L12-v2"

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
