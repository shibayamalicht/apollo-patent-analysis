
import streamlit as st
import pandas as pd
import textwrap

def generate_ai_insight_prompt(role, context, data_summary, instructions, extra_content=""):
    """
    外部AI用のプロンプトを生成する
    """
    
    # Dedent and clean inputs
    role = textwrap.dedent(str(role)).strip()
    context = textwrap.dedent(str(context)).strip()
    instructions = textwrap.dedent(str(instructions)).strip()
    
    # Data Section Construction
    data_text = ""
    if data_summary:
        if isinstance(data_summary, str):
            data_text = data_summary
        elif isinstance(data_summary, dict):
            # Parse standardized dictionary
            for k, v in data_summary.items():
                if k in ['module', 'chart_data', 'ai_insight_context']: continue # Skip large dumps and duplicates
                if isinstance(v, list) and k == 'representatives_raw': continue # Skip raw dict list
                
                if isinstance(v, (dict, list)):
                    import json
                    try:
                        formatted_v = json.dumps(v, ensure_ascii=False, indent=2)
                        data_text += f"{k}:\n{formatted_v}\n"
                    except:
                        data_text += f"{k}: {v}\n"
                else:
                    data_text += f"{k}: {v}\n"
            
            # Formatted Representatives
            if 'representatives' in data_summary:
                data_text += "\n[Representative Documents]\n"
                for r in data_summary['representatives']:
                    data_text += f"{r}\n"
    
    prompt = f"""
# 役割 (Role)
{role}

# コンテキスト (Context)
{context}

# データ (Data)
{data_text}
{extra_content}

# 指示 (Instructions)
{instructions}
"""
    return prompt.strip()

def render_ai_insight_button(prompt_text, unique_key):
    """
    プロンプトを表示・コピーするためのUIレンダラー
    """
    with st.expander("✨ AI Insight (プロンプト生成)", expanded=False):
        st.markdown("以下のプロンプトをコピーして、ChatGPT / Claude / Gemini 等に貼り付けてください。\n(可能であればマップ画像も添付するとより正確な分析が可能です)")
        st.code(prompt_text, language="markdown")
