import streamlit as st
import pandas as pd
import os
import utils
import matplotlib.pyplot as plt
import japanize_matplotlib

# ライブラリの動的インポート (エラーハンドリング用)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ==================================================================
# --- クラス定義: LLM Client ---
# ==================================================================
class LLMClient:
    def __init__(self, provider, api_key, model_name=None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.error_msg = None

        if not self.api_key:
            self.error_msg = "API Keyが設定されていません。"
            return

        if self.provider == "Google Gemini":
            if genai is None:
                self.error_msg = "google-generativeai ライブラリがインストールされていません。"
            else:
                genai.configure(api_key=self.api_key)
                if not self.model_name: self.model_name = "gemini-1.5-pro"
        else:
            self.error_msg = f"未サポートのプロバイダ: {self.provider}"

    def generate_text(self, system_prompt, user_prompt):
        if self.error_msg:
            raise ValueError(self.error_msg)

        try:
            if self.provider == "Google Gemini":
                model = genai.GenerativeModel(self.model_name)
                # Gemini 1.5 Pro以降のモデル対応
                # System promptは本来 system_instruction で渡すべきですが、簡易的に結合して処理します
                full_prompt = f"【System Instructions】\n{system_prompt}\n\n【User Request】\n{user_prompt}"
                response = model.generate_content(full_prompt)
                return response.text

        except Exception as e:
            raise RuntimeError(f"LLM生成エラー: {e}")

# ==================================================================
# --- ページ設定 ---
# ==================================================================
st.set_page_config(page_title="APOLLO | VOYAGER", page_icon="📝", layout="wide")
utils.render_sidebar()

st.title("📝 VOYAGER")
st.markdown("##### Visual Output & Yield Analysis Generator for Executive Review")

st.markdown("""
**VOYAGER** は、分析データからレポートを自動生成するAIアシスタントです。
**Google Gemini** の力を借りて、複雑な特許マップからレポートを作成します。
""")

# ==================================================================
# --- サイドバー設定 (LLM設定) ---
# ==================================================================
# ==================================================================
# --- 設定 (Settings) ---
# ==================================================================
with st.expander("⚙️ AIエンジン設定 (API Key)", expanded=True):
    col_key, col_model = st.columns([2, 1])
    
    # Provider is now fixed to Google Gemini
    llm_provider = "Google Gemini"
    
    # API Key Handling
    # 1. Check Secrets/Env
    api_key_env = None
    env_key_name = "GOOGLE_API_KEY"
    
    # Try getting from st.secrets
    try:
        api_key_env = st.secrets[env_key_name]
    except:
        pass
    
    # Try getting from os.environ
    if not api_key_env:
        api_key_env = os.environ.get(env_key_name)
    
    # Logic for Secure Key Handling
    key_status_msg = ""
    default_input_value = ""
    
    if api_key_env:
        placeholder_text = "System Key Active (Leave empty to use)"
    else:
        placeholder_text = "AIza..."

    with col_key:
        api_key_input = st.text_input(
            "Google API Key", 
            type="password", 
            value="", # NEVER populate this with the secret
            placeholder=placeholder_text,
            help="Google AI Studioで取得したAPIキーを入力してください。システムキー設定済みの場合は空欄でOKです。"
        )
    
    # Final Key selection
    final_api_key = api_key_input if api_key_input else api_key_env

    with col_model:
        # Model Selection
        model_options = [
            "gemini-2.5-flash"
        ]
        llm_model = st.selectbox("Model", model_options, key="voyager_model")


# ==================================================================
# --- 3. Snapshot Curator UI ---
# ==================================================================
st.markdown("---")
st.header("📸 Snapshot Collection")

if 'snapshots' not in st.session_state or not st.session_state['snapshots']:
    st.info("スナップショットがまだありません。ATLASなどの分析画面で「📸 Capture Snapshot」ボタンを押して、重要な発見をここに集めてください。")
    snapshots = []
else:
    snapshots = st.session_state['snapshots']
    
    # Grid display for snapshots
    cols = st.columns(3)
    indices_to_remove = []
    
    for i, snap in enumerate(snapshots):
        with cols[i % 3]:
            with st.container(border=True):
                st.subheader(snap['title'])
                if snap.get('image'):
                    st.image(snap['image'], use_container_width=True)
                elif snap.get('image_error'):
                    st.error(f"Image Error: {snap['image_error']}")
                    st.caption("※ ターミナルで `pip install -U kaleido` を実行してください。")
                else:
                    st.warning("(No Image)")
                    
                st.caption(f"Source: {snap.get('module', 'Unknown')} | {snap.get('timestamp')}")
                with st.expander("Memo / Data"):
                    st.write(snap.get('description', ''))
                    ds_preview = snap.get('data_summary', '')
                    if isinstance(ds_preview, dict):
                         ds_preview = str(ds_preview)
                    st.code(str(ds_preview)[:200] + "...")
                
                if st.button("🗑️ 削除", key=f"del_{snap['id']}_{i}"):
                    indices_to_remove.append(i)
                
                # Download Button
                if snap.get('image'):
                    file_name = f"Evidence {i+1}.png"
                    st.download_button(
                        label="💾 Download Evidence",
                        data=snap['image'],
                        file_name=file_name,
                        mime="image/png",
                        key=f"dl_{snap['id']}_{i}"
                    )

    if indices_to_remove:
        for i in sorted(indices_to_remove, reverse=True):
            del st.session_state['snapshots'][i]
        st.rerun()

# ==================================================================
# --- 4. Mission Control (Prompt) ---
# ==================================================================
st.markdown("---")
st.header("📡 Mission Objective")

col_obj, col_act = st.columns([3, 1])

with col_obj:
    mission_objective = st.text_area(
        "今回の分析レポートの目的 (問い) を設定してください:",
        height=100,
        placeholder="例: 競合A社の直近3年の出願傾向から、彼らが注力している新規事業領域を特定し、自社の対抗策を提案してください。",
        value=st.session_state.get('voyager_objective', '')
    )
    st.session_state['voyager_objective'] = mission_objective

# ==================================================================
# --- 5. Report Generation ---
# ==================================================================
# --- 5. Report Generation ---
# ==================================================================
report_placeholder = st.empty()
generated_report = ""

with col_act:
    st.write("")
    
    # Analysis Depth Selection
    report_mode = st.radio(
        "分析の深さ (Analysis Depth):",
        ["Standard Analysis (標準)", "Strategic Deep Dive (詳細・戦略的)"],
        horizontal=False,
        help="Standard: 要点を絞ったエグゼクティブサマリー形式。\nDeep Dive: 詳細な考察、シナリオ分析、将来予測を含む長文の戦略レポート形式。"
    )
    
    st.write("")
    
    # Validation
    missing_items = []
    if not final_api_key:
        missing_items.append("API Key (Google API Key)")
    if len(snapshots) == 0:
        missing_items.append("Snapshots (分析の証拠画像)")
    if len(mission_objective) <= 5:
        missing_items.append("Mission Objective (5文字以上の目的記述)")

    is_ready = len(missing_items) == 0
    
    if not is_ready:
        st.warning(f"⚠️ レポート生成には以下が必要です: {', '.join(missing_items)}")

    if st.button("🚀 Analyze & Generate Report", type="primary", disabled=not is_ready):
        
        # 1. Context Construction
        context_str = f"## Mission Objective\n{mission_objective}\n\n## Collected Evidence (Snapshots)\n"
        for i, snap in enumerate(snapshots):
            context_str += f"\n### Evidence {i+1}: {snap['title']}\n"
            context_str += f"- Description: {snap['description']}\n"
            context_str += f"- Source Module: {snap['module']}\n"
            
            # --- STRUCTURED DATA HANDLING (v5.1 High-Res) ---
            data_sum = snap.get('data_summary', '')
            if isinstance(data_sum, dict):
                # 統計情報
                if 'stats' in data_sum:
                    s = data_sum['stats']
                    context_str += f"- [Statistics]\n"
                    if 'cagr' in s: context_str += f"  - CAGR: {s['cagr']} (Trend: {s.get('trend', 'N/A')})\n"
                    if 'hhi' in s: context_str += f"  - HHI: {s['hhi']:.3f} ({s.get('hhi_status', 'N/A')})\n"
                
                # 代表特許
                if 'representatives' in data_sum and data_sum['representatives']:
                     context_str += f"- [Representative Patents (Top {len(data_sum['representatives'])})]\n"
                     for rep in data_sum['representatives']:
                         context_str += f"  {rep}\n"
                
                # Chart Data (Numerical values)
                if 'chart_data' in data_sum:
                    context_str += f"- [Chart Data]\n{data_sum['chart_data']}\n"

                # Network Statistics (Graph Analysis)
                if 'network_stats' in data_sum:
                    ns = data_sum['network_stats']
                    context_str += f"- [Network Structure Analysis]\n"
                    if 'hubs' in ns: context_str += f"  - Top Hubs (Centrality): {ns['hubs']}\n"
                    if 'edges' in ns: context_str += f"  - Strongest Connections: {ns['edges']}\n"
                    if 'communities' in ns: context_str += f"  - Community Groups: {ns['communities']}\n"
                
                # エラー情報など
                if 'error' in data_sum:
                     context_str += f"- [Note] Data extraction partial error: {data_sum['error']}\n"
            else:
                # Legacy String
                context_str += f"- Data Summary: {data_sum}\n"
        
        # 2. System Prompt Selection
        system_prompt_std = """
        あなたは熟練した特許分析官 (Senior Patent Analyst) です。
        ユーザーから提供された「Mission Objective (分析の目的)」と、その証拠となる「Collected Snapshots (グラフやデータの集合)」に基づいて、
        経営層向けの洞察に満ちたレポートを作成してください。

        ### ルール
        1. **証拠の解釈:** 各Snapshotの `Title` (例: "技術ライフサイクル", "Treemap", "Network") と `Description` を注意深く読み取り、それがどのような分析視点（時系列、シェア、関係性、成長度など）を提供しているかを正確に理解すること。
        2. **証拠に基づく論証:** 必ず提供されたスナップショットの内容を引用・参照して論を展開すること。
        3. **画像の配置:** 文中で特定の証拠（スナップショット）に言及する際は、その場所に `[[Evidence X]]` というプレースホルダーを挿入すること（XはEvidence番号）。これにより、レポート表示時に実際の画像がそこに埋め込まれます。
           - 例: "...右肩上がりの推移を示しています。[[Evidence 1]] このことから..."
        4. **目的志向:** ユーザーの「問い」に対して明確な答えや仮説を提示すること。
        5. **構造化:** 以下の構成で出力すること。
           - **Executive Summary:** 3行要約。
           - **Key Findings:** 箇条書きで3〜5点。証拠を紐付けること。
           - **Strategic Recommendation:** 今後のアクション提案。
        6. **言語:** 日本語 (ビジネスプロフェッショナルなトーン)。
        7. **情報の正確性:** ランキング上位に含まれていない企業や技術（圏外データ）について言及する場合は、必ず「**ランキング上位には含まれていませんが**」や「**圏外ですが**」といった前置きを行い、ユーザーに誤解を与えないこと。
        """

        system_prompt_deep = """
        あなたは世界トップクラスの戦略コンサルタント兼Chief Strategy Officer (CSO)です。
        提供された「Mission Objective」と「Collected Evidence」に基づき、経営会議でそのまま使用できるレベルの、極めて詳細かつ長文の戦略分析レポートを作成してください。

        ### Core Mandate
        表面的なデータの羅列は一切不要です。データの背後にある「構造的変化」「競合の意図」「市場の空白地帯」を深く読み解き、論理的かつ大胆な仮説を構築してください。

        ### Guidelines
        1.  **Deep Dive & Exhaustiveness (徹底的な深掘り):**
            -   要約してはいけません。思考の過程を省略せず、可能な限り詳細に記述してください。
            -   各Evidenceについて、単に「何が起きているか」だけでなく「なぜ起きているか（技術的・事業的背景）」「次に何が起きるか」まで踏み込んで考察してください。
            -   **すべてのEvidence** を必ず論証に組み込んでください。

        2.  **Scenario Planning (シナリオプランニング):**
            -   単一の予測だけでなく、以下の3つのシナリオを提示してください。
                -   **Probable Scenario (蓋然性が高い未来):** 現状の延長線。
                -   **Best Case (自社にとっての好機):** 自社技術が市場標準となる、または競合が失速するケース。
                -   **Risk Scenario (脅威の顕在化):** 技術パラダイムシフトや新規参入によるディスラプション。

        3.  **Strict Evidence Linking:**
            -   すべての主張は、提供された `[[Evidence X]]` によって裏付けられている必要があります。
            -   画像プレースホルダー `[[Evidence X]]` を文脈に合わせて適切な位置に配置してください（各セクションに最低1つ以上）。

        ### Report Structure (Output Format)
        以下のセクション構成を厳守してください。

        # 1. Executive Insight (総括)
        -   **Strategic Verdict:** 結論を一言で（Go/No-Go、撤退、攻勢など）。
        -   **Critical Drivers:** 意思決定を左右する決定的な要因（3点）。

        # 2. Comprehensive Evidence Analysis (詳細分析)
        各分析視点（時系列推移、プレイヤー比較、ネットワーク構造など）ごとにサブセクションを設け、徹底的に論じてください。
        -   *Observation:* データから読み取れる事実。
        -   *Insight:* その事実が意味する戦略的含意。
        -   *Evidence Reference:* ここで関連する `[[Evidence X]]` を使用。

        # 3. Competitive Landscape & Power Dynamics (競争環境)
        -   主要プレイヤーの意図と能力の評価。
        -   自社の立ち位置 (Strength/Weakness)。
        -   支配率ネットワーク等があれば、支配構造の脆さを指摘。

        # 4. Strategic Scenarios (未来予測)
        -   Probable / Best / Risk の3シナリオ提示。

        # 5. Action Plan & Roadmap (提言)
        -   Actionable Steps (直ちに着手すべきこと)。
        -   Mid-term Strategy (中期的布石)。

        **言語:** 日本語 (極めて高度で洗練された戦略ビジネス用語を使用)
        **情報の正確性:** ランキング上位に含まれていない企業や技術（圏外データ）について言及する場合は、必ず「**ランキング上位には含まれていませんが**」や「**圏外ですが**」といった前置きを行い、ユーザーに誤解を与えないこと。
        """

        system_prompt = system_prompt_deep if "Deep Dive" in report_mode else system_prompt_std
        
        with st.spinner("VOYAGER AI is analyzing your snapshots..."):
            try:
                client = LLMClient(llm_provider, final_api_key, llm_model)
                generated_report = client.generate_text(system_prompt, context_str)
                st.session_state['last_report'] = generated_report
                st.success("Analysis Complete!")
            except Exception as e:
                st.error(f"Error: {e}")

# Display Report
if 'last_report' in st.session_state:
    generated_report = st.session_state['last_report']
    with report_placeholder.container():
        st.markdown("### 📝 Analysis Report")
        
        # 1. Parse and Render Report with Images
        import re
        parts = re.split(r'\[\[Evidence (\d+)\]\]', generated_report)
        
        for part in parts:
            # Check if part is a digit (Evidence ID) or text
            if part.isdigit():
                # This is an evidence ID captured by the group in split
                ev_id = int(part) - 1 # 0-indexed
                if 0 <= ev_id < len(snapshots):
                    snap = snapshots[ev_id]
                    with st.container(border=True):
                        st.caption(f"Evidence {ev_id + 1}: {snap['title']}")
                        if snap.get('image'):
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.image(snap['image'], use_container_width=True)
                        elif snap.get('image_error'):
                             st.error(f"Image Error: {snap['image_error']}")
                        else:
                            st.warning("(No Image)")
                        st.caption(snap.get('description', ''))
                else:
                    st.caption(f"※ AIが Evidence {part} を参照しましたが、該当するスナップショットは見つかりませんでした。この部位はハルシネーションの可能性があるため、検討してください。")
            else:
                # Normal text
                if part.strip():
                    st.markdown(part)

        # 2. Copy Functionality
        st.markdown("---")
        with st.expander("📋 Copy Report Text (Markdown)", expanded=False):
            st.code(generated_report, language="markdown")
            st.info("右上のコピーボタンでテキストをクリップボードにコピーできます。画像は「右クリック→画像をコピー」で取得してください。")
