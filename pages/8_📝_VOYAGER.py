import streamlit as st
import pandas as pd
import os
import utils
import matplotlib.pyplot as plt
import japanize_matplotlib
import pdf_generator

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

    def generate_text(self, system_prompt, user_prompt, images=None):
        if self.error_msg:
            raise ValueError(self.error_msg)

        import time
        import re
        import io
        from PIL import Image

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.provider == "Google Gemini":
                    model = genai.GenerativeModel(self.model_name)
                    # Gemini 1.5 Pro以降のモデル対応
                    full_prompt = f"【System Instructions】\n{system_prompt}\n\n【User Request】\n{user_prompt}"
                    
                    if images and isinstance(images, list) and len(images) > 0:
                        content_parts = [full_prompt]
                        for img_bytes in images:
                            try:
                                if img_bytes:
                                    pil_img = Image.open(io.BytesIO(img_bytes))
                                    content_parts.append(pil_img)
                            except Exception as e:
                                print(f"Image load error in LLMClient: {e}")
                        
                        response = model.generate_content(content_parts)
                    else:
                        response = model.generate_content(full_prompt)
                        
                    return response.text

            except Exception as e:
                error_str = str(e)
                last_error = e
                        # レート制限 (429) またはクォータ超過のチェック
                if "429" in error_str or "Quota exceeded" in error_str or "Resource has been exhausted" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 60 # 安全なデフォルト値
                        # エラーメッセージから待機時間をパース
                        match = re.search(r'retry in (\d+(\.\d+)?)s', error_str)
                        if match:
                            wait_time = float(match.group(1)) + 10 # 10秒のバッファを追加
                        
                        st.toast(f"⏳ レート制限に達しました。{int(wait_time)}秒後に再試行します... ({attempt+1}/{max_retries})", icon="⚠️")
                        
                        # 待機中にプログレスバーを使用するか、単純にスリープ
                        with st.empty():
                            for i in range(int(wait_time), 0, -1):
                                st.write(f"⚠️ API制限に達しました。再試行まであと {i} 秒待機中...")
                                time.sleep(1)
                        continue
                
                # リトライ不可能なエラーまたは最大試行回数に到達
                break
        
        raise RuntimeError(f"LLM Generation Failed: {last_error}")

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

with st.expander("⚙️ AIエンジン設定 (API Key)", expanded=True):
    col_key, col_model = st.columns([2, 1])
    
    # プロバイダはGoogle Geminiに固定
    llm_provider = "Google Gemini"
    
    # APIキー処理
    # 1. Secrets/Envの確認
    api_key_env = None
    env_key_name = "GOOGLE_API_KEY"
    
    # st.secretsから取得を試行
    try:
        if env_key_name in st.secrets:
            api_key_env = st.secrets[env_key_name]
    except (FileNotFoundError, Exception):
        # secrets.tomlが存在しない、またはキーがない場合は無視
        pass
    
    # os.environから取得を試行
    if not api_key_env:
        api_key_env = os.environ.get(env_key_name)
    
    # セキュアキー処理ロジック
    key_status_msg = ""
    default_input_value = ""
    
    if api_key_env:
        placeholder_text = "システムキー設定済み（空欄のままで使用可能）"
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
    
    # 最終キー選択
    final_api_key = api_key_input if api_key_input else api_key_env

    with col_model:
        # モデル選択
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
    
    # スナップショットのグリッド表示
    cols = st.columns(3)
    indices_to_remove = []
    
    for i, snap in enumerate(snapshots):
        with cols[i % 3]:
            with st.container(border=True):
                st.subheader(snap['title'])
                
                # 複数画像表示ロジック
                images = snap.get('images', [])
                main_image = snap.get('image')
                
                if images and len(images) > 1:
                    # 複数画像用タブ
                    tab_names = [f"Img {j+1}" for j in range(len(images))]
                    img_tabs = st.tabs(tab_names)
                    for j, tab in enumerate(img_tabs):
                        with tab:
                             try:
                                 st.image(images[j], caption=f"Evidence {i+1}-{j+1}", use_container_width=True)
                             except Exception as e:
                                 st.error(f"Img Error: {e}")
                
                # フォールバック / 単一画像
                elif main_image:
                    st.image(main_image, use_container_width=True)
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
                
                # ダウンロードボタン
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
report_placeholder = st.empty()
generated_report = ""

with col_act:
    st.write("")
    
    # 分析深度の選択
    report_mode = st.radio(
        "分析の深さ (Analysis Depth):",
        ["Standard Analysis (標準)", "Strategic Deep Dive (詳細・戦略的)", "Market Intelligence (市場統合分析)"],
        horizontal=False,
        help="Standard: 要点を絞ったエグゼクティブサマリー形式。\nDeep Dive: 詳細な考察、シナリオ分析、将来予測を含む長文の戦略レポート形式。"
    )
    
    st.write("")
    
    # バリデーション
    missing_items_common = []
    if len(snapshots) == 0:
        missing_items_common.append("Snapshots (分析の証拠画像)")
    if len(mission_objective) <= 5:
        missing_items_common.append("Mission Objective (5文字以上の目的記述)")

    missing_items_gen = missing_items_common.copy()
    if not final_api_key:
        missing_items_gen.append("API Key (Google API Key)")

    is_ready_preview = len(missing_items_common) == 0
    is_ready_gen = len(missing_items_gen) == 0
    
    if not is_ready_gen:
        if not is_ready_preview:
             st.warning(f"⚠️ プレビュー・生成には以下が必要です: {', '.join(missing_items_common)}")
        elif not final_api_key:
             st.info("ℹ️ API Keyが未設定のため、レポート生成はできませんが、「プロンプト・プレビュー」は利用可能です。")

    # --- プロンプト構築ヘルパー ---
    def build_voyager_prompts(objective, current_snapshots, mode):
        # 1. コンテキスト構築
        c_str = f"## Mission Objective\n{objective}\n\n## Collected Evidence (Snapshots)\n"
        for i, snap in enumerate(current_snapshots):
            c_str += f"\n### Evidence {i+1}: {snap['title']}\n"
            c_str += f"- Description: {snap.get('description', '')}\n" # Safeguard get
            c_str += f"- Source Module: {snap.get('module', 'Unknown')}\n"
            
            # 複数画像のヒント
            if snap.get('images') and len(snap['images']) > 1:
                c_str += f"- [Visual Reference Note]: This evidence consists of multiple images. Refer to [Evidence {i+1}-1] for the first chart (e.g. Growth/Ranking) and [Evidence {i+1}-2] for the second (e.g. Network).\n"

            # --- 構造化データ処理 (v5.1 High-Res) ---
            
            # リストアーティファクトの再帰的クリーナー (['a', 'b'] -> "a, b")
            def clean_data_for_prompt(data, key=None):
                # カスタムフォーマットが必要な特別なリストは平滑化しない
                if key in ['representatives', 'items', 'top_growing_keywords'] and isinstance(data, list):
                     return data
                     
                if isinstance(data, dict):
                    return {k: clean_data_for_prompt(v, k) for k, v in data.items()}
                elif isinstance(data, list):
                    # リストを文字列に結合
                    return ", ".join([str(x) for x in data if x is not None])
                elif isinstance(data, (int, float)):
                    return data
                elif isinstance(data, str):
                    return data
                else:
                    return str(data)

            raw_data_sum = snap.get('data_summary', '')
            data_sum = clean_data_for_prompt(raw_data_sum) if isinstance(raw_data_sum, dict) else raw_data_sum
            
            if isinstance(data_sum, dict):
                # 統計情報
                if 'stats' in data_sum:
                    s = data_sum['stats']
                    c_str += f"- [Statistics]\n"
                    if 'cagr' in s: c_str += f"  - CAGR: {s['cagr']} (Trend: {s.get('trend', 'N/A')})\n"
                    if 'hhi' in s: c_str += f"  - HHI: {s['hhi']:.3f} ({s.get('hhi_status', 'N/A')})\n"
                
                # NEBULA: トレンドチャート (ハイプサイクル) 処理
                if data_sum.get('type') == 'trend_chart' and 'stats' in data_sum:
                    c_str += f"- [Trend Analysis Data (Hype Cycle)]\n"
                    t_stats = data_sum['stats']
                    if 'patent_trend' in t_stats and t_stats['patent_trend']:
                        c_str += f"  - Patent Trend: {t_stats['patent_trend']}\n"
                    if 'academic_trend' in t_stats and t_stats['academic_trend']:
                        c_str += f"  - Academic Trend: {t_stats['academic_trend']}\n"
                    if 'news_trend' in t_stats and t_stats['news_trend']:
                        c_str += f"  - News Trend: {t_stats['news_trend']}\n"
                
                # 代表特許
                if 'representatives' in data_sum and data_sum['representatives']:
                     c_str += f"- [Representative Patents (Top {len(data_sum['representatives'])})]\n"
                     for rep in data_sum['representatives']:
                         c_str += f"  {rep}\n"
                
                # マクロリスト項目 (政策/市場)
                if 'items' in data_sum and isinstance(data_sum['items'], list) and data_sum['items']:
                     c_str += f"- [Macro List Content (Policy/Market/Academic)]\n"
                     for item in data_sum['items']:
                         if isinstance(item, dict):
                            yr = item.get('year', '-')
                            dt = item.get('date', yr) # Use detailed date if available, else year
                            tp = item.get('type', 'Unknown')
                            ti = item.get('title', 'No Title')
                            src = item.get('source', '')
                            c_str += f"  - [{dt}] [{tp}] {ti} ({src})\n"
                         else:
                            c_str += f"  - {item}\n"
                
                # チャートデータ (数値)
                if 'chart_data' in data_sum:
                    c_str += f"- [Chart Data]\n{data_sum['chart_data']}\n"

                # ネットワーク統計 (グラフ分析)
                if 'network_stats' in data_sum:
                    ns = data_sum['network_stats']
                    c_str += f"- [Network Structure Analysis]\n"
                    
                    def clean_join(val):
                        if isinstance(val, list):
                            return ", ".join([str(x) for x in val if x])
                        return str(val)

                    if 'hubs' in ns: c_str += f"  - Top Hubs (Centrality): {clean_join(ns['hubs'])}\n"
                    if 'edges' in ns: c_str += f"  - Strongest Connections: {clean_join(ns['edges'])}\n"
                    if 'communities' in ns: c_str += f"  - Community Groups: {clean_join(ns['communities'])}\n"
                

                if 'cluster_summary' in data_sum:
                    c_str += f"- [Cluster Composition]\n{data_sum['cluster_summary']}\n"
                

                if 'matrix_context' in data_sum:
                    c_str += f"- [Context Note] {data_sum['matrix_context']}\n"
                
                # エラー情報など
                if 'error' in data_sum:
                     c_str += f"- [Note] Data extraction partial error: {data_sum['error']}\n"
                
                # --- NEBULA統合スナップショット処理 (v5.3) ---
                if data_sum.get('type') == 'trend_network_consolidated':
                    c_str += f"- [Consolidated Analysis Data]\n"
                    
                    # 1. 手法コンテキスト
                    if 'methodology' in data_sum:
                        c_str += f"  - [Methodology]: {data_sum['methodology']}\n"
                    
                    # 2. 急上昇キーワードランキング (成長率)
                    if 'ranking' in data_sum:
                        r = data_sum['ranking']
                        c_str += f"  - [Emerging Keywords (Growth Ranking)]\n"
                        c_str += f"    Period: {r.get('period_past', '')} vs {r.get('period_recent', '')}\n"
                        if 'top_growing_keywords' in r and isinstance(r['top_growing_keywords'], list):
                             c_str += f"    Top Growing:\n"
                             for k in r['top_growing_keywords']:
                                 c_str += f"      - {k.get('Keyword')} (Growth: {k.get('Growth Rate', 0):.2f}, Recent Count: {k.get('Recent')})\n"

                    # 3. ネットワーク統計
                    if 'network' in data_sum:
                        ns = data_sum['network']
                        c_str += f"  - [Network Structure Analysis]\n"
                        # 安全な結合のためのヘルパー
                        def clean_join_local(val):
                            if isinstance(val, list): return ", ".join([str(x) for x in val if x])
                            return str(val)

                        if 'nodes' in ns: c_str += f"    Nodes: {ns['nodes']}, Edges: {ns['edges']}\n"
                        if 'hubs' in ns: c_str += f"    Top Hubs (Centrality): {clean_join_local(ns['hubs'])}\n"
                        if 'strongest_edges' in ns: c_str += f"    Strongest Connections: {clean_join_local(ns['strongest_edges'])}\n"
                        if 'communities' in ns: c_str += f"    Community Groups: {clean_join_local(ns['communities'])}\n"

            else:
                # レガシー文字列
                c_str += f"- Data Summary: {data_sum}\n"
        
        # 2. システムプロンプト選択 (2段階アーキテクチャ v6.0)
        
        # --- 共通ルール ---
        common_evidence_rules = """
        ### 証拠引用の絶対ルール (Strict Evidence Rules)
        1. **形式 (Format):** 引用は **`[[Evidence X]]`** の形式（Xは番号）**のみ** を使用してください。
           - **絶対禁止 (Prohibited):** `[[NEBULA]]`, `[[Explorer]]`, `[[Saturn V]]` などのモジュール名タグは**決して使用しないでください**。
           - **禁止:** `[[Evidence 1, 2]]` のようなカンマ区切りも不可です。`[[Evidence 1]] [[Evidence 2]]` と記述してください。
        2. **配置 (Placement):** 必ず文末または段落末尾に `[[Evidence X]]` タグを配置してください。
        3. **根拠の明確化:** 「なぜそう言えるのか」を、必ず具体的な証拠データや数値を引用して説明してください。
        """
        
        # --- フェーズ1: 分析官 (証拠抽出) ---
        analyst_system_prompt = f"""
        あなたは熟練した **「特許・市場分析官 (Patent & Market Analyst)」** です。
        あなたの仕事は、与えられた複数の「証拠 (Evidence)」を含むデータグループを分析し、**構造化された洞察テキスト (Structured Insight)** を抽出することです。
        
        ### 目的
        視覚的なグラフと数値データを、後の工程で「戦略レポート」を執筆するCSOが使えるような、明確な事実と洞察に変換してください。
        
        ### 必須要件
        1. **Reference by Evidence ID:** 洞察を記述する際は、必ず **「Evidence X によれば〜」** や **「(Evidence X)」** のように、情報源となる **Evidence番号** を明記してください。モジュール名（NEBULAなど）で曖昧に参照しないでください。
        2. **Visual & Data Synthesis:** チャートの視覚的なトレンドと、「Data Summary」内の具体的な数値を組み合わせて分析してください。
        3. **Representative Citations (最重要):**
           - 提供された「代表特許/文献リスト」の中から、分析を裏付ける**具体的な事例を2〜3件**引用してください。
           - 記述形式: 「例えば、[出願人]の『[発明の名称]』(出願年) は、〜〜を示唆している。(Evidence X)」
        4. **出力フォーマット:**
           - **観測事実 (Observation):** 何が起きているか。
           - **データ裏付け (Data Backup):** 具体的な数値。
           - **情報源 (Source):** 該当する Evidence ID。
        
        ### 出力スタイル
        - 言語: 日本語
        - 箇条書きで簡潔に。
        """
        
        # --- フェーズ2: 戦略官 (レポート統合) ---
        strategist_sys_base = f"""
        あなたは **「最高戦略責任者 (Chief Strategy Officer: CSO)」** です。
        あなたは部下の分析官たちから「分析レポート（各証拠の洞察）」と、クライアントからの「Mission Objective（目的）」を受け取りました。
        あなたの任務は、これらバラバラの洞察を統合し、一つの首尾一貫した **「戦略インテリジェンス・レポート」** を執筆することです。
        
        ### Mission Objective
        {{objective}}
        
        ### Core Mandates (鉄則)
        1. **Storytelling:** 単に洞察を羅列するのではなく、Mission Objectiveに対する「答え」となるようなストーリーを構築してください。
        2. **Evidence Integration:** 主張の根拠として、必ず提供された `[[Evidence X]]` タグを文中に埋め込んでください。
        3. **Specific Citations:** 分析官のレポートに含まれている「具体的な特許事例（社名・技術名）」を、**最終レポートに必ず盛り込んでください**。具体性が説得力を生みます。
        4. **Gap Analysis (Market vs Patent):** 市場情報(NPL)のトレンドと、特許活動(Patent)の整合性・ギャップを必ず分析してください。「ニュース等の市場情報からはトレンドを読み取り、特許情報とのギャップ（市場は拡大しているが特許出願は減少している、等）を導く」ことで、リスクや機会を浮き彫りにしてください。
        
        {common_evidence_rules}
        
        ### Report Structure
        1. **Executive Strategy Brief:** 問いに対する直截的な回答・結論 (Verdict)。
        2. **Strategic Drivers (Synthesis):** 複数の証拠を掛け合わせた要因分析。
        3. **Detailed Findings:**
           - 具体的な特許・市場の事例を引用。
           - `[[Evidence X]]` タグを使用。
        4. **Future Scenarios:** リスクと機会のシナリオ分析。
        
        **言語:** 日本語 (経営層を唸らせる、格調高く論理的なビジネス文書スタイル)
        **Output Volume:** Extremely Detailed (Minimum 4000 characters). 各セクションにおいて、表面的な要約ではなく、徹底的な深掘り分析を行ってください。
        """
        
        # Phase 1タスクをモジュールごとにグループ化するヘルパー関数
        def build_phase1_tasks_grouped(all_snapshots):
            grouped_tasks = {}
            tasks = []
            
            # データ文字列を安全に抽出するヘルパー
            def get_data_str(d):
                return clean_data_for_prompt(d)
            
            # 1. スナップショットのグループ化
            for i, snap in enumerate(all_snapshots):
                 module = snap.get('module', 'General')
                 if module not in grouped_tasks: grouped_tasks[module] = []
                 grouped_tasks[module].append((i, snap))
            
            # 2. モジュールごとに1つのタスクを作成
            for module_name, snaps in grouped_tasks.items():
                
                module_content = f"### Analysis Target: {module_name} Data Group\n"
                module_content += "以下の証拠グループを統合的に分析し、このモジュール（視点）からの包括的なインサイトを抽出してください。\n\n"
                module_images = []
                
                for (index, snap) in snaps:
                    raw_data = snap.get('data_summary', {})
                    domain = raw_data.get('domain', 'Patent') if isinstance(raw_data, dict) else 'Patent'
                    
                    ref_label = "引用・参照用特許リスト"
                    if domain == 'Academic': ref_label = "引用・参照用文献リスト (Literature)"
                    elif domain == 'News': ref_label = "引用・参照用ニュース/レポートリスト (News/Market)"
                    elif domain == 'Policy/Market': ref_label = "引用・参照用 ポリシー/市場レポートリスト"

                    # --- コンテンツ構築 ---
                    module_content += f"#### Evidence {index+1}: {snap['title']}\n"
                    module_content += f"- Context/Description: {snap.get('description', '')}\n"
                    
                    # データサマリーと統合データのロジック
                    if isinstance(raw_data, dict) and raw_data.get('type') == 'trend_network_consolidated':
                         # 統合データ（ランキング + ネットワーク）
                         if 'ranking' in raw_data: 
                             module_content += f"- [Growth Data]:\n{get_data_str(raw_data['ranking'])}\n"
                         if 'network' in raw_data:
                             module_content += f"- [Network Data]:\n{get_data_str(raw_data['network'])}\n"
                    else:
                         start_data = raw_data
                        
                         if isinstance(start_data, dict):
                             # 標準処理 - 重複したキーを除外してサマリーを作成
                             exclude_keys = ['ai_insight_context', 'representatives_raw', 'representatives', 'chart_data', 'cluster_summary', 'network_stats']
                             filtered_data = {k: v for k, v in start_data.items() if k not in exclude_keys}
                             module_content += f"- [Data Summary]:\n{get_data_str(filtered_data)}\n"
                            
                             # リッチコンテンツの明示的な処理
                             if 'cluster_summary' in raw_data:
                                  module_content += f"\n- [Cluster/Group Composition]:\n{raw_data['cluster_summary']}\n"
                            
                             if 'network_stats' in raw_data:
                                  module_content += f"\n- [Network Statistics]:\n{get_data_str(raw_data['network_stats'])}\n"

                             # チャートデータ (CSV)
                             if 'chart_data' in raw_data:
                                  module_content += f"- [Chart Table (Top 30)]:\n{raw_data['chart_data']}\n"
                         else:
                             # レガシーストリングデータ
                             module_content += f"- [Data Summary]:\n{start_data}\n"

                    
                    # AI Insight コンテキスト（Saturn V / EAGLE用）
                    if isinstance(raw_data, dict) and 'ai_insight_context' in raw_data:
                        module_content += f"\n- [Advanced Landscape Context (Spatial/Method)]:\n{raw_data['ai_insight_context']}\n"

                    
                    # 代表特許/文献（整形済みリストを優先、無ければRawデータを使用）
                    if isinstance(raw_data, dict):
                        if 'representatives' in raw_data and raw_data['representatives']:
                            module_content += f"- [{ref_label} (Must Cite)]:\n"
                            for rep in raw_data['representatives']:
                                module_content += f"  {rep}\n"
                        elif 'representatives_raw' in raw_data and raw_data['representatives_raw']:
                            module_content += f"- [{ref_label} (Must Cite)]:\n"
                            for r in raw_data['representatives_raw'][:5]:
                                module_content += f"  * 【{r['title']}】 ({r['applicant']}, {r['year']}): {r['abstract'][:100]}...\n"

                    
                    module_content += "\n---\n"
                    
                    # 画像の収集
                    if snap.get('images'): module_images.extend(snap['images'])
                    elif snap.get('image'): module_images.append(snap['image'])
                
                # タスク追加
                tasks.append({
                    'id_label': f"Module Analysis: {module_name}",
                    'content': module_content,
                    'images': module_images,
                    'system_prompt_add': f"Focus on insights specific to {module_name}. (ATLAS=Macro/Landscape, NEBULA=Trends/Future, CORE=Companies)." 
                })

            return tasks

        # ラッパー: (Phase 2システムプロンプト, Phase 1タスクリスト, Phase 1システムプロンプトベース) を返す
        # グループ化されたタスクを使用
        phase1_tasks = build_phase1_tasks_grouped(current_snapshots)
        
        return strategist_sys_base, phase1_tasks, analyst_system_prompt

    col_btn_1, col_btn_2 = st.columns([1, 1])

    with col_btn_1:
        if st.button("📜 Preview Prompts (APIなし)", help="AIに送るプロンプト構成を確認します。APIは消費しません。", disabled=not is_ready_preview):
            strat_sys, p1_tasks, analyst_sys = build_voyager_prompts(mission_objective, snapshots, report_mode)
            
            # 手動利用用にフォーマットされたデータを準備
            
            # Phase 2 テンプレート
            p2_template = f"""【System Instructions】
{strat_sys.format(objective=mission_objective)}

【User Request】
以下は、各証拠に関する分析官からの報告書 (Analyst Reports) です。これらを統合し、最終レポートを作成してください。

[ここにPhase 1で得られた分析結果（インサイト）を全て貼り付けてください...]
"""
            
            # Phase 1 フルプロンプト (タスク反復)
            p1_full_prompts = []
            for task in p1_tasks:
                # タスク固有の指示があれば追加
                sys_combined = analyst_sys
                if task['system_prompt_add']:
                    sys_combined += f"\n\n(Specific Focus: {task['system_prompt_add']})"
                    
                p1_full = f"""【System Instructions】
{sys_combined}

【User Request】
{task['content']}"""
                p1_full_prompts.append({'label': task['id_label'], 'text': p1_full})

            st.session_state['voyager_prompt_preview_data'] = {
                'p2_template': p2_template,
                'p1_full_prompts': p1_full_prompts
            }
            st.toast("実用コピー用プロンプトを生成しました！下の画面で確認してください。", icon="📋")

    with col_btn_2:
        if st.button("🚀 Analyze & Generate Report (2-Stage)", type="primary", disabled=not is_ready_gen):
            
            # --- 2段階生成プロセス (v6.0) ---
            strat_sys_template, p1_tasks, analyst_sys_base = build_voyager_prompts(mission_objective, snapshots, report_mode)
            
            progress_bar = st.progress(0, text="分析を開始しています...")
            status_text = st.empty()
            
            collected_insights = []
            
            try:
                client = LLMClient(llm_provider, final_api_key, llm_model)
                
                # --- Phase 1: 分析官 (The Analyst) - 順次処理 ---
                total_tasks = len(p1_tasks)
                
                for i, task in enumerate(p1_tasks):
                    status_text.markdown(f"**[フェーズ1: 分析官]** 分析タスクを実行中 {i+1}/{total_tasks}: {task['id_label']}...")
                    
                    # システムプロンプト準備
                    current_sys = analyst_sys_base
                    if task['system_prompt_add']:
                        current_sys += f"\n\n(IMPORTANT: {task['system_prompt_add']})"
                    
                    # LLM呼び出し
                    insight = client.generate_text(current_sys, task['content'], images=task['images'])
                    collected_insights.append(f"### Insight from {task['id_label']}\n{insight}")
                    
                    # 進捗更新
                    progress_bar.progress((i + 1) / (total_tasks + 1))
                
                # --- Phase 2: 戦略官 (The Strategist) - 統合 ---
                status_text.markdown(f"**[フェーズ2: 戦略官]** 最終レポートを統合執筆中...")
                
                # Phase 2 ユーザーコンテンツ構築
                phase2_user_content = f"以下は、{total_tasks}件の分析タスクからの報告書 (Analyst Reports) です。これらを統合し、最終レポートを作成してください。\n\n" + "\n\n".join(collected_insights)
                
                phase2_sys = strat_sys_template.format(objective=mission_objective)
                
                final_report = client.generate_text(phase2_sys, phase2_user_content)
                
                st.session_state['last_report'] = final_report
                progress_bar.progress(1.0)
                status_text.success("分析完了！")
                st.rerun()

            except Exception as e:
                st.error(f"Error during generation: {e}")

    # プロンプトプレビューエリア
    if 'voyager_prompt_preview_data' in st.session_state:
        data = st.session_state['voyager_prompt_preview_data']
        
        # 旧データスキーマの安全性チェック
        if 'p1_full_prompts' not in data:
            del st.session_state['voyager_prompt_preview_data']
            st.rerun()
            
        with st.expander("📜 プロンプト確認ウィンドウ (手動分析用)", expanded=True):
            st.info("APIキーがない場合や、ChatGPT/Claudeで手動分析したい場合に利用してください。")
            
            tab1, tab2 = st.tabs(["Phase 1: Analyst (Individual)", "Phase 2: Strategist (Synthesis)"])
            
            with tab1:
                st.markdown("### 手順1: 各スナップショットの分析 (Analyst)")
                st.caption("以下のプロンプトを順番にコピーし、**「該当する画像」を添付して** AIに送信してください。Consolidatedスナップショットは「Growth」と「Network」に分割されています。")
                
                for i, item in enumerate(data['p1_full_prompts']):
                    with st.expander(f"{item['label']} 用プロンプト", expanded=(i==0)):
                        st.code(item['text'], language='markdown')
                        st.caption(f"※ ここで {item['label']} の画像をアップロードしてください。")

            with tab2:
                st.markdown("### 手順2: レポートの統合・執筆 (Strategist)")
                st.caption("手順1で得られた全てのインサイトを、以下のプロンプトの末尾（プレースホルダー部分）に貼り付け、AIに送信してください。")
                st.code(data['p2_template'], language='markdown')
            
            if st.button("プレビューを閉じる", key="close_preview"):
                del st.session_state['voyager_prompt_preview_data']
                st.rerun()

# レポート表示
if 'last_report' in st.session_state:
    generated_report = st.session_state['last_report']
    with report_placeholder.container():
        st.markdown("### 📝 Analysis Report")
        
        # 1. 画像付きレポートのパースとレンダリング
        import re
        import unicodedata
        
        last_idx = 0
        # [[Evidence 1, 5]] などをサポート
        # Regex captures the content "1, 5" or "１，５" inside the brackets
        evidence_pattern = r'\[{1,2}Evidence\s*[:：]?\s*([^\]]+)\]{1,2}'
        
        for match in re.finditer(evidence_pattern, generated_report, flags=re.IGNORECASE):
            # タグ前のテキスト
            text_segment = generated_report[last_idx:match.start()]
            
            # サニタイズ: 引用符(>)を削除
            cleaned_segment = re.sub(r'^\s*>\s?', '', text_segment, flags=re.MULTILINE)
            
            if cleaned_segment.strip(): 
                st.markdown(cleaned_segment)
            
            # タグ内のIDをパース
            ids_str = match.group(1)
            # 正規化 (全角->半角など)
            ids_str = unicodedata.normalize('NFKC', ids_str)
            
            # カンマまたは読点で分割
            ids_str = ids_str.replace('、', ',')
            raw_ids = [x.strip() for x in ids_str.split(',')]
            
            # タグで見つかった証拠画像をレンダリング
            for raw_id in raw_ids:
                e_num, e_sub = None, None
                
                # 厳密なパースを使用
                # Matches "1" or "1-2"
                m_id = re.match(r'^(\d+)(?:-(\d+))?$', raw_id)
                if m_id:
                     e_num = m_id.group(1)
                     e_sub = m_id.group(2)
                
                if e_num:
                    ev_id = int(e_num) - 1
                    if 0 <= ev_id < len(snapshots):
                        snap = snapshots[ev_id]
                        # スタイリッシュなコンテナを使用
                        with st.container(border=True):
                            # キャプション
                            cap_suffix = f" ({e_sub})" if e_sub else ""
                            st.caption(f"Evidence {ev_id + 1}{cap_suffix}: {snap['title']}")
                            
                            # 画像選択ロジック
                            target_image = None
                            if e_sub:
                                try:
                                    sub_id = int(e_sub) - 1
                                    if snap.get('images') and 0 <= sub_id < len(snap['images']):
                                        target_image = snap['images'][sub_id]
                                except: pass
                            else:
                                target_image = snap.get('image')
                                
                            if target_image:
                                c1, c2 = st.columns([4, 1])
                                with c1: st.image(target_image, use_container_width=True)
                            else:
                                st.warning("(No Image)")
                            
                            st.caption(snap.get('description', ''))
            
            last_idx = match.end()
            
        # Remaining Text
        if last_idx < len(generated_report):
            st.markdown(generated_report[last_idx:])

        # 2. コピー機能
        st.markdown("---")
        with st.expander("📋 レポート本文をコピー (Markdown)", expanded=False):
            st.code(generated_report, language="markdown")
            st.info("右上のコピーボタンでテキストをクリップボードにコピーできます。画像は「右クリック→画像をコピー」で取得してください。")

        # 3. PDF Download
        st.markdown("---")
        col_pdf, _ = st.columns([1, 2])
        with col_pdf:
            if pdf_generator.HAS_REPORTLAB:
                # Generate PDF if not already in session (lazy load for old sessions)
                if 'last_pdf' not in st.session_state or st.session_state.get('last_pdf_source') != generated_report:
                     with st.spinner("PDFドキュメントを生成中..."):
                         # Determine Subtitle based on Mode
                         subtitle = "MARKET INTELLIGENCE & IP INTELLIGENCE"
                         if "Standard" in report_mode:
                             subtitle = "EXECUTIVE SUMMARY REPORT"
                         elif "Deep Dive" in report_mode:
                             subtitle = "STRATEGIC DEEP DIVE REPORT"
                         elif "Market Intelligence" in report_mode:
                             subtitle = "MARKET INTELLIGENCE & IP INTELLIGENCE"
                         
                         pdf_bytes, pdf_err = pdf_generator.generate_pdf(generated_report, snapshots, mission_objective, subtitle=subtitle)
                         if pdf_bytes:
                             st.session_state['last_pdf'] = pdf_bytes
                             st.session_state['last_pdf_source'] = generated_report # Cache invalidation
                         else:
                             st.error(f"PDF Generation Failed: {pdf_err}")
                
                if 'last_pdf' in st.session_state:
                    st.download_button(
                        label="📄 PDFレポートをダウンロード (デザイン版)",
                        data=st.session_state['last_pdf'],
                        file_name="VOYAGER_Strategy_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("⚠️ PDF Export is unavailable. Please install `reportlab`.\n`pip install reportlab`")
