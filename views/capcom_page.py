# ==================================================================
# views/capcom_page.py — レポート生成 (CAPCOM)
#
# v9.1.0 の pages/8_📝_VOYAGER.py + pages/10_📡_CAPCOM.py の統合ビュー。
# 1ページに2つのレポート経路を並べる:
#   経路A: :material/bolt: チャットキット — プロンプトを生成して手元のチャットAIに貼るだけ（APIキー不要）
#   経路B: :material/school: 本格レポート — セッションZIPを Claude Code 等の AI エージェントに渡す
#
# 設計メモ:
# - Gemini API 自動実行（旧 VOYAGER の LLMClient / APIキー入力 / 3フェーズ生成）は完全削除
# - capcom.py の呼び出し・引数・ZIP 内部構造（voyager/mission.json 等）・
#   session_state キー名（capcom_mission_objective, voyager_objective,
#   capcom_query_intent, _capcom_zip_key 等）は APOLLO と完全互換のまま維持
# - ミッション目的は1つの入力欄に統一し、旧 VOYAGER→CAPCOM 継承のねじれを解消
#   （voyager_objective / capcom_mission_objective の両キーへ同値を書き込み互換維持）
# ==================================================================
import io
import re
import json
import base64
import zipfile
import datetime

import streamlit as st
import pandas as pd

import capcom
import apollo_ui
import flight_recorder
import utils


# ==================================================================
# --- 定数（旧 CAPCOM ページから移設・値は不変） ---
# ==================================================================

# AI エージェント選択肢: 表示ラベル → 内部キー
# ツール選択 UI と context.json 構築の両方で使う単一ソース
CAPCOM_TOOL_OPTIONS = {
    "Claude Code（Anthropic）": "claude_code",
    "Codex CLI（OpenAI）": "codex",
    "Antigravity IDE（Google）": "antigravity",
}

# レポート生成の進め方: 表示ラベル → 内部キー
# （context.json の report_mode・ZIP キャッシュキー・export_session_zip 引数で使う）
CAPCOM_REPORT_MODES = {
    ":material/smart_toy: 自律生成モード（従来・4フェーズ自動進行）": "autonomous",
    ":material/chat: 対話型レポート作成モード（KATHERINE）": "interactive",
}

# ZIP 内 data/ ファイルの説明対応表（キー=内部ファイル名は不変。説明は機能名）
_DATA_FILE_DESC = {
    "patents.csv": "全特許データ（タイトル・要約・出願人・クラスタ情報等）",
    "atlas_statistics.json": "基本統計のマクロ統計（時系列・ランキング・HHI/Entropy/Gini）",
    "atlas_grant_rate.json": "権利化率マップ（出願数×権利化率の象限分析）",
    "core_classification.json": "ルール分類（ルールベース分類）の結果",
    "saturnv_clusters.json": "俯瞰図分析のAIクラスタリング結果（ノイズ分析・動態マップ含む）",
    "saturnv_drilldown.json": "俯瞰図分析のドリルダウン（サブクラスタ精査）",
    "mega_momentum.json": "動態分析（CAGR×活動量4象限）",
    "mega_drilldown.json": "動態分析ドリルダウンのポートフォリオ詳細",
    "explorer_global_network.json": "キーワード共起ネットワーク（全体）",
    "explorer_trend.json": "キーワードトレンドネットワーク",
    "explorer_dominance.json": "キーワードドミナンスネットワーク",
    "eagle_clusters.json": "探索的クラスタリング結果（旧版互換）",
    "eagle_cluster_dynamics.json": "探索的クラスタリングのクラスタ動態マップ（旧版互換）",
    "nebula_hype_cycle.json": "環境分析のハイプサイクル分析",
    "nebula_macro_events.json": "環境分析のマクロイベント分析",
    "nebula_academic_clusters.json": "環境分析の学術ランドスケープ",
    "crew_network.json": "出願人・発明者ネットワーク分析（媒介中心性・コミュニティ）",
}


def _describe_data_file(filename):
    """ZIP 内 data/ ファイルの日本語説明を返す（旧 CAPCOM と同一）"""
    return _DATA_FILE_DESC.get(filename, filename)


# ==================================================================
# --- プロンプト構築（旧 VOYAGER build_voyager_prompts・ロジック不変） ---
# ==================================================================

def build_voyager_prompts(objective, current_snapshots):
    """スナップショット群からチャットAI用の2段階プロンプトを構築する。

    Returns:
        tuple: (strategist_sys_base, phase1_tasks, analyst_system_prompt)
               strategist_sys_base は {objective} プレースホルダを含み .format() で埋める
    """
    # 1. コンテキスト構築
    c_str = f"## Mission Objective\n{objective}\n\n## Collected Evidence (Snapshots)\n"
    for i, snap in enumerate(current_snapshots):
        c_str += f"\n### Evidence {i+1}: {snap['title']}\n"
        c_str += f"- Description: {snap.get('description', '')}\n"  # Safeguard get
        c_str += f"- Source Module: {snap.get('module', 'Unknown')}\n"

        # 複数画像のヒント
        if snap.get('images') and len(snap['images']) > 1:
            c_str += f"- [Visual Reference Note]: This evidence consists of multiple images. Refer to [Evidence {i+1}-1] for the first chart (e.g. Growth/Ranking) and [Evidence {i+1}-2] for the second (e.g. Network).\n"

        # --- 構造化データ処理 ---

        # リストアーティファクトの再帰的クリーナー (['a', 'b'] -> "a, b")
        def clean_data_for_prompt(data, key=None):
            # カスタムフォーマットが必要な特別なリストは平滑化しない
            if key in ['representatives', 'items', 'top_growing_keywords',
                       'emerging_keywords', 'declining_keywords',
                       'hubs_ranked', 'edges_ranked', 'communities', 'bridge_edges'] and isinstance(data, list):
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
                        dt = item.get('date', yr)  # Use detailed date if available, else year
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
                        # 構造化リスト対応: dictの場合はキー情報を抽出
                        parts = []
                        for x in val:
                            if isinstance(x, dict):
                                # hubs_ranked形式: keyword + centrality
                                if 'keyword' in x:
                                    parts.append(f"{x['keyword']}({x.get('degree_centrality', '')})")
                                # edges_ranked形式: source-target + jaccard
                                elif 'source' in x and 'target' in x:
                                    parts.append(f"{x['source']}-{x['target']}(J={x.get('jaccard', '')})")
                                # communities形式: id + members
                                elif 'members' in x:
                                    members_str = ', '.join(x['members'][:5])
                                    parts.append(f"Group {x.get('id', '?')+1}: {members_str}")
                                else:
                                    parts.append(str(x))
                            elif x is not None:
                                parts.append(str(x))
                        return ", ".join(parts)
                    return str(val)

                # ノード数・エッジ数・密度
                if 'nodes' in ns:
                    c_str += f"  - Nodes: {ns['nodes']}, Edges: {ns.get('edges', 'N/A')}"
                    if 'density' in ns:
                        c_str += f", Density: {ns['density']}"
                    c_str += "\n"
                # ハブ（新形式 hubs_ranked / 旧形式 hubs 両対応）
                if 'hubs_ranked' in ns:
                    top_hubs = ns['hubs_ranked'][:10] if isinstance(ns['hubs_ranked'], list) else ns['hubs_ranked']
                    c_str += f"  - Top Hubs (Centrality): {clean_join(top_hubs)}\n"
                elif 'hubs' in ns:
                    c_str += f"  - Top Hubs (Centrality): {clean_join(ns['hubs'])}\n"
                # エッジ（新形式 edges_ranked / 旧形式 edges 両対応）
                if 'edges_ranked' in ns:
                    top_edges = ns['edges_ranked'][:10] if isinstance(ns['edges_ranked'], list) else ns['edges_ranked']
                    c_str += f"  - Strongest Connections: {clean_join(top_edges)}\n"
                elif 'edges' in ns:
                    c_str += f"  - Strongest Connections: {clean_join(ns['edges'])}\n"
                # コミュニティ（新形式 communities リスト / 旧形式 文字列 両対応）
                if 'communities' in ns:
                    c_str += f"  - Community Groups: {clean_join(ns['communities'])}\n"
                # ブリッジエッジ
                if 'bridge_edges' in ns and ns['bridge_edges']:
                    bridges = ns['bridge_edges'][:5] if isinstance(ns['bridge_edges'], list) else ns['bridge_edges']
                    c_str += f"  - Bridge Edges (Cross-Community): {clean_join(bridges)}\n"

            # トレンド分析データ（Explorer強化版）
            if 'trend_analysis' in data_sum:
                ta = data_sum['trend_analysis']
                c_str += f"- [Trend Analysis]\n"
                c_str += f"  - Period: {ta.get('period_past', 'N/A')} vs {ta.get('period_recent', 'N/A')}\n"
                if 'emerging_keywords' in ta:
                    top_emerging = ta['emerging_keywords'][:10]
                    for ek in top_emerging:
                        c_str += f"  - Emerging: {ek.get('keyword', '')} (Growth: {ek.get('growth_rate', '')}, Recent: {ek.get('recent_count', '')})\n"

            # 支配率分析データ（Explorer強化版）
            if 'dominance_analysis' in data_sum:
                da = data_sum['dominance_analysis']
                c_str += f"- [Dominance Analysis: {da.get('my_company', '')} vs {da.get('target_company', '')}]\n"
                if da.get('my_exclusive'):
                    c_str += f"  - My Exclusive Keywords: {', '.join(da['my_exclusive'][:10])}\n"
                if da.get('target_exclusive'):
                    c_str += f"  - Competitor Exclusive: {', '.join(da['target_exclusive'][:10])}\n"
                if da.get('contested'):
                    c_str += f"  - Contested (0.4-0.6): {', '.join(da['contested'][:10])}\n"

            if 'cluster_summary' in data_sum:
                c_str += f"- [Cluster Composition]\n{data_sum['cluster_summary']}\n"

            if 'matrix_context' in data_sum:
                c_str += f"- [Context Note] {data_sum['matrix_context']}\n"

            # エラー情報など
            if 'error' in data_sum:
                c_str += f"- [Note] Data extraction partial error: {data_sum['error']}\n"

            # --- NEBULA統合スナップショット処理 ---
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

    # 2. システムプロンプト選択 (2段階アーキテクチャ)

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


# ==================================================================
# --- チャットキット構築（経路A・APIキー不要） ---
# ==================================================================

def _build_chatkit_markdown(objective, snapshots):
    """2段階プロンプト（分析官→戦略官）を「1回貼るだけ」で使える1本のMarkdownに束ねる。

    プロンプト内容そのものは build_voyager_prompts（旧 VOYAGER Preview Prompts）を流用。
    """
    strat_sys, p1_tasks, analyst_sys = build_voyager_prompts(objective, snapshots)
    p2_body = strat_sys.format(objective=objective)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    parts = []
    parts.append(f"""# APOLLO レポート作成キット（チャットAI用）
生成日時: {now} ・ エビデンス {len(snapshots)} 枚 ・ 分析グループ {len(p1_tasks)} 組

【この文書の使い方】
この文書を丸ごとチャットAI（ChatGPT / Claude / Gemini 等）に貼り付けてください。
AI は STEP 1（グループ別分析）→ STEP 2（統合レポート）の順に、この1つの依頼の中で実行します。
画像ファイル（evidence_N*.png）も一緒に渡されている場合、ファイル名の N は本文中の Evidence N に
対応します。チャートの視覚的特徴も必ず分析に使ってください。
分量が多い場合は、STEP 1 の各グループを1メッセージずつ送り、最後に STEP 2 を送っても構いません。

---

## STEP 1 — グループ別分析（まず「分析官」として実行）

<分析官への指示>
{analyst_sys.strip()}
</分析官への指示>
""")

    for gi, task in enumerate(p1_tasks, start=1):
        focus = f"\n（このグループの焦点: {task['system_prompt_add']}）\n" if task.get('system_prompt_add') else ""
        parts.append(f"""### 分析グループ {gi}/{len(p1_tasks)} — {task['id_label']}{focus}
{task['content']}""")

    parts.append(f"""---

## STEP 2 — 統合レポート（次に「戦略官」として実行）

STEP 1 で抽出した全グループの洞察（Analyst Reports）を統合し、以下の指示に従って
最終レポートを執筆してください。

<戦略官への指示>
{p2_body.strip()}
</戦略官への指示>
""")
    return "\n\n".join(parts)


def _build_chatkit_zip(prompt_md, snapshots):
    """フル版キット: prompt.md + スナップショットPNG群をインメモリZIPに束ねる。

    画像ソースは session_state の snapshots リスト内の画像バイト
    （snap['images'] / snap['image']・旧 VOYAGER と同じ取り出し方）。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('prompt.md', prompt_md)
        for i, snap in enumerate(snapshots):
            imgs = snap.get('images') or ([snap['image']] if snap.get('image') else [])
            for j, img_bytes in enumerate(imgs):
                if not img_bytes:
                    continue
                if len(imgs) == 1:
                    name = f"images/evidence_{i+1}.png"
                else:
                    name = f"images/evidence_{i+1}_{j+1}.png"
                zf.writestr(name, img_bytes)
    return buf.getvalue()


def _chatkit_cache_key(objective, snapshots):
    """チャットキット再構築要否の判定キー（スナップショット増減・差し替え・目的変更で更新）"""
    return (
        objective,
        tuple(
            (s.get('id'), str(s.get('timestamp')), len(s.get('images') or []), bool(s.get('image')))
            for s in snapshots
        ),
    )


# ==================================================================
# --- 貼り戻しビューア（旧 VOYAGER レポート表示部を流用） ---
# ==================================================================

# 整形表示用CSS（配色は apollo_ui のネイビー基調に合わせて調整・構造は旧 VOYAGER と同一）
_REPORT_CSS = """<style>
.report-container { background:#fff; border:1px solid #DEE5EC; border-radius:12px; padding:32px 40px;
  margin:8px 0 16px; line-height:1.9; color:#1C2733; }
.report-container h1 { color:#141C24; font-size:1.5rem; border-bottom:3px solid #141C24; padding-bottom:8px; margin-top:30px; }
.report-container h2 { color:#134A77; font-size:1.2rem; border-bottom:2px solid #DEE5EC; padding-bottom:6px; margin-top:24px; }
.report-container h3 { color:#333; font-size:1.05rem; margin-top:18px; }
.report-container p { margin:10px 0; text-align:justify; }
.report-container ul, .report-container ol { margin:8px 0 8px 24px; }
.report-container li { margin:4px 0; }
.report-container blockquote { border-left:4px solid #141C24; background:#F2F6FA; padding:12px 16px; margin:12px 0; border-radius:0 8px 8px 0; }
.report-container table { border-collapse:collapse; width:100%; margin:16px 0; }
.report-container th { background:#141C24; color:#fff; padding:9px 13px; text-align:left; font-weight:600; }
.report-container td { border:1px solid #ddd; padding:8px 13px; }
.report-container tr:nth-child(even) { background:#F6F8FA; }
.report-container strong { color:#141C24; }
.report-container .evidence-badge { background:#FDF3EF; color:#C2461A; padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.9em; }
.report-container .evidence-img { border:1px solid #DEE5EC; border-radius:8px; margin:12px 0; max-width:100%; }
</style>"""


def _render_report_html(md_text, snap_lookup):
    """Markdownテキストを処理し、[[Evidence X]] 引用に画像をインライン展開する（各画像は初回のみ表示）"""
    inserted_images = set()  # 既に画像を挿入済みのEvidence ID

    def _replace_evidence(match):
        ev_id = int(match.group(1))
        badge = f'<span class="evidence-badge"><span class=msym>push_pin</span> Evidence {ev_id}</span>'

        # 2回目以降の引用はバッジのみ（画像重複を防止）
        if ev_id in inserted_images:
            return badge

        snap = snap_lookup.get(ev_id)
        if snap:
            img_bytes = snap.get('image')
            if not img_bytes and snap.get('images') and len(snap['images']) > 0:
                img_bytes = snap['images'][0]

            if img_bytes:
                inserted_images.add(ev_id)
                b64 = base64.b64encode(img_bytes).decode()
                title = snap.get('title', '')
                img_tag = f'<br><img class="evidence-img" src="data:image/png;base64,{b64}" alt="Evidence {ev_id}: {title}">'
                caption = f'<div style="text-align:center; color:#666; font-size:0.85em; margin-bottom:16px;">Fig.{ev_id}: {title}</div>'
                return badge + img_tag + caption
        return badge

    return re.sub(r'\[\[Evidence\s*(\d+)\]\]', _replace_evidence, md_text)


# ==================================================================
# --- セクション描画 ---
# ==================================================================

def _render_mission_objective():
    """ミッション目的（1入力欄に統一）。

    旧仕様: VOYAGER の voyager_objective を CAPCOM の capcom_mission_objective が
    初期値として継承し、Export 時に capcom 優先・voyager フォールバックだった。
    統合後は1つの欄から両キーへ同値を書き込み、ねじれを解消しつつ
    capcom.py まわりの読み出しロジック（キー名・優先順位）はそのまま生かす。
    """
    st.markdown("### :material/crisis_alert: レポートの目的")
    _default = (
        st.session_state.get('capcom_mission_objective', '')
        or st.session_state.get('voyager_objective', '')
    )
    mission_objective = st.text_area(
        "このレポートで答えを出したい「問い」を書いてください:",
        height=100,
        value=_default,
        placeholder=(
            "例: 競合A社の直近3年の出願傾向から、彼らが注力している新規事業領域を特定し、"
            "自社の対抗策を提案してください。"
        ),
        key="logbook_mission_objective_input",
        help=(
            "ここに書いた問いが、経路A（チャットキット）と経路B（本格レポート）の両方で"
            "レポートの軸 (Mission Objective) になります。具体的なほど答えが鋭くなります。"
        ),
    )
    # 旧 VOYAGER / CAPCOM の両キーへ同値を書き込み（外部レポート生成スキーマとの互換維持）
    st.session_state['voyager_objective'] = mission_objective
    st.session_state['capcom_mission_objective'] = mission_objective
    return mission_objective


def _render_snapshot_gallery():
    """収集済みスナップショットの一覧・削除・個別ダウンロード（旧 VOYAGER Snapshot Collection）"""
    st.markdown("### :material/photo_camera: 収集済みスナップショット")

    snapshots = st.session_state.get('snapshots') or []
    pages = st.session_state.get('ap_pages', {})

    if not snapshots:
        st.info(
            "スナップショットがまだありません。"
            "分析ページで「:material/photo_camera: レポート用に保存」を押すと、ここに集まります。",
            icon=":material/photo_camera:",
        )
        if 'saturnv' in pages:
            st.page_link(pages['terra'], label="俯瞰図分析 で図を集めに行く", icon=":material/public:")
        return snapshots

    st.caption(
        f"現在 {len(snapshots)} 枚。ここに並んだ図が、両経路のレポートの材料（エビデンス）になります。"
        "番号（Evidence N）はレポート内の引用番号と対応します。"
    )

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
                    tab_names = [f"画像 {j+1}" for j in range(len(images))]
                    img_tabs = st.tabs(tab_names)
                    for j, tab in enumerate(img_tabs):
                        with tab:
                            try:
                                st.image(images[j], caption=f"Evidence {i+1}-{j+1}", use_container_width=True)
                            except Exception as e:
                                st.error(f"画像エラー: {e}")

                # フォールバック / 単一画像
                elif main_image:
                    st.image(main_image, caption=f"Evidence {i+1}", use_container_width=True)
                elif snap.get('image_error') or snap.get('image_errors'):
                    st.error(f"画像エラー: {snap.get('image_error') or snap.get('image_errors')}")
                    st.caption("※ ターミナルで `pip install -U kaleido` を実行してください。")
                else:
                    st.warning("（画像なし）")

                st.caption(f"取得元: {apollo_ui.module_display(snap.get('module', 'Unknown'))} ・ {snap.get('timestamp')}")
                with st.expander("メモ・データ"):
                    st.write(snap.get('description', ''))
                    ds_preview = snap.get('data_summary', '')
                    if isinstance(ds_preview, dict):
                        ds_preview = str(ds_preview)
                    st.code(str(ds_preview)[:200] + "...")

                if st.button(":material/delete: 削除", key=f"del_{snap['id']}_{i}"):
                    indices_to_remove.append(i)

                # ダウンロードボタン
                if snap.get('image'):
                    st.download_button(
                        label=":material/save: 画像を保存",
                        data=snap['image'],
                        file_name=f"Evidence {i+1}.png",
                        mime="image/png",
                        key=f"dl_{snap['id']}_{i}",
                    )

    if indices_to_remove:
        for i in sorted(indices_to_remove, reverse=True):
            del st.session_state['snapshots'][i]
        st.rerun()

    return snapshots


def _render_chatkit(snapshots, mission_objective):
    """経路A: チャットキット（プロンプト生成のみ・API呼び出しなし）"""
    with st.container(border=True):
        st.markdown("### :material/bolt: 経路A — チャットキット（所要5分・APIキー不要）")
        st.markdown(
            "普段お使いのチャットAI（ChatGPT / Claude / Gemini 等）に**貼るだけ**で、"
            "レポートの骨格が得られます。**APIキーは不要**です。"
        )

        # バリデーション（旧 VOYAGER Preview Prompts と同じ条件）
        missing_items = []
        if len(snapshots) == 0:
            missing_items.append("スナップショット（分析の証拠画像）")
        if len(mission_objective) <= 5:
            missing_items.append("レポートの目的（上の欄に6文字以上）")
        if missing_items:
            st.warning(":material/warning: キットを作るには次が必要です: " + " / ".join(missing_items))
            return

        # --- キット構築（内容が変わったときだけ再構築） ---
        cache_key = _chatkit_cache_key(mission_objective, snapshots)
        cache = st.session_state.get('_logbook_chatkit_cache')
        if not isinstance(cache, dict) or cache.get('key') != cache_key:
            kit_md = _build_chatkit_markdown(mission_objective, snapshots)
            cache = {
                'key': cache_key,
                'md': kit_md,
                'zip': _build_chatkit_zip(kit_md, snapshots),
            }
            st.session_state['_logbook_chatkit_cache'] = cache
        kit_md = cache['md']
        kit_zip = cache['zip']
        _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- ライト版: プロンプト全文（右上のコピーで1発） ---
        st.markdown(
            "**:material/content_paste: ライト版** — 下のプロンプトを**右上のコピーボタン**で丸ごとコピーして、"
            "チャットAIに貼り付けます。"
        )
        with st.container(height=380):
            st.code(kit_md, language="markdown")

        col_light, col_full = st.columns(2)
        with col_light:
            st.download_button(
                ":material/download: プロンプトを .md で保存（ライト版）",
                data=kit_md,
                file_name=f"mercator_chatkit_prompt_{_ts}.md",
                mime="text/markdown",
                key="logbook_kit_light_dl",
                use_container_width=True,
            )
            st.caption("プロンプトのみ。まずはこれだけでレポートの骨格が得られます。")
        with col_full:
            st.download_button(
                f":material/inventory_2: フル版キットを ZIP で保存（{len(kit_zip) / (1024 * 1024):.1f} MB）",
                data=kit_zip,
                file_name=f"mercator_chatkit_{_ts}.zip",
                mime="application/zip",
                key="logbook_kit_full_dl",
                use_container_width=True,
            )
            st.caption(
                "プロンプト + スナップショット画像一式。"
                "**画像も一緒に渡すとチャットAIの読解が良くなります。**"
            )

        # --- 貼り戻しビューア（任意） ---
        with st.expander(":material/description: 貼り戻しビューア — チャットAIの回答をきれいに表示（任意）"):
            st.caption(
                "チャットAIの回答（Markdown）を貼ると整形表示します。"
                "本文中の [[Evidence N]] タグの位置には、該当するスナップショット画像を差し込みます。"
            )
            pasted_md = st.text_area(
                "チャットAIの回答（Markdown）",
                height=220,
                key="logbook_pasteback_md",
                label_visibility="collapsed",
                placeholder="ここにチャットAIの回答（Markdown）を貼り付けてください…",
            )
            if pasted_md and pasted_md.strip():
                snap_lookup = {i + 1: s for i, s in enumerate(snapshots)}
                st.markdown(_REPORT_CSS, unsafe_allow_html=True)
                rendered_html = _render_report_html(pasted_md, snap_lookup)
                st.markdown(f'<div class="report-container">{rendered_html}</div>', unsafe_allow_html=True)
                st.download_button(
                    ":material/download: このレポートを .md で保存",
                    data=pasted_md,
                    file_name=f"mercator_report_{_ts}.md",
                    mime="text/markdown",
                    key="logbook_pasteback_dl",
                )


def _render_capcom_standby():
    """経路B: 記録セッション未開始のときの案内（旧 CAPCOM STANDBY 画面を縮約）"""
    pages = st.session_state.get('ap_pages', {})
    st.info(
        "記録セッションがまだ始まっていません。"
        "**前処理を完了すると自動で記録セッションが始まります。**"
        "Mission Control で前処理を実行してから、もう一度このページに戻ってきてください。",
        icon=":material/menu_book:",
    )
    if 'mission_control' in pages:
        st.page_link(pages['mission_control'], label="データ準備を開く", icon=":material/anchor:")

    with st.expander("仕組み（詳細）— 本格レポートができるまで"):
        st.markdown("""
記録セッション（内部名: CAPCOM）は、各分析ページの実行結果（データ・図・プロンプト）を
舞台裏で自動収集し、AI レポート執筆エージェントに橋渡しする仕組みです。

```
APOLLO（分析・可視化）
    ↓ 分析結果を構造化して自動記録（CAPCOM）
ZIP ダウンロード（選択した AI ツール用の資材が同梱済み）
    ↓ ユーザーが展開
Claude Code / Codex CLI / Antigravity IDE（レポート執筆）
    ↓ 自律生成（4フェーズ自動進行）または対話型（KATHERINE）
Typst で PDF 完成
```

**流れ**
1. Mission Control でデータを読み込み、前処理を実行（ここで記録セッションが自動開始）
2. 各分析ページを実行 → 結果が自動で蓄積・図は「:material/photo_camera: レポート用に保存」で収集
3. このページでレポートの目的・分析対象データの情報（任意）・進め方・使用ツールを設定
4. 「:material/inventory_2: レポート用ZIPを作成する」→ ZIP をダウンロード
5. ZIP を展開して選択ツールで開く → レポート生成
""")


def _render_capcom_route(snapshots):
    """経路B: 本格レポート（旧 CAPCOM ページの機能を移植・処理とキーは不変）"""
    with st.container(border=True):
        st.markdown("### :material/school: 経路B — 本格レポート（Claude Code 等・数十ページ品質）")
        st.markdown(
            "分析セッション一式を ZIP に書き出し、AI エージェント（Claude Code / Codex CLI / "
            "Antigravity IDE）に渡して**数十ページ品質**の本格レポートを生成します。"
            "進め方は **自律生成モード（おまかせ）** と **対話型レポート作成モード（KATHERINE）** "
            "の2つから選べます。"
        )
        st.info(
            "レポートの生成エンジンは **v9.1.0 から実績のあるレポートスキーマ（品質ゲート Check 1〜37 を含む）**を"
            "そのまま使っています。レポート本文の分析名は機能名（俯瞰図分析・動態分析 など）で出力されます。",
            icon=":material/info:",
        )

        if not capcom.is_active():
            _render_capcom_standby()
            return

        # --- セッションステータス（旧 CAPCOM ステータスパネルの要点を軽量表示） ---
        session_id = capcom.get_session_id()
        _tel = capcom.get_telemetry()
        st.caption(
            f":material/menu_book: 記録セッション ON ・ ID: `{session_id}` ・ "
            f"蓄積状況: 画像 {_tel['snapshots']} / プロンプト {_tel['prompts']} / データ {_tel['data']} ファイル"
        )

        # ==========================================================
        # 母集団メタ情報 — 入力は Mission Control（母集団設計タブ）に集約。
        # ここでは確定内容の確認と、飛行記録（分析条件）の一覧を出す。
        # ==========================================================
        st.markdown("#### 分析の前提 — 母集団設計と飛行記録")
        _meta = [
            ("母集団論理式の設計意図", st.session_state.get('capcom_query_intent', '')),
            ("母集団論理式", st.session_state.get('capcom_query_logic', '')),
            ("収録年情報", st.session_state.get('capcom_coverage_years', '')),
            ("使用した特許データベース名", st.session_state.get('capcom_database_name', '')),
        ]
        _filled = [(k, v) for k, v in _meta if str(v).strip()]
        if _filled:
            st.caption(
                f"母集団設計 {len(_filled)}/4 項目を記録済み。論理式は付録に、"
                "設計意図・収録年・データベース名は本文と付録の両方に反映されます。")
        else:
            st.info(
                "母集団設計（論理式・設計意図・収録年・データベース名）が未入力です。"
                "レポートの前提条件と付録が薄くなるため、Mission Control での記録を推奨します。",
                icon=":material/target:")
        _pages_meta = st.session_state.get('ap_pages', {})
        if "mission_control" in _pages_meta:
            st.page_link(_pages_meta["mission_control"], label="Mission Control で母集団設計を編集",
                         icon=":material/rocket_launch:")

        # 飛行記録: この ZIP に同梱される分析条件そのもの
        flight_recorder.render_card(title="この ZIP に同梱される分析条件")
        st.caption(
            "母集団の来歴と各モジュールの実行条件は flight_recorder.json として ZIP に同梱され、"
            "レポート付録の「分析条件」表に転記されます。")

        # --- 画像・スライドに関する指示（任意） ---
        capcom_image_directive = st.text_area(
            ":material/image: 画像・スライドに関する指示（任意）",
            value=st.session_state.get('capcom_image_directive', ''),
            placeholder="例: 表紙にクラスタ動態マップを使う / 権利化率マップはスライド必須 / CREWは媒介中心性の図を優先 など。",
            key="capcom_image_directive_input",
            height=90,
            help=(
                "どの画像をどこで使うか等の指示をAIに渡せます。context.json 経由でレポート/スライド生成に"
                "反映されます。レポート本文の図キャプションはAIが執筆するため要点記入は不要、"
                "スライドの要点はAIが本指示に沿って作成します。"
            ),
        )
        st.session_state['capcom_image_directive'] = capcom_image_directive
        # 収集済みスナップショット一覧（指示でタイトルを参照できるよう提示）
        _snaps_for_list = st.session_state.get('snapshots', [])
        if _snaps_for_list:
            with st.expander(f":material/image: 収集済みスナップショット {len(_snaps_for_list)} 枚（上の指示で名前を参照できます）"):
                for _s in _snaps_for_list:
                    st.caption(f"・{_s.get('module', '?')}: {_s.get('title', '(無題)')}")

        # ==========================================================
        # レポート生成の進め方（自律生成 / 対話型 KATHERINE）
        # 選択は context.json の report_mode と ZIP キャッシュキーに反映される。
        # ==========================================================
        st.markdown("#### :material/explore: レポート生成の進め方")
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
                ":material/chat: 対話型（KATHERINE）では、母集団タイプの判定・クロス分析の仮説と検証・結論の確定などの"
                "判断ポイントで AI が「提案・根拠・別の見方」を提示し、分析者が確定しながら進みます。"
                "各ポイントで「おまかせ」も選べます。所要時間は自律生成より長くなります（複数セッション推奨）。"
            )

        # ==========================================================
        # 使用ツール選択（複数選択可）
        # ==========================================================
        st.markdown("#### :material/handshake: 使用する AI ツール（複数選択可）")
        st.markdown(
            "レポート生成に使用する AI エージェントを選択してください。"
            "**ZIP に各エージェント用の資材が自動同梱**され、手動でのパッチ適用は不要です。"
        )

        default_tools = st.session_state.get('capcom_tools_selected', ["Claude Code（Anthropic）"])
        selected_tool_labels = st.multiselect(
            "使用する AI ツール",
            options=list(CAPCOM_TOOL_OPTIONS.keys()),
            default=default_tools,
            key="capcom_tools_selected_input",
            help=(
                "Claude Code は `capcom_schema/` 一式で動作（追加資材なし）。"
                "Codex / Antigravity を選択した場合、対応するオーバーレイ資材が ZIP 直下に展開済みで同梱されます。"
            ),
            label_visibility="collapsed",
        )
        st.session_state['capcom_tools_selected'] = selected_tool_labels
        selected_tool_keys = [CAPCOM_TOOL_OPTIONS[lbl] for lbl in selected_tool_labels]

        if not selected_tool_keys:
            st.warning(":material/warning: 最低1つの AI ツールを選択してください（デフォルト: Claude Code）。")
            # 最低限 Claude Code は動作するので、後段はフォールバック
            selected_tool_keys = ["claude_code"]

        if report_mode == 'interactive' and any(k != 'claude_code' for k in selected_tool_keys):
            st.info(
                ":material/info: 対話型レポート作成モード（KATHERINE）は **Claude Code で検証済み**です。"
                "Codex CLI / Antigravity IDE 向けの対話型対応（Codex 用補遺・Antigravity 用対話 Artifact 雛形）"
                "も ZIP に同梱されますが、両ツールでの実機検証は未了です。"
            )

        with st.expander(":material/info: 選択ツールごとの起動方法"):
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

        # ==========================================================
        # レポート用 ZIP の作成（旧: CAPCOM Export 実行 → ZIPダウンロード の2段を1つの流れに）
        # 内部処理（voyager/ 書き出し・ZIP 構築・キャッシュキー）は旧 CAPCOM と同一。
        # ==========================================================
        st.markdown("#### :material/inventory_2: レポート用 ZIP の作成")

        # ミッション目的の読み出しは旧仕様のまま（capcom 優先・voyager フォールバック。
        # 上の統一入力欄が両キーへ同値を書くため、実質どちらも同じ値になる）
        mission_objective = (
            st.session_state.get('capcom_mission_objective', '').strip()
            or st.session_state.get('voyager_objective', '').strip()
        )

        export_ready = len(snapshots) > 0 and len(mission_objective) > 5
        if not export_ready:
            if len(snapshots) == 0:
                st.warning(
                    ":material/warning: スナップショットが集まっていません。"
                    "各分析ページで「:material/photo_camera: レポート用に保存」を押して図を集めてください。"
                )
            elif len(mission_objective) <= 5:
                st.warning(":material/warning: ページ上部の「:material/crisis_alert: レポートの目的」を入力してください（6文字以上）。")

        if st.button(":material/inventory_2: レポート用ZIPを作成する", type="primary", key="capcom_export_btn",
                     disabled=not export_ready):
            try:
                def _get_period_str():
                    df = st.session_state.get('df_main')
                    if df is None or 'year' not in df.columns:
                        return "不明"
                    try:
                        years = df['year'].dropna().astype(int)
                        return f"{years.min()}-{years.max()}" if len(years) > 0 else "不明"
                    except Exception:
                        return "不明"

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
                        st.caption(f":material/warning: WARN Evidence画像パスの解決に失敗しました（要確認）: {e}")

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

                _abs_col = col_map.get('abstract')
                context = {
                    "session_id": st.session_state.get('capcom_session_id', ''),
                    "dataset": {
                        "total_patents": len(df_main) if df_main is not None else 0,
                        "period": _get_period_str(),
                        "column_mapping": col_map,
                        # 要約なしデータ（J-PlatPat の要約なしDL等）を AI 側が明示的に認識できるようにする。
                        # False のときは深掘り分析が発明の名称ベースになる（voyager_schema.md / data_notes.md 参照）
                        "abstract_available": bool(_abs_col and df_main is not None and _abs_col in df_main.columns),
                        # 分析に使っている分類コードの実効表記（'IPC' または 'FI'）。レポート本文の表記はこれに従う
                        "classification_code_label": utils.classification_code_label(col_map),
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

                st.success(
                    ":material/check_circle: レポート用データ一式を書き出しました"
                    "（voyager/mission.json + voyager/evidence/ + voyager/context.json）。"
                    "続けて下の「:material/inventory_2: ZIPをダウンロード」で保存してください。"
                )
            except Exception as e:
                st.error(f"レポート用ZIPの作成エラー: {e}")

        # --- ZIP 構築とダウンロード（書き出し済みのときだけ表示） ---
        if capcom.get_voyager_mission() is not None:
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
                    f":material/inventory_2: ZIPをダウンロード ({file_size_mb:.1f} MB)",
                    data=zip_bytes,
                    file_name=zip_filename,
                    mime="application/zip",
                    type="primary",
                    key="capcom_page_zip_download",
                )
                st.caption(
                    f"同梱ツール: {', '.join(selected_tool_labels) if selected_tool_labels else 'Claude Code（フォールバック）'}"
                    " ・ 展開後の手順は下の「準備チェックリスト」を参照"
                )
            else:
                st.warning("セッションフォルダが見つかりません。分析モジュールを実行してデータを蓄積してください。")

        # ==========================================================
        # 準備チェックリスト（ZIP を渡す前に）
        # ==========================================================
        _instruction_line = (
            "「`capcom_schema/interactive/SKILL_INTERACTIVE.md` を読んで対話型でレポートを作りましょう」と指示する"
            if report_mode == 'interactive'
            else "「CLAUDE.md を読んでレポートを書いて」と指示する（詳しくは上の「選択ツールごとの起動方法」）"
        )
        with st.expander(":material/check_circle: 準備チェックリスト — ZIP を渡す前に（初回のみ）"):
            st.markdown(f"""
**① 選択した AI ツールを手元の PC にインストール**
- Claude Code / Codex CLI / Antigravity IDE のうち、上で選んだものを各公式手順でインストールし、ターミナル等で起動できることを確認します。

**② Typst をインストール**（レポートの PDF 化に必要・分析だけなら不要）
- macOS: `brew install typst`
- Windows: `winget install --id Typst.Typst`（または `scoop install typst`）
- Linux: `snap install typst`（または cargo / GitHub リリース版）
- 確認: `typst --version`
- レポート完成後の PDF 化: `typst compile --root ".." reports/report.typ reports/report.pdf`
  （多くの場合 AI エージェントが自動で実行してくれます）

**③ ZIP を渡して指示する（3ステップ）**
1. ダウンロードした ZIP を任意の場所に展開する
2. 展開したフォルダを選択したツールで開く（Claude Code ならそのフォルダで `claude` を起動）
3. {_instruction_line}
   （Codex / Antigravity では ZIP 直下の AGENTS.md / GEMINI.md が優先されます）
""")

        # ==========================================================
        # セッション内ファイル一覧 (session_state 上の In-Memory store)
        # ==========================================================
        st.markdown("#### :material/folder: ZIP に入る内容（セッション内ファイル一覧）")
        st.caption("記録セッションがメモリ上に蓄積している内容です。ZIP にはこの一式が入ります。")

        _store = st.session_state.get('capcom_store', {})

        # data/
        data_files = sorted(_store.get('data', {}).keys())
        if data_files:
            with st.expander(f":material/folder_open: data/ ({len(data_files)} ファイル)", expanded=False):
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
            with st.expander(f":material/folder_open: snapshots/ ({len(snap_keys)} ファイル)"):
                for skey in snap_keys:
                    size_kb = len(_store['snapshots'][skey]['image']) / 1024
                    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                    st.caption(f"`snapshots/{skey}.png` — {size_str}")

        # prompts/
        prompt_keys = sorted(_store.get('prompts', {}).keys())
        if prompt_keys:
            with st.expander(f":material/folder_open: prompts/ ({len(prompt_keys)} ファイル)"):
                for pkey in prompt_keys:
                    size_kb = len(_store['prompts'][pkey].encode('utf-8')) / 1024
                    st.caption(f"`prompts/{pkey}` — {size_kb:.1f} KB")

        # voyager/
        _voy = _store.get('voyager', {})
        voy_count = (1 if _voy.get('mission') else 0) + (1 if _voy.get('context') else 0) + len(_voy.get('evidence', {}))
        if voy_count > 0:
            with st.expander(f":material/folder_open: voyager/ ({voy_count} ファイル)"):
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
            ":material/info: ZIP には `capcom_schema/`、`CLAUDE.md`、`.claude/skills/` も同梱されます "
            f"(リポジトリ資産){_tool_assets_str}"
        )

        # ==========================================================
        # トークン節約ガイド
        # ==========================================================
        with st.expander(":material/lightbulb: Claude Code でのトークン節約ガイド（ツァーリ・ボンバ対策）"):
            st.markdown("""
            レポート生成時の Claude Code トークン消費を最小化するためのガイドです。

            #### 問題の本質
            Claude Codeはメッセージを送るたびにコンテキスト全体をAPIに再送信します。
            SKILL.md + スキーマ + exemplar + 会話履歴が毎回「再印刷」されるため、
            **トークン消費の90%がキャッシュ読み取り**になりえます。

            #### 適用済みの対策
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


# ==================================================================
# --- ページ本体 ---
# ==================================================================

def render():
    apollo_ui.page_header(
        "レポート生成", "CAPCOM",
        "収集したスナップショットと分析条件をパッケージ化 — チャットAI用キット（経路A）／ AIエージェント連携ZIP（経路B）",
    )
    apollo_ui.require_data()

    # セッション消失の注意（旧 CAPCOM 冒頭警告の要点を維持）
    st.warning(
        "スナップショットと記録セッションの内容は**ブラウザを閉じると失われます**。"
        "図を集め終えたら、このページでプロンプトの保存（経路A）または ZIP のダウンロード（経路B）"
        "まで済ませてください。",
        icon=":material/warning:",
    )

    # 1. ミッション目的（両経路共通・1入力欄に統一）
    mission_objective = _render_mission_objective()

    st.markdown("---")

    # 2. 収集済みスナップショット
    snapshots = _render_snapshot_gallery()

    st.markdown("---")

    # 3. 経路A: チャットキット（かんたん）
    _render_chatkit(snapshots, mission_objective)

    st.markdown("")

    # 4. 経路B: 本格レポート（未開始時は案内に切替）
    _render_capcom_route(snapshots)
