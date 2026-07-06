---
name: phase-a-data-intake
description: >
  APOLLO CAPCOM Phase A: ミッション理解 + データ精読 + 母集団設計読解。
  Mission Objective / Dataset Context / Evidence Inventory / Key AI Insights /
  Population Meta / query_logic 構造化読解 / query_intent 3 点整理 /
  サブクエスチョン / 意図↔論理整合性 / データ Level 2 逆読み /
  母集団タイプ判定 / NEBULA 戦略判定 を implementation_plan.md に埋める。
  Artifact Review ゲート複数あり。
---

# Phase A: Mission & Data Intake + 母集団設計読解（4 Artifact Review Gate）

本ワークフローは **Phase A のみ** を実行します。前提として `apollo-capcom` スキルの初期化（task.md / implementation_plan.md / walkthrough.md 生成）が完了していること。

## 参照

- スキル本体の詳細: `.agent/skills/apollo-capcom/SKILL.md` § 4. Phase A
- 母集団タイプ分類: `capcom_schema/analysis/population_type_metrics.md`
- 検索式読解: `capcom_schema/analysis/query_logic_reading.md`
- 用語統一: `capcom_schema/analysis/terminology.md` §1-6
- 絶対遵守ルール: `.agent/skills/apollo-capcom/SKILL.md` § 0

## 実行ステップ

### STEP 0: 用語統一と母集団メタ情報の読み込み

1. **terminology.md 読了** (`capcom_schema/analysis/terminology.md`): §1-6 すべて
   - §1: 内部識別子の露出禁止
   - §4: Mission Objective ベタ貼り禁止
   - §5: 母集団メタ情報の扱い
   - §6: スコープ限定ルール（本母集団 vs 業界全体）
   - §5-A-2: サブクエスチョン化ルール

2. **population_meta 4 フィールド確認**: `voyager/context.json` の以下を `implementation_plan.md` § Population Meta に転記
   - `query_intent` / `query_logic` / `coverage_years` / `database_name`
   - **`database_name` 未指定 → 「提供された特許データセット」と汎用表記**（J-PlatPat 等を勝手に補わない）

### 🛑 ARTIFACT GATE: 経営層向け要約版（別冊）の生成確認

3. `implementation_plan.md` § Executive Summary Edition Decision に 3 択を記入:
   ```markdown
   - [ ] ✅ 両方生成（本編 + 別冊）
   - [ ] 📘 本編のみ
   - [ ] ❓ 相談したい
   ```
4. `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を直前に配置し、Artifact Review でユーザー承認を待機
5. 「両方生成」選択時 → **別冊生成フラグ = ON**、Phase D で `reports/report_executive.typ` を生成

### 🛑 ARTIFACT GATE A: query_logic 構造化読解（`query_logic` 指定時のみ）

6. `capcom_schema/analysis/query_logic_reading.md` を読了（7 DB 構文リファレンス）
7. 4 ステップを実施し § query_logic Reading に記入:
   - **Step 1 DB 識別**: `database_name` + 構文特徴（`/TX` → J-PlatPat、`HTX=` → JP-NET、`$Wn` → PatSnap 等）
   - **Step 2 構文分解**: AND/OR/NOT で節分割 + カテゴリ分類（分類 / キーワード / 出願人 / 日付 / その他）
   - **Step 3 意図推定**: 各条件の意図（例: `NOT A23*/IC` → 食品分野除外）
   - **Step 4 Artifact Review**: `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` 配置、ユーザー承認待ち

### 🛑 ARTIFACT GATE: `query_intent` 3 点整理（`query_intent` 指定時のみ）

8. § query_intent 3-Point Summary に記入:
   ```markdown
   - 分析目的 (1 行): ...
   - 母集団の輪郭 (2-3 行): ...
   - 分析の視座 (1-2 行): ...
   ```
9. `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、Artifact Review でユーザー合意を待機
10. **ベタ貼り禁止**: 原文のままレポートに書かず、Phase B 以降で「分析の視座」として内在化

### 🛑 ARTIFACT GATE: サブクエスチョン化（`query_intent` 指定時のみ）

11. § Sub-Questions に 3-5 個の観点を箇条書きで起草（各観点にキーワード 1-3 個を付記）
12. `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、Artifact Review でユーザー確認
13. 確定後、`reports/_phase_a_decisions.json` の `sub_questions` に保存
14. **⚠️ 絶対制約**: サブクエスチョンは**内部メモ専用**。レポート本文に「Q1 / A1 / SQ1 / 問い 1」等の記号・形式は禁止

### 🛑 ARTIFACT GATE B: 意図 ↔ 論理 整合性検査（両方指定時のみ）

15. `query_logic_reading.md` §4 の 8 項目（技術領域 / 用途 / 対象期間 / 地域 / 出願人絞り込み / 除外条件 / 公報種別 / 分類階層）で対比
16. 乖離を 3 段階分類（🔴 Critical / 🟡 Warning / 🔵 Info）して § Intent-Logic Divergences に記入
17. Critical / Warning には **具体的な改善提案**を添える（例: 「末尾に `* NOT (A23*/IC)` を追加すると意図に沿う」）
18. `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` 配置、対処方針選択（[A] 修正して再抽出 / [B] このまま進めて「範囲と限界」章で明記 / [C] 無視 / ✅ 乖離なし）

### データ精読（基本 8 ステップ）

19. **Mission 読込**: `voyager/mission.json` → § Mission Objective に転記
20. **Context 読込**: `voyager/context.json` のデータセットメタ情報と `capcom_tools` → § Dataset Context に記録
21. **Evidence 走査**: `evidence_list` 全件を § Evidence Inventory に一覧化（Evidence ID / Module / Title / 優先度 / 関連 Snapshot）
22. **Snapshots 取得**: `ls snapshots/` でファイル一覧取得、Evidence と紐付け
23. **patents.csv 把握**:
    - `head -5` + `wc -l` でカラム構成・件数確認
    - pandas で出願人上位 10 社・クラスタ別件数・年別件数 → § Dataset Context に記録
    - **`print(df)` / `cat` は禁止**
24. **モジュール JSON 抽出**: `data/` 以下の全 JSON から主要数値抽出（クラスタ数・ノイズ率・HHI/Entropy/Gini・CAGR 等）
25. **AI Insights 読了**: `prompts/` から最低 3-5 件選定、1 件ずつ読む:
    - 50KB 以下 → 全量読み込み可
    - 50KB 超 → 部分読み込み（grep）
    - `saturn_drill_insight.md` (最大 220KB) と `crew_network_insight.md` (最大 400KB) は必ず部分読み込み
    - 要点を § Key AI Insights に記録

### 🛑 ARTIFACT GATE C: データ側からの母集団実態確認 + 母集団タイプ判定（必須）

26. **Level 2 項目算出** → § Data Level 2 Reverse-Read に記入:
    - 総件数・対象期間・使用 DB / 上位 10 出願人・シェア / 主要 IPC/FI 上位 10 / 出願年分布 / 出願人集中度 HHI / 国・地域分布
27. **自動偏り警告**: 上位 1 社 30% 超 / 上位 1 IPC 40% 超 / 直近 2 年 50% 超集中 / HHI > 0.25 / 特定国 95% 超 を検出
28. **母集団タイプ判定** (`population_type_metrics.md` §4-2) → § Population Type に記入:
    - A 業界全体 / A' 技術領域 / B 競合限定 / C 単一企業 / D 特定製品・技術テーマ
    - タイプ C では出願人 HHI 算出無意味（HHI=1.0）
    - タイプ B/C/D では「市場集中」「業界シェア」等の市場・業界解釈は禁止（`population_type_metrics.md` §3）
29. `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` 配置、Artifact Review で「データ実態 + タイプ推定」を確認（✅ 進める / ✏️ タイプ変更 / 💬 偏りあり・範囲と限界に明記 / 🔙 再抽出）

### 🛑 ARTIFACT GATE D: NEBULA 戦略判定（必須）

30. `data/nebula_*.json` の存在確認
    - 存在すれば `nebula_strategy.selected_mode = "execute"` を自動決定
31. 存在しない場合、§ NEBULA Strategy に 2 択を記入:
    - **🌐 Web 補完モード**: Phase B で 4 カテゴリ必須カバー（市場規模 / 政策・規制 / 学術動向 / 主要企業動向）→ 「外部環境分析（Web 調査）」章、`#footnote[...]` で出所明記
    - **📘 省略モード**: NEBULA 章なし + 「本分析の範囲と限界」章で「特許情報のみ対象」と注記
32. `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` 配置、Artifact Review でモード選択を待機

### 永続化

33. **`reports/_phase_a_decisions.json` への保存**:
    ```json
    {
      "phase_a_version": "v9.0",
      "phase_a_completed_at": "{ISO8601}",
      "population_type": { "code": "...", "label": "...", "confirmed_by_user": true },
      "query_intent_summary": { "purpose": "...", "population_outline": "...", "perspective": "..." },
      "sub_questions": [{"id": "SQ1", "content": "...", "keywords": [...]}],
      "query_logic_structure": { "database": "...", "main_conditions": [...] },
      "intent_logic_divergences": [{"severity": "...", "description": "...", "user_decision": "..."}],
      "data_level2_warnings": [{"category": "...", "value": "...", "note": "..."}],
      "forbidden_expressions": ["市場は寡占", "業界シェア", ...],
      "nebula_strategy": { "selected_mode": "execute|web_compensation|omit", ... },
      "user_notes": "..."
    }
    ```

34. **task.md 更新**: Phase A 全チェックボックスを `[x]` に更新

## 完了条件

- [ ] `implementation_plan.md` § Mission Objective 完成
- [ ] `implementation_plan.md` § Dataset Context 完成（population_meta 4 フィールド含む）
- [ ] `implementation_plan.md` § Evidence Inventory 完成
- [ ] `implementation_plan.md` § Key AI Insights 完成
- [ ] `implementation_plan.md` § Executive Summary Edition Decision 確定
- [ ] `implementation_plan.md` § query_logic Reading（指定時）Artifact Review 承認済
- [ ] `implementation_plan.md` § query_intent 3-Point Summary（指定時）Artifact Review 承認済
- [ ] `implementation_plan.md` § Sub-Questions（指定時）Artifact Review 承認済
- [ ] `implementation_plan.md` § Intent-Logic Divergences（両方指定時）Artifact Review 承認済
- [ ] `implementation_plan.md` § Data Level 2 Reverse-Read 完成、Artifact Review 承認済
- [ ] `implementation_plan.md` § Population Type 確定
- [ ] `implementation_plan.md` § NEBULA Strategy 確定、Artifact Review 承認済
- [ ] `reports/_phase_a_decisions.json` 永続化済
- [ ] `task.md` Phase A 全チェック `[x]`

## 次ステップ

完了後、`.agent/workflows/02_phase_a2_title_selection.md` を呼び出して Phase A-2（タイトル選定ゲート）に進む。
