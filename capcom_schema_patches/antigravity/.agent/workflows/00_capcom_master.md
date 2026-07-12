---
name: capcom-master
description: >
  APOLLO CAPCOM レポート生成の全 Phase（A → A-2 → B → C → D）を順次実行するマスターワークフロー。
  Artifact 駆動で task.md / implementation_plan.md / walkthrough.md を更新しながら進行する。
---

# CAPCOM Master Workflow

本ワークフローは `apollo-capcom` スキルの **全 Phase 実行** 用マスターです。通常はチャットで `apollo-capcom` スキルを起動すれば自動で実行されますが、Antigravity の workflows UI から明示的に起動したい場合は本ファイルを呼び出してください。

## 進行様式の分岐（最初に確認）

**`voyager/context.json` の `report_mode` を必ず確認する**:

- `"autonomous"`（既定・未指定含む）: 本ワークフローの実行手順（自律生成モード）でそのまま進行する
- `"interactive"`（またはユーザーが対話型を明示した場合）: **本ワークフローではなく `.agent/workflows/06_interactive.md`（対話型 KATHERINE の起動点）から開始**する。正本 2 ファイル（`capcom_schema/interactive/SKILL_INTERACTIVE.md` + `dialogue_points.md`）読了 → Phase 0 深さゲート → 下記ステップ 2〜6 を対話ポイント CP-1〜8 のオーバーレイ付きで実行する。品質ゲート・成果物形式・トークン効率制約は自律生成モードと完全に同一（対話型では Review Policy = "Request Review" が必須）

## 実行手順

1. **初期化**: Artifact 3ファイル + フェーズ間引き継ぎ日誌を生成
   - `cp artifacts_templates/task.md.tmpl task.md`
   - `cp artifacts_templates/implementation_plan.md.tmpl implementation_plan.md`
   - `cp artifacts_templates/walkthrough.md.tmpl walkthrough.md`
   - `cp capcom_schema/templates/carryover_template.md reports/_carryover.md`（既にあれば上書きしない）
   - 以降、各フェーズ完了時・`/compact`/タスク中断前に `reports/_carryover.md` を更新（Web は1件ごとに即 出所台帳へ）。詳細は `capcom_schema/SKILL.md` §フェーズ間引き継ぎ日誌

> 🔄 **セッション・チェックポイント（各フェーズ境界で必須・枯渇/クォータの予防）**: 下記ステップ 2〜6 の **各フェーズ完了後、および Phase C（ステップ5）の各モジュール（deep_dive 1本）完了ごと** に、ゲート/Artifact 通過＋`reports/_carryover.md` 更新を確認したうえで、ユーザーに「**新タスクに切り替えますか？**（続行／切替／`/compact`）」と提案して **一旦停止** する。「枯渇しそうな時だけ」ではなく **予防として各境界で必ず出す**。切替時は新タスクが `_carryover.md`＋Artifact から自動再開（→ 後述「実行中断時の復帰」）。詳細はスキル本体 `.agent/skills/apollo-capcom/SKILL.md` の `### 🔄 セッション・チェックポイント`。

2. **Phase A 実行**: `.agent/workflows/01_phase_a_data_intake.md` を呼び出し
   - 🛑 Artifact Review ゲート: **以下の一式（構成・個数の正本はスキル本体 Phase A 節。全て指定時は最大 7）**
     - 別冊（経営層向け要約版）生成確認
     - STOP-GATE A: query_logic 構造化読解（指定時）
     - query_intent 3 点整理（指定時）
     - サブクエスチョン化（指定時）
     - STOP-GATE B: 意図 ↔ 論理 整合性検査（両方指定時）
     - STOP-GATE C: データ Level 2 逆読み + 母集団タイプ判定
     - STOP-GATE D: NEBULA 戦略判定（execute / web_compensation / omit）
   - 成果物: `reports/_phase_a_decisions.json` 永続化

3. **Phase A-2 実行**: `.agent/workflows/02_phase_a2_title_selection.md` を呼び出し
   - 🛑 Artifact Review ゲート: タイトル確定までユーザー承認待ち

4. **Phase B 実行**: `.agent/workflows/03_phase_b_evidence_cross.md` を呼び出し
   - 🛑 Artifact Review ゲート x 2: クロスパターン選定 + Web 調査可否
   - Web 調査は `nebula_strategy.selected_mode` に応じて分岐（`web_compensation` 時は 4 カテゴリ必須）

5. **Phase C 実行**: `.agent/workflows/04_phase_c_deep_dive.md` を呼び出し
   - 🛑 Artifact Review ゲート: Deep Dive Plan 承認
   - **決定的選定（Phase C 冒頭・1回だけ）**: `python3 capcom_schema/scripts/select_representatives.py` → `reports/representative_patents.json` 生成。ミクロ分析A の引用はこの JSON の番号のみ（gate Check 35）
   - 🛑 Scripted Gate: `bash capcom_schema/scripts/phase_c_gate.sh`
   - Step 0（NEBULA deep_dive）は `nebula_strategy` で分岐: execute / web_compensation（外部環境分析章として）/ omit（省略）

6. **Phase D 実行**: `.agent/workflows/05_phase_d_integration.md` を呼び出し
   - 🛑 Artifact Review ゲート: Report Plan 承認
   - 🛑 Scripted Gate: `bash capcom_schema/scripts/phase_d_gate.sh` — **Check 1〜37 で自動検証（項目の正本は gate スクリプト本体。以下は代表例のみ）**
     - Check 1: 内容量（**非空白文字数 45000字以上**・行数は参考）/ Check 2: 代表特許15件 / Check 4: クロス分析（120行以上＝5パターン）/ Check 5: snapshot 8枚
     - Check 8: 用語統一（内部識別子・工程ナレーション節 8e・技法名露出 8f）/ Check 10〜13: スコープ限定・母集団タイプ別禁止表現・設計意図の一貫性・NEBULA 戦略検証
     - Check 19/19a: 反復・水増し検出＋本文スクリプト生成 / Check 25/26: 走査層（`#point-lead` / `#chapter-summary`）
     - Check 30/31: 結論の検証（別解釈＋決め手）・結論の前提と見直しのサイン / Check 33/34: 仮説検証サマリ・提言の検証結論参照
     - Check 35: ミクロ分析A の引用が決定的選定リスト（`reports/representative_patents.json`）由来のみ
   - 別冊フラグ ON 時は `reports/report_executive.typ` も生成

7. **最終成果物確認**: `reports/report.pdf` (+ 別冊 `report_executive.pdf`) + `reports/*_deep_dive.typ` + `walkthrough.md` 完成

## 前提条件

- 本ワークフローは `session_YYYYMMDD_HHMMSS/` を cwd として実行すること
- Antigravity の Review Policy が **"Request Review"** または **"Agent Decides"** に設定されていること
- `capcom_schema/` 配下の全ファイルが存在すること（ZIP展開時に自動配置）
- `voyager/mission.json` が存在すること（APOLLOの CAPCOM Export 実行済み）

## 実行中断時の復帰

いずれかの Phase で中断した場合（別タスク・コンテキスト枯渇・`/compact` 含む）、該当 Phase の workflow を個別に呼び出して再開できます：
- **まず `reports/_carryover.md`（引き継ぎ日誌）を通読** → STATUS で現在地・次アクション、直近フェーズ節・WEB出所台帳・申し送りで分析の記憶を復元
- `ls reports/` と `task.md` のチェックボックス状態から到達 Phase を判定
- `reports/_phase_a_decisions.json` で母集団タイプ等の確定値を復元 → Artifact/brain を再ロード
- 不完全な Phase の workflow を個別実行（Phase C/D は最後に成功した `walkthrough.md` のゲート結果から復帰判断）

## 参照

- スキル本体: `.agent/skills/apollo-capcom/SKILL.md`
- 対話型（KATHERINE）起動点: `.agent/workflows/06_interactive.md`（正本: `capcom_schema/interactive/SKILL_INTERACTIVE.md` + `dialogue_points.md`）
- Artifact 雛形: `artifacts_templates/`
- 共通ルール: `GEMINI.md` / `AGENTS.md`
- 品質ゲート: `capcom_schema/scripts/phase_c_gate.sh`, `phase_d_gate.sh`
