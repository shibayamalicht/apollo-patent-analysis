---
name: capcom-interactive
description: >
  APOLLO CAPCOM 対話型レポート作成モード（KATHERINE）の起動点。
  正本2ファイル（SKILL_INTERACTIVE.md + dialogue_points.md）読了 → Phase 0 深さゲート →
  以降は 01〜05 の各ワークフローを対話オーバーレイ（CP-1〜8）付きで実行する。
  品質ゲート・成果物形式・トークン効率制約は自律生成モードと完全に同一。
---

# 対話型レポート作成モード（KATHERINE）🛑 ARTIFACT GATE

本ワークフローは **対話型モードの起動点** です。発動条件（二重ガード）:

- `voyager/context.json` の `report_mode` が `"interactive"`、または
- ユーザーが対話型での進行を明示（例: 「対話型でレポートを作りましょう」）

片方しか満たされない場合は、進行を始める前に Artifact Review（またはチャット）で意図を 1 回確認する
（**ユーザーの明示指示が `context.json` より優先**。確認結果は判断ログに記録）。

> ⚠️ 対話型モードは **Claude Code で検証済み**。Antigravity IDE では確定手段を Artifact Review 承認に
> 読み替えるベストエフォート対応（実機検証未了）。読替の正本は SKILL_INTERACTIVE.md 冒頭の読替表。

## 参照（進行の正本）

- **対話型の進行手順書（正本）**: `capcom_schema/interactive/SKILL_INTERACTIVE.md`
- **対話ポイント CP-1〜8 の実施定義（正本）**: `capcom_schema/interactive/dialogue_points.md`
- フェーズ・ゲートの正本: `capcom_schema/SKILL.md` + `.agent/skills/apollo-capcom/SKILL.md`（Antigravity 翻案）
- Artifact 翻案ルール: `GEMINI.md`「レポート生成の進行様式」
- CP-4/6/7 のローリング Artifact 雛形: `artifacts_templates/dialogue_review.md.tmpl`

## 実行ステップ

### STEP 1: 正本 2 ファイルの読了（省略不可）

1. `capcom_schema/interactive/SKILL_INTERACTIVE.md` を **全文読了**（§0-I 対話型の絶対遵守ルール
   I-1〜I-3 / 読替表 / 進行様式 / 判断ログ / 完了条件）
2. `capcom_schema/interactive/dialogue_points.md` を **全文読了**（4部構成テンプレート /
   確定の選択肢設計 / 委任の粒度・抱き合わせ禁止 / CP-1〜8 の実施定義）
3. `capcom_schema/analysis/terminology.md` を読了（対話の用語にも適用するため、対話型では
   セッション冒頭に読む — SKILL_INTERACTIVE.md §4）

### STEP 2: Review Policy 確認（対話型では "Request Review" 必須）

4. Antigravity の Review Policy が **"Request Review"** であることを確認する。
   **"Agent Decides" / "Always Proceed" を検出したらここで停止**し、ユーザーに設定変更を依頼する
   （対話ポイントの確定が Artifact Review 承認で成立する前提のため。`review_policy_recommendation.md` 参照）

### STEP 3: Artifact 初期化（自律生成モードの 3 ファイル + dialogue_review.md）

5. `cp artifacts_templates/task.md.tmpl task.md`（既にあればスキップ）
6. `cp artifacts_templates/implementation_plan.md.tmpl implementation_plan.md`（同上）
7. `cp artifacts_templates/walkthrough.md.tmpl walkthrough.md`（同上）
8. `cp artifacts_templates/dialogue_review.md.tmpl dialogue_review.md`（対話型のみ）
9. `cp capcom_schema/templates/carryover_template.md reports/_carryover.md`（既にあれば上書きしない）。
   判断ログに `[対話]` タグでセッション開始を記録する

### 🛑 STEP 4: Phase 0 — 対話の深さ選択ゲート

10. `implementation_plan.md` § Phase 0 Gate（対話の深さ選択）を記入し、
    `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` で **Artifact Review 承認を待機**:
    - **標準（推奨）**: 全 8 対話ポイント（CP-1〜CP-8）で提示 → 確定
    - **ライト**: CP-2（母集団タイプ）・CP-6（仮説と検証）・CP-8（結論の確定）のみ対話
    - 時間配分（1 セッション / 複数セッション〈推奨〉）も同時に確定
11. 確定値を判断ログに記録し、進め方を一言で説明する:
    「各判断ポイントで、私の案と根拠をお見せします。選ぶ・直す・おまかせ、どれでも構いません。
    疑問点はいつでも聞いてください。」

### STEP 5: Phase A〜D — 01〜05 を対話オーバーレイ付きで実行

12. 以降は既存ワークフロー **01 → 02 → 03 → 04 → 05 を順に実行**し、次の 2 点だけを重ねる
    （SKILL_INTERACTIVE.md §5。STOP-GATE・ワンショット統計・`_phase_a_decisions.json`・
    `select_representatives.py`・`phase_c_gate.sh` / `phase_d_gate.sh`・成果物・完了条件は一切変えない）:

    **(a) 8 つの対話ポイント（CP）を該当ステップに重ねる**（定義の正本: `dialogue_points.md`）

    | CP | 内容 | 重ねる先 | 記入媒体 | ライト時 |
    |---|---|---|---|---|
    | CP-1 | 検索式の読解 | 01: Gate A（query_logic 指定時のみ） | `implementation_plan.md` CP-1 対話欄 | 自律進行 |
    | CP-2 | 母集団タイプの判定 | 01: Gate C | `implementation_plan.md` CP-2 対話欄 | **対話** |
    | CP-3 | サブクエスチョン | 01: サブクエスチョン化（query_intent 指定時のみ） | `implementation_plan.md` CP-3 対話欄 | 自律進行 |
    | CP-4 | マップ読解 | 03: Evidence 精読 | `dialogue_review.md` CP-4 ブロック | 自律進行 |
    | CP-5 | クロスパターン選択 | 03: GATE 1 | `implementation_plan.md` CP-5 対話欄 | 自律進行 |
    | CP-6 | 仮説と検証 | 03: クロス分析実行 | `dialogue_review.md` CP-6 ブロック（2段） | **対話**（第1段のみ） |
    | CP-7 | 統合インサイト | 04: 各モジュール deep_dive の統合インサイト節の前 | `dialogue_review.md` CP-7 ブロック | 自律進行※ |
    | CP-8 | 結論の確定 + WARN トリアージ | 05: 結論章執筆前 + gate 実行後 | `implementation_plan.md` § CP-8 (a)(b) | **対話** |

    ※ ライトでも分析者が希望すれば 2〜3 モジュールに限定して CP-7 を実施してよい。

    **(b) 既存 STOP-GATE の提示に根拠説明を添える**: CP が重ならない既存ゲート（別冊確認・
    NEBULA 戦略・タイトル 3 案・Web 調査テーマ・Phase C/D 計画確認等）も、提示時に
    「なぜこの案か」の根拠を 1〜3 行添える。選択肢・確定手順は変えない。

13. **マーカー規約**: 確定必須ブロック = `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->`、
    応答不要の情報提示（CP-7 の委任時・ライト時の突合サマリ等）= `<!-- ANTIGRAVITY_INFO_ONLY -->`
    （INFO_ONLY では承認待ちにしない）。**抱き合わせ禁止**: 既存 STOP-GATE の確認と CP の
    進め方・委任の確認を 1 つのレビューにまとめない（`dialogue_points.md` §確定の選択肢設計）

14. **判断ログ**: 各 CP の確定ごとに `reports/_carryover.md` の判断ログへ `[対話]` タグ付きで
    1 行記録する（委任は `[対話→委任]`。形式の正本: `dialogue_points.md`）

15. **コンテキスト運用**: 対話型は往復が増えるため **「1 タスク = 1 フェーズ」分割を標準**とする
    （自律生成モードでは推奨、対話型では標準。セッション・チェックポイントの手順は
    スキル本体 `### 🔄 セッション・チェックポイント` と同一）

## 完了条件（SKILL_INTERACTIVE.md §8 — 自律生成モードの完了条件に追加）

- [ ] SKILL.md の各 Phase 完了条件・`phase_c_gate.sh` / `phase_d_gate.sh` 合格（同一基準・無改修）
- [ ] **発動した**全対話ポイント（CP-1 / CP-3 は発動条件を満たした場合のみ対象。ライト時は
      CP-2 / CP-6 / CP-8）の確定または委任が判断ログに記録されている。発動しなかった
      CP-1 / CP-3 は「未指定のためスキップ」を判断ログに 1 行記録
- [ ] Phase D gate の WARN について、分析者の直す/残す判断と理由（または `[対話→委任]` の記録。
      委任時は「WARN を AI 判断で処理した」旨を最終報告に明記）が判断ログと
      `implementation_plan.md` § CP-8 (b) に記録されている
- [ ] `reports/report.typ`・`reports/report_executive.typ`（生成した場合は PPTX も）に
      `SKILL_INTERACTIVE` / `dialogue_points` / `_analyst_notes` の文字列が **0 件**であることを
      grep で確認した（内部ガイドファイル名の露出は用語統一ルール違反として修正する）
- [ ] 対話サマリが `reports/_carryover.md` の申し送り節に追記されている

## 禁止事項（SKILL_INTERACTIVE.md §0-I）

- **根拠なしの提案・確定**: 4部構成（提案・根拠・別の見方・確認したいこと）を欠く提示は違反
- **実際のユーザー応答なしで対話ポイントを確定扱い**: Artifact Review を発行せず「承認された」と
  仮定して進むのは重大違反（「おまかせ」も実際に選ばれて初めて成立）
- **迎合**: 分析者の修正がデータ・正本基準と矛盾する場合は根拠を添えて指摘する。品質ゲートの
  機械基準に反する確定は分析者の同意があっても不可
