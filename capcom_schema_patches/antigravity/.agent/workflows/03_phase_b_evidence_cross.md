---
name: phase-b-evidence-cross
description: >
  APOLLO CAPCOM Phase B: エビデンス精読 + クロスモジュール分析。
  2つの Artifact ゲート（クロスパターン3つ選定 / Web調査可否）を経て本体作業を実施。
---

# Phase B: Evidence & Cross-Module 🛑 ARTIFACT GATE x2

本ワークフローは **Phase B のみ** を実行します。前提: Phase A-2 完了済み（Confirmed Title が確定していること）。

## 参照

- スキル本体の詳細: `.agent/skills/apollo-capcom/SKILL.md` § 4. Phase B
- クロス分析パターン: `capcom_schema/analysis/cross_module.md` (13種)
- Web調査ルール: `capcom_schema/analysis/data_notes.md` セクション3

## 実行ステップ

### 準備（リファレンス読了）
1. `capcom_schema/analysis/common_framework.md` 読了 → 4層分析モデル把握
2. `capcom_schema/analysis/data_notes.md` 読了 → 特許/NPL非対称性、Web調査ルール把握
3. `capcom_schema/analysis/cross_module.md` 読了 → 13種クロスパターン把握

### 🛑 GATE 1: クロスパターン3つ選定

4. **Artifact 更新**: `implementation_plan.md` § Cross-Module Pattern Selection に13パターン + Agent推奨3つ（★）を記載

   Agent推奨の選び方:
   - Mission Objective と最も直結するパターン 1つ
   - データの強み（例: Saturn V クラスタ多数）を活かすパターン 1つ
   - 不明点を解消できるパターン（例: NEBULA ギャップ）1つ

5. **🛑 Artifact Review GATE 1**: ユーザーが3つ選定するまで待機（☐ → ✅、または Other でカスタム）

6. **Confirmed Cross Patterns 記録**: ユーザー選択後、§ Confirmed Cross Patterns に3パターン記入

### 🛑 GATE 2: Web調査可否

7. **候補テーマ導出**: Mission Objective から Web 調査テーマ3-5件を導出

   例:
   - 〈出願人名〉の2024-2025年プレスリリース
   - 〈技術分野〉市場規模予測（調査会社レポート）
   - 政策動向（経産省/NEDO の戦略文書）
   - 【立場が competitor/buyer/supplier（関係性立場）かつ own_company 指定時は**必須**】自社「〈own_company〉」の事業・技術/特許ポジション・市場での立ち位置（対象企業との対比材料。buyer/supplier は対象ドメインのHHI・集中度を交渉力シグナルとして読む。SKILL.md STOP-GATE 2、`terminology.md` §6-2-B）

8. **Artifact 更新**: `implementation_plan.md` § Web Research Themes に候補記載

9. **🛑 Artifact Review GATE 2**: `task.md` § Phase B Gates の Web Research チェックボックス（「実施/スキップ/修正」）をユーザーが選択するまで待機

10. **Confirmed Web Research 記録**: ユーザー選択後、§ Confirmed Web Research に採用テーマと出所記録規約を記入

### 本体作業

11. **Evidence 優先順位付け**: 全Evidenceに Mission Objective への直結度で 1-3 のランク → § Evidence Inventory の優先度列を更新

12. **Evidence 精読**: 優先度の高い **5-8件** を1件ずつ読む
    - 各Evidence について: AIインサイト照合 / `map_reading.md` 該当セクション読解 / 代表特許抽出 / スナップショット画像パス記録

13. **代表特許取得**: `data/patents.csv` を pandas で条件検索、**最低15件** 取得（タイトル・出願人・公開番号）

14. **クロス分析実行**: 選定した3パターンを実行
    - 各パターンで 仮説 → 検証 → 結論 (15-20行)
    - 結果を § Phase B Output Summary に記録

15. **Web調査実行**（実施する場合のみ）: 
    - Agent 本体で Web 検索（または Browser subagent 経由）
    - 全情報に出所（URL・サイト名・取得日）を記録
    - footnote で `phase_d_gate.sh` の Check 6 に引っかからない書式にする

16. **task.md 更新**: Phase B の全チェックボックス `[x]`

## 完了条件

- [ ] リファレンス3ファイル読了済み
- [ ] § Confirmed Cross Patterns に3パターン記入済み（ユーザー承認経由）
- [ ] § Confirmed Web Research 決定済み（ユーザー承認経由）
- [ ] Evidence 5件以上精読済み
- [ ] 代表特許15件以上取得済み
- [ ] 3クロスパターンの仮説→検証→結論完了
- [ ] `task.md` Phase B 全チェック `[x]`

## 禁止事項

- **Agent 側で勝手に3パターン確定して進めるのは禁止**
- **「Web調査不要と判断しました」AI自己判断も禁止**
- ユーザーの明示的な承認（Artifact Review）を必ず経由

## 次ステップ

完了後、`.agent/workflows/04_phase_c_deep_dive.md` を呼び出して Phase C（モジュール別 deep_dive 生成）に進む。
