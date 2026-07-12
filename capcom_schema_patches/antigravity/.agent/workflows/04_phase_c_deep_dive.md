---
name: phase-c-deep-dive
description: >
  APOLLO CAPCOM Phase C: モジュール別 deep_dive 生成。
  Deep Dive Plan の Artifact Review 承認 + Phase C 完了時に phase_c_gate.sh 実行。
---

# Phase C: Module Deep Dive 🛑 ARTIFACT + SCRIPTED GATE

本ワークフローは **Phase C のみ** を実行します。前提: Phase B 完了済み（Confirmed Cross Patterns と Web調査方針確定）。

## 参照

- スキル本体の詳細: `.agent/skills/apollo-capcom/SKILL.md` § 4. Phase C
- Deep Dive ガイド: `capcom_schema/analysis/deep_dive_guide.md`（Step 0-6 の必須セクション・最低行数・ミクロ分析ルール）
- exemplars: `capcom_schema/exemplars/` 配下の7 Typst ファイル（nebula / saturnv / explorer / mega / atlas / core / crew）

## 実行ステップ

### 🛑 GATE: Deep Dive Plan 承認

1. **リファレンス読了**: `capcom_schema/analysis/deep_dive_guide.md` を読む

2. **Artifact 更新**: `implementation_plan.md` § Deep Dive Plan にテーブル形式で計画記入
   
   | Step | モジュール | 最低行数 | 必須セクション数 | 実行 |
   |---|---|---|---|---|
   | 0 | NEBULA | 120行 | 8 | ✅ |
   | 1 | Saturn V | 250行 | 13 | ✅ |
   | 2 | Explorer | 200行 | 11 | ✅ |
   | 3 | MEGA | 120行 | 9 | ✅ |
   | 4 | ATLAS | 120行 | 9 | ✅ |
   | 5 | CORE | 80行 | 7 | ✅ |
   | 6 | CREW | 60行 | 任意 | ✅ |

3. **🛑 Artifact Review GATE**: ユーザーが Plan に承認コメントを付けるまで待機

### 代表特許の決定的選定（Phase C の最初に1回だけ・v9 必須）

3-2. **選定スクリプトを実行する**（代表特許は「決まった手順」で選ぶ＝**自由選択の禁止**。モデルが自由に選ぶ限り、結論に都合の良い特許を選べてしまう＝つまみ食い）:

   ```bash
   python3 capcom_schema/scripts/select_representatives.py
   ```

   `reports/representative_patents.json` が生成される（Saturn V=SBERT重心近傍〔patiroha算出済み〕、MEGA=象限内で出願年昇順→番号昇順、Explorer/CREW=中心性上位ノード対応。全て決定的でタイブレークまで固定）

3-3. **ミクロ分析A で引用する特許番号は、この JSON に載った番号だけ**とする（`phase_d_gate.sh` Check 35 が、ミクロ分析節にリスト外の番号があれば警告する）。JSON の `cite_as` / `title` / `applicant` を使い、技術的意義と戦略的文脈を1-2文添えて引用する。プレースホルダ番号（`特開2023-XXXXXX` 等）・出願年から推測した捏造番号は自動不合格。ミクロ分析A **以外**の箇所（クロスモジュール検証等）で条件検索により特定の特許を根拠として引くことは引き続き可（その場合も実番号ルールは適用）。詳細: `capcom_schema/analysis/deep_dive_guide.md` ミクロ分析A

### Deep Dive 生成（Step 0 → 6 順序）

各 Step で:
- 対応する exemplar を読む（例: Step 1 なら `capcom_schema/exemplars/saturnv_exemplar.typ`）
- `reports/<module>_deep_dive.typ` を生成
- 必須: ミクロ分析A（代表特許15件以上・**引用は `reports/representative_patents.json` の番号のみ**）、B（出願人5社以上、各5行以上）、`#snapshot-figure()` 1枚以上
- **走査層（v9）**: 各番号セクション冒頭に `#point-lead[...]` を1個、各章末に `#chapter-summary[...]` を置く（散文の代替にしない。gate Check 25/26）
- 🔄 **モジュール完了ごとにセッション・チェックポイント**: deep_dive を1本生成したら、`reports/_carryover.md` を更新（完了モジュール・次モジュール・直近の固有事実）したうえで、ユーザーに「**新タスクに切り替えますか？**（続行／切替／`/compact`）」と提案して一旦停止する。Phase C は最も枯渇しやすいので**1モジュール=1タスク**を基本とし、切替時は新タスクが `_carryover.md`＋`ls reports/` で次モジュールから自動再開。詳細はスキル本体 `### 🔄 セッション・チェックポイント`

4. **Step 0 NEBULA**: `reports/nebula_deep_dive.typ` 生成（120行以上 / 8セクション）
   - `capcom_schema/exemplars/nebula_exemplar.typ` を参考
   - 環境分析（特許/学術/ニュース）のギャップ可視化

5. **Step 1 Saturn V**: `reports/saturnv_deep_dive.typ` 生成（250行以上 / 13セクション）
   - `capcom_schema/exemplars/saturnv_exemplar.typ` を参考
   - ノイズ分析は `capcom_schema/analysis/noise_analysis.md` 参照
   - 13セクション必須、巨大

6. **Step 2 Explorer**: `reports/explorer_deep_dive.typ` 生成（200行以上 / 11セクション）
   - `capcom_schema/exemplars/explorer_exemplar.typ` を参考
   - 共起ネットワーク、急上昇キーワード、トルネードチャート

7. **Step 3 MEGA**: `reports/mega_deep_dive.typ` 生成（120行以上 / 9セクション）
   - `capcom_schema/exemplars/mega_exemplar.typ` を参考
   - 4象限分析、CAGR、寡占度

8. **Step 4 ATLAS**: `reports/atlas_deep_dive.typ` 生成（120行以上 / 9セクション）
   - `capcom_schema/exemplars/atlas_exemplar.typ` を参考
   - 時系列、ランキング、ライフサイクル

9. **Step 5 CORE**: `reports/core_deep_dive.typ` 生成（80行以上 / 7セクション）
   - `capcom_schema/exemplars/core_exemplar.typ` を参考
   - ルールベース分類結果の解釈。ミクロ分析Aは `data/patents.csv` を重点セルで絞り込み実特許を引用

10. **Step 6 CREW**: `reports/crew_deep_dive.typ` 生成（60行以上、最低行数指定なし）
    - `capcom_schema/exemplars/crew_exemplar.typ` を参考（指標ごとに節立て）
    - ネットワーク分析（媒介中心性・コミュニティ）。色分け指標ごとの解釈、ミクロ分析は `data/patents.csv` を `inventor_main`/`applicant_main` で絞り込み実特許を引用

### 🛑 SCRIPTED GATE: Phase C 完了判定

11. **bash gate 実行**:
    ```bash
    bash capcom_schema/scripts/phase_c_gate.sh
    ```

12. **結果を walkthrough.md に転記**: `walkthrough.md` § Phase C Gate Result にスクリプト出力を **全文コピペ**（要約・加工禁止）

13. **判定**:
    - exit 0 → Gate 通過、Phase D に進む
    - exit 1 → FAIL。不足モジュールを補強してから再度 step 11 を実行

14. **task.md 更新**: Phase C 全チェックボックス `[x]`

## 完了条件

- [ ] § Deep Dive Plan にユーザー承認済み
- [ ] `reports/representative_patents.json` 生成済み（決定的選定・Phase C 冒頭に1回）
- [ ] 7モジュール分の `reports/*_deep_dive.typ` 生成済み
- [ ] `phase_c_gate.sh` exit 0
- [ ] `walkthrough.md` § Phase C Gate Result に全文転記済み
- [ ] `task.md` Phase C 全チェック `[x]`

## 禁止事項

- **`phase_c_gate.sh` の実行スキップは禁止**
- **「実質的にOK」と質的判断で進むのは禁止**
- deep_dive を exemplar なしで書き始めるのは禁止（Step 0-6 すべて exemplar あり）
- **Agent tool / サブエージェント起動禁止**（Browser subagent も Phase C では使わない）

## 次ステップ

完了後、`.agent/workflows/05_phase_d_integration.md` を呼び出して Phase D（統合レポート生成）に進む。
