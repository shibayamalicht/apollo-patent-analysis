---
name: phase-d-integration
description: >
  APOLLO CAPCOM Phase D: 統合レポート生成 + 品質検証。
  Report Plan の Artifact Review 承認 + Phase D 完了時に phase_d_gate.sh 実行 + PDF 出力。
---

# Phase D: Integration & Quality Gate 🛑 ARTIFACT + SCRIPTED GATE

本ワークフローは **Phase D のみ** を実行します。前提: Phase C 完了済み（`phase_c_gate.sh` exit 0、`reports/*_deep_dive.typ` 4ファイル以上生成済み）。

## 参照

- スキル本体の詳細: `.agent/skills/apollo-capcom/SKILL.md` § 4. Phase D
- レポート構造: `capcom_schema/analysis/report_structure.md`（章構成・deep_dive コピールール・付録テンプレート）
- 品質チェック: `capcom_schema/analysis/quality_checklist.md`（定量チェックコマンド・全項目）
- 特許引用書式: `capcom_schema/analysis/patent_citation.md` セクション2-3

## 実行ステップ

### 🛑 GATE: Report Structure & Quality Plan 承認

1. **リファレンス読了**:
   - `capcom_schema/analysis/report_structure.md`
   - `capcom_schema/analysis/quality_checklist.md`

2. **Artifact 更新**: `implementation_plan.md` § Report Structure & Quality Plan に記入
   - 章構成（10章 + 付録）
   - 品質チェック7項目の閾値

3. **🛑 Artifact Review GATE**: ユーザーが承認コメントを付けるまで待機

### 統合レポート生成

4. **ファイル存在確認**: `ls reports/*_deep_dive.typ` で Phase C 出力を確認（最低4ファイル必要）

5. **引用書式準備**: `capcom_schema/analysis/patent_citation.md` セクション2-3 を読み、特許番号の書式（特開/特許第/WO20/JP20）と出所表記を確認

6. **templates コピー**: `cp capcom_schema/templates/report_style.typ reports/` （既存ならスキップ）

7. **report.typ 生成**: 以下の構造で `reports/report.typ` を生成
   ```typst
   #import "report_style.typ": *
   #show: apollo-report.with(
     title: "〈Confirmed Title〉",
     subtitle: "〈Confirmed Subtitle〉",
     ...
   )

   // 1. 表紙 + エグゼクティブサマリー
   #exec-summary[...]

   // 2. 背景・目的
   ...

   // 3. データセット概要
   ...

   // 4. NEBULA deep_dive
   #include "nebula_deep_dive.typ"
   #pagebreak()

   // 5-9. Saturn V / Explorer / MEGA / ATLAS / CORE / CREW deep_dive
   ...

   // 10. クロスモジュール統合分析
   ...

   // 11. 結論・推奨アクション
   #conclusion-box(...)[...]

   // 付録
   ...
   ```

8. **deep_dive 全文コピー**: 全 deep_dive ファイルを **要約・圧縮・省略せず** そのまま `#include` または内容コピー
   - `capcom_schema/analysis/report_structure.md` セクション2 の deep_dive コピールール厳守

### 🛑 SCRIPTED GATE: Phase D 品質判定

9. **bash gate 実行**:
    ```bash
    bash capcom_schema/scripts/phase_d_gate.sh
    ```

10. **結果を walkthrough.md に転記**: `walkthrough.md` § Phase D Gate Result にスクリプト出力を **全文コピペ**

11. **判定**:
    - exit 0 → Gate 通過、PDF 出力へ
    - exit 1 → FAIL。不合格 Check 項目を修正して再実行
      - Check 1 (行数 < 800): deep_dive の全文コピー確認、結論章の充実化
      - Check 2 (代表特許 < 15): 各章の代表特許引用を追加
      - Check 4 (クロス統合 < 80行): クロスモジュール統合分析章を充実
      - Check 5 (snapshot-figure < 8): スナップショット画像を追加挿入
      - Check 6 (Web情報出所不足): 全Web情報に footnote で出所追加

### PDF 出力

12. **Typst コンパイル**:
    ```bash
    typst compile --root ".." reports/report.typ reports/report.pdf
    ```
    - 注意: `--root ".."` 必須。画像パスが `reports/` からの相対パスで解決される

13. **PDF 検証**:
    - ファイルサイズが0KB でないことを確認
    - ページ数が想定範囲内（30-100ページ）
    - 全ての `#snapshot-figure()` が正しく表示されているか
    - 日本語フォントが埋め込まれているか

14. **PPTX 出力（任意）**: 
    - 🔑 **デッキは完成レポート `reports/report.typ` を土台に**作る（evidence の寄せ集め禁止）。各章の主張→根拠→示唆を凝縮し章順に沿わせ、出所はモジュール名でなくデータにする
    - `capcom_schema/templates/apollo_template.pptx` を `reports/` にコピー
    - `capcom_schema/templates/slides_spec.md` を熟読（とくに §0.9「レポートを土台にする」）
    - python-pptx で 25-40 枚のスライド生成
    - 出力: `reports/presentation.pptx`

15. **walkthrough.md § Final Deliverables 記入**:
    - ファイル一覧、サイズ、ページ数
    - Typst コンパイル結果
    - PDF 検証項目 `[x]`

16. **task.md 更新**: Phase D 全チェックボックス `[x]` + 最終成果物チェック `[x]`

## 完了条件

- [ ] § Report Structure & Quality Plan にユーザー承認済み
- [ ] `reports/report.typ` 生成済み（全 deep_dive 全文コピー、クロス統合章、結論章、付録）
- [ ] `phase_d_gate.sh` exit 0 （全 Check 通過）
- [ ] `walkthrough.md` § Phase D Gate Result 転記済み
- [ ] `reports/report.pdf` 生成済み
- [ ] `walkthrough.md` § Final Deliverables 完成
- [ ] `task.md` Phase D 全チェック `[x]`

## 禁止事項

- **`phase_d_gate.sh` の実行スキップは禁止**
- **「全項目OKです」と自前判定してスクリプトを飛ばすのは禁止**
- **FAIL時に「軽微なので許容範囲です」と AI の質的判断で進むのは禁止**
- **deep_dive を圧縮・要約するのは禁止**（全文コピー）
- **ドル記号使用禁止**: `kpi-card` 内で `$` / `\$` は使えない。「ドル」「USD」と表記

## 次ステップ

Phase D 完了 = レポート完成。以下を最終確認してユーザーに報告:
- `reports/report.pdf`（本レポート）
- `reports/report.typ`（ソース）
- `reports/*_deep_dive.typ`（モジュール別）
- `reports/presentation.pptx`（任意）
- `task.md` / `implementation_plan.md` / `walkthrough.md`（Artifact 完成版）

Antigravity UI 上で、`task.md` の全チェック `[x]` と `walkthrough.md` の Gate PASSED 表示を確認して作業完了。
