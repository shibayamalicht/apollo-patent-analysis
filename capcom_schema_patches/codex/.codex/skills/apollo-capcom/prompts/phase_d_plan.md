# Phase D STOP-GATE — レポート構造・品質チェック計画確認テンプレ

SKILL.md §Phase D STOP-GATE の `ask_user_question` 呼び出し用雛形。

---

## 前提

- `capcom_schema/analysis/report_structure.md` を読了済み（章構成・deep_diveコピールール把握）
- `capcom_schema/analysis/quality_checklist.md` を読了済み（定量チェックコマンド・チェック項目把握）
- Phase C で最低4つの `reports/*_deep_dive.typ` が生成済み

---

## 使い方

1. 以下の Report Structure & Quality Plan をテキストで先に提示
2. 構成と品質基準について `ask_user_question` で承認を取る
3. ユーザーが承認したら `report.typ` 生成作業に進む

---

## Report Structure & Quality Plan（提示テキスト）

```
=== Phase D レポート構造・品質計画 ===

report.typ 章構成（標準10章構成）:
  1. 表紙 + エグゼクティブサマリー
  2. 背景・目的（Mission Objective）
  3. データセット概要（件数・期間・出願人統計）
  4. NEBULA deep_dive（全文コピー）
  5. Saturn V deep_dive（全文コピー）
  6. Explorer deep_dive（全文コピー）
  7. MEGA deep_dive（全文コピー）
  8. ATLAS deep_dive（全文コピー）
  9. CORE / CREW deep_dive（全文コピー、省略モジュールあり）
 10. クロスモジュール統合分析（80行以上、3パターン）
 11. 結論・推奨アクション
 付録: Evidence 一覧 / 代表特許リスト / Web調査出所一覧

deep_dive コピールール:
  - 全モジュールの deep_dive.typ を **全文そのままコピー**
  - 要約・圧縮・省略は一切禁止（SKILL.md §Phase D 前提）
  - 各 deep_dive 末尾に `#pagebreak()` を挿入

品質チェック項目（phase_d_gate.sh で自動判定）:
  Check 1: report.typ 行数 ≥ 800行
  Check 2: 代表特許引用 ≥ 15件（特開/特許第/WO20/JP20 パターン）
  Check 3: 4層モデル（解釈/洞察/提言）各5件以上
  Check 4: クロスモジュール統合分析 ≥ 80行
  Check 5: snapshot-figure ≥ 8枚
  Check 6: Web情報使用件数 ≤ 出所記載件数（出所漏れチェック）
  Check 7: 仮説件数 vs 検証件数のバランス（情報のみ）

完了判定:
  bash capcom_schema/scripts/phase_d_gate.sh
  → exit 0 が必須
```

---

## ask_user_question 呼び出しフォーマット

```yaml
question: "上記の Report Structure & Quality Plan で Phase D に進めて良いですか？"
header: "Phase D 計画"
multiSelect: false
options:
  - label: "この構成・品質基準で進める"
    description: "10章構成 + 7 品質チェックで report.typ を生成"
  - label: "章構成を変更したい"
    description: "Other で変更内容を指定（章の順序/省略/追加）"
  - label: "品質チェックを緩和したい"
    description: "Other で緩和項目を指定（ただし Check 1-6 は必須、警告付き）"
```

---

## 応答別の挙動

| ユーザー回答 | Agent の次アクション |
|---|---|
| 「この構成で進める」 | `report.typ` 生成開始 → `phase_d_gate.sh` 実行 → 結果報告 |
| 「章構成を変更したい」 | 変更内容を反映（ただし deep_dive コピーと結論章は必須）。変更内容を反映してから再度 `ask_user_question` で確認を取り直す |
| 「品質チェックを緩和したい」 | Check 1-6 は機械的判定で緩和不可と説明。「判定結果を見た上で修正するか/緩和して提出するかを決める」運用を提案 |

---

## 重要: Phase D ゲートの客観性

`phase_d_gate.sh` は `capcom_schema/analysis/quality_checklist.md` セクション1の定量チェックコマンドを統合実行する。**「自前のチェックで代替」は禁止**（再現性のないチェックは無効、SKILL.md §Phase D 注意）。

**FAIL 時の扱い**:
1. スクリプト出力をユーザーに **そのまま転記**（整形・要約禁止）
2. FAIL 項目（例: Check 1 不合格: 行数720行、要800行以上）を箇条書きで整理
3. 修正方針を提示し、ユーザーに「修正する / このまま提出する」を `ask_user_question` で確認
4. 修正後は再度 `phase_d_gate.sh` を実行

---

## レポート出力コマンド（Phase D 完了後）

```bash
# Typst → PDF
typst compile --root ".." reports/report.typ reports/report.pdf

# PPTX生成（python-pptx使用）
python3 reports/generate_pptx.py  # capcom_schema/templates/apollo_template.pptx を使用
```

> 🔑 **ヘルパーは import して使う（コピーしない）**: `generate_pptx.py` の冒頭で `import sys; sys.path.insert(0, "capcom_schema/templates"); from apollo_slides import *` し、`add_title_shape` / `add_kpi_slide` 等のヘルパーを呼ぶ。`slides_spec.md` は設計ガイド（いつ・どのヘルパーを・どんな主張骨格で使うか）として読み、ヘルパー実装を本文へ写経しない。
> 🔑 **PPTX はレポート（`reports/report.typ`）を土台に**作る（evidence の寄せ集め禁止）。レポート各章の主張→根拠→示唆を凝縮し章順に沿わせる。出所はモジュール名でなくデータにする。詳細は `capcom_schema/templates/slides_spec.md` §0.9 を熟読。

出力ファイル:
- `reports/report.pdf` — 本レポート（Typst PDF）
- `reports/presentation.pptx` — スライド（python-pptx、推奨25-40枚）

---

## 禁止事項

- **`phase_d_gate.sh` の実行をスキップするのは絶対禁止**
- 「全項目OKです」と自前判定してスクリプトを飛ばすのは禁止（SKILL.md §0 第3項）
- FAIL 時に「軽微なので許容範囲です」と AI の質的判断で進めるのは禁止
- deep_dive を「長いので要約します」と圧縮するのは禁止（`analysis/report_structure.md` セクション2 参照）
