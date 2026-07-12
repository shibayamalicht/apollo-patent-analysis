# Phase C STOP-GATE — Deep Dive 計画確認テンプレ

SKILL.md §Phase C STOP-GATE の `ask_user_question` 呼び出し用雛形。

---

## 前提

`capcom_schema/analysis/deep_dive_guide.md` を読了済みであること。各モジュールの必須セクション数・最低行数を把握した上でこのテンプレを使う。

---

## 使い方

1. 以下の Deep Dive Plan 一覧（クイックリファレンス）をテキストで先に提示
2. 一覧の妥当性について `ask_user_question` で承認を取る
3. ユーザーが承認したら Phase C 本体作業に進む

---

## Deep Dive Plan クイックリファレンス（提示テキスト）

```
=== Phase C Deep Dive 実行計画 ===

Step 順序:
  Step 0: NEBULA
  Step 1: Saturn V
  Step 2: Explorer
  Step 3: MEGA
  Step 4: ATLAS
  Step 5: CORE
  Step 6: CREW

最低行数/必須セクション数:
  NEBULA    : 120行 / 8セクション
  Saturn V  : 250行 / 13セクション
  Explorer  : 200行 / 11セクション
  MEGA      : 120行 / 9セクション
  ATLAS     : 120行 / 9セクション
  CORE      :  80行 / 7セクション
  CREW      :  60行 / （最低行数指定なし、ファイル存在のみ）

各モジュール必須内容:
  - ミクロ分析A: 代表特許15件以上（公開番号・タイトル・出願人）
  - ミクロ分析B: 出願人5社以上、各5行以上の分析
  - スナップショット画像 1枚以上を #snapshot-figure() で挿入
  - Phase B で選定したクロスパターン（5つ以上）に該当するモジュールは特に厚く書く

完了判定:
  bash capcom_schema/scripts/phase_c_gate.sh
  → exit 0 でなければ Phase D 開始不可（不足モジュール補強後再実行）
```

---

## ask_user_question 呼び出しフォーマット

```yaml
question: "上記の Deep Dive Plan で Phase C に進めて良いですか？"
header: "Phase C 計画"
multiSelect: false
options:
  - label: "このPlanで進める"
    description: "7モジュール全て、上記の行数/セクション要件で生成する"
  - label: "一部のモジュールを省略したい"
    description: "Other で省略モジュールを指定（CREW のみ省略等）"
  - label: "最低行数を調整したい"
    description: "Other で調整内容を指定（例: CORE を 80 → 50行に緩和）"
```

---

## 応答別の挙動

| ユーザー回答 | Agent の次アクション |
|---|---|
| 「このPlanで進める」 | Phase C 本体作業開始。Step 0 → Step 6 の順で実行 |
| 「一部を省略したい」 | 省略モジュールを確認し、**ただし** Saturn V / Explorer / MEGA / ATLAS の4つは `phase_c_gate.sh` の最低要件なので省略不可と説明。他モジュール（CORE / CREW / NEBULA）の省略は受け入れる |
| 「行数調整したい」 | 調整内容を聞くが、**`phase_c_gate.sh` の閾値は変更不可** であることを説明。ユーザーが「じゃあゲート通らなくてもOK」と言った場合は、ゲートスクリプト実行結果をそのまま報告して終了（SKILL.md §0 第3項により量的基準を質的判断で上書きしない） |

---

## 重要: ゲートの客観性

`phase_c_gate.sh` は **bash 3.2 互換スクリプト** で、全ファイルの行数を数値比較する。**AIの主観で「実質的にOK」と判定するのは禁止**（SKILL.md §0 第3項）。

ユーザーが「行数要件を緩和したい」と言った場合でも、スクリプト自体は元の閾値で実行する。緩和はユーザーの自己責任で、その旨を必ず報告に含める：

> 「ユーザー指定により CORE の最低行数を 80 → 50 行に緩和しましたが、`phase_c_gate.sh` は 80 行基準で FAIL を返しています。以下のモジュールが不足: core_deep_dive.typ (52行 / 要 80行)」

---

## 禁止事項

- **Phase C 完了ゲート（`phase_c_gate.sh`）の実行をスキップするのは絶対禁止**
- FAIL 時に「大丈夫です、進めます」と宣言してPhase D に進むのは禁止
- FAIL 時は必ず Phase C に戻り、不足モジュールを補強してから再実行する
