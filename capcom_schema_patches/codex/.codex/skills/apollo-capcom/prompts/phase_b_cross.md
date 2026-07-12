# Phase B STOP-GATE 1 — クロスモジュールパターン選定用テンプレ（最低5つ）

SKILL.md §Phase B STOP-GATE 1 の `ask_user_question` 呼び出し用雛形。

---

## 前提

`capcom_schema/analysis/cross_module.md` を読了済みであること。13種のクロスパターン（P1-P13）の内容を把握した上でこのテンプレを使う。

---

## 使い方

1. Mission Objective との整合性で **Agent推奨5つ** を選ぶ（★マーク）
2. 13パターン全てを提示し、ユーザーが**5つ以上**選択する形式にする（最低5つは Phase B 完了条件・gate Check 4 の必須基準）
3. `ask_user_question` の `multiSelect: true` で呼び出し

---

## ask_user_question 呼び出しフォーマット

```yaml
question: "採用するクロスモジュール分析パターンを5つ以上選んでください（★がAgent推奨）。選んだパターンでPhase B本体を進めます。"
header: "クロス選定"
multiSelect: true
options:
  - label: "★ P1: <パターン名>"
    description: "<モジュール組み合わせ + 1行要旨 + Mission Objective との関連度>"
  - label: "★ P4: <パターン名>"
    description: "<同上>"
  - label: "★ P13: <パターン名>"
    description: "<同上>"
  - label: "P2: <パターン名>"
    description: "<同上>"
  - label: "P3: <パターン名>"
    description: "<同上>"
  # ... 以下 P5-P12 を全て列挙
```

**Agent推奨（★）の選び方**（計5つ）:
- Mission Objective と最も直結するパターンを2つ
- データの強み（例: Saturn V クラスタ多数、MEGA 4象限の QI が多い等）を活かすパターンを2つ
- 不明点を解消できるパターン（例: ノイズ分析 → 学術論文ギャップ）を1つ

---

## 具体例

Mission Objective が「全固体電池市場の競争優位分析」の場合：

```yaml
question: "採用するクロスモジュール分析パターンを5つ以上選んでください（★がAgent推奨）。"
header: "クロス選定"
multiSelect: true
options:
  - label: "★ P1: Saturn V × MEGA（クラスタ動態）"
    description: "AI分類クラスタごとのCAGR比較で成長/衰退を判定"
  - label: "★ P4: CORE × ATLAS（出願人集中度）"
    description: "ルール分類×出願人HHIで寡占状況を時系列可視化"
  - label: "★ P13: NEBULA × Saturn V（学術ギャップ）"
    description: "学術クラスタと特許クラスタの対応関係から未特許化領域を抽出"
  - label: "★ P2: Saturn V × Explorer（技術×キーワード）"
    description: "クラスタ内の頻出キーワードで技術トレンド言語化"
  # ★は計5つ付ける（残り1つは Mission Objective に応じて P3-P12 から）。P3-P12 も全て列挙
```

---

## ゲート通過の条件

- ユーザーが5つ以上選択 → 確定、Phase B 本体作業へ（6つ以上も歓迎。多い分には問題ない）
- ユーザーが4つ以下 → 「クロス分析は最低5パターンが品質ゲートの必須基準です。追加で選んでください」と再要求
- Other で自由記入 → 選ばれたカスタムパターンを受け入れ、既存の Px と合わせて計5つ以上になるよう確認

---

## 禁止事項

- **Agent側で勝手にパターンを確定して進めるのは禁止**（SKILL.md §0 第2項）
- 「Agent推奨で進めます」とユーザー確認なしに宣言してはならない。必ず `ask_user_question` を呼び出す
