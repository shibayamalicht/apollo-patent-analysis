# ワードクラウドスキーマ

## 対象ファイル
`data/*_wordcloud.json`

## モジュール概要
ワードクラウド分析結果。各モジュールのテキスト分析から生成された単語頻度データ。
スナップショット画像（`snapshots/*_wordcloud.png`）と対になる。

## 生成元モジュール

| ファイル名パターン | 生成元 | 内容 |
|------------------|--------|------|
| `saturnv_drill_wordcloud.json` | Saturn V PROBE | ドリルダウン対象クラスタの特徴語 |
| `explorer_global_wordcloud.json` | Explorer 全体俯瞰 | 全特許の頻出キーワード |
| `mega_drill_wordcloud.json` | MEGA TELESCOPE | ドリルダウン対象エンティティの特徴語 |
| `eagle_drill_wordcloud.json` | EAGLE ドリルダウン | 手動クラスタの特徴語 |
| `crew_wordcloud.json` | CREW ネットワーク | 選択ノード（発明者/出願人）のキーワード |

## JSONスキーマ

```json
{
  "metadata": {
    "module": "string",   // 生成元モジュール名（Saturn V / Explorer / MEGA / EAGLE / CREW）
    "title": "string",    // ワードクラウドのタイトル
    "top_n": "integer",   // 表示上位語数（通常20-30）
    "type": "string"      // (CREW のみ) "node_keywords"
  },
  "word_frequencies": {
    "単語A": 150,         // {単語: 出現回数} の辞書
    "単語B": 98,          // 上位100語まで
    ...
  }
}
```

## トップレベル構造

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `metadata` | object | 生成メタ情報 |
| `word_frequencies` | object | 単語→出現回数の辞書。上位100語に制限 |

## 分析での活用

- 各モジュールの分析対象がどのような技術用語で特徴づけられるかを把握
- 複数モジュール間のワードクラウドを比較し、技術領域の共通性・差異を分析
- patents.csv のクラスタラベルと組み合わせ、クラスタの技術的特徴をテキストレベルで深掘り
