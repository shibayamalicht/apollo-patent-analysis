# metadata.json スキーマ

## 対象ファイル
- `metadata.json`（セッションフォルダ直下。ZIP エクスポート時に `capcom.py` が生成）

## モジュール概要

`metadata.json` は CAPCOM セッションの**エクスポート時メタデータ**を保持する。セッション基本情報に加え、**解析環境（使用した文埋め込みモデル等）**が記録される。レポート付録「分析条件一覧」や VOYAGER 章で、再現性のための環境情報として参照する。

## JSON スキーマ

| フィールド | 型 | 説明 |
|-----------|-----|------|
| （セッション基本情報） | object | `store['metadata']` の内容（セッション ID・エクスポート日時・データ件数等。セッションにより異なる） |
| `analysis_environment` | object | 解析環境情報（下記）。SBERT モデル付与に失敗した場合は欠落することがある |
| `analysis_environment.sbert_model` | string | 使用した文埋め込みモデル名（例: `"intfloat/multilingual-e5-base"` / `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`） |
| `analysis_environment.embedding_dim` | integer | 埋め込み次元数（例: `768`（quality）/ `384`（fast）） |
| `analysis_environment.text_preprocessing` | string | テキスト前処理の説明（固定値 `"Janome形態素解析 + 複合名詞結合"`） |

### サンプル構造

```json
{
  "session_id": "session_20260101_120000_ab12cd",
  "exported_at": "2026-01-01T12:00:00",
  "analysis_environment": {
    "sbert_model": "intfloat/multilingual-e5-base",
    "embedding_dim": 768,
    "text_preprocessing": "Janome形態素解析 + 複合名詞結合"
  }
}
```

## レポートでの利用

- **付録「分析条件一覧」**: 使用した文埋め込みモデルと次元数を「分析環境」として明記する（再現性のため）。
- モデルにより SBERT の表現力が異なる（`fast`=MiniLM 384 次元 / `quality`=multilingual-e5-base 768 次元）。クラスタリング結果の解像度を解釈する際の補助情報。
- **用語ルール**: 本文ではモデル名を「文埋め込みモデル（multilingual-e5-base, 768 次元）」のように補足付きで自然に書く。内部ファイル名 `metadata.json` 自体はレポート本文に書かない（terminology.md）。
