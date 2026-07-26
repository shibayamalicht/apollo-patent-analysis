# MEGA スキーマ

## 対象ファイル
- `data/mega_momentum_<軸>.json` (PULSE 4象限。**軸別に保存される**: 出願人=`mega_momentum_applicant.json` / IPC=`mega_momentum_ipc.json` / Fターム=`mega_momentum_fterm.json`)。※旧バージョンは軸非依存の `mega_momentum.json` 単一ファイル（最後の軸で上書きされる不具合があった）
- `data/mega_drilldown_<対象>.json` (ポートフォリオ詳細。**対象別に保存**: 例 `mega_drilldown_トヨタ自動車.json`。旧: 固定名 `mega_drilldown.json`)

> 💡 複数軸を実行した場合、軸ごとに別ファイルが残る。**存在する `mega_momentum_*.json` を全て読み、各軸を個別に分析する**（`metadata.axis` で軸名を確認）。

## モジュール概要

MEGA (動態分析) モジュールは、特許出願の時系列動態をマクロ・ミクロの2視点で分析する。

- **PULSE (4象限マップ)**: 出願人・IPC・Fタームを分析軸とし、CAGR（成長率）× 現在の活動量（直近N年の出願件数）の散布図で4象限に分類。各エンティティの戦略的ポジションを可視化する。
- **TELESCOPE (ドリルダウン)**: PULSE上で選択した特定エンティティの特許群をSBERT+UMAP+HDBSCANでクラスタリングし、技術ポートフォリオの内部構造を可視化する。

## JSONスキーマ

### mega_momentum_<軸>.json (PULSE 4象限。軸別: applicant/ipc/fterm)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `metadata` | object | 分析メタデータ |
| `metadata.module` | string | 固定値 `"MEGA"` |
| `metadata.mode` | string | 固定値 `"PULSE"` |
| `metadata.axis` | string | 分析軸ラベル。`"出願人"` / `"IPC (メイングループ)"` / `"Fターム (テーマコード)"` のいずれか |
| `metadata.total_entities` | integer | entities配列の要素数（フィルタ後のエンティティ総数） |
| `entities` | array | 各エンティティ（出願人・IPC等）の動態データ |
| `entities[].name` | string | エンティティ名（`metadata.axis` に対応＝出願人名／IPC メイングループ／F タームテーマコード）。※v9 で修正済み（旧版は `X_Present` が誤格納されていた） |
| `entities[].cagr` | float | CAGR 値（X 軸＝過去の勢い）。※v9 で修正済み（旧版は常に `0` だった） |
| `entities[].activity` | integer | 現在の活動量（Y_Present: 直近N年の出願件数合計） |
| `entities[].total` | integer | 累計総出願件数（Bubble_Present: バブルサイズに対応） |
| `entities[].quadrant` | string | 所属する象限ラベル（下記「4象限の定義」参照） |
| `quadrant_summary` | object | 象限別のエンティティ件数集計 |
| `quadrant_summary["リーダー (Leaders)"]` | integer | リーダー象限のエンティティ数 |
| `quadrant_summary["新興・高ポテンシャル (Emerging)"]` | integer | 新興象限のエンティティ数 |
| `quadrant_summary["成熟・既存勢力 (Established)"]` | integer | 成熟象限のエンティティ数 |
| `quadrant_summary["衰退・ニッチ (Declining/Niche)"]` | integer | 衰退象限のエンティティ数 |

#### サンプル構造

```json
{
  "metadata": {
    "module": "MEGA",
    "mode": "PULSE",
    "axis": "出願人",
    "total_entities": 45
  },
  "entities": [
    {
      "name": "トヨタ自動車",
      "cagr": 0.0523,
      "activity": 128,
      "total": 1540,
      "quadrant": "リーダー (Leaders)"
    }
  ],
  "quadrant_summary": {
    "リーダー (Leaders)": 12,
    "新興・高ポテンシャル (Emerging)": 8,
    "成熟・既存勢力 (Established)": 15,
    "衰退・ニッチ (Declining/Niche)": 10
  }
}
```

### mega_drilldown_<対象>.json (TELESCOPE ドリルダウン)

> 💡 v9 以降、**対象別ファイル名で保存**される（`mega_drilldown_<対象>.json`）ため、複数対象を続けて深掘りしても上書きされず全て残る。存在する `mega_drilldown_*.json` を各々分析する。レポートでは各ファイルの `metadata.drilldown_target` を確認し、その対象の話として書く（旧版は固定名 `mega_drilldown.json` で最後の対象のみ）。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `metadata` | object | 分析メタデータ |
| `metadata.module` | string | 固定値 `"MEGA"` |
| `metadata.mode` | string | 固定値 `"TELESCOPE"` |
| `metadata.drilldown_target` | string | ドリルダウン対象のエンティティ名（出願人名・IPC等） |
| `metadata.n_clusters` | integer | 検出されたクラスタ数（ノイズクラスタ含む） |
| `clusters` | array | 各クラスタの情報 |
| `clusters[].cluster_id` | integer | クラスタID。`-1` はノイズ（HDBSCANが未分類とした特許群） |
| `clusters[].label` | string | クラスタラベル。TF-IDFの上位語から自動生成。形式: `"[ID] 単語1, 単語2, 単語3"`（ノイズの場合は `"ノイズ"`） |
| `clusters[].count` | integer | クラスタ内の特許件数 |
| `clusters[].representatives` | array | 代表特許のリスト（文字列配列）。各要素は `"  * [公開番号]【タイトル】(出願人, 年, IPC:分類): 要約先頭200文字..."` 形式。※v9 で修正済み（旧版は変数未定義で空リストになっていた） |

#### サンプル構造

```json
{
  "metadata": {
    "module": "MEGA",
    "mode": "TELESCOPE",
    "drilldown_target": "トヨタ自動車",
    "n_clusters": 5
  },
  "clusters": [
    {
      "cluster_id": 0,
      "label": "[0] 電池, 電極, 正極材料",
      "count": 42,
      "representatives": [
        "  * [JP2023-123456]【全固体電池用正極材料】(トヨタ自動車, 2023, IPC:H01M): 本発明は全固体電池に用いる正極活物質に関し..."
      ]
    },
    {
      "cluster_id": -1,
      "label": "ノイズ",
      "count": 8,
      "representatives": []
    }
  ]
}
```

## 分析軸（出願人 / IPC / F ターム）— ⚠️ レポート執筆時の最重要注意

動態分析（内部名 MEGA PULSE）は **3 つの分析軸**で実行できる（UI のセレクトボックス）。各軸が独立した 4 象限マップを生成する。

| 軸 | `metadata.axis` の値 | 説明 | 利用条件 |
|----|--------------------|------|---------|
| 出願人 | `"出願人"` | 出願人ごとの成長×活動量 | 常時 |
| 分類コード | `"分類コード IPC (メイングループ)"` または `"分類コード FI (メイングループ)"`（旧セッションは `"IPC (メイングループ)"`） | 分類コードのメイングループごとの動態。IPC/FI の別は `context.json` の `classification_code_label` と一致 | 常時 |
| F ターム | `"Fターム (テーマコード)"` | F タームテーマごとの動態 | F ターム列がマッピングされている場合のみ |

**⚠️ 執筆時に必ず踏まえる**: v9 以降、JSON は**軸別ファイル名で保存**される（`mega_momentum_applicant.json` / `_ipc.json` / `_fterm.json`）ため上書き衝突は解消した。スナップショット画像のタイトルにも軸名が入る（「動態分析マップ（〇〇軸）」。分類コード軸の表記は「分類コード IPC (メイングループ)」または「分類コード FI (メイングループ)」で、IPC/FI は `context.json` の `classification_code_label` と一致する）。このため:
- **存在する `mega_momentum_*.json` を全て読み、各軸を個別に分析する**。JSON の `metadata.axis` で軸名を確認し、「どの軸の数値か」を明示する（出願人の数値を F タームの話に混ぜない）。
- **スナップショット画像を貼るなら、その軸を本文で必ず分析する**。スナップショットのタイトルにある軸名と本文の軸を一致させる。分析しない図は貼らない（前回レポートの「F タームのみ分析なのに出願人マップも貼付」はこの取り違えが原因）。
- 旧バージョンの単一 `mega_momentum.json` しか無いセッションでは、その 1 軸（`metadata.axis`）のみを分析対象とする。

> 旧版にあった `name`/`cagr`/`representatives` の不具合は **v9 で修正済み**（`name`=軸のインデックス名、`cagr`=X_Present の CAGR、`representatives`=代表特許が入る）。

## 解釈ガイドライン

### 4象限の定義

象限の境界は、全エンティティの **CAGR平均値** と **活動量平均値** で動的に決定される（固定閾値ではない）。

| 象限 | CAGR | 活動量 | 戦略的意味 |
|------|------|--------|-----------|
| **リーダー (Leaders)** | 高（平均超） | 高（平均超） | 成長期・注力領域。CAGR・活動量ともに高く、市場をリードするプレイヤー。継続的投資と技術深耕が推奨される。 |
| **新興・高ポテンシャル (Emerging)** | 高（平均超） | 低（平均以下） | 新興・ニッチ。成長率は高いが現時点の活動量は少ない。将来のリーダー候補であり、動向監視と早期参入が重要。 |
| **成熟・既存勢力 (Established)** | 低（平均以下） | 高（平均超） | 成熟・防衛的出願。大量の出願実績があるが成長は鈍化。既存技術の維持・防衛的ポジション。 |
| **衰退・ニッチ (Declining/Niche)** | 低（平均以下） | 低（平均以下） | 衰退・撤退候補。成長率・活動量ともに低い。ポートフォリオ見直し・撤退判断の対象。 |

### 補足事項

- **Y軸（活動量）が 0 以下** のエンティティは、象限判定ロジックにより自動的に「衰退・ニッチ」に分類される。
- **CAGR** は `(end_value / start_value)^(1/years) - 1` で計算される。出願が途切れた期間がある場合、活動期間内の最初と最後の年の値が使用される。
- **バブルサイズ（total）** は全期間の累計出願件数であり、4象限の分類基準には使用されない（視覚的な参考情報）。
- **TELESCOPE のクラスタラベル** はTF-IDFの上位語から自動生成されるため、技術内容と完全に一致しない場合がある。ユーザーがUI上で手動編集可能。

## レポートでの典型的な言及パターン

### PULSE（4象限分析）からの戦略的示唆

- **市場構造の把握**: 「{分析軸}を4象限で分類した結果、リーダー象限に{N}者、新興象限に{M}者が位置する。市場は{集中型/分散型}の構造を示す。」
- **成長プレイヤーの特定**: 「CAGR {X}%以上の高成長エンティティとして{名前}が挙げられ、直近{Y}年で活動量{Z}件を記録している。」
- **成熟市場の警告**: 「成熟象限に{N}者が集中しており、技術分野全体の成長鈍化が示唆される。特に{名前}はCAGR {X}%と低迷している。」
- **ポートフォリオバランス**: 「4象限の分布から、{分析軸}のポートフォリオは{領域}に偏重しており、{領域}での活動強化が推奨される。」

### TELESCOPE（ドリルダウン）からの戦略的示唆

- **技術ポートフォリオの多様性**: 「{対象名}の技術ポートフォリオは{N}個のクラスタに分かれ、主要テーマは{ラベル1}（{件数}件）、{ラベル2}（{件数}件）である。」
- **注力技術の特定**: 「最大クラスタ（{ラベル}、{件数}件）が全体の{割合}%を占め、{対象名}の中核技術と推定される。」
- **新規技術の萌芽**: 「小規模クラスタ{ラベル}（{件数}件）は、{対象名}における新規技術開発の兆候を示す。」
- **ノイズ特許の解釈**: 「ノイズクラスタ（{件数}件）は他クラスタに分類されなかった特許であり、独自性の高い発明や分野横断的な技術を含む可能性がある。」
