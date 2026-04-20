# Saturn V スキーマ

## 対象ファイル
- `data/saturnv_clusters.json` (TELESCOPE)
- `data/saturnv_drilldown.json` (PROBE)

## モジュール概要
Saturn V はSBERT+UMAP+HDBSCANによるAI特許ランドスケープモジュール。
- **TELESCOPE**: 全特許の自動クラスタリング。技術領域の全体俯瞰を提供する。
- **PROBE**: 特定クラスタのドリルダウン分析。選択したクラスタ内部の細分化構造を明らかにする。

## JSONスキーマ

### TELESCOPE (`saturnv_clusters.json`)

#### metadata

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `module` | string | 固定値 `"Saturn V"` |
| `mode` | string | 固定値 `"TELESCOPE"` |
| `n_clusters` | int | クラスタ数（ノイズクラスタ除く） |
| `noise_count` | int | ノイズに分類された特許数 |
| `total_patents` | int | 分析対象の全特許数 |

#### clusters（配列）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `cluster_id` | int | クラスタID（-1はノイズ） |
| `label` | string | ユーザー編集可能ラベル |
| `auto_label` | string | TF-IDF自動生成ラベル |
| `count` | int | クラスタ内の特許数 |
| `centroid` | [float, float] | UMAP 2D空間の重心座標 `[umap_x, umap_y]` |
| `tfidf_top_terms` | [string] | TF-IDFスコア上位10語のリスト |
| `representative_patents` | [string] | 代表特許の要約テキスト（各200文字以内） |

#### トップレベル

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `spatial_context` | string | UMAP空間におけるクラスタ配置の自然言語記述 |
| `noise_patents` | [object] | ノイズ特許の詳細データ（上限200件）。→ `analysis/noise_analysis.md` で分析 |

#### noise_patents（配列の各要素）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `umap_x` | float | UMAP X座標（小数4桁） |
| `umap_y` | float | UMAP Y座標（小数4桁） |
| `title` | string | 特許タイトル（100文字截断） |
| `applicant` | string | 主出願人（50文字截断） |
| `year` | int | 出願年 |

#### metadata 追加フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `noise_ratio` | float | ノイズ率（noise_count / total_patents）。0-1の小数 |

---

### PROBE (`saturnv_drilldown.json`)

#### metadata

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `module` | string | 固定値 `"Saturn V"` |
| `mode` | string | 固定値 `"PROBE"` |
| `parent_cluster` | string | ドリルダウン元の親クラスタ名 |
| `n_clusters` | int | サブクラスタ数 |
| `total_patents` | int | 親クラスタ内の特許数 |

#### clusters（配列）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `cluster_id` | int | サブクラスタID |
| `label` | string | サブクラスタのラベル |
| `count` | int | サブクラスタ内の特許数 |
| `centroid` | [float, float] | UMAP 2D空間の重心座標 `[umap_x, umap_y]` |
| `representative_patents` | [string] | 代表特許の要約テキスト（各200文字以内） |

#### トップレベル

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `spatial_context` | string | サブクラスタ間の空間配置の自然言語記述 |

## 解釈ガイドライン

- **クラスタID -1 はノイズ**: HDBSCANの密度閾値以下の小グループ。どのクラスタにも属さない特許群。ノイズ比率が高い場合、データの多様性が高いか、パラメータ調整が必要であることを示唆する。`noise_patents` 配列にノイズ特許の座標・タイトル・出願人・年が含まれており、`analysis/noise_analysis.md` の5手法（ノイズ率解釈・空間分布・時系列・出願人分析・新興領域抽出）で深掘り分析が可能。
- **centroid はUMAP 2D空間の重心座標**: 意味的に近いクラスタは座標上でも近接する。ただしUMAPの非線形変換により、距離の絶対値に厳密な意味はない。相対的な近接関係に着目すること。
- **representative_patents は200文字截断済み**: 要約テキストが途中で切れている場合がある。詳細な特許情報が必要な場合は `patents.csv` で公開番号等により検索可能。
- **spatial_context はクラスタ間の空間配置を自然言語で記述**: `utils_spatial.py` により生成される。クラスタ間の距離関係、密集度、孤立クラスタの有無などを含む。
- **label はユーザー編集可能**: EAGLEモジュール等でユーザーが手動で付与・修正したラベル。`auto_label` はTF-IDFスコアに基づく機械的な自動生成ラベルであり、TELESCOPEのみに存在する。

## レポートでの典型的な言及パターン

### 技術領域の俯瞰
- 「SBERT+HDBSCANにより**N個**の技術クラスタが自動識別された」
- 「最大クラスタは『{label}』（**X件**, 全体の**Y%**）であり、当該技術分野の中核領域を形成している」
- 「上位3クラスタで全体の**Z%**を占め、技術集中度が高い/低いことが確認された」

### クラスタ間関係の分析
- 「クラスタAとクラスタBはUMAP空間上で近接しており、技術的な関連性が示唆される」
- 「クラスタCは他クラスタから孤立した位置にあり、独自の技術領域を形成している」
- 「spatial_contextによれば、{空間配置の要約}」

### ノイズ比率の解釈
- 「ノイズ特許は**N件**（全体の**X%**）であり、既存クラスタに分類できない多様な技術が存在する」
- 「ノイズ比率が低い（**X%**以下）ことから、分析対象の技術分野は比較的均質であると考えられる」
- 「ノイズ比率が高い（**X%**超）場合、ニッチ技術や新興領域が含まれている可能性がある」

### PROBEドリルダウン分析
- 「親クラスタ『{parent_cluster}』をドリルダウンした結果、**N個**のサブクラスタが検出された」
- 「サブクラスタの分布から、当該技術領域内の細分化された研究テーマが明らかになった」

## 新規フィールド: ノイズ分析

`saturnv_clusters.json` に `noise_analysis` オブジェクトが追加:

```json
{
  "noise_analysis": {
    "noise_count": 150,
    "noise_ratio": 0.12,
    "noise_interpretation": "標準的・安定構造（5-15%）",
    "temporal_pattern": "recent_surge | historical | uniform",
    "year_distribution": {"2020": 5, "2021": 8, ...},
    "top_applicants": [{"applicant": "A社", "count": 12}, ...],
    "emerging_keywords": [{"keyword": "ペロブスカイト", "frequency": 15}, ...]
  }
}
```

### ノイズ率の解釈基準
| ノイズ率 | 解釈 | 戦略的意味 |
|---------|------|----------|
| < 5% | 成熟・均質 | 技術体系が確立。ディスラプション余地は小さい |
| 5-15% | 標準的 | 安定したクラスタ構造。通常の技術進化 |
| 15-30% | 多様・融合活発 | 技術融合が進行中。新テーマが生まれやすい |
| > 30% | 萌芽・黎明期 | 技術体系が未確立。大きな変化の前兆 |

### 萌芽テーマの分析手順
1. `emerging_keywords` から頻出語彙を確認
2. `temporal_pattern` が `recent_surge` なら新興テーマの可能性大
3. `top_applicants` で特定企業に集中していれば戦略的ニッチ
4. `patents.csv` で cluster==-1 のレコードを直接参照し、具体的な特許タイトルを確認

## 新規フィールド: クラスタ動態マップ

`saturnv_clusters.json` に `cluster_dynamics` オブジェクトが追加:

```json
{
  "cluster_dynamics": {
    "x_axis": "cumulative_count",
    "y_axis": "cagr",
    "cagr_window_years": 5,
    "x_threshold": 150,
    "y_threshold": 0.05,
    "clusters": [
      {"cluster_id": 0, "label": "...", "cumulative_count": 245, "cagr": 0.12, "quadrant": "成長リーダー", "share": 0.15}
    ],
    "quadrant_summary": {
      "成長リーダー": {"count": 3, "total_patents": 520},
      "新興クラスタ": {"count": 5, "total_patents": 180},
      "成熟クラスタ": {"count": 2, "total_patents": 800},
      "ニッチ/衰退": {"count": 4, "total_patents": 100}
    }
  }
}
```

### 4象限の解釈ガイド
| 象限 | 位置 | 解釈 | 戦略的示唆 |
|------|------|------|----------|
| 成長リーダー | 右上(大規模×高成長) | 主要技術が活発に発展中 | 投資継続、競合動向を注視 |
| 新興クラスタ | 左上(小規模×高成長) | 新テーマが急速に立ち上がり中 | 早期参入の機会、技術動向を追跡 |
| 成熟クラスタ | 右下(大規模×低成長) | 確立された技術、出願ペースは鈍化 | 改良発明中心、差別化が重要 |
| ニッチ/衰退 | 左下(小規模×低成長) | 小規模で伸びていない | 撤退検討 or ニッチ戦略の深掘り |

### クロスモジュール分析での活用
- **× MEGA PULSE**: クラスタの成長とエンティティ（出願人）の成長を対比。「成長クラスタに属しているが衰退している出願人」を特定
- **× NEBULA学術クラスタ**: 特許クラスタと学術クラスタの動態を比較。研究→特許化のパイプライン遅延を検出
