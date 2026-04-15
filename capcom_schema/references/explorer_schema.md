# Explorer スキーマ

## 対象ファイル
- `data/explorer_global_network.json` (全体共起ネットワーク)
- `data/explorer_trend.json` (トレンドネットワーク)
- `data/explorer_dominance.json` (ドミナンスネットワーク)

## モジュール概要

Explorer はキーワード共起分析モジュール。TF-IDF特徴語の共起関係をネットワーク可視化し、技術キーワード間の関連性・トレンド・競合優位性を分析する。3つの分析モード（全体ネットワーク・トレンド分析・競合比較戦略）に対応し、それぞれ独立したJSONファイルを出力する。

---

## JSONスキーマ

エッジは全モードともweight（Jaccard係数）降順で**top100**に制限される。`metadata.n_edges_total` で元データの全体規模を確認可能。

### 1. explorer_global_network.json (全体共起ネットワーク)

```jsonc
{
  "metadata": {
    "module": "Explorer",           // 固定値
    "mode": "global_network",       // 固定値
    "n_nodes": int,                 // ネットワーク内のノード（キーワード）数
    "n_edges_total": int,           // 元データの全エッジ数
    "n_edges_exported": int,        // 出力されたエッジ数（top100以下）
    "density": float,               // ネットワーク密度（0.0〜1.0）
    "top_n": int,                   // 上位N件のキーワードを対象（UIスライダー値）
    "threshold": float              // Jaccard係数の下限閾値（UIスライダー値）
  },

  // 全ノードの中心性ランキング（degree_centrality降順）
  "nodes": [
    {
      "keyword": string,            // キーワード文字列
      "degree_centrality": float,   // 次数中心性（0.0〜1.0、小数4桁）
      "frequency": int,             // 全文書中の出現回数
      "community": int              // 所属コミュニティID（-1 = 未所属）
    }
  ],

  // エッジ上位100件（Jaccard係数降順）
  "edges": [
    {
      "source": string,             // キーワードA
      "target": string,             // キーワードB
      "jaccard": float,             // Jaccard係数（0.0〜1.0、小数4桁）
      "cooccurrence_count": int     // 同一文書内での共起回数
    }
  ],

  // Louvain法によるコミュニティ検出結果
  "communities": [
    {
      "id": int,                    // コミュニティID（0始まり）
      "size": int,                  // コミュニティ内のノード数
      "members": [string],          // 所属キーワード一覧
      "hub": string,                // コミュニティ内ハブ（次数中心性最大のノード）
      "hub_centrality": float       // ハブの次数中心性（小数4桁）
    }
  ],

  // コミュニティ間を結ぶブリッジエッジ上位20件（Jaccard係数降順）
  "bridge_edges": [
    {
      "source": string,             // キーワードA
      "target": string,             // キーワードB
      "communities": [int, int],    // 接続するコミュニティIDのペア
      "jaccard": float              // Jaccard係数（小数4桁）
    }
  ]
}
```

### 2. explorer_trend.json (トレンドネットワーク)

```jsonc
{
  "metadata": {
    "module": "Explorer",           // 固定値
    "mode": "trend_network",        // 固定値
    "target": string,               // 分析対象（"全体 (Market)" または出願人名）
    "n_nodes": int,                 // ネットワーク内のノード数
    "n_edges_total": int,           // 元データの全エッジ数
    "n_edges_exported": int,        // 出力されたエッジ数（top100以下）
    "density": float,               // ネットワーク密度
    "top_n": int,                   // 上位N件のキーワードを対象
    "threshold": float,             // Jaccard係数の下限閾値
    "period_past": string,          // 過去期間（例: "2015-2019"）
    "period_recent": string         // 直近期間（例: "2020-2024"）
  },

  // 急上昇キーワード（growth_rate降順、growth_rate >= 0）
  "emerging_keywords": [
    {
      "keyword": string,            // キーワード文字列
      "growth_rate": float,         // 成長率（小数3桁、0以上）
      "past_count": int,            // 過去期間の出現回数
      "recent_count": int,          // 直近期間の出現回数
      "community": int,             // 所属コミュニティID（-1 = 未所属）
      "degree_centrality": float    // 次数中心性（小数4桁）
    }
  ],

  // 衰退キーワード（growth_rate昇順、growth_rate < 0）
  "declining_keywords": [
    {
      "keyword": string,            // キーワード文字列
      "growth_rate": float,         // 成長率（小数3桁、負値）
      "past_count": int,            // 過去期間の出現回数
      "recent_count": int,          // 直近期間の出現回数
      "community": int,             // 所属コミュニティID（-1 = 未所属）
      "degree_centrality": float    // 次数中心性（小数4桁）
    }
  ],

  // Louvain法によるコミュニティ検出結果
  "communities": [
    {
      "id": int,                    // コミュニティID（0始まり）
      "size": int,                  // コミュニティ内のノード数
      "members": [string],          // 所属キーワード一覧
      "hub": string,                // コミュニティ内ハブ
      "hub_centrality": float       // ハブの次数中心性（小数4桁）
    }
  ],

  // エッジ上位100件（Jaccard係数降順）
  "edges": [
    {
      "source": string,             // キーワードA
      "target": string,             // キーワードB
      "jaccard": float,             // Jaccard係数（小数4桁）
      "cooccurrence_count": int     // 共起回数
    }
  ]
}
```

### 3. explorer_dominance.json (ドミナンスネットワーク)

```jsonc
{
  "metadata": {
    "module": "Explorer",           // 固定値
    "mode": "dominance_network",    // 固定値
    "my_company": string,           // 自社名（UIで選択した出願人）
    "target_company": string,       // 競合他社名（UIで選択した出願人）
    "n_nodes": int,                 // ネットワーク内のノード数
    "n_edges_total": int,           // 元データの全エッジ数
    "n_edges_exported": int,        // 出力されたエッジ数（top100以下）
    "density": float,               // ネットワーク密度
    "top_n": int,                   // 上位N件のキーワードを対象
    "threshold": float              // Jaccard係数の下限閾値
  },

  // 全キーワードの支配率情報
  "keywords": [
    {
      "keyword": string,            // キーワード文字列
      "my_count": int,              // 自社の出現回数
      "target_count": int,          // 競合の出現回数
      "dominance": float,           // 支配率（0.0〜1.0、小数3桁）
                                    //   1.0 = 完全に自社優位
                                    //   0.5 = 拮抗
                                    //   0.0 = 完全に競合優位
      "community": int,             // 所属コミュニティID（-1 = 未所属）
      "degree_centrality": float    // 次数中心性（小数4桁）
    }
  ],

  "my_exclusive": [string],         // 自社のみが使用するキーワード（target_count=0）
  "target_exclusive": [string],     // 競合のみが使用するキーワード（my_count=0）
  "contested": [string],            // 拮抗キーワード（dominance 0.4〜0.6）

  // Louvain法によるコミュニティ検出結果
  "communities": [
    {
      "id": int,                    // コミュニティID（0始まり）
      "size": int,                  // コミュニティ内のノード数
      "members": [string],          // 所属キーワード一覧
      "hub": string,                // コミュニティ内ハブ
      "hub_centrality": float       // ハブの次数中心性（小数4桁）
    }
  ],

  // エッジ上位100件（Jaccard係数降順）
  "edges": [
    {
      "source": string,             // キーワードA
      "target": string,             // キーワードB
      "jaccard": float,             // Jaccard係数（小数4桁）
      "cooccurrence_count": int     // 共起回数
    }
  ]
}
```

---

## 解釈ガイドライン

### Jaccard係数
キーワードAとBの共起強度を表す。`|A ∩ B| / |A ∪ B|` で算出。
- **0.3以上**: 非常に強い共起関係。ほぼ同一文脈で使用される技術概念
- **0.1〜0.3**: 中程度の共起。関連技術領域
- **0.1未満**: 弱い共起。間接的な関連

`cooccurrence_count` と合わせて解釈する。Jaccard係数が高くても共起回数が少ない場合はニッチな組み合わせ、回数が多い場合は主流の技術組み合わせを示す。

### コミュニティ（Louvain法）
ネットワーク内で密に接続されたキーワード群を自動検出する。
- 各コミュニティは**技術サブドメイン**に対応する傾向がある
- `hub` はそのサブドメインの**中核概念**を示す
- `hub_centrality` が高いほど、そのキーワードがサブドメイン内外で多数のキーワードと結びついている
- コミュニティサイズの偏りは、技術領域の成熟度や注力度の差を反映する

### ブリッジエッジ（global_networkのみ）
異なるコミュニティ間を結ぶエッジ。技術領域間の**融合点・連携点**を示す。
- ブリッジエッジのJaccard係数が高い場合、2つの技術サブドメインが統合されつつある
- `communities` フィールドで接続先のコミュニティIDを確認し、それぞれのhubキーワードと合わせて解釈する
- ブリッジエッジが少ない/弱い場合、技術領域が独立して発展している

### 急上昇・衰退キーワード（trend）
`growth_rate` は `(recent_count - past_count) / (past_count + 1)` で算出。
- **急上昇 (emerging)**: `growth_rate >= 0`。直近期間に使用頻度が増加した技術キーワード
  - `past_count=0, recent_count>0` の場合は完全な新出キーワード
  - 高い `degree_centrality` を持つ急上昇キーワードはネットワーク全体に影響を与える新潮流
- **衰退 (declining)**: `growth_rate < 0`。使用頻度が減少した技術キーワード
  - 技術の陳腐化、用語の変遷、市場からの撤退を示唆
- `period_past` / `period_recent` でどの時間窓での比較かを確認する

### ドミナンス分析（dominance）
2社間のキーワード支配率を比較する。
- **dominance > 0.6**: 自社優位。自社がより多く使用している技術キーワード
- **dominance 0.4〜0.6**: 拮抗領域（`contested`リストに含まれる）。競争が激しい技術領域
- **dominance < 0.4**: 競合優位。競合がリードしている技術領域
- **my_exclusive**: 自社のみが出願しているキーワード。独自技術やニッチ戦略を示す
- **target_exclusive**: 競合のみが出願しているキーワード。自社の技術ギャップや潜在的脅威を示す
- コミュニティと組み合わせることで、どの技術サブドメインで優位/劣位かを構造的に把握できる

### ネットワーク密度
SKILL.md の共通ルールに準拠。
- **< 0.1**: 疎なネットワーク（キーワードが独立的に使用されている）
- **0.1〜0.3**: 中程度（標準的な技術ネットワーク）
- **> 0.3**: 密なネットワーク（キーワードが相互に強く関連している）

---

## レポートでの典型的な言及パターン

### 技術キーワード戦略（global_network使用）
- コミュニティ構造から**主要技術クラスタ**を特定し、各クラスタのhubキーワードを中核技術として報告する
- ブリッジエッジから**技術融合の兆候**を読み取り、異分野連携の可能性を提言する
- 次数中心性の高いキーワードを**基盤技術**、低いが特定コミュニティに集中するキーワードを**専門技術**として区別する
- エッジのJaccard係数と共起回数から、技術の組み合わせパターンを抽出する

### トレンド分析（trend使用）
- 急上昇キーワードから**注目すべき技術動向**を列挙し、past/recent期間の具体的な件数推移を示す
- 衰退キーワードから**成熟・陳腐化技術**を特定し、代替技術の出現と関連づける
- 急上昇キーワードのコミュニティ分布から、成長がどの技術サブドメインに集中しているかを分析する
- 特定出願人を対象とした場合、その企業の技術戦略の方向転換を時系列で把握する

### 競合比較（dominance使用）
- my_exclusive / target_exclusive から**独自技術領域**と**技術ギャップ**を特定する
- contested キーワードから**競争激化領域**を抽出し、投資判断の材料とする
- 支配率 + コミュニティの組み合わせで、技術サブドメイン単位での優劣を可視化する
- 自社の弱いコミュニティ（競合優位のキーワードが集中するコミュニティ）を**重点強化領域**として提言する

### クロスモジュール分析での活用
- **Saturn V / EAGLE クラスタとの対応**: Explorerのコミュニティが、Saturn VのクラスタやEAGLEの手動クラスタとどう対応するかを比較し、技術分類の妥当性を検証する
- **MEGA 4象限との組み合わせ**: 急上昇キーワードがMEGAのどの象限の出願人に多いかを分析し、成長領域と出願人の動態を関連づける
- **CREW ネットワークとの統合**: Explorerの技術キーワードクラスタとCREWの人的ネットワーク（発明者コミュニティ）を突き合わせ、どの研究チームがどの技術領域を推進しているかを把握する
