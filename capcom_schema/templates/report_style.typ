// ==================================================================
// report_style.typ — APOLLO v8.0.0 レポート共通スタイル
// コンサルレポート風デザイン（McKinsey/BCG調）
// ==================================================================

// --- カラーパレット ---
#let color-navy = rgb("#1B2A4A")
#let color-blue = rgb("#2E5090")
#let color-accent = rgb("#3B7DD8")
#let color-light-blue = rgb("#E8F0FE")
#let color-dark-gray = rgb("#333333")
#let color-medium-gray = rgb("#666666")
#let color-light-gray = rgb("#F5F5F5")
#let color-border = rgb("#DDDDDD")
#let color-zebra = rgb("#F8F9FA")

// ==================================================================
// メインテンプレート関数
// #show: apollo-report.with(...) で全設定をドキュメント全体に適用
// ==================================================================
#let apollo-report(
  title: "",
  subtitle: "",
  date: "",
  author: "APOLLO",
  body,
) = {
  // --- フォント設定（日本語対応）---
  // show/set ルールはこの関数内で定義すると body 全体に伝播する
  set text(
    font: ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "BIZ UDGothic", "Noto Sans JP", "Meiryo", "sans-serif"),
    size: 10pt,
    fill: color-dark-gray,
    lang: "ja",
  )

  // --- ページ設定 ---
  set page(
    paper: "a4",
    margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
    header: context {
      if counter(page).get().first() > 1 {
        set text(size: 8pt, fill: color-medium-gray)
        grid(
          columns: (1fr, 1fr),
          align(left)[APOLLO],
          align(right)[#counter(page).display()],
        )
        line(length: 100%, stroke: 0.5pt + color-border)
      }
    },
  )

  // --- 段落設定 ---
  set par(leading: 0.8em, justify: true)

  // --- 見出しスタイル ---
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(1cm)
    block(width: 100%)[
      #line(length: 100%, stroke: 2pt + color-navy)
      #v(0.3cm)
      #text(size: 20pt, weight: "bold", fill: color-navy)[#it.body]
      #v(0.5cm)
    ]
  }
  show heading.where(level: 2): it => {
    v(0.5cm)
    block(width: 100%)[
      #text(size: 14pt, weight: "bold", fill: color-blue)[#it.body]
      #line(length: 60%, stroke: 1pt + color-accent)
      #v(0.3cm)
    ]
  }
  show heading.where(level: 3): it => {
    v(0.3cm)
    text(size: 12pt, weight: "bold", fill: color-dark-gray)[#it.body]
    v(0.2cm)
  }

  // --- テーブルのデフォルトスタイル ---
  set table(stroke: none)
  show table: set text(size: 9pt)

  // --- カバーページ ---
  {
    set page(margin: 0pt)
    block(
      width: 100%,
      height: 100%,
      fill: color-navy,
    )[
      #align(center + horizon)[
        #block(width: 85%)[
          #v(2cm)
          #line(length: 40%, stroke: 2pt + white)
          #v(1cm)
          // タイトル: 長さに応じてフォントサイズを動的調整し、不自然な改行を防ぐ
          #set par(justify: false, leading: 0.6em)
          #block(width: 100%, breakable: false)[
            #align(center)[
              #context {
                let title-len = title.clusters().len()
                let title-size = if title-len <= 15 { 28pt }
                  else if title-len <= 25 { 24pt }
                  else if title-len <= 35 { 20pt }
                  else { 18pt }
                text(size: title-size, weight: "bold", fill: white)[#title]
              }
            ]
          ]
          #v(0.5cm)
          #block(width: 100%, breakable: false)[
            #align(center)[
              #context {
                let sub-len = subtitle.clusters().len()
                let sub-size = if sub-len <= 30 { 14pt }
                  else if sub-len <= 50 { 12pt }
                  else { 11pt }
                text(size: sub-size, fill: rgb("#AABBDD"))[#subtitle]
              }
            ]
          ]
          #v(2cm)
          #line(length: 40%, stroke: 2pt + white)
          #v(1cm)
          #text(size: 14pt, weight: "bold", fill: rgb("#8899BB"), tracking: 1pt)[APOLLO]
          #v(0.15cm)
          #block(width: 100%, breakable: false)[
            #align(center)[
              #text(size: 8pt, fill: rgb("#7788AA"), tracking: 0.3pt)[
                Advanced Patent & Overall Landscape-analytics Logic Orbiter
              ]
            ]
          ]
          #v(0.5cm)
          #text(size: 11pt, fill: rgb("#8899BB"))[#date]
        ]
      ]
    ]
  }

  // --- 本文 ---
  body
}

// ==================================================================
// ユーティリティ関数
// ==================================================================

// --- エグゼクティブサマリーボックス ---
#let exec-summary(body) = {
  block(
    width: 100%,
    inset: 1.2em,
    radius: 4pt,
    fill: color-light-blue,
    stroke: 1pt + color-accent,
  )[
    #text(size: 11pt, weight: "bold", fill: color-navy)[Executive Summary]
    #v(0.3cm)
    #text(size: 10pt)[#body]
  ]
}

// --- KPIカード（改善版：左辺アクセント + 動的フォントサイズ、breakable: false） ---
// value のテキスト長に応じてフォントサイズを自動調整し、日本語の長い値も崩れず収まる。
#let kpi-card(label, value, note: none) = {
  block(
    width: 100%,
    breakable: false,
    inset: 0.8em,
    radius: 4pt,
    fill: color-light-gray,
    stroke: (left: 3pt + color-accent, rest: 1pt + color-border),
  )[
    #text(size: 8pt, fill: color-medium-gray, weight: "bold")[#label]
    #v(0.15cm)
    #context {
      let val-len = str(value).clusters().len()
      let val-size = if val-len <= 3 { 24pt }
        else if val-len <= 5 { 20pt }
        else if val-len <= 7 { 16pt }
        else if val-len <= 10 { 14pt }
        else { 12pt }
      text(size: val-size, weight: "bold", fill: color-navy)[#value]
    }
    #if note != none {
      v(0.1cm)
      text(size: 8pt, fill: color-medium-gray)[#note]
    }
  ]
}

// --- KPIダッシュボード（ページまたぎ防止ラッパー） ---
// 使い方: #kpi-dashboard(3,
//   kpi-card("総特許数", "1,234", note: "2018-2024年"),
//   kpi-card("クラスタ数", "12"),
//   kpi-card("HHI", "0.097"),
// )
#let kpi-dashboard(cols: 3, ..cards) = {
  block(breakable: false, width: 100%)[
    #grid(
      columns: (1fr,) * cols,
      gutter: 1em,
      ..cards.pos()
    )
  ]
}

// --- Evidenceボックス（breakable対応） ---
#let evidence-box(ev-num, title, body) = {
  block(
    width: 100%,
    inset: 1em,
    radius: 4pt,
    stroke: 1pt + color-border,
    breakable: true,
  )[
    #grid(
      columns: (auto, 1fr),
      gutter: 0.8em,
      block(
        inset: 0.4em,
        radius: 2pt,
        fill: color-blue,
      )[
        #text(size: 9pt, weight: "bold", fill: white)[Evidence #ev-num]
      ],
      text(size: 11pt, weight: "bold", fill: color-dark-gray)[#title],
    )
    #v(0.3cm)
    #body
  ]
}

// --- インサイトボックス（breakable対応） ---
#let insight-box(body) = {
  block(
    width: 100%,
    inset: 1em,
    radius: 4pt,
    fill: rgb("#FFF8E1"),
    stroke: 1pt + rgb("#FFC107"),
    breakable: true,
  )[
    #text(size: 9pt, weight: "bold", fill: rgb("#F57F17"))[💡 Key Insight]
    #v(0.2cm)
    #body
  ]
}

// --- 注釈ボックス ---
#let note-box(body) = {
  block(
    width: 100%,
    inset: 0.8em,
    radius: 4pt,
    fill: color-light-gray,
    stroke: (left: 3pt + color-accent),
  )[
    #text(size: 9pt)[#body]
  ]
}

// --- スナップショット画像挿入 ---
#let snapshot-figure(path, caption: none) = {
  figure(
    image(path, width: 100%),
    caption: if caption != none { text(size: 8pt)[#caption] },
  )
}

// --- BCG風スタイルテーブル ---
// 使い方:
//   #styled-table(
//     columns: (1fr, 2fr, 1fr),
//     header: ([順位], [出願人], [件数]),
//     [1], [トヨタ自動車], [245],
//     [2], [本田技研工業], [198],
//   )
#let styled-table(columns: auto, header: (), align: auto, ..rows) = {
  let header-cells = header.map(h =>
    table.cell(fill: color-navy)[
      #text(size: 9pt, weight: "bold", fill: white)[#h]
    ]
  )
  let body-cells = rows.pos()

  table(
    columns: columns,
    align: if align != auto { align } else { left },
    inset: 0.6em,
    stroke: (x, y) => {
      if y == 0 {
        // ヘッダー下に太線
        (bottom: 2pt + color-navy)
      } else {
        // ボディ行は薄い下線のみ
        (bottom: 0.5pt + color-border)
      }
    },
    fill: (x, y) => {
      if y == 0 { color-navy }
      else if calc.rem(y, 2) == 0 { color-zebra }
      else { white }
    },
    table.header(..header-cells),
    ..body-cells,
  )
}

// ==================================================================
// 結論・提言セクション用関数
// ==================================================================

// --- 結論ボックス（戦略的提言ハイライト用） ---
#let conclusion-box(title, body) = {
  block(
    width: 100%,
    inset: 1.2em,
    radius: 4pt,
    fill: rgb("#EDE7F6"),
    stroke: 1pt + rgb("#7E57C2"),
    breakable: true,
  )[
    #text(size: 10pt, weight: "bold", fill: rgb("#4527A0"))[#title]
    #v(0.3cm)
    #body
  ]
}

// --- 推奨アクションカード（優先度別: 高=赤、中=橙、低=緑） ---
#let recommendation-card(priority, title, description, timeframe: none) = {
  let border-color = if priority == "高" { rgb("#E53935") } else if priority == "中" { rgb("#FB8C00") } else { rgb("#43A047") }
  let badge-fill = if priority == "高" { rgb("#FFEBEE") } else if priority == "中" { rgb("#FFF3E0") } else { rgb("#E8F5E9") }
  block(
    width: 100%,
    inset: 0.8em,
    radius: 4pt,
    fill: color-light-gray,
    stroke: (left: 3pt + border-color),
    breakable: false,
  )[
    #grid(
      columns: (auto, 1fr),
      gutter: 0.5em,
      block(
        inset: 0.3em,
        radius: 2pt,
        fill: badge-fill,
      )[
        #text(size: 8pt, weight: "bold")[優先度: #priority]
      ],
      text(size: 11pt, weight: "bold", fill: color-dark-gray)[#title],
    )
    #v(0.2cm)
    #text(size: 9pt)[#description]
    #if timeframe != none {
      v(0.1cm)
      text(size: 8pt, fill: color-medium-gray)[推奨実施時期: #timeframe]
    }
  ]
}

// --- アクションアイテムリスト ---
#let action-items(..items) = {
  block(
    width: 100%,
    inset: 1em,
    radius: 4pt,
    stroke: 1pt + color-border,
    breakable: true,
  )[
    #text(size: 10pt, weight: "bold", fill: color-navy)[Action Items]
    #v(0.3cm)
    #for item in items.pos() {
      block(inset: (left: 1em, bottom: 0.3em))[
        #text(size: 9pt)[☐ #item]
      ]
    }
  ]
}
