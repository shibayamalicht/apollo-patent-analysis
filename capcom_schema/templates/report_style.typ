// ==================================================================
// report_style.typ — APOLLO v9.0.0 レポート共通スタイル
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
// 走査層: 本文の「数値＋単位」を navy 太字で強調する正規表現と、その打ち消しヘルパー。
// 【設計原則 — 必読】num-rx の navy ハイライトは「本文（地の文）のみ」に効かせる。
//   ・大域 `show num-rx`（apollo-report 内・body 直前で宣言）は body 全体に効くため、
//     “色を持つ文脈”（濃い背景・色付きテキスト）では文字が背景に溶けたり色が浮いたりする。
//   ・そうした文脈では必ず `no-num-hl(...)` で打ち消す（= ローカル show が大域 show を上書き）。
//   ・打ち消し適用済みの場所: 見出し H1/H2/H3 ／ styled-table ヘッダ ／
//     conclusion-box・recommendation-card・evidence-box のタイトル。
//   ・カバーページは「num-rx の show 規則をカバー描画より後で宣言する」ことで対象外化している。
//   ★新たに「色付き背景＋テキスト」のコンポーネントを足す時は、そのラベル/見出しを
//     必ず `no-num-hl(...)` で囲むこと（でないと数値が溶ける/色が浮く同種バグが再発する）。
// ==================================================================
// 数値＋単位を拾う。先頭の `(?:…[〜～~–-]…)?` は範囲表記（例「20〜50件」）の前半を任意で取り込み、
// 範囲全体を1マッチにする（これが無いと前半「20」に単位が付かず、後半「50件」だけが強調されて不揃いになる）。
#let num-rx = regex("(?:[0-9][0-9,.]*\s*[〜～~–-]\s*)?[0-9][0-9,.]*\s*(件|％|%|倍|社|ポイント)")
// no-num-hl: 大域 `show num-rx`（navy 太字）の再マッチを“数字と単位の間に語結合子
//   U+2060（WORD JOINER・幅ゼロ・不可視）を挿入して断ち切る”ことで打ち消すヘルパー。
//   旧実装 `{ show num-rx: c => c; body }` は機能しない（`c => c` の出力を大域 `show num-rx`
//   が再マッチして見出し数値が navy になる）。U+2060 を入れると `[0-9]…\s*(社)` の連続が
//   切れ、グローバル規則が再マッチできなくなる。本文（no-num-hl 非適用箇所）の強調は従来どおり効く。
#let no-num-hl(body) = {
  show num-rx: it => {
    let s = it.text
    for u in ("件", "％", "%", "倍", "社", "ポイント") { s = s.replace(u, "\u{2060}" + u) }
    s
  }
  body
}

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

  // --- 段落設定（段落間に余白を確保し、壁のような密度を緩和）---
  // leading=行内行間 / spacing=段落間。spacing を広げると「地の文の塊」が分節されて読みやすくなる。
  set par(leading: 0.82em, spacing: 1.5em, justify: true)

  // --- 見出しスタイル ---
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(1cm)
    block(width: 100%)[
      #line(length: 100%, stroke: 2pt + color-navy)
      #v(0.3cm)
      #text(size: 20pt, weight: "bold", fill: color-navy)[#no-num-hl(it.body)]
      #v(0.5cm)
    ]
  }
  show heading.where(level: 2): it => {
    v(0.6cm)
    block(width: 100%)[
      #grid(
        columns: (auto, 1fr),
        gutter: 0.5em,
        align: (left + horizon, left + horizon),
        rect(width: 5pt, height: 1.0em, fill: color-accent, stroke: none, radius: 1pt),
        text(size: 14pt, weight: "bold", fill: color-blue)[#no-num-hl(it.body)],
      )
      #v(0.1cm)
      #line(length: 100%, stroke: 0.7pt + color-border)
      #v(0.35cm)
    ]
  }
  show heading.where(level: 3): it => {
    v(0.35cm)
    text(size: 12pt, weight: "bold", fill: color-dark-gray)[#no-num-hl(it.body)]
    v(0.2cm)
  }

  // --- テーブルのデフォルトスタイル ---
  set table(stroke: none)
  show table: set text(size: 9pt)
  // 表セル内は両端揃えを無効化（狭い列で和文の字間が間延びするのを防ぐ）
  show table: set par(justify: false, leading: 0.7em)

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

  // --- 走査層: 本文中の「数値＋単位」を控えめに強調（地の文を走査可能にする）---
  // 件数・百分率・倍率・社数など “意味のあるキー数値” だけを太字＋ネイビーに。
  // 年号（2024年）や特許番号（2026-094730）は単位を伴わないため対象外＝過剰強調を防ぐ。
  // ⚠️ この規則は必ず「カバーページの後・body の直前」に置くこと。カバーは濃紺背景なので、
  //    数値を濃紺太字にすると副題等が背景に溶けて見えなくなる（過去のバグ）。本文だけに適用する。
  show num-rx: it => text(weight: "bold", fill: color-navy)[#it]

  // 本文の太字（*…*＝strong）も強調色をネイビーに統一する。
  // これをしないと、会社名等の *太字*（既定の濃灰 #333）と数値ハイライト（ネイビー）が
  // 隣り合って「二色」になり、色が不揃いに見える（例: ミクロ分析B の出願人プロファイル）。
  // ※ 見出し・各種ボックスのラベル・カバーは strong でなく明示 text() を使うため影響を受けない。
  show strong: set text(fill: color-navy)

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
// 使い方: #kpi-dashboard(cols: 3,
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
      text(size: 11pt, weight: "bold", fill: color-dark-gray)[#no-num-hl(title)],
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

// --- 章末まとめボックス（各章末の「本章のまとめ」を薄黄で囲む。走査層）---
// 各モジュール章（= 見出し単位）の末尾に1個だけ置く。本章で読み取った要点（事実→洞察）を
// 3〜5行で凝縮する。insight-box と同系の薄黄だがラベルが「本章のまとめ」で役割が異なる
// （insight-box=章中の個別の気づき／chapter-summary=章全体の締め）。
// phase_d_gate.sh Check 25 が各章末での有無を検査する（任意WARN）。
#let chapter-summary(body) = {
  block(
    width: 100%,
    inset: 1em,
    radius: 4pt,
    fill: rgb("#FFF8E1"),
    stroke: 1pt + rgb("#FFC107"),
    breakable: true,
  )[
    #text(size: 9pt, weight: "bold", fill: rgb("#F57F17"))[📌 本章のまとめ]
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

// --- 要点ストリップ（セクション冒頭の「結論先出し」1行。走査層）---
// 思想: 散文（精読層）は下にそのまま残し、その“1文サマリ”だけを枠で先出しする。
//       これは散文の代替ではない。要点だけを書いて散文を削る＝品質不合格（内容が薄くなる）。
// 使い方: == セクション見出し の直後に1個だけ置く。本文の結論を1〜2文で。
#let point-lead(body) = {
  block(
    width: 100%,
    inset: (left: 0.9em, right: 0.9em, top: 0.5em, bottom: 0.5em),
    radius: (top-right: 3pt, bottom-right: 3pt),
    fill: color-light-blue,
    stroke: (left: 3pt + color-accent),
    breakable: false,
  )[
    #grid(
      columns: (auto, 1fr),
      gutter: 0.7em,
      align: (left + horizon, left + horizon),
      text(size: 8pt, weight: "bold", fill: color-accent, tracking: 0.5pt)[要点],
      {
        set par(justify: false, leading: 0.75em)
        text(size: 9.5pt, fill: color-navy)[#body]
      },
    )
  ]
  v(0.2cm)
}

// --- インライン強調（地の文中のキー数値・キーワードを選択的に。多用禁止）---
// 自動の数値強調（数値＋単位）に加え、文脈上とくに効かせたい語に手動で使う。
// 1段落に何個も使うと強調が無意味化する。1セクションあたり数語まで。
#let hl(body) = text(weight: "bold", fill: color-blue)[#body]

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
      #text(size: 9pt, weight: "bold", fill: white)[#no-num-hl(h)]
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
    #text(size: 10pt, weight: "bold", fill: rgb("#4527A0"))[#no-num-hl(title)]
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
      text(size: 11pt, weight: "bold", fill: color-dark-gray)[#no-num-hl(title)],
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

// ==================================================================
// 構造化分析技法（ACH／リンチピン）用関数
// 使い方・執筆ルールは analysis/structured_techniques.md を参照。
// 位置づけ: これらの枠は結論の「正しさ」を保証しない。判断（対立仮説・弁別材料・
// 要の前提・崩れる条件）を人間が安く監査できる形に外部化する＝注意の誘導である。
// ==================================================================

// --- 競合仮説ボックス（ACH 軽量版・inline） ---
// 使い方:
//   #competing-hypotheses(
//     main: "採用した結論（主仮説）",
//     rivals: (
//       (hypothesis: "退けた対立仮説1（実際に支持のあった、もっともらしいもの）",
//        counter_evidence: "対立仮説1と食い違う具体的な材料（HHI値・特許番号・#footnote 等）"),
//       (hypothesis: "退けた対立仮説2", counter_evidence: "..."),
//     ),
//   )
// ⚠️ counter_evidence は「主結論を支持する材料」ではなく「その別解釈と矛盾する事実」。
//    ほとんどの事実は複数の解釈と両立するので判断を動かさない。別解釈と食い違う事実だけが決め手になる。
// ⚠️ かかし禁止: 退けるために立てた（どの材料にも支持されていない）弱い別解釈を並べない。
// ⚠️ 用語: 本文・ラベルは読者向けの日本語（別解釈・決め手）。技法名（ACH・対立仮説・弁別材料）は
//    スキーマ内部用語であり、レポート出力に書かない（terminology.md「構造化分析の読者向け用語」）。
// phase_d_gate.sh Check 30（有無）・Check 30b（決め手の裏付け）が検査する。
#let competing-hypotheses(main: "", rivals: ()) = {
  block(
    width: 100%,
    inset: 1em,
    radius: 4pt,
    fill: rgb("#E0F2F1"),
    stroke: 1pt + rgb("#26A69A"),
    breakable: true,
  )[
    #text(size: 9pt, weight: "bold", fill: rgb("#00695C"))[⚖️ 結論の検証 — 別解釈との比較]
    #v(0.25cm)
    #text(size: 9.5pt)[#text(weight: "bold")[採用した結論:] #main]
    #for r in rivals {
      v(0.25cm)
      block(
        width: 100%,
        inset: (left: 0.8em, top: 0.45em, bottom: 0.45em, right: 0.6em),
        fill: white,
        radius: 3pt,
        stroke: (left: 2.5pt + rgb("#26A69A")),
      )[
        #text(size: 9pt)[#text(weight: "bold", fill: rgb("#00695C"))[検討した別解釈:] #r.hypothesis]
        #v(0.15cm)
        #text(size: 9pt)[#text(weight: "bold", fill: rgb("#00695C"))[決め手（この解釈と食い違う事実）:] #r.counter_evidence]
      ]
    }
  ]
}

// --- リンチピン（要の前提）ボックス ---
// 一つの結論について (1) 結論が依存する要の前提 (2) その前提を確認したデータ
// (3) この結論を修正すべき具体的な観測（崩れる条件）を外部化する。
// 使い方:
//   #linchpin(
//     conclusion: "本報告書の主要結論",
//     pillars: (
//       (premise: "結論が依存する要の前提1",
//        verified_by: "確認した具体的なデータ（特許番号・数値・ファミリー情報など）",
//        overturns_if: "この結論を修正すべき具体的な観測（例: 主要3極で権利化が進行していたら）"),
//     ),
//   )
// ⚠️ overturns_if は観測できる具体的な出来事を書く（「新しい情報が出たら」のような空文は禁止）。
//    見直しのサインを定義できない結論は過信の兆候である。
// ⚠️ 用語: ラベルは読者向け日本語。「リンチピン」はスキーマ内部用語＝レポート出力に書かない。
// phase_d_gate.sh Check 31（有無）・Check 31b（見直しサインの裏付け）が検査する。
#let linchpin(conclusion: "", pillars: ()) = {
  block(
    width: 100%,
    inset: 1em,
    radius: 4pt,
    fill: rgb("#FBE9E7"),
    stroke: 1pt + rgb("#FF7043"),
    breakable: true,
  )[
    #text(size: 9pt, weight: "bold", fill: rgb("#D84315"))[🧷 結論の前提と、見直しのサイン]
    #v(0.25cm)
    #text(size: 9.5pt)[#text(weight: "bold")[結論:] #conclusion]
    #for p in pillars {
      v(0.25cm)
      block(
        width: 100%,
        inset: (left: 0.8em, top: 0.45em, bottom: 0.45em, right: 0.6em),
        fill: white,
        radius: 3pt,
        stroke: (left: 2.5pt + rgb("#FF7043")),
      )[
        #text(size: 9pt)[#text(weight: "bold", fill: rgb("#D84315"))[この結論が依存する前提:] #p.premise]
        #v(0.15cm)
        #text(size: 9pt)[#text(weight: "bold", fill: rgb("#D84315"))[確認した事実:] #p.verified_by]
        #v(0.15cm)
        #text(size: 9pt)[#text(weight: "bold", fill: rgb("#D84315"))[この結論を見直すべきサイン:] #p.overturns_if]
      ]
    }
  ]
}

// --- ACH マトリクス（仮説検証サマリ・中心的な問い1つに限って描く） ---
// 列=仮説、行=材料（出所をたどれる観測）、セル="○"（両立）/"×"（食い違い）/"—"（判断保留）。
// 自動描画: 各仮説の×の数の集計行、×が最も少ない仮説に ◎（残った仮説）、
//           全列○の行（弁別力ゼロ）を灰色化。
// 使い方:
//   #ach-matrix(
//     question: "本報告書の中心的な問い",
//     hypotheses: ("仮説1", "仮説2", "仮説3"),
//     evidence: (
//       (item: "材料1（HHIの経年推移・出所つき）", cells: ("×", "○", "○")),
//       (item: "材料2（新規出願人の有無）",       cells: ("×", "○", "—")),
//     ),
//   )
// ⚠️ 非弁別の行（全列○）だけで表を埋めない。×の判断を結論に合わせて後付けしない（先に材料を見る）。
// ⚠️ 表の直後に「残った仮説」を結論として明示する段落を必ず書く（マトリクス→結論→提言の接続）。
// phase_d_gate.sh Check 33（×が皆無の非弁別マトリクス）・Check 34（提言との接続）が検査する。
#let ach-matrix(question: "", hypotheses: (), evidence: ()) = {
  let n = hypotheses.len()
  // 各仮説（列）の「食い違い（×）」の数を集計する
  let xcounts = range(n).map(i => evidence.filter(e => e.cells.at(i) == "×").len())
  let minx = if xcounts.len() > 0 { calc.min(..xcounts) } else { 0 }
  block(width: 100%, breakable: true)[
    #text(size: 10pt, weight: "bold", fill: color-navy)[⚖️ 仮説の比較検証]
    #v(0.15cm)
    #text(size: 9.5pt)[#text(weight: "bold")[中心的な問い:] #question]
    #v(0.25cm)
    #table(
      columns: (2.4fr,) + range(n).map(i => 1fr),
      inset: 0.5em,
      stroke: (x, y) => if y == 0 { (bottom: 2pt + color-navy) } else { (bottom: 0.5pt + color-border) },
      fill: (x, y) => if y == 0 { color-navy } else if calc.rem(y, 2) == 0 { color-zebra } else { white },
      table.header(
        table.cell(fill: color-navy)[#text(size: 8.5pt, weight: "bold", fill: white)[事実（出所つき）]],
        ..range(n).map(i => table.cell(fill: color-navy)[
          #text(size: 8.5pt, weight: "bold", fill: white)[#no-num-hl(hypotheses.at(i))#if xcounts.at(i) == minx [ ◎]]
        ]),
      ),
      ..evidence.map(e => {
        // 全列○ = どの仮説とも両立 = 弁別力ゼロの行（灰色で示す）
        let nondiscr = e.cells.all(c => c == "○")
        let tcol = if nondiscr { color-medium-gray } else { color-dark-gray }
        ([#text(size: 8.5pt, fill: tcol)[#e.item#if nondiscr [（決め手にならず）]]],) + e.cells.map(c => [
          #align(center)[#text(size: 9pt,
            weight: if c == "×" { "bold" } else { "regular" },
            fill: if c == "×" { rgb("#C62828") } else { tcol })[#c]]
        ])
      }).flatten(),
      table.cell(fill: color-light-blue)[#text(size: 8.5pt, weight: "bold")[矛盾（×）の数]],
      ..range(n).map(i => table.cell(fill: color-light-blue)[
        #align(center)[#text(size: 9pt, weight: "bold",
          fill: if xcounts.at(i) == minx { rgb("#00695C") } else { color-dark-gray })[#xcounts.at(i)]]
      ]),
    )
    #v(0.1cm)
    #text(size: 8pt, fill: color-medium-gray)[○=その解釈と整合 ／ ×=その解釈と矛盾 ／ —=判断保留。◎=矛盾が最も少ない解釈（採用）。全列○の行は判断の決め手にならないため灰色で示す。○×の判定は分析者の判断であり正しさの証明ではない——×のセルこそ根拠を確認されたい。]
  ]
}
