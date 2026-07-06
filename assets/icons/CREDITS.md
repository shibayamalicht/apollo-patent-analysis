# モジュールアイコン 画像クレジット

APOLLO 各モジュールのアイコン（`assets/icons/<key>_icon.png`、256px 円形）は、
NASA 等が公開する**パブリックドメイン（PD）**画像をもとに作成しています。
原画は `assets/icons/_raw/` に保管（差し替え前の旧原画は `_raw/_replaced/`）。
取得元はすべて Wikimedia Commons、取得日は 2026-06-21。
加工内容：正方センタークロップ → 円形アルファマスク → 256px 縮小（Pillow）。
一部（EAGLE）は黒い宇宙背景で被写体が暗いため、被写体へのズームと
明るさ・コントラスト・彩度の補正を追加（原画は同一の NASA PD 画像）。

| key | モジュール | 題材 | 出所（クレジット） | ライセンス |
|---|---|---|---|---|
| home | Home / Mission Control | ハッブル宇宙望遠鏡（STS-125/SM4 撮影） | NASA | パブリックドメイン |
| earth | ATLAS | 地球（DSCOVR EPIC・西半球＝太平洋／南北アメリカ, 2021-06-21） | NASA / DSCOVR EPIC | パブリックドメイン |
| core | CORE | 太陽 全面（SDO, プロミネンス／CME） | NASA Goddard / SDO | パブリックドメイン |
| saturnv | Saturn V | アポロ11号 サターンV 打上げ（機体クローズアップ） | NASA | パブリックドメイン |
| eagle | EAGLE | 月着陸船「イーグル」（着陸形態, コロンビアから撮影） | NASA | パブリックドメイン |
| mega | MEGA | 観測ロケットの光跡（NASA VISIONS 等） | NASA | パブリックドメイン |
| explorer | Explorer | 天の川 銀河中心 コンポジット | NASA/JPL-Caltech/ESA/CXC/STScI | パブリックドメイン |
| crew | CREW | 球状星団 ケンタウルス座オメガ星団 | NASA/ESA（ハッブル） | パブリックドメイン |
| nebula | NEBULA | ジェイムズ・ウェッブ「宇宙の崖」（カリーナ星雲） | NASA/ESA/CSA/STScI | パブリックドメイン |
| record | VOYAGER | ボイジャー ゴールデンレコード | NASA | パブリックドメイン |
| capcom | CAPCOM | ゴールドストーン深宇宙通信施設 DSS-14 アンテナ | NASA/JPL | パブリックドメイン |

## 注記
- 旧バージョンでは CORE に ESO（eso1703e, CC BY 4.0）、Explorer に CC BY の天の川写真を
  使用していたが、**厳密なパブリックドメインに統一**するため NASA 系 PD 画像へ差し替えた
  （ユーザー方針 2026-06-21）。差し替え前の原画は `_raw/_replaced/` に保管。
- NASA が作成した著作物は原則としてパブリックドメイン（米国政府著作物）。
  ハッブル／ウェッブ等で ESA・STScI 等が共同クレジットされる場合も、本セットで採用した
  各画像は Wikimedia Commons 上でパブリックドメインとして提供されているものを選定した。
