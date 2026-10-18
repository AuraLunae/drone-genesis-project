# ドローン自律飛行プロジェクト

## プロジェクトの現状
* 目的: 風外乱下でも安定してホバリング・目標到達できる、自律飛行ドローンの強化学習方策を作ること。
* 現在の構成: Unityは完全に廃止しました。Python完結で、Genesis World(GPU並列物理シミュレータ)+ rsl-rl-lib(PPO)
* 物理モデル: Genesisの組み込みのツールである、DroneEntity(Crazyflie 2.X / CF2Xモデル)による4プロペラのRPM直接制御(3軸の単純化された推力モデルではなく、実機に近いローレベル制御)
* 環境構築: uv(Pythonのバージョン指定も行う)
* 風: Ornstein-Uhlenbeck過程(OU過程)による時間変化する乱気流。エピソードごとに強さ・方向・乱れ方を再抽選(ドメインランダム化)
* 16384並列での学習を成功させようとしている。現在は報酬モデルを「遠さへの継続減点+生存ボーナス+目標到達ボーナス」の構成に拡張したところです。
* やりたいこと:
  * seabornで、綺麗なグラスを描きたい
  * より良い報酬モデルを考えたい
  * より、リアルに近い実装を心がけたい

## ファイル構成
```
drone-genesis-project/
├── wind_hover_env.py   # 環境定義(公式HoverEnv + OU過程の風 + 拡張報酬)
├── train.py             # 学習スクリプト(公式hover_train.pyベース)
├── eval.py               # 評価スクリプト(公式hover_eval.pyベース)
├── wind_flag.py          # 風の強さと向きを表示する旗(布/板)
├── pyproject.toml       # uv用の依存関係定義
└── README.md
```

## 風の可視化

評価スクリプトでは、シミュレーション中の風の強さと向きを白い旗で表示します。
既定はGenesisのPBDソルバーで動く布旗を、粒子ではなく変形メッシュとして表示します。支柱は剛体ソルバーを使い、
布とドローンなどの剛体間の衝突連成を無効にすることで、旗が飛行に干渉しないようにしています。
旗の表示は評価時だけで、学習の物理や報酬には影響しません。

```powershell
# PBD布旗を変形メッシュで表示(既定)。--wind-flag も同じオプションとして使えます
uv run python eval.py -e wind-hovering --run latest

# PBD粒子を表示する(粒が見えるため、通常は上の既定表示を推奨)
uv run python eval.py -e wind-hovering --run latest --flag-style cloth

# 変形しない板旗にする
uv run python eval.py -e wind-hovering --run latest --flag-style plate

# AI・PID・人間の比較アリーナ(旗の種類を指定可能)
uv run python compare_arena.py
```

旗が垂れているとほぼ無風、持ち上がるほど風が強く、旗が向く方向が風下です。
`cloth-mesh` は布の物理にGenesisのPBDソルバーを使い、描画は変形する面にするため、大きな粒子が並ぶ表示になりません。


## ベースのコード
* [Genesis world公式 ](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_train.py)
* [Qiita WSL2+Ubuntu 24.04でGenesis環境構築～学習までやってみた](https://qiita.com/t_tanno/items/a17f0717d136a4f96da0)
