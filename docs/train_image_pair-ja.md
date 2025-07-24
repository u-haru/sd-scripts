# 画像のペアを用いた学習ガイド

画像のペアを用いた学習のガイドです。ADDifTとFLUX.1 Kontextの2つの手法をサポートしています。

[学習についての共通ドキュメント](./train_README-ja.md) もあわせてご覧ください。

## ADDifTの概要と学習方法について

ADDifTはAlternating Direct Difference Trainingの略で、画像ペアを用いて差分を学習する手法です。細かな理論については以下のnoteを参照してください。
[交替直接差分学習法ADDifT(Alternating Direct Difference Training)の解説](https://note.com/hakomikan/n/n716397e39d56)

### 学習の手順

あらかじめこのリポジトリのREADMEを参照し、環境整備を行ってください。

#### データの準備

[学習データの準備について](./train_README-ja.md) を参照してください。

このスクリプトでは、ADDifTの学習を行うために、画像ペアを用意する必要があります。具体的には、画像のファイル名に`_target`が含まれる画像をターゲット画像、含まれない画像をソース画像としてペアにします。

ここでは、コマンドライン引数により設定を行う方法を例示しています。tomlの設定ファイルを用いた設定も可能です。\
例として、画像フォルダは以下のような構成になります。

```
50_large_ears/
├── img.png
├── img.txt
├── img_target.png
└── img_target.txt
```

例えば"耳を大きくする"という学習を行う場合、ソース画像(`img.png`)に耳が小さい画像、ターゲット画像(`img_target.png`)に耳が大きい画像を用意します。\
キャプションは同じでも構いません。ターゲット(`img_target.txt`)にのみ特定のキャプションを含めると、その部分に差分が紐づくようになります。

ターゲット画像とソース画像は、学習したい部分以外は同じ内容である必要があります。ピクセル単位で一致していることが望ましいです。

#### 学習の実行

スクリプトを実行します。以下に例を示します。必要に応じて他のオプションを追加してください。

```
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
  --network_module=networks.lora \
  --output_dir=<学習したモデルの出力先フォルダ> \
  --output_name=<学習したモデル出力時のファイル名> \
  --train_data_dir=<上記の画像フォルダが含まれる親フォルダ> \
  --pretrained_model_name_or_path=<.ckptまたは.safetensordのパス> \
  --caption_extension=.txt \
  --save_model_as=safetensors \
  --optimizer_type=Lion \
  --lr_scheduler=cosine \
  --resolution=1024 \
  --learning_rate=1e-04 \
  --text_encoder_lr=1e-05 \
  --image_pair_training=addift \  # ADDifTを有効化
  --min_timestep=500 \  # 最小タイムステップ
  --max_timestep=1000 \  # 最大タイムステップ
  --network_dim=4
```

重要なオプションは以下の通りです。
- `--image_pair_training=addift`: ADDifTを有効化します。
- `--min_timestep` と `--max_timestep`: タイムステップの範囲を指定します。これにより、ADDifTの学習が行われます。
  0 ~ 1000のような広い範囲での学習は推奨されません。概念の学習などの場合は、500 ~ 1000のように狭い範囲での学習が効果的です。
- `--addift_timesteps_segments`: タイムステップのセグメント数を指定します。ADDifTの学習において、タイムステップを分割して学習するために使用します。デフォルトは5です。
- `--addift_scale`: ADDifTのスケールを指定します。これは、ターゲット画像とソース画像の差分を学習する際の重み付けに使用されます。デフォルトは0.5です。

### 参考: ADDifTでのステップ数について

ADDifTでは画像の差分を直接学習するため、通常の学習よりも少ないステップ数で効果的な学習が可能です。\
学習率にもよりますが、50ステップ程度でも効果が得られる場合が多いです。


## Kontextの概要と学習方法について

FLUX.1 Kontextは画像の編集を可能にするモデルです。このスクリプトでは、ADDifTと同様に画像ペアを用いて学習します。ファイル名などのルールはADDifTと同じです。
FLUX.1 Kontextモデルの学習にのみ対応しています。SDXLや他のモデルでは動作しません。

### 学習の手順

あらかじめこのリポジトリのREADMEを参照し、環境整備を行ってください。

#### データの準備

[学習データの準備について](./train_README-ja.md) を参照してください。

基本的にADDifTと同様の手順でデータを準備します。Kontextでは、ターゲット画像とソース画像のペアを用意し、ファイル名に`_target`を含めることで識別します。注意として、Kontextでは以下のリソースのみが学習に利用されます。

- ソース画像 (`image.png`など)
- ターゲット画像 (`image_target.png`など)
- ターゲット画像のキャプション (`image_target.txt`など)

Kontextでは、ソース画像のキャプションは使用されません。ソース画像は画像入力としてのみ使用され、ターゲット画像のキャプションが学習に影響を与えます。

先ほどと同様に"耳を大きくする"という学習を行う場合、以下のようなフォルダ構成になります。

```
50_large_ears/
├── img.png
├── img_target.png
└── img_target.txt
```

img_target.txtには、画像の編集に使いたいプロンプトを記述します。例えば、"Make this character's ears larger"のように、ターゲット画像に対する具体的な指示を記述します。

なお、"ソース画像のキャプションは使用されない"という点から、次のような構成にすることも可能です。

```
50_Make this character's ears larger/
├── img.png
└── img_target.png
```

キャプションがないファイルにはフォルダのプロンプトが利用される仕様を利用しています。

#### 学習の実行
スクリプトを実行します。以下に例を示します。必要に応じて他のオプションを追加してください。
```
accelerate launch --num_cpu_threads_per_process 1 flux_train_network.py \
  --network_module=networks.lora_flux \
  --output_dir=<学習したモデルの出力先フォルダ> \
  --output_name=<学習したモデル出力時のファイル名> \
  --train_data_dir=<上記の画像フォルダが含まれる親フォルダ> \
  --pretrained_model_name_or_path=<.ckptまたは.safetensordのパス> \
  --clip_l=<clip_lのパス> \
  --t5xxl=<t5xxlのパス> \
  --ae=<aeのパス> \
  --caption_extension=.txt \
  --save_model_as=safetensors \
  --optimizer_type=Lion \
  --lr_scheduler=cosine \
  --resolution=1024 \
  --learning_rate=1e-04 \
  --text_encoder_lr=1e-05 \
  --network_dim=4 \
  --timestep_sampling=flux_shift \
  --guidance_scale=1.0 \
  --model_prediction_type=raw \
  --image_pair_training=kontext
```

重要なオプションは以下の通りです。

- `--image_pair_training=kontext`: Kontextを有効化します。