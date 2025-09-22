# TiffEditor

> **🚀 新しいリポジトリ**: より良い名前とクリーンな構造を持つ新しいホームへようこそ！

メモリ効率の良い巨大 TIFF ファイルの部分編集を可能にする Python ライブラリです。

## ✨ 主な特徴

- 🚀 **メモリ効率**: メモリに乗らない巨大な TIFF ファイルでも部分的な読み書きが可能
- 🔧 **タイル構造**: TIFF のタイル機能を活用した高速アクセス
- 🎯 **スライス記法**: NumPy ライクなスライス記法でのデータアクセス
- 📊 **データ整合性**: 読み書きの整合性チェック機能内蔵
- 💾 **部分更新**: 既存ファイルの一部のみを効率的に更新可能
- 🎨 **BGR 形式統一**: OpenCV（cv2）との完全互換
- ⚡ **ScalableTiffEditor**: 仮想的な大画像操作でプロトタイプ開発を効率化

## 📦 インストール

### PyPI からインストール（推奨）

```bash
pip install tiffeditor
```

### GitHub から直接インストール

```bash
pip install git+https://github.com/vitroid/tiffeditor.git
```

### 開発環境セットアップ

```bash
# プロジェクトをクローン
git clone https://github.com/vitroid/tiffeditor.git
cd tiffeditor

# 依存関係をインストール（開発用）
make install-dev
# または
poetry install

# 仮想環境をアクティブ化
poetry shell
```

## 🚀 基本的な使用方法

### 通常の TIFF 編集

```python
from tiffeditor import TiffEditor
import numpy as np

# 新しいTIFFファイルを作成
with TiffEditor(
    filepath="large_image.tiff",
    mode="w",
    shape=(10000, 10000, 3),  # 高さ×幅×チャンネル
    dtype=np.uint8,
    create_if_not_exists=True,
) as editor:
    # 部分的にデータを書き込み（BGR形式）
    test_data = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    editor[1000:2000, 2000:3000] = test_data

    print(f"ファイル情報: {editor.get_info()}")
```

### 仮想的な大画像操作 (ScalableTiffEditor)

```python
from tiffeditor import ScalableTiffEditor
import numpy as np

# 仮想的に8000x6000の画像だが、実際は800x600で保存
with ScalableTiffEditor(
    filepath="virtual_large.tiff",
    virtual_shape=(6000, 8000, 3),  # 仮想的なサイズ
    scale_factor=0.1,                # 実際のサイズ = 仮想サイズ × 0.1
    mode="w",
    dtype=np.uint8,
    create_if_not_exists=True,
) as editor:
    # ユーザーは大きな座標で操作
    test_data = np.zeros((1000, 1000, 3), dtype=np.uint8)
    test_data[:, :, 2] = 255  # BGR形式で赤色

    # 仮想座標で書き込み（実際には100x100にリサイズされて保存）
    editor[1000:2000, 1500:2500] = test_data

    # 仮想座標で読み込み（実際のデータが1000x1000にリサイズされて返される）
    read_data = editor[1000:2000, 1500:2500]
    print(f"読み込んだデータの形状: {read_data.shape}")  # (1000, 1000, 3)
```

## 🎯 主要クラス

### `TiffEditor`

メインクラス。BGR 形式で TIFF ファイルの読み書きを担当。OpenCV（cv2）との完全互換。

### `ScalableTiffEditor`

仮想的に大きな画像を扱いながら、実際には縮小されたファイルで操作を行う拡張クラス。

**主要な特徴:**

- 仮想座標系での操作
- 自動的なスケーリング（拡大・縮小）
- メモリ効率の大幅改善
- 透明な座標変換

## 🔧 テストと開発

### クイックテスト

```bash
# 使用例を実行
python example_usage.py

# 基本的なエディタテスト
python tiffeditor.py test_editor

# ScalableTiffEditorのテスト
python tiffeditor.py test_scalable

# メモリ効率テスト（大容量ファイル）
python tiffeditor.py large_test
```

### 開発者向け Make コマンド

```bash
# ヘルプを表示
make help

# 全チェック実行（フォーマット + lint + 型チェック + テスト）
make all-checks

# コミット前チェック
make pre-commit

# パッケージビルド
make build

# TestPyPIにアップロード
make upload-test

# 本番PyPIにアップロード
make upload
```

## 📈 パフォーマンス

**メモリに乗らないサイズ（6.47GB）のファイルを、わずか 1.31GB のメモリで処理！**

```
INFO: 利用可能メモリ: 4.24 GB
INFO: 作成予定サイズ: 47717x47717x3
INFO: 予想ファイルサイズ: 6.36 GB
INFO: 実際のファイルサイズ: 6.47 GB
INFO: メモリ使用量変化: 52.8MB -> 1311.8MB
INFO: ✅ 大きなTIFFファイルの作成・整合性チェックが成功しました！
```

## 🏗️ 技術仕様

- **依存関係**: numpy, rasterio, tifffile, opencv-python
- **対応形式**: タイル化 TIFF（Tiled TIFF）
- **データ型**: uint8, uint16, float32 など（NumPy 対応型）
- **色形式**: BGR（OpenCV 互換）
- **最大ファイルサイズ**: システムディスク容量に依存

## 🔄 移行ガイド

### 既存の `rasterio_tiff` からの移行

```python
# 旧: rasterio_tiff
from rasterio_tiff import TiffEditor

# 新: tiffeditor
from tiffeditor import TiffEditor

# API は 100% 互換！追加コードは必要ありません
```

## 📄 ライセンス

MIT License

## 🤝 コントリビューション

Issue や Pull Request を歓迎します！

---

**注意**: BGR 形式での統一により、OpenCV（cv2）との混在利用が完全に安全になりました。
