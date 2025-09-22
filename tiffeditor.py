"""
TiffEditor

メモリ効率の良い巨大TIFFファイルの部分編集を可能にするライブラリ。
OpenCV（cv2）との完全互換性のため、すべてのカラー画像データはBGR形式で扱います。
"""

import logging
import os
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import tifffile
import rasterio
from rasterio.windows import Window
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio import warp
import cv2


__version__ = "0.2.0"
__author__ = "User"
__email__ = ""

# 公開するシンボルを明示
__all__ = ["TiffEditor", "ScalableTiffEditor", "Range", "Rect", "create_sample_tiff"]


@dataclass
class Range:
    """数値の範囲を表現するクラス

    Attributes:
        min_val (int): 範囲の最小値
        max_val (int): 範囲の最大値（min_val + width）
    """

    min_val: int
    max_val: int

    @property
    def width(self) -> int:
        """範囲の幅を返す"""
        return self.max_val - self.min_val

    def overlaps(self, other: "Range") -> bool:
        """他の範囲と重複があるかどうかを判定する"""
        return self.min_val < other.max_val and other.min_val < self.max_val

    def get_overlap(self, other: "Range") -> "Range | None":
        """他の範囲との重複部分を返す。重複がない場合はNoneを返す"""
        if self.overlaps(other):
            return Range(
                max(self.min_val, other.min_val), min(self.max_val, other.max_val)
            )
        return None

    def __and__(self, other: "Range") -> "Range | None":
        """&演算子で範囲の重複を取得する

        Example:
            r1 = Range(0, 10)
            r2 = Range(5, 15)
            overlap = r1 & r2  # Range(5, 10)
        """
        return self.get_overlap(other)

    def as_list(self) -> list[int]:
        """範囲をリストに変換する"""
        return [self.min_val, self.max_val]

    def validate(self, min_size: int = 0):
        if self.min_val >= self.max_val:
            raise ValueError("Invalid region")
        if self.max_val - self.min_val < min_size:
            raise ValueError("Region is too small")


@dataclass
class Rect:
    """2次元の領域を表現するクラス

    Attributes:
        x_range (Range): X方向の範囲
        y_range (Range): Y方向の範囲
    """

    x_range: Range
    y_range: Range

    @classmethod
    def from_bounds(cls, left: int, right: int, top: int, bottom: int) -> "Rect":
        """座標からAreaを作成する

        Args:
            left: X方向の最小値
            right: X方向の最大値
            top: Y方向の最小値
            bottom: Y方向の最大値
        """
        return cls(Range(left, right), Range(top, bottom))

    @property
    def width(self) -> int:
        """領域の幅を返す"""
        return self.x_range.width

    @property
    def height(self) -> int:
        """領域の高さを返す"""
        return self.y_range.width

    @property
    def left(self) -> int:
        """領域の左端を返す"""
        return self.x_range.min_val

    @property
    def right(self) -> int:
        """領域の右端を返す"""
        return self.x_range.max_val

    @property
    def top(self) -> int:
        """領域の上端を返す"""
        return self.y_range.min_val

    @property
    def bottom(self) -> int:
        """領域の下端を返す"""
        return self.y_range.max_val

    def overlaps(self, other: "Rect") -> bool:
        """他の領域と重複があるかどうかを判定する"""
        return self.x_range.overlaps(other.x_range) and self.y_range.overlaps(
            other.y_range
        )

    def get_overlap(self, other: "Rect") -> "Rect | None":
        """他の領域との重複部分を返す。重複がない場合はNoneを返す"""
        x_overlap = self.x_range & other.x_range
        y_overlap = self.y_range & other.y_range
        if x_overlap is not None and y_overlap is not None:
            return Rect(x_overlap, y_overlap)
        return None

    def __and__(self, other: "Rect") -> "Rect | None":
        """&演算子で領域の重複を取得する

        Example:
            a1 = Area.from_coords(0, 10, 0, 10)
            a2 = Area.from_coords(5, 15, 5, 15)
            overlap = a1 & a2  # Area(Range(5, 10), Range(5, 10))
        """
        return self.get_overlap(other)

    def __or__(self, other: "Rect") -> "Rect | None":
        """&演算子で領域の包含を取得する

        Example:
            a1 = Area.from_coords(0, 10, 0, 10)
            a2 = Area.from_coords(5, 15, 5, 15)
            overlap = a1 & a2  # Area(Range(0, 15), Range(0, 15))
        """
        return Rect(
            x_range=Range(
                min_val=min(self.left, other.left),
                max_val=max(self.right, other.right),
            ),
            y_range=Range(
                min_val=min(self.top, other.top),
                max_val=max(self.bottom, other.bottom),
            ),
        )

    def trim(self, image_shape: tuple[int, int]):
        """
        画像の範囲を超える領域をtrimする。
        """
        top = max(0, self.y_range.min_val)
        bottom = min(image_shape[0], self.y_range.max_val)
        left = max(0, self.x_range.min_val)
        right = min(image_shape[1], self.x_range.max_val)
        return Rect(
            x_range=Range(min_val=left, max_val=right),
            y_range=Range(min_val=top, max_val=bottom),
        )

    def to_cvrect(self) -> tuple[int, int, int, int]:
        return (
            self.x_range.min_val,
            self.y_range.min_val,
            self.width,
            self.height,
        )

    def validate(self, min_size: tuple[int, int] = (0, 0)):
        self.x_range.validate(min_size[0])
        self.y_range.validate(min_size[1])


class TiffEditor:
    """
    TIFFファイルの部分編集を可能にするクラス（BGR形式でデータを扱う）

    メモリ効率の良い方法で巨大なTIFFファイルの部分的な読み書きを行う。
    TiledImageの設計思想を参考に、ディスク上のTIFFファイルを直接操作する。
    OpenCV（cv2）との互換性のため、すべてのカラー画像データはBGR形式で扱う。

    Features:
    - 部分的な読み込み（メモリ効率、BGR形式で返す）
    - 部分的な書き込み（既存ファイルの更新、BGR形式で受け取る）
    - タイル構造の活用
    - スライス記法での操作
    - OpenCV（cv2）との完全互換
    """

    def __init__(
        self,
        filepath: str,
        mode: str = "r+",
        tilesize: Union[int, Tuple[int, int]] = 512,
        dtype: Optional[np.dtype] = None,
        shape: Optional[Tuple[int, int, int]] = None,
        create_if_not_exists: bool = False,
    ):
        """
        TiffEditorを初期化する

        Args:
            filepath: TIFFファイルのパス
            mode: ファイルのオープンモード ('r', 'r+', 'w')
            tilesize: タイルサイズ（int または (width, height)のタプル）
            dtype: データ型（新規作成時）
            shape: 画像の形状 (height, width, channels)（新規作成時）
            create_if_not_exists: ファイルが存在しない場合に新規作成するか
        """
        self.filepath = filepath
        self.mode = mode

        if isinstance(tilesize, int):
            self.tilesize = (tilesize, tilesize)
        else:
            self.tilesize = tilesize

        self._tiff_handle = None
        self._rasterio_handle = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # ファイルが存在しない場合の処理
        if not os.path.exists(filepath):
            if mode == "w" or (create_if_not_exists and mode in ["w", "r+"]):
                if shape is None or dtype is None:
                    raise ValueError("新規作成時はshapeとdtypeを指定してください")
                self._create_tiff_file(shape, dtype)
                # ファイル作成後に再度オープンを試行
                self._open_file()
            elif mode != "w":
                raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
        else:
            # 既存ファイルがある場合
            if mode == "w":
                # 書き込みモードの場合は既存ファイルを上書き
                if shape is None or dtype is None:
                    raise ValueError("mode='w'時はshapeとdtypeを指定してください")
                self._create_tiff_file(shape, dtype)
                self._open_file()
            else:
                self._open_file()

    def _create_tiff_file(self, shape: Tuple[int, int, int], dtype: np.dtype):
        """新しいタイル化TIFFファイルを作成する"""
        height, width, channels = shape

        # ダミーデータで初期化
        dummy_data = np.zeros((height, width, channels), dtype=dtype)

        # タイル化TIFFとして保存
        tifffile.imwrite(
            self.filepath,
            dummy_data,
            tile=self.tilesize,
            photometric="rgb" if channels == 3 else "minisblack",
        )

        self.logger.info(f"新しいTIFFファイルを作成しました: {self.filepath}")

    def _open_file(self):
        """ファイルを開く"""
        try:
            if self.mode == "r":
                # 読み込み専用の場合はtifffileを使用
                if os.path.exists(self.filepath):
                    self._tiff_handle = tifffile.TiffFile(self.filepath)
                else:
                    raise FileNotFoundError(
                        f"読み取り用ファイルが存在しません: {self.filepath}"
                    )
            else:
                # 読み書きの場合はrasterioを使用
                if os.path.exists(self.filepath):
                    self._rasterio_handle = rasterio.open(self.filepath, "r+")
                else:
                    raise FileNotFoundError(
                        f"読み書き用ファイルが存在しません: {self.filepath}"
                    )

            # ハンドルが正しく設定されたかチェック
            if not self._tiff_handle and not self._rasterio_handle:
                raise IOError("ファイルハンドルの初期化に失敗しました")

        except Exception as e:
            raise IOError(f"ファイルを開けませんでした: {e}")

    def _ensure_handle_initialized(self):
        """ハンドルが初期化されていることを保証する"""
        if not self._tiff_handle and not self._rasterio_handle:
            self.logger.warning(
                "ハンドルが初期化されていません。再初期化を試行します。"
            )
            self._open_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """ファイルハンドルを閉じる"""
        if self._tiff_handle:
            self._tiff_handle.close()
            self._tiff_handle = None
        if self._rasterio_handle:
            self._rasterio_handle.close()
            self._rasterio_handle = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """画像の形状を取得"""
        # ハンドルが初期化されていることを確認
        self._ensure_handle_initialized()

        if self._rasterio_handle:
            height, width = self._rasterio_handle.height, self._rasterio_handle.width
            channels = self._rasterio_handle.count
        elif self._tiff_handle:
            page = self._tiff_handle.pages[0]
            height, width = page.shape[:2]
            channels = page.shape[2] if len(page.shape) > 2 else 1
        else:
            raise ValueError("ファイルが開かれていません")

        return (height, width, channels)

    @property
    def dtype(self) -> np.dtype:
        """データ型を取得"""
        # ハンドルが初期化されていることを確認
        self._ensure_handle_initialized()

        if self._rasterio_handle:
            return self._rasterio_handle.dtypes[0]
        elif self._tiff_handle:
            return self._tiff_handle.pages[0].dtype
        else:
            raise ValueError("ファイルが開かれていません")

    def _parse_slice(self, key) -> Rect:
        """スライスを解析してRectに変換する（TiledImageと同じ）"""
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("2次元のスライスを指定してください")

        y_slice, x_slice = key
        if not (isinstance(y_slice, slice) and isinstance(x_slice, slice)):
            raise IndexError("スライスを指定してください")

        # 画像の実際のサイズを取得
        height, width, _ = self.shape

        # スライスの開始と終了を取得
        y_start = y_slice.start if y_slice.start is not None else 0
        y_stop = y_slice.stop if y_slice.stop is not None else height
        x_start = x_slice.start if x_slice.start is not None else 0
        x_stop = x_slice.stop if x_slice.stop is not None else width

        # 範囲チェック
        y_start = max(0, min(y_start, height))
        y_stop = max(0, min(y_stop, height))
        x_start = max(0, min(x_start, width))
        x_stop = max(0, min(x_stop, width))

        # ステップは未対応
        if y_slice.step is not None or x_slice.step is not None:
            raise NotImplementedError("ステップ付きスライスには未対応です")

        return Rect.from_bounds(x_start, x_stop, y_start, y_stop)

    def __getitem__(self, key) -> np.ndarray:
        """スライスで領域を取得する（BGR形式で返す）"""
        region = self._parse_slice(key)
        return self.get_region(region)

    def __setitem__(self, key, value: np.ndarray):
        """スライスで領域を設定する（BGR形式で受け取る）"""
        if not isinstance(value, np.ndarray):
            raise TypeError("NumPy配列を指定してください")

        region = self._parse_slice(key)
        self.put_region(region, value)

    def get_region(self, region: Rect) -> np.ndarray:
        """指定された領域のデータを読み込む（BGR形式で返す）"""
        x_start = region.x_range.min_val
        x_stop = region.x_range.max_val
        y_start = region.y_range.min_val
        y_stop = region.y_range.max_val

        width = x_stop - x_start
        height = y_stop - y_start

        if width <= 0 or height <= 0:
            return np.array([])

        if self._rasterio_handle:
            # rasterioを使用した読み込み
            window = Window(x_start, y_start, width, height)
            data = self._rasterio_handle.read(window=window)

            # rasterioは(channels, height, width)で返すので転置
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data[0]  # 単一チャンネルの場合

        elif self._tiff_handle:
            # tifffileを使用した読み込み
            page = self._tiff_handle.pages[0]
            data = page.asarray()[y_start:y_stop, x_start:x_stop]

        else:
            raise ValueError("ファイルが開かれていません")

        # RGB形式で保存されているデータをBGR形式に変換（CV2互換）
        if data.ndim == 3 and data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        elif data.ndim == 3 and data.shape[2] == 4:
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)

        self.logger.debug(
            f"領域を読み込みました（BGR形式）: {region}, shape: {data.shape}"
        )
        return data

    def put_region(self, region: Rect, data: np.ndarray):
        """指定された領域にデータを書き込む（入力はBGR形式、内部でRGBに変換）"""
        if self.mode == "r":
            raise ValueError("読み込み専用モードでは書き込みできません")

        x_start = region.x_range.min_val
        x_stop = region.x_range.max_val
        y_start = region.y_range.min_val
        y_stop = region.y_range.max_val

        width = x_stop - x_start
        height = y_stop - y_start

        if width <= 0 or height <= 0:
            return

        # データサイズの検証
        expected_shape = (height, width)
        if data.ndim == 3:
            expected_shape = (height, width, data.shape[2])
        elif data.ndim == 2:
            expected_shape = (height, width)
        else:
            raise ValueError(f"不正なデータ形状: {data.shape}")

        if data.shape[:2] != (height, width):
            raise ValueError(
                f"データサイズが一致しません。期待: {expected_shape}, 実際: {data.shape}"
            )

        # BGR形式で入力されたデータをRGB形式に変換してからTIFFに保存
        write_data = data.copy()
        if data.ndim == 3 and data.shape[2] == 3:
            write_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        elif data.ndim == 3 and data.shape[2] == 4:
            write_data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)

        if self._rasterio_handle:
            # rasterioを使用した書き込み
            window = Window(x_start, y_start, width, height)

            if write_data.ndim == 3:
                # (height, width, channels) -> (channels, height, width)
                formatted_data = np.transpose(write_data, (2, 0, 1))
                for i in range(formatted_data.shape[0]):
                    self._rasterio_handle.write(formatted_data[i], i + 1, window=window)
            else:
                # 単一チャンネル
                self._rasterio_handle.write(write_data, 1, window=window)

        else:
            raise ValueError("書き込みにはrasterioハンドルが必要です")

        self.logger.debug(
            f"領域に書き込みました（BGR→RGB変換済み）: {region}, shape: {data.shape}"
        )

    def get_info(self) -> dict:
        """ファイルの情報を取得する"""
        shape = self.shape
        dtype = self.dtype

        info = {
            "filepath": self.filepath,
            "shape": shape,
            "dtype": str(dtype),
            "size_mb": os.path.getsize(self.filepath) / (1024 * 1024),
            "tilesize": self.tilesize,
        }

        if self._rasterio_handle:
            info.update(
                {
                    "compression": getattr(
                        self._rasterio_handle, "compression", "unknown"
                    ),
                    "photometric": getattr(
                        self._rasterio_handle, "photometric", "unknown"
                    ),
                    "is_tiled": getattr(self._rasterio_handle, "is_tiled", False),
                }
            )

        return info

    def get_scaled_image(
        self,
        scale_factor: Optional[float] = None,
        target_width: Optional[int] = None,
        resampling: Resampling = Resampling.bilinear,
    ) -> np.ndarray:
        """
        TIFF画像全体の縮小版を効率的に取得する（読み取り専用対応）

        Args:
            scale_factor: 縮小倍率（0.0-1.0）。target_widthが指定された場合は無視される
            target_width: 目標となる画像幅。指定された場合、scale_factorは自動計算される
            resampling: リサンプリング方法（デフォルト: bilinear）

        Returns:
            np.ndarray: 縮小された画像データ（BGR形式、shape: (height, width, channels)）

        Raises:
            ValueError: scale_factorとtarget_widthの両方が未指定、または不正な値の場合

        Note:
            この関数は元のTIFFファイルを変更しません。読み取り専用モードでも動作します。
            rasterioのout_shapeパラメータを使用してメモリ効率的に縮小を行います。
        """

        if scale_factor is None and target_width is None:
            raise ValueError(
                "scale_factorまたはtarget_widthのいずれかを指定してください"
            )

        # 元画像の情報を取得
        original_height, original_width, channels = self.shape

        # スケールファクターの計算
        if target_width is not None:
            if target_width <= 0 or target_width > original_width:
                raise ValueError(
                    f"target_widthは1以上{original_width}以下である必要があります"
                )
            scale_factor = target_width / original_width
        else:
            if scale_factor <= 0 or scale_factor > 1:
                raise ValueError("scale_factorは0より大きく1以下である必要があります")

        # 出力サイズの計算
        output_width = int(original_width * scale_factor)
        output_height = int(original_height * scale_factor)

        self.logger.info(
            f"画像を縮小中: {original_width}x{original_height} -> {output_width}x{output_height} (scale: {scale_factor:.3f})"
        )

        # rasterioハンドルの確認（読み取り専用でも可）
        handle = self._rasterio_handle or self._tiff_handle
        if not handle:
            raise ValueError("ファイルハンドルが利用できません")

        if self._rasterio_handle:
            # rasterioハンドルを使用してメモリ効率的に縮小
            # out_shapeパラメータを使用して読み込み時に縮小
            scaled_data = self._rasterio_handle.read(
                out_shape=(channels, output_height, output_width), resampling=resampling
            )
        else:
            # tifffileハンドルの場合は全体を読み込んでからリサイズ
            # （この場合はメモリ効率が劣るが、読み取り専用で動作）
            page = self._tiff_handle.pages[0]
            full_data = page.asarray()

            if full_data.ndim == 3:
                # (height, width, channels) -> (channels, height, width)
                full_data = np.transpose(full_data, (2, 0, 1))
            else:
                # 単一チャンネルの場合
                full_data = full_data[np.newaxis, :, :]
                channels = 1

            # OpenCVを使用してリサイズ
            scaled_data = np.zeros(
                (channels, output_height, output_width), dtype=full_data.dtype
            )
            for i in range(channels):
                scaled_data[i] = cv2.resize(
                    full_data[i],
                    (output_width, output_height),
                    interpolation=(
                        cv2.INTER_LINEAR
                        if resampling == Resampling.bilinear
                        else cv2.INTER_NEAREST
                    ),
                )

        # チャンネル順を変更: (channels, height, width) -> (height, width, channels)
        if channels > 1:
            scaled_data = np.transpose(scaled_data, (1, 2, 0))
        else:
            scaled_data = scaled_data[0]  # 単一チャンネルの場合

        # TIFFファイルはRGB形式で保存されているため、BGR形式に変換（OpenCV互換）
        if channels == 3:
            scaled_data = cv2.cvtColor(scaled_data, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            scaled_data = cv2.cvtColor(scaled_data, cv2.COLOR_RGBA2BGRA)

        self.logger.info(f"縮小完了（BGR形式）: 出力形状 {scaled_data.shape}")
        return scaled_data


def create_sample_tiff(filepath: str, shape: Tuple[int, int, int], tilesize: int = 512):
    """サンプルのタイル化TIFFファイルを作成する関数（RGB形式でTIFFに保存）"""
    height, width, channels = shape

    # グラデーションパターンを作成
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    if channels == 3:
        # カラーグラデーション（RGB順序で作成）
        r = (x_coords / width * 255).astype(np.uint8)
        g = (y_coords / height * 255).astype(np.uint8)
        b = ((x_coords + y_coords) / (width + height) * 255).astype(np.uint8)
        data = np.stack([r, g, b], axis=2)
    else:
        # グレースケールグラデーション
        data = ((x_coords + y_coords) / (width + height) * 255).astype(np.uint8)

    # タイル化TIFFとして保存（RGB形式）
    tifffile.imwrite(
        filepath,
        data,
        tile=(tilesize, tilesize),
        photometric="rgb" if channels == 3 else "minisblack",
    )

    return filepath


def test_tiff_editor():
    """TiffEditorのテスト関数"""
    import tempfile
    import cv2

    logging.basicConfig(level=logging.DEBUG)

    # テスト用の一時ファイル
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        temp_filepath = tmp.name

    try:
        # サンプルTIFFファイルを作成
        print("サンプルTIFFファイルを作成中...")
        create_sample_tiff(temp_filepath, (2000, 3000, 3), tilesize=256)

        # TiffEditorでファイルを開く
        with TiffEditor(temp_filepath, mode="r+") as editor:
            print(f"ファイル情報: {editor.get_info()}")

            # 部分的に読み込み
            print("部分読み込みテスト...")
            region_data = editor[100:300, 200:400]  # 200x200の領域
            print(f"読み込んだ領域の形状: {region_data.shape}")

            # 読み込んだ領域を変更
            print("部分書き込みテスト...")
            modified_data = np.zeros_like(region_data)
            modified_data[:, :, 0] = 255  # 赤色にする
            editor[100:300, 200:400] = modified_data

            # 変更を確認
            print("変更確認...")
            verification_data = editor[100:300, 200:400]
            print(f"変更後の平均値 (R,G,B): {np.mean(verification_data, axis=(0,1))}")

            # 別の領域を読み込んで表示用に保存（TiffEditorはBGR形式を返すのでそのまま保存）
            display_region = editor[0:500, 0:500]
            cv2.imwrite(
                "tiff_editor_test_output.png",
                display_region,
            )
            print("テスト結果を 'tiff_editor_test_output.png' に保存しました")

        print("TiffEditorのテストが完了しました！")

    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_filepath):
            os.unlink(temp_filepath)


def test():
    import sys
    import cv2
    import logging

    logging.basicConfig(level=logging.DEBUG)

    png = sys.argv[1]
    tilesize = int(sys.argv[2])

    # 元画像を読み込んでサイズを取得
    original_image = cv2.imread(png)
    if original_image is None:
        print(f"エラー: 画像ファイル '{png}' が見つかりません")
        return

    height, width, channels = original_image.shape
    print(f"元画像サイズ: {height}x{width}x{channels}")

    # TIFFファイルを新規作成（十分な大きさで）
    tiff_height = height + 100  # 余裕を持たせる
    tiff_width = width + 100

    with TiffEditor(
        filepath=png + ".tiff",
        mode="r+",
        tilesize=tilesize,
        shape=(tiff_height, tiff_width, channels),
        dtype=np.uint8,
        create_if_not_exists=True,
    ) as tiff_editor:
        print(f"TIFFファイル情報: {tiff_editor.get_info()}")

        # CV2で読み込んだBGR画像をそのまま使用（TiffEditorはBGR形式を受け取る）
        bgr_image = original_image

        # 画像を配置
        tiff_editor[20 : 20 + height, 40 : 40 + width] = bgr_image
        tiff_editor[10 : 10 + height, 20 : 20 + width] = bgr_image

        # 結果を取得して表示（TiffEditorはBGR形式を返す）
        result_image = tiff_editor[0:tiff_height, 0:tiff_width]

        # 結果はすでにBGR形式なのでそのまま表示
        cv2.imshow("TIFF Editor Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"TIFFファイル '{png}.tiff' が作成されました")


def test_large_tiff():
    """メモリに乗らない大きなTIFFファイルを作成・テストする関数"""
    import psutil
    import gc

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # システムメモリの情報を取得
    memory_info = psutil.virtual_memory()
    available_memory_gb = memory_info.available / (1024**3)
    logger.info(f"利用可能メモリ: {available_memory_gb:.2f} GB")

    # メモリより大きなサイズを設定（GB単位で指定）
    # 例：利用可能メモリの1.5倍のサイズの画像を作成
    target_memory_gb = min(available_memory_gb * 1.5, 8.0)  # 最大8GBに制限

    # RGB画像で1ピクセル3バイトとして計算
    bytes_per_pixel = 3
    total_pixels = int(target_memory_gb * 1024**3 / bytes_per_pixel)

    # 正方形に近い形状で計算
    side_length = int(np.sqrt(total_pixels))
    height = width = side_length
    channels = 3

    logger.info(f"作成予定サイズ: {height}x{width}x{channels}")
    logger.info(f"予想ファイルサイズ: {height * width * channels / (1024**3):.2f} GB")

    large_tiff_path = "large_test.tiff"
    tilesize = 512

    try:
        # メモリ使用量を監視しながら大きなTIFFファイルを作成
        logger.info("大きなTIFFファイルを作成中...")
        initial_memory = psutil.Process().memory_info().rss / (1024**2)

        with TiffEditor(
            filepath=large_tiff_path,
            mode="r+",
            tilesize=tilesize,
            shape=(height, width, channels),
            dtype=np.uint8,
            create_if_not_exists=True,
        ) as editor:

            logger.info(f"TIFFファイル情報: {editor.get_info()}")

            # タイルサイズで分割して段階的に書き込み
            tile_h, tile_w = tilesize, tilesize
            tiles_written = 0
            total_tiles = (height // tile_h + 1) * (width // tile_w + 1)

            for y in range(0, height, tile_h):
                for x in range(0, width, tile_w):
                    # 実際のタイル範囲を計算
                    y_end = min(y + tile_h, height)
                    x_end = min(x + tile_w, width)
                    actual_tile_h = y_end - y
                    actual_tile_w = x_end - x

                    # パターン化されたタイルデータを作成
                    tile_data = create_pattern_tile(
                        actual_tile_h, actual_tile_w, channels, x, y
                    )

                    # タイルを書き込み
                    editor[y:y_end, x:x_end] = tile_data

                    tiles_written += 1
                    if tiles_written % 100 == 0:
                        current_memory = psutil.Process().memory_info().rss / (1024**2)
                        logger.info(
                            f"タイル進捗: {tiles_written}/{total_tiles} "
                            f"(メモリ使用量: {current_memory:.1f}MB)"
                        )
                        gc.collect()  # ガベージコレクション

            final_memory = psutil.Process().memory_info().rss / (1024**2)
            logger.info(
                f"メモリ使用量変化: {initial_memory:.1f}MB -> {final_memory:.1f}MB"
            )

        # ファイルが正しく作成されたかチェック
        logger.info("ファイル整合性チェック中...")
        consistency_check_passed = check_tiff_consistency(large_tiff_path, tilesize)

        if consistency_check_passed:
            logger.info("✅ 大きなTIFFファイルの作成・整合性チェックが成功しました！")
        else:
            logger.error("❌ ファイル整合性チェックに失敗しました")

        # ファイルサイズ情報
        actual_size_gb = os.path.getsize(large_tiff_path) / (1024**3)
        logger.info(f"実際のファイルサイズ: {actual_size_gb:.2f} GB")

    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
        raise
    finally:
        # クリーンアップ
        if os.path.exists(large_tiff_path):
            logger.info(f"テストファイル {large_tiff_path} を削除中...")
            os.unlink(large_tiff_path)


def create_pattern_tile(
    height: int, width: int, channels: int, offset_x: int, offset_y: int
) -> np.ndarray:
    """パターン化されたタイルデータを作成"""
    if channels == 3:
        # 座標に基づくカラーパターン
        r = ((offset_x % 256) * np.ones((height, width))).astype(np.uint8)
        g = ((offset_y % 256) * np.ones((height, width))).astype(np.uint8)
        b = (((offset_x + offset_y) % 256) * np.ones((height, width))).astype(np.uint8)
        return np.stack([r, g, b], axis=2)
    else:
        # グレースケールパターン
        return (((offset_x + offset_y) % 256) * np.ones((height, width))).astype(
            np.uint8
        )


def check_tiff_consistency(filepath: str, tilesize: int) -> bool:
    """TIFFファイルの整合性をチェック"""
    logger = logging.getLogger(__name__)

    try:
        with TiffEditor(filepath, mode="r") as editor:
            height, width, channels = editor.shape
            logger.info(f"読み込んだファイル形状: {height}x{width}x{channels}")

            # ランダムな位置のタイルをいくつかサンプリングしてチェック
            import random

            sample_count = min(10, (height // tilesize) * (width // tilesize))

            for i in range(sample_count):
                # ランダムなタイル位置を選択
                tile_y = random.randint(0, height // tilesize) * tilesize
                tile_x = random.randint(0, width // tilesize) * tilesize

                tile_y_end = min(tile_y + tilesize, height)
                tile_x_end = min(tile_x + tilesize, width)

                # タイルを読み込み
                tile_data = editor[tile_y:tile_y_end, tile_x:tile_x_end]

                # 期待値と比較
                expected_tile = create_pattern_tile(
                    tile_y_end - tile_y, tile_x_end - tile_x, channels, tile_x, tile_y
                )

                if not np.array_equal(tile_data, expected_tile):
                    logger.error(f"タイル ({tile_x}, {tile_y}) のデータが一致しません")
                    return False

                logger.debug(f"タイル ({tile_x}, {tile_y}) の整合性チェック完了")

            logger.info(f"整合性チェック完了: {sample_count}個のタイルをテストしました")
            return True

    except Exception as e:
        logger.error(f"整合性チェック中にエラー: {e}")
        return False


class ScalableTiffEditor(TiffEditor):
    """
    縮小画像を扱えるTiffEditorの拡張クラス（BGR形式でデータを扱う）

    ユーザーには大きな画像を扱っているように見せかけて、
    実際には指定されたスケールで縮小された画像で操作を行う。
    これにより、メモリ効率を保ちながら大きな画像のシミュレーションが可能。

    Example:
        # 仮想的に10000x8000の画像だが、実際は1000x800で保存される
        editor = ScalableTiffEditor(
            "test.tiff",
            virtual_shape=(8000, 10000, 3),
            scale_factor=0.1
        )

        # ユーザーは大きな座標で操作
        region = editor[1000:2000, 1500:2500]  # 仮想座標
        # 実際には100:200, 150:250の領域から読み込まれる
    """

    def __init__(
        self,
        filepath: str,
        virtual_shape: Tuple[int, int, int],
        scale_factor: float,
        mode: str = "r+",
        tilesize: Union[int, Tuple[int, int]] = 512,
        dtype: Optional[np.dtype] = None,
        create_if_not_exists: bool = False,
    ):
        """
        ScalableTiffEditorを初期化する

        Args:
            filepath: TIFFファイルのパス
            virtual_shape: 仮想的な画像の形状 (height, width, channels)
            scale_factor: 実際のファイルサイズに対するスケール (0.0-1.0)
            mode: ファイルのオープンモード ('r', 'r+', 'w')
            tilesize: タイルサイズ
            dtype: データ型（新規作成時）
            create_if_not_exists: ファイルが存在しない場合に新規作成するか
        """
        if not (0.0 < scale_factor <= 1.0):
            raise ValueError("scale_factorは0.0より大きく1.0以下である必要があります")

        self.virtual_shape = virtual_shape
        self.scale_factor = scale_factor

        # 実際のファイルサイズを計算
        virtual_height, virtual_width, channels = virtual_shape
        actual_height = int(virtual_height * scale_factor)
        actual_width = int(virtual_width * scale_factor)
        actual_shape = (actual_height, actual_width, channels)

        # 親クラスを実際のサイズで初期化
        super().__init__(
            filepath=filepath,
            mode=mode,
            tilesize=tilesize,
            dtype=dtype,
            shape=actual_shape,
            create_if_not_exists=create_if_not_exists,
        )

        self.logger.info(
            f"ScalableTiffEditor初期化: 仮想サイズ{virtual_shape} -> 実際サイズ{actual_shape} (scale: {scale_factor})"
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """仮想的な画像の形状を返す"""
        return self.virtual_shape

    @property
    def actual_shape(self) -> Tuple[int, int, int]:
        """実際のファイルの形状を返す"""
        return super().shape

    def _virtual_to_actual_coords(self, virtual_rect: Rect) -> Rect:
        """仮想座標を実際の座標に変換する"""
        actual_left = int(virtual_rect.left * self.scale_factor)
        actual_right = int(virtual_rect.right * self.scale_factor)
        actual_top = int(virtual_rect.top * self.scale_factor)
        actual_bottom = int(virtual_rect.bottom * self.scale_factor)

        # 実際のファイルサイズ内に収める
        actual_height, actual_width, _ = self.actual_shape
        actual_left = max(0, min(actual_left, actual_width))
        actual_right = max(0, min(actual_right, actual_width))
        actual_top = max(0, min(actual_top, actual_height))
        actual_bottom = max(0, min(actual_bottom, actual_height))

        return Rect.from_bounds(actual_left, actual_right, actual_top, actual_bottom)

    def _actual_to_virtual_size(
        self, actual_height: int, actual_width: int
    ) -> Tuple[int, int]:
        """実際のサイズを仮想サイズに変換する"""
        virtual_height = int(actual_height / self.scale_factor)
        virtual_width = int(actual_width / self.scale_factor)
        return virtual_height, virtual_width

    def get_region(self, region: Rect) -> np.ndarray:
        """指定された仮想領域のデータを読み込む（BGR形式で返す）"""
        # 仮想座標を実際の座標に変換
        actual_region = self._virtual_to_actual_coords(region)

        # 実際のデータを読み込み
        actual_data = super().get_region(actual_region)

        if actual_data.size == 0:
            return actual_data

        # 仮想サイズにリサイズ
        virtual_height = region.height
        virtual_width = region.width

        if actual_data.ndim == 3:
            # カラー画像
            resized_data = cv2.resize(
                actual_data,
                (virtual_width, virtual_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            # グレースケール画像
            resized_data = cv2.resize(
                actual_data,
                (virtual_width, virtual_height),
                interpolation=cv2.INTER_LINEAR,
            )

        self.logger.debug(
            f"仮想領域を読み込み: {region} -> 実際{actual_region}, リサイズ後{resized_data.shape}"
        )
        return resized_data

    def put_region(self, region: Rect, data: np.ndarray):
        """指定された仮想領域にデータを書き込む（入力はBGR形式）"""
        if self.mode == "r":
            raise ValueError("読み込み専用モードでは書き込みできません")

        # データサイズの検証
        expected_shape = (region.height, region.width)
        if data.ndim == 3:
            expected_shape = (region.height, region.width, data.shape[2])

        if data.shape[:2] != (region.height, region.width):
            raise ValueError(
                f"データサイズが一致しません。期待: {expected_shape}, 実際: {data.shape}"
            )

        # 仮想座標を実際の座標に変換
        actual_region = self._virtual_to_actual_coords(region)

        if actual_region.width <= 0 or actual_region.height <= 0:
            self.logger.warning(f"実際の領域が無効です: {actual_region}")
            return

        # データを実際のサイズにリサイズ
        if data.ndim == 3:
            # カラー画像
            resized_data = cv2.resize(
                data,
                (actual_region.width, actual_region.height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            # グレースケール画像
            resized_data = cv2.resize(
                data,
                (actual_region.width, actual_region.height),
                interpolation=cv2.INTER_LINEAR,
            )

        # 実際のファイルに書き込み
        super().put_region(actual_region, resized_data)

        self.logger.debug(
            f"仮想領域に書き込み: {region} -> 実際{actual_region}, 元データ{data.shape} -> リサイズ後{resized_data.shape}"
        )

    def get_info(self) -> dict:
        """ファイルの情報を取得する（仮想サイズと実際サイズの両方）"""
        info = super().get_info()
        info.update(
            {
                "virtual_shape": self.virtual_shape,
                "actual_shape": self.actual_shape,
                "scale_factor": self.scale_factor,
                "virtual_size_mb": (
                    self.virtual_shape[0]
                    * self.virtual_shape[1]
                    * self.virtual_shape[2]
                )
                / (1024 * 1024),
            }
        )
        return info


def test_scalable_tiff_editor():
    """ScalableTiffEditorのテスト関数"""
    import tempfile
    import cv2

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # テスト用の一時ファイル
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        temp_filepath = tmp.name

    try:
        logger.info("ScalableTiffEditorのテストを開始...")

        # 仮想的に大きなサイズ（8000x6000）だが、実際は800x600で保存
        virtual_shape = (6000, 8000, 3)
        scale_factor = 0.1

        with ScalableTiffEditor(
            temp_filepath,
            virtual_shape=virtual_shape,
            scale_factor=scale_factor,
            mode="w",
            dtype=np.uint8,
            create_if_not_exists=True,
        ) as editor:
            logger.info(f"ファイル情報: {editor.get_info()}")

            # 仮想座標でテストデータを作成（1000x1000の赤いボックス）
            test_data = np.zeros((1000, 1000, 3), dtype=np.uint8)
            test_data[:, :, 2] = 255  # BGR形式で赤

            # 仮想座標で書き込み（実際には100x100にリサイズされて保存）
            logger.info("仮想座標でデータを書き込み中...")
            editor[1000:2000, 1500:2500] = test_data

            # 異なる色で別の領域に書き込み
            test_data2 = np.zeros((800, 1200, 3), dtype=np.uint8)
            test_data2[:, :, 1] = 255  # BGR形式で緑
            editor[3000:3800, 2000:3200] = test_data2

            # 仮想座標で読み込み
            logger.info("仮想座標でデータを読み込み中...")
            read_data1 = editor[1000:2000, 1500:2500]
            read_data2 = editor[3000:3800, 2000:3200]

            logger.info(f"読み込んだデータ1の形状: {read_data1.shape}")
            logger.info(f"読み込んだデータ2の形状: {read_data2.shape}")

            # 全体の縮小画像を取得（仮想座標ベース）
            logger.info("全体画像の一部を取得...")
            overview = editor[0:3000, 0:4000]  # 仮想座標で3000x4000の領域

            # 結果を保存
            cv2.imwrite("scalable_test_overview.png", overview)
            cv2.imwrite("scalable_test_region1.png", read_data1)
            cv2.imwrite("scalable_test_region2.png", read_data2)

            logger.info("テスト画像を保存しました:")
            logger.info("- scalable_test_overview.png")
            logger.info("- scalable_test_region1.png")
            logger.info("- scalable_test_region2.png")

        logger.info("ScalableTiffEditorのテストが完了しました！")

    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_filepath):
            os.unlink(temp_filepath)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "large_test":
        test_large_tiff()
    elif len(sys.argv) > 1 and sys.argv[1] == "test_editor":
        test_tiff_editor()
    elif len(sys.argv) > 1 and sys.argv[1] == "test_scalable":
        test_scalable_tiff_editor()
    else:
        test()
