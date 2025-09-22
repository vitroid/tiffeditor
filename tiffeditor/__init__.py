"""
TiffEditor

メモリ効率の良い巨大TIFFファイルの部分編集を可能にするライブラリ。
OpenCV（cv2）との完全互換性のため、すべてのカラー画像データはBGR形式で扱います。
"""

__version__ = "0.2.0"
__author__ = "User"
__email__ = ""

from dataclasses import dataclass


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


# メインクラスをインポート
from .tiffeditor import TiffEditor, ScalableTiffEditor, create_sample_tiff

# 公開するシンボルを明示
__all__ = ['TiffEditor', 'ScalableTiffEditor', 'Range', 'Rect', 'create_sample_tiff']
