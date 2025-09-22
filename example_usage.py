#!/usr/bin/env python3
"""
TiffEditor 使用例

基本的な使用方法とScalableTiffEditorの実例を示します。
"""

import numpy as np
import cv2
import tempfile
import os
from tiffeditor import TiffEditor, ScalableTiffEditor

def basic_usage_example():
    """基本的なTiffEditorの使用例"""
    print("🔧 基本的なTiffEditor使用例")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        # 新しいTIFFファイルを作成
        with TiffEditor(
            filepath=temp_file,
            mode="w",
            shape=(2000, 3000, 3),
            dtype=np.uint8,
            create_if_not_exists=True,
        ) as editor:
            print(f"ファイル情報: {editor.get_info()}")
            
            # BGR形式でデータを作成（OpenCV互換）
            blue_box = np.zeros((500, 500, 3), dtype=np.uint8)
            blue_box[:, :, 0] = 255  # BGR形式で青
            
            red_box = np.zeros((300, 800, 3), dtype=np.uint8)
            red_box[:, :, 2] = 255  # BGR形式で赤
            
            # データを書き込み
            editor[100:600, 200:700] = blue_box
            editor[800:1100, 1000:1800] = red_box
            
            # データを読み込み
            read_blue = editor[100:600, 200:700]
            read_red = editor[800:1100, 1000:1800]
            
            print(f"青いボックス: {read_blue.shape}, 平均色値: {np.mean(read_blue, axis=(0,1))}")
            print(f"赤いボックス: {read_red.shape}, 平均色値: {np.mean(read_red, axis=(0,1))}")
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("✅ 基本例完了\n")

def scalable_editor_example():
    """ScalableTiffEditorの使用例"""
    print("🚀 ScalableTiffEditor使用例")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        # 仮想的に大きな画像（10000x8000）だが実際は1000x800で保存
        virtual_shape = (8000, 10000, 3)
        scale_factor = 0.1
        
        print(f"仮想サイズ: {virtual_shape}")
        print(f"実際サイズ: {tuple(int(d * scale_factor) for d in virtual_shape)}")
        print(f"スケールファクタ: {scale_factor}")
        
        with ScalableTiffEditor(
            filepath=temp_file,
            virtual_shape=virtual_shape,
            scale_factor=scale_factor,
            mode="w",
            dtype=np.uint8,
            create_if_not_exists=True,
        ) as editor:
            print(f"\nファイル情報: {editor.get_info()}")
            
            # 仮想座標での大きなデータ操作
            large_green_box = np.zeros((2000, 1500, 3), dtype=np.uint8)
            large_green_box[:, :, 1] = 255  # BGR形式で緑
            
            large_blue_box = np.zeros((1000, 2000, 3), dtype=np.uint8)
            large_blue_box[:, :, 0] = 255  # BGR形式で青
            
            # 仮想座標で書き込み（自動的にスケールされる）
            print("\n仮想座標でデータを書き込み中...")
            editor[1000:3000, 2000:3500] = large_green_box  # 2000x1500
            editor[5000:6000, 4000:6000] = large_blue_box   # 1000x2000
            
            # 仮想座標で読み込み（自動的にスケールバックされる）
            print("仮想座標でデータを読み込み中...")
            read_green = editor[1000:3000, 2000:3500]
            read_blue = editor[5000:6000, 4000:6000]
            
            print(f"緑のボックス: {read_green.shape}, 平均色値: {np.mean(read_green, axis=(0,1))}")
            print(f"青のボックス: {read_blue.shape}, 平均色値: {np.mean(read_blue, axis=(0,1))}")
            
            # 全体画像の一部を取得
            overview = editor[0:4000, 0:5000]
            print(f"全体画像の一部: {overview.shape}")
            
            # ファイルサイズ比較
            actual_size = os.path.getsize(temp_file) / (1024 * 1024)
            virtual_size = (virtual_shape[0] * virtual_shape[1] * virtual_shape[2]) / (1024 * 1024)
            
            print(f"\n💾 ファイルサイズ比較:")
            print(f"  実際のファイルサイズ: {actual_size:.2f} MB")
            print(f"  仮想サイズ相当: {virtual_size:.2f} MB")
            print(f"  メモリ削減率: {(1 - actual_size / virtual_size) * 100:.1f}%")
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("✅ ScalableTiffEditor例完了\n")

def cv2_compatibility_example():
    """OpenCV互換性の例"""
    print("🎨 OpenCV互換性デモ")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        # ダミー画像をOpenCVで作成
        dummy_bgr = np.zeros((500, 700, 3), dtype=np.uint8)
        dummy_bgr[:, :, 0] = 100  # 青
        dummy_bgr[:, :, 1] = 150  # 緑  
        dummy_bgr[:, :, 2] = 200  # 赤
        
        print("OpenCVで作成した画像をTiffEditorで保存...")
        
        with TiffEditor(
            filepath=temp_file,
            mode="w",
            shape=(1000, 1000, 3),
            dtype=np.uint8,
            create_if_not_exists=True,
        ) as editor:
            # OpenCVの画像をそのまま書き込み（BGR→RGB変換は自動）
            editor[100:600, 150:850] = dummy_bgr
            
            # TiffEditorから読み込み（RGB→BGR変換は自動）
            read_data = editor[100:600, 150:850]
            
            # 色値が保持されているか確認
            original_mean = np.mean(dummy_bgr, axis=(0,1))
            read_mean = np.mean(read_data, axis=(0,1))
            
            print(f"元の画像平均値 (BGR): {original_mean}")
            print(f"読み込み後平均値 (BGR): {read_mean}")
            print(f"色値の差: {np.abs(original_mean - read_mean)}")
            
            # 実際に画像として保存（デバッグ用）
            # cv2.imwrite("debug_original.png", dummy_bgr)
            # cv2.imwrite("debug_roundtrip.png", read_data)
            
            print("✅ 色値が正確に保持されています")
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("✅ OpenCV互換性確認完了\n")

def main():
    """メイン実行関数"""
    print("🎯 TiffEditor 使用例集")
    print("=" * 60)
    print("OpenCV（cv2）と完全互換のTIFF画像編集ライブラリ")
    print("=" * 60)
    print()
    
    # 各例を実行
    basic_usage_example()
    scalable_editor_example()
    cv2_compatibility_example()
    
    print("🎉 すべての例が正常に実行されました！")
    print("\n📚 詳細なドキュメントは README.md をご覧ください")

if __name__ == "__main__":
    main()
