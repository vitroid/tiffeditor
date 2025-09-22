# TiffEditor Makefile
# 開発、テスト、デプロイメント用のコマンド集

.PHONY: help install install-dev test test-cov lint format type-check clean build upload upload-test all-checks pre-commit

# デフォルトターゲット（ヘルプを表示）
help:
	@echo "📦 TiffEditor 開発用Makefile"
	@echo ""
	@echo "利用可能なコマンド:"
	@echo "  install      - プロダクション依存関係をインストール"
	@echo "  install-dev  - 開発用依存関係をインストール"
	@echo "  test         - テストを実行"
	@echo "  test-cov     - カバレッジ付きでテストを実行"
	@echo "  lint         - コードの静的解析（flake8）"
	@echo "  format       - コードフォーマット（black）"
	@echo "  type-check   - 型チェック（mypy）"
	@echo "  all-checks   - すべてのチェックを実行（lint + format + type-check + test）"
	@echo "  pre-commit   - コミット前のすべてのチェック"
	@echo "  clean        - ビルド成果物を削除"
	@echo "  build        - パッケージをビルド"
	@echo "  upload-test  - TestPyPIにアップロード"
	@echo "  upload       - PyPIにアップロード"
	@echo ""
	@echo "使用例:"
	@echo "  make install-dev  # 開発環境セットアップ"
	@echo "  make pre-commit   # コミット前チェック"
	@echo "  make build        # パッケージビルド"
	@echo "  make upload-test  # テスト環境にアップロード"

# 依存関係のインストール
install:
	@echo "📥 プロダクション依存関係をインストール中..."
	poetry install --only=main

install-dev:
	@echo "🔧 開発用依存関係をインストール中..."
	poetry install
	@echo "✅ 開発環境の準備が完了しました"

# テスト実行
test:
	@echo "🧪 テストを実行中..."
	poetry run python -m pytest tests/ -v

test-cov:
	@echo "📊 カバレッジ付きでテストを実行中..."
	poetry run python -m pytest tests/ --cov=tiffeditor --cov-report=html --cov-report=term-missing
	@echo "📈 カバレッジレポートが htmlcov/ に生成されました"

# コードの静的解析とフォーマット
lint:
	@echo "🔍 コードの静的解析中（flake8）..."
	poetry run flake8 tiffeditor.py example_usage.py

format:
	@echo "🎨 コードフォーマット中（black）..."
	poetry run black tiffeditor.py example_usage.py
	@echo "✅ コードフォーマット完了"

format-check:
	@echo "🎨 コードフォーマットをチェック中..."
	poetry run black --check tiffeditor.py example_usage.py

type-check:
	@echo "🔍 型チェック中（mypy）..."
	poetry run mypy tiffeditor.py

# 全チェック実行
all-checks: format-check lint type-check test
	@echo "✅ すべてのチェックが完了しました"

pre-commit: format lint type-check test
	@echo "🚀 コミット前チェック完了！"

# クリーンアップ
clean:
	@echo "🧹 ビルド成果物を削除中..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "✅ クリーンアップ完了"

# パッケージビルド
build: clean
	@echo "📦 パッケージをビルド中..."
	poetry build
	@echo "✅ ビルド完了！ dist/ ディレクトリを確認してください"

# バージョン確認とファイルリスト
check-build: build
	@echo "📋 ビルド結果の確認:"
	@echo "ビルドされたファイル:"
	@ls -la dist/
	@echo ""
	@echo "パッケージ内容の確認:"
	@poetry run python -c "import tarfile; tar = tarfile.open('dist/tiffeditor-0.2.0.tar.gz'); print('📁 含まれるファイル:'); [print(f'  {member.name}') for member in tar.getmembers()]"

# TestPyPIにアップロード
upload-test: build
	@echo "🚀 TestPyPIにアップロード中..."
	@echo "⚠️  TestPyPI用のAPIトークンが必要です"
	@echo "設定方法: poetry config repositories.testpypi https://test.pypi.org/legacy/"
	@echo "トークン設定: poetry config pypi-token.testpypi pypi-xxxxxxxx"
	poetry publish -r testpypi
	@echo "✅ TestPyPIにアップロード完了！"
	@echo "🔗 https://test.pypi.org/project/tiffeditor/ で確認できます"

# 本番PyPIにアップロード
upload: build
	@echo "🚨 本番PyPIにアップロード中..."
	@echo "⚠️  これは取り消しできない操作です！"
	@echo "PyPI用のAPIトークンが必要です"
	@echo "トークン設定: poetry config pypi-token.pypi pypi-xxxxxxxx"
	@read -p "本当にアップロードしますか？ (y/N): " confirm && [ "$$confirm" = "y" ]
	poetry publish
	@echo "🎉 PyPIにアップロード完了！"
	@echo "🔗 https://pypi.org/project/tiffeditor/ で確認できます"

# 開発用コマンド
demo:
	@echo "🎯 デモ実行中..."
	poetry run python example_usage.py

test-basic:
	@echo "🧪 基本機能テスト中..."
	poetry run python tiffeditor.py test_editor

test-scalable:
	@echo "🧪 ScalableTiffEditorテスト中..."
	poetry run python tiffeditor.py test_scalable

test-large:
	@echo "🧪 大容量ファイルテスト中..."
	poetry run python tiffeditor.py large_test

# バージョン管理
version-patch:
	@echo "📈 パッチバージョンを上げます..."
	poetry version patch
	@echo "新しいバージョン: $$(poetry version -s)"

version-minor:
	@echo "📈 マイナーバージョンを上げます..."
	poetry version minor
	@echo "新しいバージョン: $$(poetry version -s)"

version-major:
	@echo "📈 メジャーバージョンを上げます..."
	poetry version major
	@echo "新しいバージョン: $$(poetry version -s)"

# リリース用ワークフロー
release-patch: version-patch all-checks build upload-test
	@echo "🎉 パッチリリース準備完了！"
	@echo "TestPyPIで確認後、'make upload' で本番リリースしてください"

release-minor: version-minor all-checks build upload-test
	@echo "🎉 マイナーリリース準備完了！"
	@echo "TestPyPIで確認後、'make upload' で本番リリースしてください"

# 情報表示
info:
	@echo "📋 プロジェクト情報:"
	@echo "名前: $$(poetry version | cut -d' ' -f1)"
	@echo "バージョン: $$(poetry version -s)"
	@echo "Python: $$(poetry env info --python)"
	@echo "仮想環境: $$(poetry env info --path)"
