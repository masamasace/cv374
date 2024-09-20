# CV374
東京測振製の3成分速度計CV374-AVの生データ処理用のライブラリです。非公式で完全なコードチェックは行っておりませんので、ご利用は自己責任でお願いします。

## 使い方
```python
import cv374

# ファイル読み込み
file_path = "path/to/file"
cv374 = cv374.DataFormatter(file_path)

# HVSRの計算
cv374.calculate_HVSR()

# 複数記録の統合と結果の出力
cv374.merge_HVSR()
```

## TODO
- [ ] テストコードの追加
- [ ] ドキュメントの整備
- [ ] パッケージ化
- [ ] ファイルの読み込み方法の変更
    - `cv374.read_file(file_path)`のように使いたい
- [ ] ファイルの書き出し方法の変更
    - `cv374.export_result(type="figure", ext="png")`のように使いたい
