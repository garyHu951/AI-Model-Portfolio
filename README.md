# 智能模型作品集（AI-Model-Portfolio）

## 項目概述

本倉庫包含人工智慧課程的三個主要項目：
1. **8-Puzzle 遊戲求解器**（Project 1）
2. **Othello 黑白棋遊戲**（Project 2）
3. **甲狀腺疾病分類器**（Project 3）

---

## Project 1: 8-Puzzle Games

### 項目描述
實現一個完整的8-拼圖遊戲求解系統，使用 A* 搜索算法找到最優解。

### 功能模塊
- **Part 1**: 可解性判斷 (Solvability)
- **Part 2**: 後繼狀態生成函數 (Successor function)
- **Part 3**: 曼哈頓距離啟發式函數 (Manhattan distance)
- **Part 4**: 優先佇列實現 (Priority queue)
- **完整系統**: 8-拼圖遊戲求解器

### 算法特性
- 使用 A* 搜索算法
- 曼哈頓距離作為啟發式函數
- 優先佇列管理搜索節點
- 支持無解狀態檢測

### 使用方法
```bash
# 編譯並運行
g++ -o puzzle "Fullsystem_an 8-puzzle game.cpp"
./puzzle
# 輸入初始狀態，例如：312457680
```

### 測試用例
- `724506831`: 26步解
- `104782563`: 17步解
- `817365204`: 25步解
- `426031785`: 17步解
- `102345678`: 1步解

---

## Project 2: Othello (黑白棋 Reversi)

### 項目描述
實現一個完整的黑白棋遊戲系統，包括遊戲規則實現和 AI 對手。

### 遊戲規則
- 6×6 棋盤（而非傳統的8×8）
- 黑子先行，輪流下棋
- 必須夾取對手棋子才能落子
- 遊戲結束時棋子數多者獲勝

### 功能模塊

#### Part 1: 計算可翻面棋子數
```python
# part1.py
countFlipPieces(gameboard, player_color, grid, direction_id)
```
- 計算在指定方向上可翻面的對手棋子數量

#### Part 2: 執行翻棋操作
```python
# part2.py
flipPieces(gameboard, player_color, grid)
```
- 在棋盤上放置棋子並翻轉所有被夾取的對手棋子

#### Part 3: 計算棋子數量
```python
# part3.py
countColorPieces(gameboard, player_color)
```
- 計算指定顏色棋子的數量
- 實現簡單的評估函數

#### Part 4: Minimax 搜索 AI
```cpp
# part4.cpp
minimaxSearch(gameboard, player_color, depth)
```
- 使用 Minimax 算法實現 AI 對手
- 支援 Alpha-Beta 剪枝優化
- 可設定搜索深度

### 座標系統
- 行：A-F (對應 0-5)
- 列：a-f (對應 0-5)
- 方向ID：0-7 (上、右上、右、右下、下、左下、左、左上)

### 使用方法
```bash
# Python 模塊
python part1.py  # 測試翻子計算
python part2.py  # 測試翻子執行
python part3.py  # 測試棋子計數

# C++ AI 模塊
g++ -o othello_ai part4-2.cpp
./othello_ai
```

---

## Project 3: 甲狀腺疾病分類器

### 項目描述
使用多種機器學習算法對甲狀腺疾病進行分類，並比較不同算法的性能表現。

### 實驗設計
對比10種不同的機器學習算法：
1. 決策樹 (Decision Tree)
2. 隨機森林 (Random Forest)
3. 梯度提升 (Gradient Boosting)
4. AdaBoost
5. 支持向量機-線性核 (SVM Linear)
6. 深度神經網路 (Deep Neural Network)
7. 邏輯回歸 (Logistic Regression)
8. 支持向量機-RBF核 (SVM RBF)
9. K近鄰分類器 (K-Nearest Neighbors)
10. 樸素貝葉斯 (Naive Bayes)

### 實驗結果
| 算法 | 準確率 | 特點 |
|------|--------|------|
| 決策樹 | 98.83% | 解釋性強，訓練快速 |
| 隨機森林 | 98.83% | 泛化能力強，穩定性好 |
| 梯度提升 | 98.83% | 學習能力強 |
| AdaBoost | 98.53% | 集成學習 |
| SVM線性核 | 97.95% | 高維空間表現穩定 |

### 數據預處理
- 處理缺失值（'?' 替換為 NaN）
- 特徵分類：數值型、二元型、類別型
- 標準化數值特徵
- 獨熱編碼類別特徵
- 標籤編碼目標變量

### 使用方法
```bash
# 安裝依賴
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn

# 運行實驗
python project3.py
```

### 文件需求
- `hypothyroid_cjlin2025_training.arff`
- `hypothyroid_cjlin2025_test.arff`

---

## 實驗報告

詳細的實驗結果和分析請參考：`01157123_衡家豪_實驗結果報告.docx`

### 主要發現
- **集成學習方法**表現最優（決策樹、隨機森林、梯度提升）
- **深度神經網路**在複雜特徵關係建模上有優勢
- **SVM線性核**在高維空間中表現穩定
- **樸素貝葉斯**在醫療診斷中具有良好解釋性

---

## 開發環境

### 系統要求
- Python 3.7+
- C++ 11 或更高版本
- 足夠的內存運行深度學習模型

### Python 依賴
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### C++ 編譯
```bash
g++ -std=c++11 -O2 -o program_name source_file.cpp
```

---


## 授權說明

本項目僅供學術研究和學習使用，請勿用於商業用途。
