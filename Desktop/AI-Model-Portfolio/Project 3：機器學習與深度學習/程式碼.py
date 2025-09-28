import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_arff_data(filename):
    """載入並解析ARFF格式數據，處理重複欄位名稱"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到屬性定義和數據部分
        attributes = []
        data_start = -1
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.lower().startswith('@attribute'):
                # 解析屬性名稱
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1].strip("'\"")
                    attributes.append(attr_name)
            elif line.lower().startswith('@data'):
                data_start = i + 1
                break
        
        if data_start == -1:
            raise ValueError(f"在檔案 {filename} 中找不到 @data 標記")
        
        # 處理重複的欄位名稱
        seen_names = {}
        unique_attributes = []
        for attr in attributes:
            if attr in seen_names:
                seen_names[attr] += 1
                unique_attributes.append(f"{attr}_{seen_names[attr]}")
            else:
                seen_names[attr] = 0
                unique_attributes.append(attr)
        
        print(f"原始欄位: {attributes}")
        print(f"處理後欄位: {unique_attributes}")
        
        # 讀取數據行
        data_rows = []
        for line in lines[data_start:]:
            line = line.strip()
            if line and not line.startswith('%'):
                # 分割數據，注意處理可能包含逗號的字符串
                row = [item.strip() for item in line.split(',')]
                data_rows.append(row)
        
        if not data_rows:
            raise ValueError(f"檔案 {filename} 中沒有找到有效數據")
        
        # 創建DataFrame
        df = pd.DataFrame(data_rows, columns=unique_attributes)
        print(f"成功載入 {filename}: {df.shape[0]} 行, {df.shape[1]} 列")
        return df
        
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {filename}")
        return pd.DataFrame()
    except Exception as e:
        print(f"載入檔案 {filename} 時發生錯誤: {e}")
        return pd.DataFrame()

def preprocess_thyroid_data(df):
    """專門針對甲狀腺數據的預處理函數"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # 處理缺失值（將'?'替換為NaN）
    df = df.replace('?', np.nan)
    
    # 識別數值型和類別型特徵
    # 根據甲狀腺數據集的特徵進行分類
    numeric_features = []
    binary_features = []
    categorical_features = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['age', 'tsh', 't3', 'tt4', 't4u', 'fti']):
            numeric_features.append(col)
        elif any(keyword in col_lower for keyword in ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                           'sick', 'pregnant', 'thyroid_surgery', 'i131_treatment', 'query_hypothyroid',
                           'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']):
            binary_features.append(col)
        elif 'class' not in col_lower:
            categorical_features.append(col)
    
    print(f"數值型特徵: {numeric_features}")
    print(f"二元特徵: {binary_features}")
    print(f"類別型特徵: {categorical_features}")
    
    # 處理數值型特徵
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 使用中位數填充缺失值
            if df[col].notna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(0, inplace=True)
    
    # 處理二元特徵和其他可能的二元欄位
    for col in df.columns:
        if col in binary_features or col not in numeric_features + categorical_features:
            if 'class' not in col.lower():
                # 檢查該欄位的唯一值
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 3:  # 可能是二元特徵
                    df[col] = df[col].astype(str).str.lower()
                    # 嘗試多種二元值映射
                    mapping_dict = {'f': 0, 't': 1, 'false': 0, 'true': 1, 
                                  'no': 0, 'yes': 1, 'm': 1, 'male': 1, 'female': 0,
                                  'n': 0, 'y': 1}
                    df[col] = df[col].map(mapping_dict).fillna(0)
                    df[col] = df[col].astype(int)
                else:
                    categorical_features.append(col)
    
    # 處理類別型特徵（使用獨熱編碼）
    for col in categorical_features:
        if col in df.columns and 'class' not in col.lower():
            # 填充缺失值
            df[col].fillna('unknown', inplace=True)
            # 獨熱編碼
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    return df

def evaluate_classifier(name, model, X_train, y_train, X_test, y_test):
    """評估分類器性能"""
    try:
        print(f"正在訓練 {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            'name': name,
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
    except Exception as e:
        print(f"評估 {name} 時發生錯誤: {e}")
        return None

def create_neural_network(input_dim, num_classes):
    """創建改進的神經網路模型"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 主程式開始
print("="*60)
print("甲狀腺疾病分類器實驗")
print("="*60)

# 載入數據檔案
train_file = 'hypothyroid_cjlin2025_training.arff'
test_file = 'hypothyroid_cjlin2025_test.arff'

print("開始載入數據...")
train_df = load_arff_data(train_file)
test_df = load_arff_data(test_file)

if train_df.empty or test_df.empty:
    print("錯誤: 無法載入數據文件，請檢查文件路徑和格式")
    print("請確保以下檔案存在:")
    print(f"- {train_file}")
    print(f"- {test_file}")
    exit()

print(f"\n原始數據資訊:")
print(f"訓練集大小: {train_df.shape}")
print(f"測試集大小: {test_df.shape}")
print(f"訓練集欄位: {list(train_df.columns)}")

# 數據預處理
print("\n開始數據預處理...")
train_df_processed = preprocess_thyroid_data(train_df)
test_df_processed = preprocess_thyroid_data(test_df)

# 找出目標變數欄位
target_columns = [col for col in train_df_processed.columns if 'class' in col.lower()]
if not target_columns:
    print("錯誤: 找不到目標變數欄位")
    exit()

target_col = target_columns[0]
print(f"目標變數欄位: {target_col}")

# 確保訓練集和測試集有相同的特徵列
train_features = set(train_df_processed.columns) - {target_col}
test_features = set(test_df_processed.columns) - {target_col}

# 找出缺少的特徵並補齊
missing_in_test = train_features - test_features
missing_in_train = test_features - train_features

for feature in missing_in_test:
    test_df_processed[feature] = 0
    
for feature in missing_in_train:
    train_df_processed[feature] = 0

# 確保列順序一致
common_features = sorted(list(train_features | test_features))

# 分離特徵和標籤
X_train = train_df_processed[common_features]
y_train = train_df_processed[target_col]
X_test = test_df_processed[common_features]
y_test = test_df_processed[target_col]

print(f"\n預處理後數據資訊:")
print(f"特徵數量: {X_train.shape[1]}")
print(f"訓練樣本數: {X_train.shape[0]}")
print(f"測試樣本數: {X_test.shape[0]}")
print(f"類別分佈 (訓練集): {y_train.value_counts().to_dict()}")

# 標籤編碼
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"編碼後的類別: {label_encoder.classes_}")

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定義分類器
classifiers = {
    '樸素貝葉斯': GaussianNB(),
    '支持向量機(RBF核)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    '支持向量機(線性核)': SVC(kernel='linear', C=1.0, random_state=42),
    '決策樹': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
    '隨機森林': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    '梯度提升': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    '邏輯回歸': LogisticRegression(max_iter=1000, random_state=42),
    'K近鄰分類器': KNeighborsClassifier(n_neighbors=5)
}

# 評估所有分類器
print("\n" + "="*50)
print("開始評估傳統機器學習分類器...")
print("="*50)

results = []
for name, classifier in classifiers.items():
    result = evaluate_classifier(name, classifier, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded)
    if result:
        results.append(result)

# 神經網路評估
print("\n開始評估深度神經網路...")
try:
    # 轉換標籤為one-hot編碼
    num_classes = len(label_encoder.classes_)
    y_train_onehot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=num_classes)

    # 創建神經網路模型
    nn_model = create_neural_network(X_train_scaled.shape[1], num_classes)

    # 設置回調函數
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    print("正在訓練神經網路...")
    # 訓練模型
    history = nn_model.fit(
        X_train_scaled, y_train_onehot,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # 預測
    nn_pred = nn_model.predict(X_test_scaled, verbose=0)
    nn_pred_classes = np.argmax(nn_pred, axis=1)
    
    # 計算指標
    nn_accuracy = accuracy_score(y_test_encoded, nn_pred_classes)
    nn_precision = precision_score(y_test_encoded, nn_pred_classes, average='weighted', zero_division=0)
    nn_recall = recall_score(y_test_encoded, nn_pred_classes, average='weighted', zero_division=0)
    nn_f1 = f1_score(y_test_encoded, nn_pred_classes, average='weighted', zero_division=0)

    results.append({
        'name': '深度神經網路',
        'model': nn_model,
        'accuracy': nn_accuracy,
        'precision': nn_precision,
        'recall': nn_recall,
        'f1_score': nn_f1,
        'predictions': nn_pred_classes
    })

except Exception as e:
    print(f"神經網路評估時發生錯誤: {e}")

# 結果展示
print("\n" + "="*80)
print("分類器性能比較結果")
print("="*80)
print(f"{'模型名稱':<20} {'準確率':<10} {'精確率':<10} {'召回率':<10} {'F1分數':<10}")
print("-"*80)

# 按準確率排序
results.sort(key=lambda x: x['accuracy'], reverse=True)

for result in results:
    print(f"{result['name']:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
          f"{result['recall']:<10.4f} {result['f1_score']:<10.4f}")

print("="*80)

# 顯示最佳模型的詳細報告
if results:
    best_result = results[0]
    print(f"\n最佳模型: {best_result['name']}")
    print(f"準確率: {best_result['accuracy']:.4f}")
    
    print(f"\n{best_result['name']} 詳細分類報告:")
    print(classification_report(y_test_encoded, best_result['predictions'], 
                               target_names=label_encoder.classes_, zero_division=0))

    # 混淆矩陣
    print(f"\n{best_result['name']} 混淆矩陣:")
    conf_mat = confusion_matrix(y_test_encoded, best_result['predictions'])
    conf_mat_df = pd.DataFrame(conf_mat, 
                              index=label_encoder.classes_, 
                              columns=label_encoder.classes_)
    print(conf_mat_df)

# 實驗分析與結論
print("\n" + "="*60)
print("實驗分析與結論")
print("="*60)

print("\n1. 模型性能排名:")
for i, result in enumerate(results[:5]):
    rank = f"第{i+1}名"
    print(f"   {rank} {result['name']}: 準確率 {result['accuracy']:.4f}")

print(f"\n2. 數據集特性分析:")
print(f"   - 總特徵數: {X_train.shape[1]}")
print(f"   - 樣本數: 訓練集 {X_train.shape[0]}, 測試集 {X_test.shape[0]}")
print(f"   - 類別數: {len(label_encoder.classes_)}")
print(f"   - 數據類型: 混合型(數值+類別)")

print("\n3. 模型適用性分析:")
if results:
    top_3_models = [r['name'] for r in results[:3]]
    
    if any('森林' in name or '提升' in name for name in top_3_models):
        print("   - 集成學習方法表現優異，適合處理複雜的特徵關係")
    
    if any('神經網路' in name for name in top_3_models):
        print("   - 深度學習能夠捕捉非線性特徵交互")
    
    if any('SVM' in name or '支持向量機' in name for name in top_3_models):
        print("   - SVM在高維特徵空間中表現穩定")
    
    if any('樸素貝葉斯' in name for name in top_3_models):
        print("   - 樸素貝葉斯在醫療診斷中具有良好的解釋性")

print("\n4. 醫療應用建議:")
print("   - 優先考慮高召回率模型，避免漏診風險")
print("   - 建議使用集成學習提高預測穩定性") 
print("   - 可考慮模型融合進一步提升性能")
print("   - 需要進行更多的特徵工程和調參優化")

print(f"\n實驗完成！共評估了 {len(results)} 個分類器")