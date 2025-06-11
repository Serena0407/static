import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

# 設置中文字型（macOS 適配）
try:
    plt.rcParams['font.family'] = 'Arial Unicode MS'  # macOS 預設中文字型
except:
    print("⚠️ Arial Unicode MS 字型不可用，使用預設字型")
plt.rcParams['axes.unicode_minus'] = False  # 確保負號顯示正常

# 讀取資料
try:
    df = pd.read_excel("processed_data.xlsx")
except FileNotFoundError:
    print("❌ 找不到 processed_data.xlsx，請確認檔案在 /Users/chenpeichi/Desktop/統計/")
    exit()

# 檢查欄位
required_columns = ['檢傷級數', 'pH', '年齡', '呼吸次數', '意識程度E', '心跳', '血壓(SBP)', '血氧濃度(%)', 'Y']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"❌ 資料缺少欄位：{missing_columns}")
    exit()

# 選擇重點欄位並移除缺失值
df = df[required_columns].dropna()

# 特徵與目標變數
X = df.drop(columns=['Y'])
y = df['Y']

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 SMOTE 平衡樣本數
try:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
except ValueError as e:
    print(f"❌ SMOTE 失敗：{e}，請檢查類別分佈或資料量")
    exit()

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 建立 XGBoost 模型
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("✅ 準確率：", accuracy_score(y_test, y_pred))
print("📊 分類報告：\n", classification_report(y_test, y_pred))

# AUC 計算
try:
    y_pred_proba = model.predict_proba(X_test)
    auc = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, multi_class='ovr')
    print("🔥 AUC：", auc)
except Exception as e:
    print(f"❌ AUC 計算失敗：{e}")

# SHAP 特徵解釋
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=required_columns[:-1], show=False)
    plt.title("特徵重要性分析（SHAP）")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"❌ SHAP 分析失敗：{e}")