from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"

# 1. 載入數據
keystats = pd.read_csv(ROOT / 'keystats.csv')
keystats['Date'] = pd.to_datetime(keystats['Date'])
keystats = keystats.sort_values('Date').reset_index(drop=True)

# 2. 原版切分 (Random Split - Leaky)
# 我們模擬原版，把 15% 數據作為測試集
indices = np.arange(len(keystats))
train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
orig_train_dates = keystats.iloc[train_idx]['Date']
orig_test_dates = keystats.iloc[test_idx]['Date']

# 3. 改良版切分 (Time Split)
val_end = int(len(keystats) * 0.85)
time_train_dates = keystats.iloc[:val_end]['Date']
time_test_dates = keystats.iloc[val_end:]['Date']

# 4. 畫圖 (甘特圖風格)
fig, ax = plt.subplots(figsize=(14, 6))

# 畫原版 (隨機切分)
ax.scatter(orig_train_dates, np.ones(len(orig_train_dates)) * 2, label='Original Model: Train Data', color='gray', alpha=0.5, s=10)
ax.scatter(orig_test_dates, np.ones(len(orig_test_dates)) * 1.5, label='Original Model: Test Data', color='blue', alpha=0.5, s=10)

# 畫改良版 (時間切分)
ax.scatter(time_train_dates, np.ones(len(time_train_dates)) * 0.5, label='Improved Model: Train Data', color='green', alpha=0.5, s=10)
ax.scatter(time_test_dates, np.ones(len(time_test_dates)) * 0.0, label='Improved Model: Test Data', color='red', alpha=0.5, s=10)

# 標註「紅色警戒區」
min_test = keystats.iloc[test_idx]['Date'].min()
max_test = keystats.iloc[test_idx]['Date'].max()
ax.axvspan(min_test, max_test, color='red', alpha=0.1, label='Data Leakage Zone (Overlapping Dates)')

ax.set_yticks([2, 1.5, 0.5, 0.0])
ax.set_yticklabels(['Original Train', 'Original Test', 'Improved Train', 'Improved Test'])
ax.set_title('Proof of Data Leakage: Original vs Improved Data Split', fontsize=16, fontweight='bold')
ax.set_xlabel('Time (Date)')
ax.grid(axis='x', alpha=0.3)

# 自定義圖例 (讓它更整齊)
handles = [
    mpatches.Patch(color='gray', alpha=0.5, label='Original Train Data'),
    mpatches.Patch(color='blue', alpha=0.5, label='Original Test Data'),
    mpatches.Patch(color='green', alpha=0.5, label='Improved Train Data'),
    mpatches.Patch(color='red', alpha=0.5, label='Improved Test Data'),
    mpatches.Patch(color='red', alpha=0.1, label='Data Leakage Zone (Future seen by Train)')
]
ax.legend(handles=handles, loc='upper left')

plt.tight_layout()
plt.savefig(str(OUT / "chart5_leakage_proof.png"), dpi=150)
plt.close()
print("✅ 已生成 Data Leakage 實錘圖: chart5_leakage_proof.png")
