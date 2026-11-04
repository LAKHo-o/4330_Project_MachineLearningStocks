import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

def draw_architecture():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off') # 隱藏座標軸

    # 定義方塊的樣式
    box_style = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2, alpha=0.9)
    text_style = dict(ha="center", va="center", fontsize=12, fontweight="bold")

    # 定義顏色
    color_data = "#E0F7FA"    # 藍綠色 (數據)
    color_proc = "#FFF9C4"    # 黃色 (處理)
    color_model = "#E8F5E9"   # 綠色 (模型)
    color_out = "#FCE4EC"     # 粉色 (輸出)

    # ==========================================
    # 1. 數據源 (Data Ingestion)
    # ==========================================
    ax.add_patch(patches.FancyBboxPatch((0.5, 4.5), 2.5, 1.5, boxstyle="round,pad=0.3", fc=color_data, ec="black", lw=2))
    ax.text(1.75, 5.25, "1. Data Ingestion\n\n(Historical CSV &\nReal-time API)", **text_style)

    # 2. 時間序列切分 (Time Splitting)
    ax.add_patch(patches.FancyBboxPatch((3.8, 4.5), 2.5, 1.5, boxstyle="round,pad=0.3", fc=color_proc, ec="black", lw=2))
    ax.text(5.05, 5.25, "2. Chronological Split\n\n(Train / Val / Test\nwith 20-day Gap)", **text_style)

    # 3. 特徵工程 (Feature Engineering)
    ax.add_patch(patches.FancyBboxPatch((7.1, 4.5), 2.5, 1.5, boxstyle="round,pad=0.3", fc=color_proc, ec="black", lw=2))
    ax.text(8.35, 5.25, "3. Feature Engineering\n\n(Base Features +\nMA6 Smoothing)", **text_style)

    # 4. 模型訓練 (Model Training)
    ax.add_patch(patches.FancyBboxPatch((10.4, 4.5), 2.5, 1.5, boxstyle="round,pad=0.3", fc=color_model, ec="black", lw=2))
    ax.text(11.65, 5.25, "4. Model Training\n\n(Random Forest +\nOOB Calibration)", **text_style)

    # ==========================================
    # 下排輸出 (Outputs)
    # ==========================================
    # 5. 回測與審計 (Backtest & Audit)
    ax.add_patch(patches.FancyBboxPatch((10.4, 1.0), 2.5, 1.5, boxstyle="round,pad=0.3", fc=color_out, ec="black", lw=2))
    ax.text(11.65, 1.75, "5a. Portfolio Backtest\n& Audit Trail\n(Trade Ledger)", **text_style)

    # 6. 實時推論 (Real-time Inference)
    ax.add_patch(patches.FancyBboxPatch((7.1, 1.0), 2.5, 1.5, boxstyle="round,pad=0.3", fc=color_out, ec="black", lw=2))
    ax.text(8.35, 1.75, "5b. Real-time Inference\n\n(Top 10 Predictions\n& Probability)", **text_style)

    # 7. 可視化與決策 (Visualization)
    ax.add_patch(patches.FancyBboxPatch((3.8, 1.0), 2.5, 1.5, boxstyle="round,pad=0.3", fc="#E1BEE7", ec="black", lw=2))
    ax.text(5.05, 1.75, "6. Visualization\n\n(Equity Curves &\nMetrics Dashboard)", **text_style)

    # ==========================================
    # 畫箭頭 (Arrows)
    # ==========================================
    arrow_style = dict(arrowstyle="->", color="black", lw=2, mutation_scale=20)

    # 上排橫向箭頭
    ax.annotate("", xy=(3.8, 5.25), xytext=(3.0, 5.25), arrowprops=arrow_style)
    ax.annotate("", xy=(7.1, 5.25), xytext=(6.3, 5.25), arrowprops=arrow_style)
    ax.annotate("", xy=(10.4, 5.25), xytext=(9.6, 5.25), arrowprops=arrow_style)

    # 向下箭頭 (模型 -> 輸出)
    ax.annotate("", xy=(11.65, 2.5), xytext=(11.65, 4.5), arrowprops=arrow_style)
    ax.annotate("", xy=(8.35, 2.5), xytext=(11.65, 4.5), arrowprops=dict(arrowstyle="->", color="black", lw=2, mutation_scale=20, connectionstyle="arc3,rad=0.2"))

    # 橫向箭頭 (輸出 -> 可視化)
    ax.annotate("", xy=(6.3, 1.75), xytext=(7.1, 1.75), arrowprops=arrow_style)
    ax.annotate("", xy=(6.3, 1.75), xytext=(10.4, 1.75), arrowprops=dict(arrowstyle="->", color="black", lw=2, mutation_scale=20, connectionstyle="arc3,rad=0.1"))

    plt.title("System Architecture: Automated Machine Learning Pipeline", fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    
    save_path = OUT / "chart6_system_architecture.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 系統架構圖已生成: {save_path}")

if __name__ == "__main__":
    draw_architecture()
