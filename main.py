import sys
import os
import time
import json
import re  # 处理文本
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder # 用于JSON序列化

try:
    import sys 
    import matplotlib
    # 强制设置后端
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    HAS_REPORT_DEPS = True
    print(">>> 成功导入 matplotlib 和 fpdf")
except Exception as e:
    HAS_REPORT_DEPS = False
    print("\n" + "="*50)
    print("【严重错误】导入库失败！")
    print(f"真实报错信息: {e}")
    print(f"当前代码运行的 Python 位置: {sys.executable}")
    print("="*50 + "\n")

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline, Pipeline

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QStackedWidget, QFrame, 
    QFileDialog, QComboBox, QSlider, QScrollArea, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QGridLayout, 
    QToolButton, QGroupBox, QListWidget, QListWidgetItem, 
    QAbstractItemView, QGraphicsDropShadowEffect, QTabWidget, 
    QSizePolicy, QTextEdit, QGraphicsOpacityEffect, # 用于遮罩层的淡出动画
    QStyleFactory, QStyledItemDelegate
)

from PyQt5.QtCore import (
    Qt, QDate, QSize, QPropertyAnimation, QEasingCurve, 
    QPointF, QTimer, QThread, pyqtSignal, QTime, QEvent,
    QObject
)
from PyQt5.QtGui import QIcon, QFont, QColor, QTextDocument, QTextCursor, QPixmap, QPainter

# 尝试导入 WebEngine
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtPrintSupport import QPrinter
except ImportError:
    print("错误：缺少必要组件。请在终端运行: pip install PyQtWebEngine")
    sys.exit(1)

# 主题与配色
LIGHT_PALETTE = {
    "PRIMARY": "#0061A4",
    "ON_PRIMARY": "#FFFFFF",
    "PRIMARY_CONTAINER": "#D1E4FF",
    "ON_PRIMARY_CONTAINER": "#001D36",
    "SURFACE_VARIANT": "#E1E2EC",
    "BACKGROUND": "#FDFCFF",
    "SURFACE": "#FFFFFF",
    "ON_SURFACE": "#1C1B1F",
    "OUTLINE": "#74777F",
    "TEXT": "#1A1C1E",
    "CARD_BG": "#FFFFFF",
    "CARD_BORDER": "#CAC4D0",
    "PLOTLY_THEME": "plotly_white",
    "METRIC_COLORS": [("#D1E4FF", "#0061A4"), ("#C4EED0", "#006E1C"), ("#FFDCC2", "#8A3600"), ("#E8DEF8", "#65558F")],
    "LOG_BG": "#F0F2F5",
    "LOG_TEXT": "#333333"
}

DARK_PALETTE = {
    "PRIMARY": "#A0C8FF", 
    "ON_PRIMARY": "#003258",
    "PRIMARY_CONTAINER": "#00497D",
    "ON_PRIMARY_CONTAINER": "#D1E4FF",
    "SURFACE_VARIANT": "#44474F", 
    "BACKGROUND": "#1A1C1E",
    "SURFACE": "#1A1C1E",
    "ON_SURFACE": "#E6E1E5",
    "OUTLINE": "#8E9199",
    "TEXT": "#E2E2E6",
    "CARD_BG": "#26292D",
    "CARD_BORDER": "#44474F",
    "PLOTLY_THEME": "plotly_dark",
    "METRIC_COLORS": [("#00497D", "#D1E4FF"), ("#005210", "#C4EED0"), ("#622500", "#FFDCC2"), ("#4A4458", "#E8DEF8")],
    "LOG_BG": "#121212",
    "LOG_TEXT": "#E0E0E0" 
}

CURRENT_THEME_MODE = "Light" 
CURRENT_COLORS = LIGHT_PALETTE
ARROW_DOWN_SVG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNDQ0NzRmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBvbHlsaW5lIHBvaW50cz0iNiA5IDEyIDE1IDE4IDkiLz48L3N2Zz4="

def hex_to_rgba(hex_code, alpha):
    hex_code = hex_code.lstrip('#')
    return f"rgba({int(hex_code[0:2], 16)}, {int(hex_code[2:4], 16)}, {int(hex_code[4:6], 16)}, {alpha})"

def get_stylesheet():
    C = CURRENT_COLORS
    hover_icon_color = "#000000" if CURRENT_THEME_MODE == "Light" else "#FFFFFF"

    # 颜色转换
    def hex_to_rgba(hex_code, alpha):
        h = hex_code.lstrip('#')
        return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"

    glass_bg_alpha = 0.85 if CURRENT_THEME_MODE == "Light" else 0.90
    popup_bg_color = hex_to_rgba(C["CARD_BG"], glass_bg_alpha)
    active_item_bg = C["PRIMARY"]         
    active_item_text = C["ON_PRIMARY"]     

    return f"""
    QMainWindow {{ background-color: {C["BACKGROUND"]}; }}
    QWidget {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: {C["TEXT"]}; font-size: 13px; }}
    QFrame.Card {{ background-color: {C["CARD_BG"]}; border-radius: 32px; border: 1px solid {C["CARD_BORDER"]}; }}
    
    #SidebarContainer {{ background-color: {C["SURFACE_VARIANT"]}; border-top-right-radius: 28px; border-bottom-right-radius: 28px; }}
    QScrollArea#SidebarScroll {{ border: none; background: transparent; }}
    QScrollArea#SidebarScroll QScrollBar:vertical {{ width: 0px; }}
    QWidget#SidebarContent {{ background: transparent; }}

    QLabel#AppTitle {{ font-size: 24px; font-weight: 700; color: {C["TEXT"]}; padding: 20px 10px; }}
    QLabel.CardTitle {{ font-size: 18px; font-weight: 600; color: {C["TEXT"]}; margin-bottom: 15px; }}

    QTabWidget::pane {{ border: none; margin-top: 20px; background: transparent; }}
    QTabWidget::tab-bar {{ left: 10px; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{ background-color: {C["SURFACE_VARIANT"]}; color: {C["TEXT"]}; border: none; min-height: 36px; border-radius: 18px; padding: 0 24px; margin-right: 12px; font-weight: 600; }}
    QTabBar::tab:selected {{ background-color: {C["PRIMARY"]}; color: {C["ON_PRIMARY"]}; }}
    QTabBar::tab:hover:!selected {{ background-color: {C["PRIMARY_CONTAINER"]}; color: {C["ON_PRIMARY_CONTAINER"]}; }}

    QGroupBox {{ border: 1px solid {C["OUTLINE"]}; border-radius: 16px; margin-top: 15px; font-weight: 600; color: {C["PRIMARY"]}; background-color: transparent; padding-top: 35px; }}
    QGroupBox::title {{ left: 15px; top: 0px; padding: 0 5px; }}

    QComboBox {{ 
        border: 1px solid {C["OUTLINE"]}; 
        border-radius: 20px; 
        padding: 5px 20px; 
        background-color: {C["CARD_BG"]}; 
        color: {C["TEXT"]}; 
        min-height: 40px; 
        font-size: 13px; 
        outline: none;
    }}
    QComboBox:hover {{ border: 1px solid {C["PRIMARY"]}; background-color: {C["SURFACE_VARIANT"]}; }}
    QComboBox:focus {{ border: 2px solid {C["PRIMARY"]}; }}
    
    QComboBox::drop-down {{ 
        subcontrol-origin: padding; 
        subcontrol-position: top right; 
        width: 30px; 
        border: none; 
        background: transparent; 
        border-top-right-radius: 20px; 
        border-bottom-right-radius: 20px;
    }}
    QComboBox::down-arrow {{ image: url({ARROW_DOWN_SVG}); width: 14px; height: 14px; }}
    
    QComboBox QAbstractItemView {{
        background-color: {popup_bg_color}; 
        border: 1px solid {hex_to_rgba(C["OUTLINE"], 0.2)};
        border-radius: 0px; 
        padding: 6px 4px;
        selection-background-color: transparent; 
        outline: none;
    }}
    
    QComboBox QAbstractItemView::item {{ 
        height: 36px; 
        padding-left: 10px; 
        color: {C["TEXT"]}; 
        border-radius: 8px; 
        margin: 2px 4px;
        border: none;
    }}
    
    QComboBox QAbstractItemView::item:selected, 
    QComboBox QAbstractItemView::item:hover {{ 
        background-color: {active_item_bg}; 
        color: {active_item_text};
        font-weight: 600;
    }}
    
    QComboBox QAbstractItemView QScrollBar:vertical {{
        width: 6px;
        background: transparent; 
        margin-right: 2px;
        border: none;
    }}
    QComboBox QAbstractItemView QScrollBar::handle:vertical {{
        background: {hex_to_rgba(C["SURFACE_VARIANT"], 0.8)};
        border-radius: 3px;
        min-height: 20px;
    }}
    QComboBox QAbstractItemView QScrollBar::add-line:vertical, 
    QComboBox QAbstractItemView QScrollBar::sub-line:vertical {{ height: 0px; background: transparent; }}
    QComboBox QAbstractItemView QScrollBar::add-page:vertical, 
    QComboBox QAbstractItemView QScrollBar::sub-page:vertical {{ background: transparent; }}
    
    QComboBox QAbstractItemView::corner {{ background: transparent; border: none; }}
    
    QPushButton {{ background-color: transparent; border: 1px solid {C["OUTLINE"]}; border-radius: 20px; padding: 6px 20px; min-height: 32px; font-weight: 600; color: {C["PRIMARY"]}; }}
    QPushButton:hover {{ background-color: {C["PRIMARY_CONTAINER"]}; color: {C["ON_PRIMARY_CONTAINER"]}; border: 1px solid transparent; }}
    QPushButton#PrimaryButton {{ background-color: {C["PRIMARY"]}; color: {C["ON_PRIMARY"]}; border: none; border-radius: 24px; min-height: 48px; }}
    QPushButton#PrimaryButton:hover {{ opacity: 0.9; }}
    QPushButton#PrimaryButton:disabled {{ background-color: {C["OUTLINE"]}; color: {C["SURFACE"]}; opacity: 0.6; }}
    
    QLineEdit#LoginInput, QLineEdit#PasswordInput {{ border: 2px solid {C["OUTLINE"]}; border-radius: 12px; padding: 12px 15px; background: rgba(255, 255, 255, 0.7); font-size: 14px; color: #1A1C1E; selection-background-color: {C["PRIMARY"]}; }}
    QLineEdit#LoginInput:focus, QLineEdit#PasswordInput:focus {{ border: 2px solid {C["PRIMARY"]}; background: #FFFFFF; }}
    QLineEdit#LoginInput:hover, QLineEdit#PasswordInput:hover {{ background: #FFFFFF; }}

    QPushButton#LoginBtn {{ background-color: {C["PRIMARY"]}; color: white; border: none; border-radius: 24px; min-height: 48px; max-height: 48px; font-size: 16px; font-weight: bold; }}
    QPushButton#LoginBtn:hover {{ background-color: {C["PRIMARY_CONTAINER"]}; color: {C["ON_PRIMARY_CONTAINER"]}; margin-top: -1px; }}
    QPushButton#LoginBtn:pressed {{ margin-top: 1px; }}

    QPushButton#IconButton {{ background-color: {C["PRIMARY_CONTAINER"]}; color: {C["PRIMARY"]}; border: 1px solid {C["PRIMARY"]}; border-radius: 20px; padding: 0; min-height: 40px; min-width: 40px; font-weight: 900; font-size: 18px; }}
    QPushButton#IconButton:hover {{ background-color: {C["PRIMARY"]} !important; color: {hover_icon_color} !important; border: 1px solid {C["PRIMARY"]}; }}

    QLineEdit {{ border: none; border-bottom: 1px solid {C["OUTLINE"]}; background: {C["SURFACE_VARIANT"]}; color: {C["TEXT"]}; padding: 10px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px; min-height: 20px; }}
    QLineEdit:focus {{ border-bottom: 2px solid {C["PRIMARY"]}; }}

    /* ---------------------------------------------------- */

    QListWidget {{ border: 1px solid {C["OUTLINE"]}; border-radius: 16px; padding: 5px; background-color: transparent; color: {C["TEXT"]}; outline: none; }}
    QListWidget::item {{ background-color: {C["SURFACE_VARIANT"]}; color: {C["TEXT"]}; border-radius: 18px; padding: 8px 15px; margin: 4px 5px; border: 1px solid transparent; }}
    QListWidget::item:hover {{ background-color: {C["PRIMARY_CONTAINER"]}; color: {C["ON_PRIMARY_CONTAINER"]}; }}
    QListWidget::item:selected {{ background-color: {C["PRIMARY"]}; color: {C["ON_PRIMARY"]}; font-weight: bold; }}
    QListWidget QScrollBar:vertical {{ width: 0px; }}

    QTableWidget {{ background-color: {C["SURFACE"]}; border: none; color: {C["TEXT"]}; }}
    QTableWidget::item {{ padding: 10px; border-bottom: 1px solid {C["SURFACE_VARIANT"]}; }}
    
    QTableWidget::item:selected {{ background-color: {C["PRIMARY"]}; color: {C["ON_PRIMARY"]}; font-weight: bold; }}

    QHeaderView::section {{ background-color: {C["SURFACE"]}; padding: 10px; border: none; border-bottom: 2px solid {C["SURFACE_VARIANT"]}; color: {C["PRIMARY"]}; }}
    
    QTextEdit#LogWindow {{ background-color: {C["LOG_BG"]}; color: {C["LOG_TEXT"]}; border: 1px solid {C["OUTLINE"]}; border-radius: 12px; font-family: "Consolas", "Monaco", monospace; font-size: 12px; padding: 10px; }}
    """

# 报表生成
class ReportGenerator:
    @staticmethod
    def clean_text(text):
        s = str(text)
        s_ascii = re.sub('[^\x00-\x7F]+', '', s)
        return s_ascii.strip()

    @staticmethod
    def create_plots(results):
        # 生成临时图片用于 PDF
        paths = {}
        if not HAS_REPORT_DEPS: return paths
        
        plt.figure(figsize=(10, 5))
        plt.style.use('ggplot')
        plt.plot(results['y_actual'], label='Actual', color='#0061A4', linewidth=2)
        plt.plot(results['y_fitted'], label='Fitted', color='#FFAB91', linestyle='--')
        
        anom_idx = results.get('anomaly_indices', [])
        if len(anom_idx) > 0:
            plt.scatter(anom_idx, results['y_actual'][anom_idx], color='red', marker='x', s=100, label='Anomaly (3σ)', zorder=5)

        if len(results['future_x']) > 0:
            plt.plot(results['future_x'], results['future_y'], label='Forecast', color='#B3261E', marker='o')
        
        plt.title('Prediction & Forecast Trend with Anomalies')
        plt.legend(); plt.grid(True, alpha=0.3)
        paths['trend'] = 'temp_trend.png'
        plt.savefig(paths['trend'], dpi=150, bbox_inches='tight')
        plt.close()

        imp = results.get('feature_importance')
        feats = results.get('features')
        if imp is not None and len(imp) == len(feats):
            plt.figure(figsize=(10, 5))
            indices = np.argsort(np.abs(imp))
            clean_feats = [ReportGenerator.clean_text(f) for f in feats]
            plt.barh(range(len(indices)), imp[indices], color='#00695C', align='center')
            plt.yticks(range(len(indices)), [clean_feats[i] for i in indices])
            plt.title('Feature Importance / Coefficients')
            plt.grid(axis='x', alpha=0.3)
            paths['imp'] = 'temp_imp.png'
            plt.savefig(paths['imp'], dpi=150, bbox_inches='tight')
            plt.close()

        resid = results.get('residuals')
        if resid is not None:
            plt.figure(figsize=(10, 4))
            plt.scatter(results['y_fitted'], resid, alpha=0.6, color='#B00020')
            plt.axhline(y=0, color='black', linestyle='--')
            std_dev = np.std(resid)
            plt.axhline(y=3*std_dev, color='red', linestyle=':', alpha=0.5)
            plt.axhline(y=-3*std_dev, color='red', linestyle=':', alpha=0.5)
            plt.title('Residuals Distribution')
            plt.xlabel('Fitted Values'); plt.ylabel('Residuals')
            paths['resid'] = 'temp_resid.png'
            plt.savefig(paths['resid'], dpi=150, bbox_inches='tight')
            plt.close()
            
        return paths

    @staticmethod
    def generate_pdf(filename, results, plot_paths):
        if not HAS_REPORT_DEPS: return False
        
        model_name_clean = ReportGenerator.clean_text(results['model_name'])
        model_name_clean = model_name_clean.replace("(", "").replace(")", "").strip()
        
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 20, "Analysis Report", 0, 1, 'C')
        pdf.set_font("Arial", '', 10)
        time_str = time.strftime('%Y-%m-%d %H:%M')
        pdf.cell(0, 10, f"Generated by PredictData Pro on {time_str}", 0, 1, 'C')
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "1. Key Performance Indicators", 0, 1)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(240, 242, 245)
        pdf.cell(60, 10, f"Model: {model_name_clean}", 1, 0, 'C', 1)
        pdf.cell(60, 10, f"R-Squared: {results['r2']:.4f}", 1, 0, 'C', 1)
        pdf.cell(60, 10, f"RMSE: {results['rmse']:.4f}", 1, 1, 'C', 1)
        
        anom_count = len(results.get('anomaly_indices', []))
        pdf.ln(5)
        pdf.set_text_color(200, 0, 0) if anom_count > 0 else pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, f"Anomaly Detection: {anom_count} anomalies found (3-Sigma Rule)", 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        if 'trend' in plot_paths:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "2. Trend & Forecast Analysis", 0, 1)
            pdf.image(plot_paths['trend'], x=10, w=190)
            pdf.ln(5)

        if 'imp' in plot_paths:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "3. Feature Drivers (Importance)", 0, 1)
            pdf.image(plot_paths['imp'], x=10, w=190)
            pdf.ln(5)

        if 'resid' in plot_paths:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "4. Model Health Check (Residuals)", 0, 1)
            pdf.image(plot_paths['resid'], x=10, w=190)
            pdf.ln(10)

        if results['future_y'] is not None and len(results['future_y']) > 0:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "5. Future Forecast Data", 0, 1)
            pdf.set_font("Arial", '', 10)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(95, 8, "Step", 1, 0, 'C', 1)
            pdf.cell(95, 8, "Forecast Value", 1, 1, 'C', 1)
            for i in range(min(15, len(results['future_y']))):
                step_val = ReportGenerator.clean_text(results['future_x'][i])
                pdf.cell(95, 8, step_val, 1, 0, 'C')
                pdf.cell(95, 8, f"{results['future_y'][i]:.4f}", 1, 1, 'C')
            if len(results['future_y']) > 15:
                pdf.cell(190, 8, "... (More data in CSV export)", 1, 1, 'C')

        pdf.output(filename)
        for p in plot_paths.values():
            if os.path.exists(p): os.remove(p)
        return True

# 分析
class AnalysisEngine:
    @staticmethod
    def clean_data(df, strategy):
        if df is None or df.empty: return None
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        if strategy == "丢弃缺失行 (Drop)":
            df_clean = df_clean.dropna()
        elif strategy == "均值填充 (Fill Mean)":
            df_clean = df_clean.fillna(df_clean.mean())
        elif strategy == "线性插值 (Interpolate)":
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
            df_clean = df_clean.dropna()
        df_clean = df_clean.dropna()
        if len(df_clean) < 3: return None
        return df_clean

    @staticmethod
    def auto_select_model(X, y):
        best_score = -np.inf
        best_model_info = None
        candidates = [
            ("Linear", LinearRegression()),
            ("Poly (Deg=2)", make_pipeline(PolynomialFeatures(2), LinearRegression())),
            ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)) 
        ]
        for name, model in candidates:
            try:
                model.fit(X, y)
                if "RandomForest" in name and not hasattr(model, "estimators_"): continue
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                if r2 > best_score:
                    best_score = r2
                    best_model_info = (name, model, y_pred, r2)
            except: continue
        return best_model_info

    @staticmethod
    def get_model_instance(model_mode):
        if "Linear" in model_mode: return LinearRegression()
        elif "Poly" in model_mode: return make_pipeline(PolynomialFeatures(2), LinearRegression())
        elif "RandomForest" in model_mode: return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        elif "SVR" in model_mode: return make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, gamma='auto'))
        return None

    @staticmethod
    def run(df, target_col, feature_cols, model_mode, forecast_steps):
        if df is None or target_col not in df.columns: return None
        y = df[target_col].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(y) == 0: return None
        
        try:
            if not feature_cols:
                X = np.arange(len(df[target_col])).reshape(-1, 1)
                y = df[target_col].values
                is_time_series = True
            else:
                X = df[feature_cols].values
                y = df[target_col].values
                is_time_series = False
            if np.isnan(X).any() or np.isnan(y).any(): return None
        except: return None

        res_info = None
        if model_mode == "自动选择 (AutoML)":
            res_info = AnalysisEngine.auto_select_model(X, y)
        else:
            try:
                m = AnalysisEngine.get_model_instance(model_mode)
                if m is None: return None
                m.fit(X, y)
                if hasattr(m, "predict"):
                    if "RandomForest" in str(type(m)) and not hasattr(m, "estimators_"): return None
                    yp = m.predict(X)
                    r2 = r2_score(y, yp)
                    res_info = (model_mode, m, yp, r2)
            except: return None

        if res_info is None: return None
        m_name, model, y_pred, r2 = res_info
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        residuals = y - y_pred

        resid_std = np.std(residuals)
        if resid_std == 0:
            anomaly_indices = []
        else:
            anomaly_mask = np.abs(residuals) > (3 * resid_std)
            anomaly_indices = np.where(anomaly_mask)[0]

        # 支持纯时间序列(is_time_series=True)或者单特征情况下的自动外推
        future_X, future_y, future_X_flat = None, [], []
        
        # 判断是否进行预测：是纯序列，或者是仅含单特征
        should_forecast = is_time_series
        last_x_val = None
        step_val = 1
        
        # 如果不是默认的时间序列模式，但特征维度为1，尝试计算步长进行外推
        if (not should_forecast) and (X.shape[1] == 1):
            should_forecast = True
            try:
                # 获取最后一个值
                last_x_val = X[-1][0]
                # 尝试根据最后两点确定步长
                step_val = X[-1][0] - X[-2][0]
                if step_val == 0: step_val = 1
            except:
                step_val = 1
        elif is_time_series:
            # 纯时间索引模式
            last_x_val = X[-1][0]
            step_val = 1

        if should_forecast and last_x_val is not None:
            # 生成未来输入数据 X
            future_range = []
            curr = last_x_val
            for _ in range(forecast_steps):
                curr += step_val
                future_range.append(curr)
            future_X = np.array(future_range).reshape(-1, 1)
            
            try:
                future_y = model.predict(future_X) # 预测
                
                # 对齐处理
                # 即使用真实的 Day (100, 101...) 训练了模型，在界面上展示时，历史曲线X轴也是从0开始的
                # 为了让红色的预测线无缝接在历史线后面，future_x 返回给界面的必须是视觉上的后续索引 (N, N+1...)
                # 这样红色折线才会出现在图表的最右侧，而不是某些随机的X轴坐标上
                start_visual_idx = len(y)
                future_X_flat = np.arange(start_visual_idx, start_visual_idx + forecast_steps)
                
            except Exception as e:
                # 如果预测报错（随机森林有时对越界数据敏感），则置空
                future_y = []
                future_X_flat = []

        feat_importance = None
        importance_type = 'unknown' 
        feat_names = feature_cols if feature_cols else ['Index']
        
        try:
            raw_weights = None
            if hasattr(model, 'feature_importances_'): 
                raw_weights = model.feature_importances_
                importance_type = 'impurity'
            elif hasattr(model, 'coef_'): 
                raw_weights = model.coef_
                importance_type = 'coef'
            elif isinstance(model, Pipeline):
                est = model.steps[-1][1]
                if hasattr(est, 'feature_importances_'): 
                    raw_weights = est.feature_importances_
                    importance_type = 'impurity'
                elif hasattr(est, 'coef_'): 
                    raw_weights = est.coef_
                    importance_type = 'coef'
            
            if raw_weights is not None:
                if len(raw_weights.flatten()) == len(feat_names): 
                    feat_importance = raw_weights.flatten()
        except: pass

        return {
            'model_name': m_name, 'y_actual': y, 'y_fitted': y_pred, 
            'rmse': rmse, 'r2': r2, 'future_x': future_X_flat, 
            'future_y': future_y, 'is_time_series': is_time_series, 
            'features': feat_names,
            'feature_importance': feat_importance,
            'importance_type': importance_type, 
            'residuals': residuals,
            'anomaly_indices': anomaly_indices, 
            'model_obj': model 
        }

# 后台工作线程
class AnalysisWorker(QThread):
    result_ready = pyqtSignal(dict)  
    error_occurred = pyqtSignal(str) 

    def __init__(self, df, target, features, model_mode, model2_mode, steps):
        super().__init__()
        self.df = df.copy(); self.target = target; self.features = features
        self.model_mode = model_mode; self.model2_mode = model2_mode; self.steps = steps

    def run(self):
        try:
            res1 = AnalysisEngine.run(self.df, self.target, self.features, self.model_mode, self.steps)
            if not res1: self.error_occurred.emit("主模型训练失败。"); return
            final_res = res1
            if self.model2_mode and self.model2_mode != "无 (None)":
                res2 = AnalysisEngine.run(self.df, self.target, self.features, self.model2_mode, self.steps)
                if res2:
                    final_res['compare_model'] = {
                        'model_name': res2['model_name'], 'y_fitted': res2['y_fitted'],
                        'future_y': res2['future_y'], 'r2': res2['r2']
                    }
            self.result_ready.emit(final_res)
        except Exception as e: self.error_occurred.emit(str(e))

# UI 组件

class StatCard(QFrame):
    def __init__(self, title, value, icon, bg_color, text_color):
        super().__init__()
        self.setObjectName("StatCard")
        self.setStyleSheet(f"QFrame#StatCard {{ background-color: {bg_color}; border-radius: 32px; border: 2px solid transparent; }}")
        self.shadow = QGraphicsDropShadowEffect(self); self.shadow.setBlurRadius(20); self.shadow.setOffset(0, 8)
        self.shadow.setColor(QColor(0, 0, 0, 0)); self.setGraphicsEffect(self.shadow)
        layout = QHBoxLayout(self); layout.setContentsMargins(24, 15, 30, 15); layout.setSpacing(18)
        icon_lbl = QLabel(icon); icon_lbl.setAlignment(Qt.AlignCenter); icon_lbl.setFixedSize(48, 48)
        icon_lbl.setStyleSheet(f"background-color: rgba(255,255,255,0.7); color: {text_color}; border-radius: 24px; font-size: 22px;")
        text_layout = QVBoxLayout()
        t_lbl = QLabel(title); t_lbl.setStyleSheet(f"font-size: 13px; color: {text_color}; opacity: 0.9; font-weight: 600;")
        v_lbl = QLabel(str(value)); v_lbl.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {text_color}; letter-spacing: -0.5px;")
        text_layout.addWidget(t_lbl); text_layout.addWidget(v_lbl); layout.addWidget(icon_lbl); layout.addLayout(text_layout)
    def enterEvent(self, event): 
        self.shadow.setColor(QColor(0, 0, 0, 50))
        super().enterEvent(event)
    def leaveEvent(self, event): 
        self.shadow.setColor(QColor(0, 0, 0, 0))
        super().leaveEvent(event)

class PasswordEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PasswordInput")
        self.setEchoMode(QLineEdit.Password)
        self.setPlaceholderText("密码")
        self.setFixedHeight(48)
        self.eye_btn = QToolButton(self)
        self.eye_btn.setCursor(Qt.PointingHandCursor)
        self.eye_btn.setStyleSheet("QToolButton { border: none; background: transparent; color: #74777F; font-size: 16px; } QToolButton:hover { color: #0061A4; }")
        self.eye_btn.setText("👁")
        self.eye_btn.clicked.connect(self.toggle_mode)
        self.setTextMargins(0, 0, 40, 0)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        btn_size = self.height()
        self.eye_btn.setGeometry(self.width() - btn_size, 0, btn_size, btn_size)
    def toggle_mode(self):
        if self.echoMode() == QLineEdit.Password: 
            self.setEchoMode(QLineEdit.Normal)
            self.eye_btn.setText("🚫")
        else: 
            self.setEchoMode(QLineEdit.Password)
            self.eye_btn.setText("👁")

class LoginWidget(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.setStyleSheet(f"QWidget#LoginRoot {{ background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 {CURRENT_COLORS['SURFACE_VARIANT']}, stop:1 {CURRENT_COLORS['PRIMARY_CONTAINER']}); }}")
        self.setObjectName("LoginRoot")
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        card = QFrame()
        card.setObjectName("LoginContainer")
        card.setFixedWidth(400)
        card.setStyleSheet("QFrame#LoginContainer { background-color: rgba(255, 255, 255, 0.85); border-radius: 28px; border: 1px solid #FFFFFF; }")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)
        main_layout.addWidget(card, 0, Qt.AlignCenter)
        vbox = QVBoxLayout(card); vbox.setContentsMargins(40, 50, 40, 50)
        vbox.setSpacing(20)
        logo = QLabel("⚡")
        logo.setStyleSheet(f"font-size:52px; color:#0061A4;")
        logo.setAlignment(Qt.AlignCenter)
        title = QLabel("PredictData Pro")
        title.setStyleSheet("font-size:26px; font-weight:800; color:#1e1e1e;")
        title.setAlignment(Qt.AlignCenter)
        hour = QTime.currentTime().hour()
        greeting = "早上好" if 5 <= hour < 12 else ("下午好" if 12 <= hour < 18 else "晚上好")
        subtitle = QLabel(f"{greeting}，欢迎回来"); subtitle.setStyleSheet("font-size:14px; color:#666666;")
        subtitle.setAlignment(Qt.AlignCenter)
        self.u_in = QLineEdit("admin")
        self.u_in.setObjectName("LoginInput")
        self.u_in.setPlaceholderText("用户名")
        self.u_in.setFixedHeight(48)
        self.u_in.returnPressed.connect(self.do_login)
        self.p_in = PasswordEdit()
        self.p_in.returnPressed.connect(self.do_login)
        btn = QPushButton("立即登录")
        btn.setObjectName("LoginBtn")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setDefault(True)
        btn.clicked.connect(self.do_login)
        vbox.addWidget(logo)
        vbox.addWidget(title)
        vbox.addWidget(subtitle)
        vbox.addSpacing(15)
        vbox.addWidget(self.u_in)
        vbox.addWidget(self.p_in)
        vbox.addSpacing(25)
        vbox.addWidget(btn)
    
    def do_login(self):
        if self.u_in.text().strip() == "admin" and self.p_in.text() == "password": 
            self.switch_callback()
        else: 
            QMessageBox.warning(self, "错误", "账号或密码错误 (admin/password)")
            self.p_in.clear()
            self.p_in.setFocus()
    
    def reset(self): 
        self.p_in.clear()
        self.p_in.setEchoMode(QLineEdit.Password)
        self.u_in.setFocus()

class DashboardWidget(QWidget):
    def __init__(self, logout_callback, toggle_theme_callback):
        super().__init__()
        self.logout_cb = logout_callback
        self.toggle_theme_cb = toggle_theme_callback
        self.df_raw = None
        self.df_clean = None
        self.results = None
        self.current_file_path = None
        self.worker = None
        self.analysis_start_time = 0
        
        self.init_ui()
        self.log("系统初始化完成。等待数据导入...", "INFO")

    def apply_slider_style(self):
        C = CURRENT_COLORS
        self.slider.setStyleSheet(f"""
            QSlider#StepSlider::groove:horizontal {{
                border: none;
                height: 8px;
                background: {C["SURFACE_VARIANT"]};
                border-radius: 4px;
            }}
            QSlider#StepSlider::sub-page:horizontal {{
                background: {C["PRIMARY"]};
                border-radius: 4px;
            }}
            QSlider#StepSlider::handle:horizontal {{
                background: {C["PRIMARY"]};
                width: 10px;                 
                height: 24px;                
                margin: -8px 0px;           
                border-radius: 5px;          
                border: none;
            }}
            QSlider#StepSlider::handle:horizontal:hover {{
                background: {C["PRIMARY"]};
                width: 10px;
                height: 24px;
                margin: -8px 0px;
                border-radius: 5px;
            }}
            QSlider#StepSlider::handle:horizontal:pressed {{
                background: {C["ON_PRIMARY_CONTAINER"]}; 
                width: 10px;
                height: 24px;
                margin: -8px 0px;
                border-radius: 5px;
            }}
        """)

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 8))
        shadow.setOffset(0, 2)
        widget.setGraphicsEffect(shadow)

    def toggle_sidebar(self):
        current_width = self.side_container.width()
        start_w, end_w = (360, 0) if current_width > 0 else (0, 360)
        if current_width > 0: 
            self.side_container.setMinimumWidth(0)
        self.anim = QPropertyAnimation(self.side_container, b"maximumWidth")
        self.anim.setDuration(400)
        self.anim.setStartValue(start_w)
        self.anim.setEndValue(end_w)
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.start()

    def init_ui(self):
        layout = QHBoxLayout(self); 
        layout.setSpacing(0); 
        layout.setContentsMargins(5,5,5,5)
        self.side_container = QFrame()
        self.side_container.setObjectName("SidebarContainer")
        self.side_container.setFixedWidth(360)
        side_main_layout = QVBoxLayout(self.side_container)
        side_main_layout.setContentsMargins(0, 20, 0, 20)
        
        h_head = QHBoxLayout()
        h_head.setContentsMargins(20, 0, 20, 0)
        title_lbl = QLabel("⚡ PredictData", objectName="AppTitle")
        title_lbl.setContentsMargins(0,0,0,0)
        self.btn_theme = QPushButton("🌞" if CURRENT_THEME_MODE=="Light" else "🌙")
        self.btn_theme.setObjectName("IconButton")
        self.btn_theme.clicked.connect(self.on_theme_toggle)
        h_head.addWidget(title_lbl)
        h_head.addStretch()
        h_head.addWidget(self.btn_theme)
        side_main_layout.addLayout(h_head)
        
        scroll = QScrollArea()
        scroll.setObjectName("SidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setObjectName("SidebarContent")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 10, 20, 20)
        scroll_layout.setSpacing(15)
        
        g_data = QGroupBox("1. 数据源与清洗")
        gl_data = QVBoxLayout(g_data)
        gl_data.setContentsMargins(15, 45, 15, 15) 
        h_file = QHBoxLayout()
        btn_load = QPushButton(" 导入 CSV / Excel")
        btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); 
        btn_load.setMinimumHeight(40)
        btn_load.clicked.connect(self.load_csv)
        btn_refresh = QPushButton("↻")
        btn_refresh.setObjectName("IconButton")
        btn_refresh.setFixedSize(40, 40)
        btn_refresh.clicked.connect(self.refresh_data)
        h_file.addWidget(btn_load)
        h_file.addSpacing(10)
        h_file.addWidget(btn_refresh)
        self.cb_clean = QComboBox()
        self.cb_clean.addItems(["丢弃缺失行 (Drop)", "均值填充 (Fill Mean)", "线性插值 (Interpolate)"])
        self.cb_clean.currentIndexChanged.connect(self.process_data)
        gl_data.addLayout(h_file)
        gl_data.addWidget(QLabel("缺失值处理:"))
        gl_data.addWidget(self.cb_clean)
        scroll_layout.addWidget(g_data)

        self.g_vars = QGroupBox("2. 变量选择")
        self.g_vars.hide()
        gl_vars = QVBoxLayout(self.g_vars)
        gl_vars.setContentsMargins(15, 45, 15, 15)
        self.cb_target = QComboBox()
        self.list_feats = QListWidget()
        self.list_feats.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_feats.setFixedHeight(160)
        self.list_feats.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        gl_vars.addWidget(QLabel("目标变量 (Y):"))
        gl_vars.addWidget(self.cb_target)
        gl_vars.addWidget(QLabel("特征变量 (X):"))
        gl_vars.addWidget(self.list_feats)
        scroll_layout.addWidget(self.g_vars)
        
        self.g_model = QGroupBox("3. 模型与预测")
        self.g_model.hide()
        gl_model = QVBoxLayout(self.g_model)
        gl_model.setContentsMargins(15, 45, 15, 15)
        self.cb_model = QComboBox()
        self.cb_model.addItems(["自动选择 (AutoML)", "线性回归 (Linear)", "多项式回归 (Poly)", "随机森林 (RandomForest)", "支持向量机 (SVR)"])
        self.cb_model_2 = QComboBox()
        self.cb_model_2.addItems(["无 (None)", "线性回归 (Linear)", "多项式回归 (Poly)", "随机森林 (RandomForest)", "支持向量机 (SVR)"])
        l_steps = QLabel("预测步数: 10")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 50)
        self.slider.setValue(10)
        self.slider.setObjectName("StepSlider")
        self.slider.setMinimumHeight(38)
        self.apply_slider_style()
        self.slider.valueChanged.connect(lambda v: l_steps.setText(f"预测步数: {v}"))
        
        self.btn_run = QPushButton("🚀 开始分析")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setFixedHeight(48)
        self.btn_run.clicked.connect(self.run_analysis)
        
        gl_model.addWidget(QLabel("主模型 (Model 1):"))
        gl_model.addWidget(self.cb_model)
        gl_model.addWidget(QLabel("对比模型 (Model 2):"))
        gl_model.addWidget(self.cb_model_2)
        gl_model.addSpacing(10); gl_model.addWidget(l_steps)
        gl_model.addWidget(self.slider)
        gl_model.addSpacing(20); 
        gl_model.addWidget(self.btn_run)
        scroll_layout.addWidget(self.g_model)

        btn_pdf = QPushButton("📄 导出报告 (PDF)")
        btn_pdf.setMinimumHeight(40)
        btn_pdf.clicked.connect(self.export_pdf)
        btn_csv = QPushButton("💾 导出数据")
        btn_csv.setMinimumHeight(40)
        btn_csv.clicked.connect(self.export_csv)
        btn_save = QPushButton("🧠 保存模型")
        btn_save.setMinimumHeight(40)
        btn_save.clicked.connect(self.save_model)
        btn_out = QPushButton("🚪 注销")
        btn_out.setMinimumHeight(40)
        btn_out.clicked.connect(self.logout)
        
        g_btm = QGridLayout()
        g_btm.setSpacing(10)
        g_btm.addWidget(btn_pdf, 0, 0)
        g_btm.addWidget(btn_csv, 0, 1)
        g_btm.addWidget(btn_save, 1, 0, 1, 2)
        g_btm.addWidget(btn_out, 2, 0, 1, 2)
        
        scroll_layout.addSpacing(10)
        scroll_layout.addLayout(g_btm)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        side_main_layout.addWidget(scroll)
        layout.addWidget(self.side_container)
        layout.addSpacing(20) 

        # 右侧
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(0)
        area.setStyleSheet("background: transparent;")
        content = QWidget()
        self.cl = QVBoxLayout(content)
        self.cl.setContentsMargins(30,30,30,30)
        self.cl.setSpacing(25)
        
        # 头部
        hl = QHBoxLayout()
        self.btn_side_toggle = QPushButton("☰")
        self.btn_side_toggle.setObjectName("IconButton")
        self.btn_side_toggle.setFixedSize(45, 45)
        self.btn_side_toggle.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.btn_side_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_side_toggle.clicked.connect(self.toggle_sidebar)
        hl.addWidget(self.btn_side_toggle)
        hl.addSpacing(10)
        hl.addWidget(QLabel("仪表盘", objectName="AppTitle"))
        hl.addStretch()
        self.lbl_info = QLabel("等待数据...")
        self.lbl_info.setStyleSheet(f"color:{CURRENT_COLORS['OUTLINE']}; font-size:14px;")
        hl.addWidget(self.lbl_info)
        self.cl.addLayout(hl)
        
        self.ph = QLabel("👋 欢迎回来\n请从左侧导入数据开始探索")
        self.ph.setAlignment(Qt.AlignCenter)
        self.ph.setStyleSheet(f"background:{CURRENT_COLORS['SURFACE_VARIANT']}; color:{CURRENT_COLORS['TEXT']}; padding:60px; border-radius:32px; font-size:18px;")
        self.cl.addWidget(self.ph)
        
        self.grid_met = QGridLayout()
        self.grid_met.setSpacing(20)
        self.w_met = QWidget()
        self.w_met.setLayout(self.grid_met)
        self.w_met.hide()
        self.cl.addWidget(self.w_met)

        self.f_chart = QFrame()
        self.f_chart.setObjectName("Card")
        self.add_shadow(self.f_chart)
        l_c = QVBoxLayout(self.f_chart)
        l_c.setContentsMargins(25, 25, 25, 25)
        l_c.addWidget(QLabel("📊 可视化分析 (点击图表点联动表格)", objectName="CardTitle"))
        self.chart_tabs = QTabWidget()
        
        self.web_trend = QWebEngineView()
        self.web_trend.setFixedHeight(420); self.web_trend.setStyleSheet("background:transparent;")
        self.web_trend.titleChanged.connect(self.on_chart_title_changed)

        self.web_corr = QWebEngineView()
        self.web_corr.setFixedHeight(420)
        self.web_corr.setStyleSheet("background:transparent;")
        self.web_imp = QWebEngineView()
        self.web_imp.setFixedHeight(420)
        self.web_imp.setStyleSheet("background:transparent;") 
        self.web_resid = QWebEngineView()
        self.web_resid.setFixedHeight(420)
        self.web_resid.setStyleSheet("background:transparent;")
        
        t1 = QWidget(); t1l = QVBoxLayout(t1); t1l.setContentsMargins(0,15,0,0); t1l.addWidget(self.web_trend)
        t2 = QWidget(); t2l = QVBoxLayout(t2); t2l.setContentsMargins(0,15,0,0); t2l.addWidget(self.web_corr)
        t3 = QWidget(); t3l = QVBoxLayout(t3); t3l.setContentsMargins(0,15,0,0); t3l.addWidget(self.web_imp)
        t4 = QWidget(); t4l = QVBoxLayout(t4); t4l.setContentsMargins(0,15,0,0); t4l.addWidget(self.web_resid)

        self.chart_tabs.addTab(t1, "趋势与预测")
        self.chart_tabs.addTab(t2, "相关性热力图")
        self.chart_tabs.addTab(t3, "🔍 模型解释")
        self.chart_tabs.addTab(t4, "残差分析")
        l_c.addWidget(self.chart_tabs)
        self.f_chart.hide()
        self.cl.addWidget(self.f_chart)

        self.table = QTableWidget()
        self.table.setFixedHeight(350)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFrameShape(QFrame.NoFrame)
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        
        self.f_tab = QFrame()
        self.f_tab.setObjectName("Card")
        self.add_shadow(self.f_tab)
        l_t = QVBoxLayout(self.f_tab)
        l_t.setContentsMargins(25, 25, 25, 25)
        l_t.addWidget(QLabel("📑 数据明细", objectName="CardTitle"))
        l_t.addWidget(self.table)
        self.f_tab.hide()
        self.cl.addWidget(self.f_tab)
        
        self.cl.addStretch()
        
        self.log_widget_container = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget_container)
        self.log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_layout.setSpacing(15) 
        self.log_header = QPushButton("  📜  系统日志 (点击展开/收起)")
        self.log_header.setFixedHeight(30)
        self.log_header.setStyleSheet(f"text-align: left; border:none; background:{CURRENT_COLORS['SURFACE_VARIANT']}; border-radius: 5px; color:{CURRENT_COLORS['TEXT']}; font-weight: bold;")
        self.log_header.setCursor(Qt.PointingHandCursor)
        self.log_header.clicked.connect(self.toggle_log_window)
        self.log_text = QTextEdit()
        self.log_text.setObjectName("LogWindow")
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(150)
        self.log_content_frame = QFrame()
        self.log_content_layout = QVBoxLayout(self.log_content_frame)
        self.log_content_layout.setContentsMargins(0, 0, 0, 0)
        self.log_content_layout.addWidget(self.log_text)
        self.log_content_frame.setMaximumHeight(0) 
        self.log_layout.addWidget(self.log_header)
        self.log_layout.addWidget(self.log_content_frame)
        self.cl.addWidget(self.log_widget_container)
        area.setWidget(content)
        layout.addWidget(area)

        self.popup_helper = RoundedPopupHelper(self)
        comboboxes = [self.cb_clean, self.cb_target, self.cb_model, self.cb_model_2]
        for cb in comboboxes:
            view = cb.view()
            cb.setItemDelegate(QStyledItemDelegate(cb))
            view.installEventFilter(self.popup_helper)
            view.viewport().setAutoFillBackground(False)
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 40))
            shadow.setOffset(0, 4)
            view.setGraphicsEffect(shadow)

    def log(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        color_map = {"INFO": "#00AA00" if CURRENT_THEME_MODE == "Dark" else "#006400", "WARN": "#FFA500", "ERROR": "#FF0000"}
        color = color_map.get(level, CURRENT_COLORS["TEXT"])
        html = f'<span style="color:{CURRENT_COLORS["OUTLINE"]}">[{timestamp}]</span> <span style="color:{color}; font-weight:bold;">[{level}]</span> <span>{message}</span>'
        self.log_text.append(html)
        self.log_text.moveCursor(QTextCursor.End)

    def toggle_log_window(self):
        current_h = self.log_content_frame.height()
        target_h = 160 if current_h == 0 else 0
        self.anim_log = QPropertyAnimation(self.log_content_frame, b"maximumHeight")
        self.anim_log.setDuration(300)
        self.anim_log.setStartValue(current_h)
        self.anim_log.setEndValue(target_h)
        self.anim_log.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim_log.start()

    def on_theme_toggle(self):
        self.toggle_theme_cb()
        self.btn_theme.setText("🌞" if CURRENT_THEME_MODE=="Light" else "🌙")
        self.log_text.setStyleSheet(f"QTextEdit#LogWindow {{ background-color: {CURRENT_COLORS['LOG_BG']}; color: {CURRENT_COLORS['LOG_TEXT']}; border: 1px solid {CURRENT_COLORS['OUTLINE']}; border-radius: 12px; padding: 10px; }}")
        self.log_header.setStyleSheet(f"text-align: left; border:none; background:{CURRENT_COLORS['SURFACE_VARIANT']}; border-radius: 5px; color:{CURRENT_COLORS['TEXT']}; font-weight: bold;")
        self.apply_slider_style()
        if self.results and self.g_vars.isVisible(): 
            self.update_dashboard(self.cb_target.currentText())

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if path: 
            self.load_csv_by_path(path)
    
    def load_csv_by_path(self, path):
        try:
            t0 = time.time(); self.current_file_path = path; self.df_raw = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
            dt = time.time() - t0
            file_name = os.path.basename(path)
            self.lbl_info.setText(f"文件: {file_name} | 行数: {len(self.df_raw)}")
            self.log(f"成功加载文件: {file_name}", "INFO")
            self.log(f"原始数据行数: {len(self.df_raw)}, 耗时: {dt:.3f}秒", "INFO")
            self.process_data()
        except Exception as e: 
            self.log(f"加载失败: {str(e)}", "ERROR"); QMessageBox.critical(self, "Load Error", str(e))
    
    def refresh_data(self):
        if not self.current_file_path or not os.path.exists(self.current_file_path): 
            return QMessageBox.warning(self, "提示", "请先导入有效文件")
        self.log("正在刷新数据...", "INFO")
        self.load_csv_by_path(self.current_file_path)
        if self.g_vars.isVisible() and self.cb_target.currentText(): 
            self.run_analysis()
        self.lbl_info.setText(self.lbl_info.text() + " (已刷新)")
    
    def process_data(self):
        if self.df_raw is None: return
        clean_method = self.cb_clean.currentText()
        self.log(f"执行数据清洗，策略: {clean_method}", "INFO")
        self.df_clean = AnalysisEngine.clean_data(self.df_raw, clean_method)
        if self.df_clean is None: 
            self.log("数据清洗后无有效数据保留，请检查数据源。", "WARN")
            return
        rows_kept = len(self.df_clean)
        rows_lost = len(self.df_raw) - rows_kept
        self.log(f"清洗完成。保留: {rows_kept} 行, 丢弃/填充: {rows_lost} 行", "INFO")
        nums = self.df_clean.select_dtypes(include=[np.number]).columns.tolist()
        prev = self.cb_target.currentText()
        self.cb_target.blockSignals(True)
        self.cb_target.clear()
        self.cb_target.addItems(nums)
        if prev in nums: 
            self.cb_target.setCurrentText(prev)
        self.cb_target.blockSignals(False)
        self.list_feats.clear()
        for c in nums: 
            self.list_feats.addItem(QListWidgetItem(c))
        self.g_vars.show()
        self.g_model.show()
        self.ph.hide()
    
    def run_analysis(self):
        if self.df_clean is None: return
        target = self.cb_target.currentText()
        feats = [i.text() for i in self.list_feats.selectedItems() if i.text() != target]
        if not target: return QMessageBox.warning(self, "Tips", "请至少选择一个目标变量 (Y)")
        
        model_name = self.cb_model.currentText()
        model2_name = self.cb_model_2.currentText()
        steps = self.slider.value()
        
        log_msg = f"开始分析 | 目标: {target} | 主模型: {model_name}"
        if model2_name != "无 (None)": log_msg += f" | 对比模型: {model2_name}"
        self.log(log_msg, "INFO")
        
        self.btn_run.setEnabled(False); self.btn_run.setText("⏳ 分析中...")
        self.lbl_info.setText("正在训练模型，请稍候...")
        self.analysis_start_time = time.time()
        
        self.worker = AnalysisWorker(self.df_clean, target, feats, model_name, model2_name, steps)
        self.worker.result_ready.connect(self.on_analysis_finished)
        self.worker.error_occurred.connect(self.on_analysis_error)
        self.worker.finished.connect(self.on_worker_finished) 
        self.worker.start()

    def on_analysis_finished(self, res):
        self.results = res
        t_cost = time.time() - self.analysis_start_time
        self.log(f"模型训练成功。耗时: {t_cost:.4f}s | RMSE: {res['rmse']:.4f} | R2: {res['r2']:.4f}", "INFO")
        
        anom_count = len(res.get('anomaly_indices', []))
        if anom_count > 0:
            self.log(f"⚠️ 报警：检测到 {anom_count} 个异常离群点 (超过3倍标准差)", "WARN")
        else:
            self.log("未检测到显著异常点", "INFO")

        self.update_metrics(res)
        self.update_table_data(res)
        
        self.lbl_info.setText("正在渲染图表...")
        QTimer.singleShot(50, lambda: self.update_dashboard(self.cb_target.currentText()))

        if self.log_content_frame.height() == 0: 
            self.toggle_log_window()

    def on_analysis_error(self, err_msg):
        self.log(f"运行时发生异常: {err_msg}", "ERROR")
        self.lbl_info.setText("分析出错 ❌")
        QMessageBox.critical(self, "Error", err_msg)

    def on_worker_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 开始分析")

    def on_table_selection(self):
        # 表格行被点击时，高亮图表中的对应点
        if not self.results: return
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        max_hist_idx = len(self.results['y_actual'])
        valid_rows = [r for r in selected_rows if r < max_hist_idx]
        if valid_rows: 
            self.update_dashboard(self.cb_target.currentText(), highlight_indices=valid_rows)
        else: 
            self.update_dashboard(self.cb_target.currentText(), highlight_indices=None)

    def on_chart_title_changed(self, title):
        if title.startswith("CLICK_EVENT:"):
            try:
                parts = title.split(":")
                idx = int(parts[1])
                if 0 <= idx < self.table.rowCount():
                    self.table.selectRow(idx)
                    self.table.scrollToItem(self.table.item(idx, 0))
            except Exception as e:
                print(f"解析点击事件失败: {e}")

    def update_metrics(self, res):
        metric_cols = CURRENT_COLORS["METRIC_COLORS"]
        for i in reversed(range(self.grid_met.count())): 
            self.grid_met.itemAt(i).widget().setParent(None)
        cards_data = [
            ("模型类型", res['model_name'], "🤖", metric_cols[0]), 
            ("拟合优度 (R²)", f"{res['r2']:.4f}", "🎯", metric_cols[1]),
            ("均方根误差 (RMSE)", f"{res['rmse']:.4f}", "📉", metric_cols[2]), 
            ("样本数量", str(len(res['y_actual'])), "🔢", metric_cols[3])
        ]
        for i, (t, v, icon, (bg, c)) in enumerate(cards_data): 
            self.grid_met.addWidget(StatCard(t, v, icon, bg, c), 0, i)
        self.w_met.show()
    
    def update_table_data(self, res):
        self.table.blockSignals(True) 
        # 当有预测值时，显示预测步数
        if len(res['future_y']) > 0:
            rows = len(res['future_y'])
            self.table.setRowCount(rows)
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Future Step", "Forecast Value"])
            for i in range(rows): 
                # future_x 可能被作为可视化的索引，因此我们做简单处理
                val_x = str(res['future_x'][i])
                self.table.setItem(i, 0, QTableWidgetItem(val_x))
                self.table.setItem(i, 1, QTableWidgetItem(f"{res['future_y'][i]:.4f}"))
        else:
            self.table.setRowCount(min(100, len(self.df_clean)))
            self.table.setColumnCount(len(self.df_clean.columns))
            self.table.setHorizontalHeaderLabels(self.df_clean.columns)
            for i in range(self.table.rowCount()):
                for j in range(self.table.columnCount()):
                    val = self.df_clean.iloc[i, j]
                    self.table.setItem(i, j, QTableWidgetItem(f"{val:.2f}" if isinstance(val, float) else str(val)))
        self.table.blockSignals(False)
        self.f_tab.show()

    def update_plotly_chart(self, web_view, fig, add_click_listener=False):
        fig.update_layout(
            template=CURRENT_COLORS["PLOTLY_THEME"], 
            margin=dict(l=10,r=10,t=40,b=10), 
            height=380, 
            hovermode="x unified",
            hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_size=12, font_family="Segoe UI"),
            font=dict(family="Segoe UI", size=12, color="#000000"), 
            legend=dict(font=dict(color="#000000")),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)"
        )

        has_content = web_view.property("has_content")
        
        js_click_handler = ""
        if add_click_listener:
            js_click_handler = """
            var graphDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (graphDiv) {
                graphDiv.on('plotly_click', function(data){
                    if(data.points && data.points.length > 0){
                        var idx = data.points[0].pointIndex;
                        document.title = "CLICK_EVENT:" + idx + ":" + Date.now();
                    }
                });
            }
            """

        if has_content:
            data_json = json.dumps(fig.data, cls=PlotlyJSONEncoder)
            layout_json = json.dumps(fig.layout, cls=PlotlyJSONEncoder)
            js_code = f"""
            var graphDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (graphDiv) {{
                Plotly.react(graphDiv, {data_json}, {layout_json}).then(function() {{
                    {js_click_handler}
                }});
            }}
            """
            web_view.page().runJavaScript(js_code)
        else:
            html = fig.to_html(include_plotlyjs='cdn', full_html=True)
            if add_click_listener:
                html += f"<script>{js_click_handler}</script>"
            web_view.setHtml(html)
            if add_click_listener:
                QTimer.singleShot(1000, lambda: web_view.page().runJavaScript(js_click_handler))
                
            web_view.setProperty("has_content", True)

    def update_dashboard(self, target_name, highlight_indices=None):
        res = self.results
        fig = go.Figure()
        plotly_template = "plotly_dark" if CURRENT_THEME_MODE == "Dark" else "plotly_white"
        fig.update_layout(
            template=plotly_template,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=CURRENT_COLORS["ON_SURFACE"]), 
            margin=dict(l=20, r=20, t=40, b=20)
        )

        idx = np.arange(len(res['y_actual']))
        
        fig.add_trace(go.Scattergl(
            x=idx, y=res['y_actual'], name='实测值', 
            mode='lines',
            line=dict(color='#0061A4', width=2.5), 
            opacity=0.3 if highlight_indices else 1.0
        ))
        
        anom_idx = res.get('anomaly_indices', [])
        if len(anom_idx) > 0 and highlight_indices is None:
             fig.add_trace(go.Scattergl(
                x=anom_idx, 
                y=res['y_actual'][anom_idx],
                mode='markers',
                name='⚠️ 异常点 (Anomaly)',
                marker=dict(color='red', symbol='x-open', size=10, line=dict(width=2))
            ))

        if highlight_indices:
            highlight_y = res['y_actual'][highlight_indices]
            fig.add_trace(go.Scattergl(
                x=highlight_indices, y=highlight_y, 
                mode='markers', 
                name='当前选中', 
                marker=dict(color='#B00020', size=14, symbol='circle', line=dict(width=3, color='white'))
            ))

        fig.add_trace(go.Scattergl(x=idx, y=res['y_fitted'], name=f"拟合: {res['model_name']}", line=dict(color='#FFAB91', dash='dash', width=2)))
        
        if 'compare_model' in res:
            cm = res['compare_model']
            fig.add_trace(go.Scattergl(x=idx, y=cm['y_fitted'], name=f"对比: {cm['model_name']}", line=dict(color='#00695C', dash='dot', width=2)))

        if len(res['future_x']) > 0: 
            fig.add_trace(go.Scattergl(x=res['future_x'], y=res['future_y'], name=f'预测: {res["model_name"]}', mode='lines+markers', line=dict(color='#B3261E')))
            if 'compare_model' in res:
                cm = res['compare_model']
                if len(cm['future_y']) > 0:
                     fig.add_trace(go.Scattergl(x=res['future_x'], y=cm['future_y'], name=f'预测: {cm["model_name"]}', mode='lines+markers', line=dict(color='#004D40')))

        self.update_plotly_chart(self.web_trend, fig, add_click_listener=True)

        if highlight_indices is None:
            df_numeric = self.df_clean.select_dtypes(include=[np.number])
            if not df_numeric.empty and df_numeric.shape[1] > 1:
                corr = df_numeric.corr()
                fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
                self.update_plotly_chart(self.web_corr, fig_corr)
            else: 
                self.web_corr.setHtml(f"<h3 style='text-align:center; color:{CURRENT_COLORS['OUTLINE']}; margin-top:50px;'>数据列不足，无法计算相关性</h3>")
                self.web_corr.setProperty("has_content", False)
            
            imp_data = res.get('feature_importance'); feat_names = res.get('features'); imp_type = res.get('importance_type', 'unknown')
            
            if imp_data is not None and len(imp_data) == len(feat_names) and len(feat_names) > 0:
                df_imp = pd.DataFrame({'Feature': feat_names, 'Value': imp_data})
                if imp_type == 'coef':
                    df_imp['Color'] = df_imp['Value'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
                    df_imp['AbsValue'] = df_imp['Value'].abs()
                    df_imp = df_imp.sort_values(by='AbsValue', ascending=True)
                    fig_imp = px.bar(df_imp, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='RdBu', color_continuous_midpoint=0, text_auto='.3f')
                else:
                    df_imp = df_imp.sort_values(by='Value', ascending=True)
                    fig_imp = px.bar(df_imp, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='Viridis', text_auto='.3f')
                self.update_plotly_chart(self.web_imp, fig_imp)
                self.chart_tabs.setTabEnabled(2, True)
            else: 
                self.web_imp.setHtml(f"<h3 style='text-align:center; color:{CURRENT_COLORS['OUTLINE']}; margin-top:50px;'>当前模型不支持或无特征重要性数据</h3>")
                self.web_imp.setProperty("has_content", False)
            
            resid = res.get('residuals')
            if resid is not None:
                fig_res = go.Figure()
                fig_res.add_trace(go.Scattergl(x=res['y_fitted'], y=resid, mode='markers', marker=dict(color='#B00020', size=8, opacity=0.7), name='残差'))
                std_resid = np.std(resid)
                fig_res.add_hline(y=3*std_resid, line_dash="dot", line_color="red", annotation_text="Upper Limit (3σ)")
                fig_res.add_hline(y=-3*std_resid, line_dash="dot", line_color="red", annotation_text="Lower Limit (3σ)")
                fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_res.update_layout(title="残差分布图 (Residuals vs Fitted)", xaxis_title="预测值", yaxis_title="残差")
                self.update_plotly_chart(self.web_resid, fig_res)
                self.chart_tabs.setTabEnabled(3, True)

            self.f_chart.show()
            self.lbl_info.setText("分析完成 ✅")

    def save_model(self):
        if not self.results or 'model_obj' not in self.results: 
            return QMessageBox.warning(self, "Warning", "没有可保存的模型")
        fname, _ = QFileDialog.getSaveFileName(self, "Save Model", "model.pkl", "Pickle Files (*.pkl)")
        if not fname: return
        try:
            joblib.dump(self.results['model_obj'], fname)
            self.log(f"模型已保存至: {fname}", "INFO")
            QMessageBox.information(self, "Success", "模型保存成功！")
        except Exception as e: 
            self.log(f"保存失败: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))

    def export_pdf(self):
        if not self.results: 
            return QMessageBox.warning(self, "Warning", "请先运行分析")
        if not HAS_REPORT_DEPS: 
            return QMessageBox.warning(self, "Error", "缺少 matplotlib 或 fpdf 库，无法生成报表")
        
        fname, _ = QFileDialog.getSaveFileName(self, "Export Executive Report", "Report.pdf", "PDF Files (*.pdf)")
        if not fname: return
        
        self.log("正在生成报表，请稍候...", "INFO")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            paths = ReportGenerator.create_plots(self.results)
            ReportGenerator.generate_pdf(fname, self.results, paths)
            self.log(f"报表已生成: {fname}", "INFO")
            QMessageBox.information(self, "Success", "报表导出成功！")
        except Exception as e:
            self.log(f"报表生成失败: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"生成失败: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def export_csv(self):
        if not self.results: 
            return QMessageBox.warning(self, "Warning", "请先运行分析")
        fname, _ = QFileDialog.getSaveFileName(self, "Export CSV", "Prediction_Data.csv", "CSV Files (*.csv)")
        if not fname: return
        res = self.results
        try:
            df_hist = pd.DataFrame({'Type': ['Historical'] * len(res['y_actual']), 'Actual': res['y_actual'], 'Fitted': res['y_fitted'], 'Residuals': res.get('residuals', [0]*len(res['y_actual']))})
            df_future = pd.DataFrame()
            # is_time_series or X.shape[1] == 1 condition
            if len(res['future_y']) > 0:
                df_future = pd.DataFrame({'Type': ['Forecast'] * len(res['future_y']), 'Step': res['future_x'], 'Forecast': res['future_y']})
            with open(fname, 'w', newline='', encoding='utf-8-sig') as f:
                f.write(f"# Model: {res['model_name']}, RMSE: {res['rmse']:.4f}, R2: {res['r2']:.4f}\n")
                df_hist.to_csv(f, index=False)
                if not df_future.empty: 
                    f.write("\n")
                    df_future.to_csv(f, index=False)
            self.log(f"数据已导出至: {fname}", "INFO")
            QMessageBox.information(self, "Success", "导出成功！")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def logout(self):
        self.df_raw = None
        self.df_clean = None
        self.results = None
        self.current_file_path = None
        self.g_vars.hide()
        self.g_model.hide()
        self.w_met.hide()
        self.f_chart.hide()
        self.f_tab.hide()
        self.ph.show()
        self.web_trend.setProperty("has_content", False); self.web_trend.setHtml("")
        self.web_corr.setProperty("has_content", False); self.web_corr.setHtml("")
        self.web_imp.setProperty("has_content", False); self.web_imp.setHtml("")
        self.web_resid.setProperty("has_content", False); self.web_resid.setHtml("")
        self.log_text.clear()
        self.log("用户已注销。", "INFO")
        self.logout_cb()

class RoundedPopupHelper(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fixed_containers = set()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show:
            container = obj.parentWidget()
            if container and container not in self.fixed_containers:
                self.fixed_containers.add(container)
                container.setObjectName("RoundedPopupContainer")
                container.setAttribute(Qt.WA_TranslucentBackground)
                container.setAttribute(Qt.WA_NoSystemBackground)
                container.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
                container.setStyleSheet("""
                    #RoundedPopupContainer {
                        background: transparent; 
                        border: none; 
                        margin: 0px; 
                        padding: 0px;
                    }
                """)
                container.show()
                return True
        return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PredictData Pro")
        self.resize(1300, 850)
        self.setWindowIcon(QIcon("logo.jpg"))
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.login = LoginWidget(self.go_dash)
        self.dash = DashboardWidget(self.go_login, self.toggle_theme) 
        self.stack.addWidget(self.login)
        self.stack.addWidget(self.dash)
        self.apply_theme()
    
    def apply_theme(self): 
        self.setStyleSheet(get_stylesheet())
    
    def toggle_theme(self):
        global CURRENT_THEME_MODE, CURRENT_COLORS
        if CURRENT_THEME_MODE == "Light": 
            CURRENT_THEME_MODE = "Dark"
            CURRENT_COLORS = DARK_PALETTE
        else: 
            CURRENT_THEME_MODE = "Light"
            CURRENT_COLORS = LIGHT_PALETTE
        self.apply_theme()
        if hasattr(self.dash, 'last_df') and self.dash.last_df is not None:
            self.dash.update_all_charts(self.dash.last_df)
    
    def go_dash(self): 
        pixmap = self.stack.grab()
        self.stack.setCurrentWidget(self.dash)
        cover = QLabel(self)
        cover.setPixmap(pixmap)
        cover.setGeometry(self.stack.geometry())
        cover.show()
        fade_effect = QGraphicsOpacityEffect(cover)
        cover.setGraphicsEffect(fade_effect)
        self.anim = QPropertyAnimation(fade_effect, b"opacity")
        self.anim.setDuration(500)              
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.InQuad)
        self.anim.finished.connect(cover.deleteLater)
        self.anim.start()

    def go_login(self): 
        self.login.reset()
        self.stack.setCurrentWidget(self.login)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling); 
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv); 
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont("Segoe UI", 10)); 
    w = MainWindow(); 
    w.show(); 
    sys.exit(app.exec_())