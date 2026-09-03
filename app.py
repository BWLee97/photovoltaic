import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ==================== 全局设置 ====================
st.set_page_config(page_title="光伏组串故障检测", layout="wide")
st.title("光伏组件无监督故障检测系统")

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建缓存文件夹
CACHE_DIR = "Cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ==================== 辅助函数 ====================
def parse_time(time_str):
    """解析时间字符串"""
    if pd.isna(time_str):
        return pd.NaT
    try:
        return pd.to_datetime(time_str, format='%d/%m月/%Y %H:%M', errors='coerce')
    except:
        return pd.to_datetime(time_str, errors='coerce')


def extract_component_and_type(col_name):
    """从列名提取组件名和类型"""
    patterns = {
        '电流': r'(.+)/输出电流\(A\)$',
        '电压': r'(.+)/输出电压\(V\)$',
        '功率': r'(.+)/输出功率\(kW\)$',
    }
    for type_name, pattern in patterns.items():
        match = re.match(pattern, col_name)
        if match:
            return match.group(1).strip(), type_name
    return None, None


def process_uploaded_files(files):
    """
    处理上传的多个 Excel 文件，合并为宽表 df_all
    参数 files: list of UploadedFile
    返回 df_all 或 None
    """
    df_list = []
    for file in files:
        try:
            df_raw = pd.read_excel(file)
        except Exception as e:
            st.error(f"读取文件 {file.name} 失败: {e}")
            continue

        if '时间' not in df_raw.columns:
            st.warning(f"文件 {file.name} 缺少'时间'列，已跳过")
            continue

        # 解析时间
        df_raw['时间'] = df_raw['时间'].apply(parse_time)

        # 识别组件列
        component_cols = []
        component_type_map = {}
        for col in df_raw.columns:
            if col == '时间':
                continue
            comp, type_name = extract_component_and_type(col)
            if comp:
                component_cols.append(col)
                component_type_map[col] = (comp, type_name)

        if len(component_cols) == 0:
            st.warning(f"文件 {file.name} 未识别到组件输出列，已跳过")
            continue

        # 保留时间 + 组件列
        keep_cols = ['时间'] + component_cols
        df_sub = df_raw[keep_cols].copy()

        # 清洗
        df_sub.replace('--', np.nan, inplace=True)
        df_sub.replace('—', np.nan, inplace=True)
        df_sub.replace('', np.nan, inplace=True)

        for col in component_cols:
            df_sub[col] = pd.to_numeric(df_sub[col], errors='coerce')
            # 负值视为 NaN
            df_sub[col] = df_sub[col].apply(lambda x: np.nan if pd.notna(x) and x < 0 else x)

        # 宽表转长表 -> 透视
        df_long = pd.melt(
            df_sub,
            id_vars=['时间'],
            value_vars=component_cols,
            var_name='原始列名',
            value_name='数值'
        )
        df_long['组件'] = df_long['原始列名'].apply(lambda x: component_type_map[x][0])
        df_long['类型'] = df_long['原始列名'].apply(lambda x: component_type_map[x][1])

        df_pivot = df_long.pivot_table(
            index=['时间', '组件'],
            columns='类型',
            values='数值',
            aggfunc='first'
        ).reset_index()
        df_pivot.columns.name = None
        df_pivot.rename(columns={'电流': '电流(A)', '电压': '电压(V)', '功率': '功率(kW)'}, inplace=True)

        df_list.append(df_pivot)

    if not df_list:
        return None

    df_all = pd.concat(df_list, ignore_index=True)
    df_all.sort_values(['时间', '组件'], inplace=True)
    df_all.drop_duplicates(subset=['时间', '组件'], keep='first', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all


def train_model(df_all):
    """
    使用 df_all 训练孤立森林模型
    返回模型和特征列名
    """
    df = df_all.dropna(subset=['电流(A)', '电压(V)', '功率(kW)']).copy()

    # 横向偏差特征
    min_components = 5
    time_counts = df.groupby('时间')['组件'].transform('count')
    df = df[time_counts >= min_components]

    df['电流中位数'] = df.groupby('时间')['电流(A)'].transform('median')
    df['电压中位数'] = df.groupby('时间')['电压(V)'].transform('median')

    df['电流偏差'] = (df['电流(A)'] - df['电流中位数']) / df['电流中位数'].replace(0, np.nan)
    df['电压偏差'] = (df['电压(V)'] - df['电压中位数']) / df['电压中位数'].replace(0, np.nan)

    df_model = df.dropna(subset=['电流偏差', '电压偏差'])

    features = ['电流偏差', '电压偏差']
    X = df_model[features].values

    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(X)

    return iso_forest, features


def classify_fault(avg_i_dev, avg_v_dev):
    """
    根据偏差均值进行故障分类（单点或窗口均值）
    返回故障类型字符串
    """
    # 接线盒异常：电压偏差显著为负，电流偏差接近0或轻微负
    if avg_v_dev < -0.15 and avg_i_dev > -0.15:
        return '接线盒异常'
    # 遮挡（热斑、积灰）：电流偏差显著为负，电压偏差基本不变
    elif avg_i_dev < -0.15 and abs(avg_v_dev) < 0.05:
        return '遮挡（热斑、积灰）'
    # 玻璃破碎：电流偏差更严重（<-0.3），且电压偏差也有轻微下降
    elif avg_i_dev < -0.3 and avg_v_dev < -0.05:
        return '玻璃破碎'
    else:
        return '其他'


def plot_comparison_bar(df_single):
    """
    绘制单点检测的横向对比条形图，高亮异常组串
    df_single: DataFrame，包含组件、电流(A)、电压(V)、是否异常等
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 电流对比
    colors = ['red' if flag else 'steelblue' for flag in df_single['是否异常']]
    axes[0].bar(df_single['组件'], df_single['电流(A)'], color=colors)
    axes[0].set_title('各组串电流对比')
    axes[0].set_xlabel('组件')
    axes[0].set_ylabel('电流 (A)')
    axes[0].tick_params(axis='x', rotation=45)
    # 电压对比
    axes[1].bar(df_single['组件'], df_single['电压(V)'], color=colors)
    axes[1].set_title('各组串电压对比')
    axes[1].set_xlabel('组件')
    axes[1].set_ylabel('电压 (V)')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig


# ==================== 模型训练标签页 ====================
st.subheader("1.模型训练")

# 上传文件
uploaded_files = st.file_uploader(
    "请上传系统导出Excel原始文件（务必一次性上传所有数据文件）：",
    type=["xlsx"],
    accept_multiple_files=True,
    key="train_uploader"
)

# 处理上传数据，并保存到缓存
if uploaded_files:
    with st.spinner('正在处理数据...'):
        df_all = process_uploaded_files(uploaded_files)
        if df_all is not None:
            # 保存到 Cache（不保存索引）
            df_all.to_excel(os.path.join(CACHE_DIR, "df_all.xlsx"), index=False)
            st.success(f"数据处理完成，共 {len(df_all)} 条记录，{df_all['组件'].nunique()} 个组件。")
        else:
            st.error("未从上传文件中提取到有效数据，请检查文件格式。")

# 从缓存加载数据
data = None
data_path = os.path.join(CACHE_DIR, "df_all.xlsx")
if os.path.exists(data_path):
    data = pd.read_excel(data_path)
    # 删除可能出现的 Unnamed: 0 列
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    # 确保时间列是 datetime 类型
    if '时间' in data.columns:
        data['时间'] = pd.to_datetime(data['时间'], errors='coerce')

# 展示数据
with st.expander('查看数据', expanded=True):
    if data is None:
        st.warning("尚未检索到有效数据，请上传数据。")
    else:
        st.dataframe(data, use_container_width=True)

# 训练按钮
train_button_disabled = (data is None) or (data.empty) or \
                        not {'电流(A)', '电压(V)', '功率(kW)'}.issubset(data.columns)

if st.button("训练模型", disabled=train_button_disabled):
    with st.spinner("训练中..."):
        model, features = train_model(data)
        # 保存模型
        with open(os.path.join(CACHE_DIR, "model.pkl"), 'wb') as f:
            pickle.dump({'model': model, 'features': features}, f)
        st.success("模型训练完成！")

# ==================== 模型应用标签页 ====================
st.subheader("2.模型应用")

# 检查模型是否存在，并加载模型（如果存在）
model_path = os.path.join(CACHE_DIR, "model.pkl")
model_available = os.path.exists(model_path)

if model_available:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        iso_forest = model_data['model']
        features = model_data['features']
    st.success("已检测到训练好的模型，已加载完成。")
else:
    st.warning("未检测到训练好的模型，请先完成训练模型步骤。")

# 选择输入方式（禁用状态取决于模型是否存在）
input_mode = st.selectbox(
    "选择输入方式",
    ["手动输入（单点检测）", "上传文件（窗口检测）"],
    disabled=not model_available
)

if input_mode == "手动输入（单点检测）":
    st.caption("请输入至少5个组串在同一时刻的电流(A)和电压(V)，组件名称可自定义。")
    
    # 使用 form 包裹编辑器，避免编辑时频繁重跑
    with st.form("manual_form"):
        init_data = pd.DataFrame({
            '组件': ['组件1', '组件2', '组件3', '组件4', '组件5'],
            '电流(A)': [0.0, 0.0, 0.0, 0.0, 0.0],
            '电压(V)': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        edited_df = st.data_editor(
            init_data,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_editor",
            disabled=not model_available
        )
        submitted = st.form_submit_button("检测", disabled=not model_available)
    
    if submitted:
        # 验证输入
        if len(edited_df) < 5:
            with st.expander('检测结果', expanded=True):
                st.error("至少需要5个组串才能进行横向对比。")
        elif edited_df['电流(A)'].isna().any() or edited_df['电压(V)'].isna().any():
            with st.expander('检测结果', expanded=True):
                st.error("电流和电压不能为空。")
        else:
            # 计算偏差
            median_i = edited_df['电流(A)'].median()
            median_v = edited_df['电压(V)'].median()
            if median_i == 0 or median_v == 0:
                with st.expander('检测结果', expanded=True):
                    st.error("中位数为0，无法计算偏差，请检查输入值。")
            else:
                edited_df['电流偏差'] = (edited_df['电流(A)'] - median_i) / median_i
                edited_df['电压偏差'] = (edited_df['电压(V)'] - median_v) / median_v
                
                # 预测异常
                X_input = edited_df[features].values
                pred_labels = iso_forest.predict(X_input)  # -1异常, 1正常
                edited_df['是否异常'] = pred_labels == -1
                
                # 分类故障
                fault_types = []
                for idx, row in edited_df.iterrows():
                    if row['是否异常']:
                        fault = classify_fault(row['电流偏差'], row['电压偏差'])
                        fault_types.append(fault)
                    else:
                        fault_types.append('正常')
                edited_df['故障类型'] = fault_types
                
                # 展示结果
                with st.expander('检测结果', expanded=True):
                    result_df = edited_df[['组件', '电流(A)', '电压(V)', '电流偏差', '电压偏差', '是否异常', '故障类型']]
                    st.dataframe(result_df, use_container_width=True)
                    # 可选：绘图
                    # fig = plot_comparison_bar(edited_df)
                    # st.pyplot(fig)
    else:
        with st.expander('检测结果'):
            st.info("请输入至少5个组串在同一时刻的电流(A)和电压(V)，编辑完成后点击“检测”。")

else:  # 上传文件窗口检测
    # 文件上传组件（禁用状态取决于模型是否存在）
    new_files = st.file_uploader(
        "请上传与训练数据格式相同的Excel文件：",
        type=["xlsx"],
        accept_multiple_files=True,
        key="predict_uploader",
        disabled=not model_available
    )
    
    if new_files:
        with st.spinner("处理数据..."):
            df_new = process_uploaded_files(new_files)
            if df_new is None or df_new.empty:
                st.error("未提取到有效数据，请检查文件。")
            else:
                st.success(f"数据加载成功，共 {len(df_new)} 条记录。")
                
                # 窗口检测流程
                df = df_new.dropna(subset=['电流(A)', '电压(V)', '功率(kW)']).copy()
                min_components = 5
                time_counts = df.groupby('时间')['组件'].transform('count')
                df = df[time_counts >= min_components]
                
                df['电流中位数'] = df.groupby('时间')['电流(A)'].transform('median')
                df['电压中位数'] = df.groupby('时间')['电压(V)'].transform('median')
                df['电流偏差'] = (df['电流(A)'] - df['电流中位数']) / df['电流中位数'].replace(0, np.nan)
                df['电压偏差'] = (df['电压(V)'] - df['电压中位数']) / df['电压中位数'].replace(0, np.nan)
                df = df.dropna(subset=['电流偏差', '电压偏差'])
                
                X_new = df[features].values
                df['异常标签'] = iso_forest.predict(X_new)
                
                # 聚合报警（1小时窗口）
                df = df.set_index('时间')
                window = '1H'
                alert_ratio = df.groupby('组件').resample(window)['异常标签'].apply(
                    lambda x: (x == -1).mean()
                )
                alerts = alert_ratio[alert_ratio > 0.5]
                
                # 诊断
                diagnosis_results = []
                for (comp, window_start), ratio in alerts.items():
                    window_end = window_start + pd.Timedelta(window)
                    mask = (df['组件'] == comp) & (df.index >= window_start) & (df.index < window_end)
                    sub = df[mask]
                    avg_i_dev = sub['电流偏差'].mean()
                    avg_v_dev = sub['电压偏差'].mean()
                    fault = classify_fault(avg_i_dev, avg_v_dev)
                    if fault == '其他':
                        fault = '未识别'
                    
                    # 严重程度
                    if 0.5 < ratio <= 0.7:
                        severity = '黄色预警'
                    elif 0.7 < ratio <= 0.9:
                        severity = '橙色预警'
                    else:
                        severity = '红色预警'
                    
                    diagnosis_results.append({
                        '组件': comp,
                        '时间窗口': window_start,
                        '异常比例': ratio,
                        '平均电流偏差': avg_i_dev,
                        '平均电压偏差': avg_v_dev,
                        '严重程度': severity,
                        '故障类型': fault
                    })
                
                if diagnosis_results:
                    diagnosis_df = pd.DataFrame(diagnosis_results)
                    with st.expander('检测结果'):
                        st.dataframe(diagnosis_df, use_container_width=True)
                else:
                    with st.expander('检测结果'):
                        st.info("未检测到持续异常报警。")
    else:
        with st.expander('检测结果'):
            st.info("请上传与训练数据格式相同的Excel文件")