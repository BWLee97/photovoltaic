import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ==================== 全局设置 ====================
st.set_page_config(page_title="光伏组串故障检测")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = "Cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== 诊断参数 ====================
GEN_REL = 0.10
MIN_GROUP_N = 5
POINT_V_LOW = -0.10
DAY_MIN_POINTS = 20
DAY_V_MED = -0.08
DAY_LOW_RATIO = 0.30
MIN_ABNORMAL_DAYS = 2

DUST_V_DEEP = -0.40
DUST_HALF = -0.30
GLASS_V_LO, GLASS_V_HI = -0.40, -0.08
GLASS_MIN_DAYS = 4
SHADE_MAX_DAYS = 3
SHADE_AM_NEAR, SHADE_PM_DEEP = -0.05, -0.12
I_CODROP = -0.10

# ==================== 辅助函数 ====================
def parse_time(time_str):
    if pd.isna(time_str):
        return pd.NaT
    try:
        return pd.to_datetime(time_str, format='%d/%m月/%Y %H:%M', errors='coerce')
    except:
        return pd.to_datetime(time_str, errors='coerce')

def extract_component_and_type(col_name):
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

        df_raw['时间'] = df_raw['时间'].apply(parse_time)

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

        keep_cols = ['时间'] + component_cols
        df_sub = df_raw[keep_cols].copy()
        df_sub.replace(['--', '—', ''], np.nan, inplace=True)

        for col in component_cols:
            df_sub[col] = pd.to_numeric(df_sub[col], errors='coerce')
            df_sub[col] = df_sub[col].apply(lambda x: np.nan if pd.notna(x) and x < 0 else x)

        df_long = pd.melt(
            df_sub,
            id_vars=['时间'],
            value_vars=component_cols,
            var_name='原始列名',
            value_name='数值'
        )
        df_long['组件'] = df_long['原始列名'].apply(lambda x: component_type_map[x][0])
        df_long['类型'] = df_long['原始列名'].apply(lambda x: component_type_map[x][1])

        df_long['组'] = df_long['组件'].apply(
            lambda comp: re.match(r"智能组件(2-\d+)-\d+", comp).group(1)
            if re.match(r"智能组件(2-\d+)-\d+", comp) else None
        )

        df_pivot = df_long.pivot_table(
            index=['时间', '组', '组件'],
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
    df_all.sort_values(['组', '组件', '时间'], inplace=True)
    df_all.drop_duplicates(subset=['组', '组件', '时间'], keep='first', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all

def add_peer_deviation(long):
    g = long.groupby(["组", "时间"], sort=False)
    df = long.copy()
    df["电流中位"] = g["电流(A)"].transform("median")
    df["电压中位"] = g["电压(V)"].transform("median")
    df["功率中位"] = g["功率(kW)"].transform("median")
    df["组内组件数"] = g["组件"].transform("count")

    mad_i = g["电流(A)"].transform(lambda s: (s - s.median()).abs().median())
    mad_v = g["电压(V)"].transform(lambda s: (s - s.median()).abs().median())

    df["电流偏差"] = (df["电流(A)"] - df["电流中位"]) / df["电流中位"].replace(0, np.nan)
    df["电压偏差"] = (df["电压(V)"] - df["电压中位"]) / df["电压中位"].replace(0, np.nan)
    df["功率偏差"] = (df["功率(kW)"] - df["功率中位"]) / df["功率中位"].replace(0, np.nan)

    df["电流rz"] = 0.6745 * (df["电流(A)"] - df["电流中位"]) / mad_i.replace(0, np.nan)
    df["电压rz"] = 0.6745 * (df["电压(V)"] - df["电压中位"]) / mad_v.replace(0, np.nan)

    peak = df.groupby("组")["电流中位"].transform("max")
    df["有效发电"] = (df["电流中位"] >= GEN_REL * peak) & (df["电流中位"] > 1e-9)
    df["低压点"] = df["电压偏差"] < POINT_V_LOW
    return df

def build_day_table(df):
    d = df[df["有效发电"] & (df["组内组件数"] >= MIN_GROUP_N)].copy()
    d["日期"] = d["时间"].dt.date
    d["小时"] = d["时间"].dt.hour

    def agg_day(s):
        return pd.Series({
            "点数": s["电压偏差"].size,
            "V日中位": s["电压偏差"].median(),
            "V日均值": s["电压偏差"].mean(),
            "低压点占比": s["低压点"].mean(),
            "I日中位": s["电流偏差"].median(),
            "P日中位": s["功率偏差"].median(),
            "V上午": s.loc[s["小时"].between(8, 12), "电压偏差"].mean(),
            "V下午": s.loc[s["小时"].between(13, 17), "电压偏差"].mean(),
        })
    day_tbl = d.groupby(["组", "组件", "日期"]).apply(agg_day).reset_index()
    day_tbl = day_tbl[day_tbl["点数"] >= DAY_MIN_POINTS].copy()
    day_tbl["异常日"] = (day_tbl["V日中位"] <= DAY_V_MED) | (day_tbl["低压点占比"] >= DAY_LOW_RATIO)
    return d, day_tbl

def classify(nbad, sv, si, am, pm):
    if sv <= DUST_V_DEEP and (am <= DUST_HALF) and (pm <= DUST_HALF):
        return "积灰/脏污"
    if nbad <= SHADE_MAX_DAYS and (
        (am > SHADE_AM_NEAR and pm <= SHADE_PM_DEEP) or (pm > SHADE_AM_NEAR and am <= SHADE_PM_DEEP)):
        return "遮挡"
    if nbad >= GLASS_MIN_DAYS and (GLASS_V_HI >= sv > GLASS_V_LO):
        return "玻璃破碎/硬件损伤"
    return "电压持续偏低(待复核)"

def diagnose(df, day_tbl):
    recs = []
    for (grp, comp), s in day_tbl.groupby(["组", "组件"]):
        ab = s[s["异常日"]]
        nbad = int(ab["日期"].nunique())
        ndays = int(s["日期"].nunique())
        sv = ab["V日中位"].median()
        si = ab["I日中位"].median()
        sp = ab["P日中位"].median()
        am = ab["V上午"].mean()
        pm = ab["V下午"].mean()
        flagged = nbad >= MIN_ABNORMAL_DAYS
        ftype = classify(nbad, sv, si, am, pm) if flagged else "正常"

        if flagged:
            if nbad >= 4 and sv <= DUST_V_DEEP:
                level = "红色预警"
            elif nbad >= 4 or sv <= -0.15:
                level = "橙色预警"
            else:
                level = "黄色预警"
        else:
            level = "—"

        recs.append({
            "组": grp,
            "组件": comp,
            "覆盖天数": ndays,
            "异常日数": nbad,
            "异常日电压中位": sv,
            "异常日电流中位": si,
            "异常日功率中位": sp,
            "异常日上午电压": am,
            "异常日下午电压": pm,
            "电流是否伴生下降": "是" if pd.notna(si) and si <= I_CODROP else "否",
            "是否报警": "是" if flagged else "否",
            "故障类型": ftype,
            "严重程度": level
        })
    comp_tbl = pd.DataFrame(recs).sort_values(
        ["是否报警", "异常日电压中位"],
        ascending=[False, True]
    ).reset_index(drop=True)
    return comp_tbl

def train_or_load_isoforest(day_points, cache_dir):
    model_path = os.path.join(cache_dir, "isolation_forest.pkl")
    X = day_points[["电压偏差", "电流偏差", "功率偏差"]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) < 100:
        st.warning("有效点太少，无法训练孤立森林，IF分数将为空")
        return None, pd.Series(dtype=float)

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except:
            st.warning("模型文件损坏，重新训练")
            model = None
    else:
        model = None

    if model is None:
        model = IsolationForest(contamination=0.02, random_state=42)
        model.fit(X.values)
        joblib.dump(model, model_path)

    scores = model.decision_function(X.values)
    out = pd.Series(scores, index=X.index)
    return model, out

# ==================== 登录认证 ====================
credentials = {
    'usernames': {
        'Admin': {
            'email': 'admin',
            'name': 'admin',
            'password': 'admin'
        }
    }
}
authenticator = stauth.Authenticate(credentials)
authenticator.login(
    'main',
    fields={
        'Form name': '光伏组件无监督故障检测系统',
        'Username': '用户名',
        'Password': '密码',
        'Login': '登录'
    }
)

if st.session_state['authentication_status'] is False:
    st.error("用户名或密码不正确！", icon="🚨")
    st.stop()
elif st.session_state['authentication_status'] is None:
    st.info('请输入用户名和密码！', icon="ℹ️")
    st.stop()

# ==================== 主界面（始终展示完整结构） ====================
st.title("光伏组串故障检测")

# 侧边栏用户信息
with st.sidebar:
    st.write(f"当前用户：{st.session_state['name']}")
    authenticator.logout(button_name='退出登录')

# 上传数据区域
# st.subheader("1. 上传数据")
uploaded_files = st.file_uploader(
    "请上传系统导出Excel原始文件（可多选，务必一次性上传所有数据文件）：",
    type=["xlsx"],
    accept_multiple_files=True,
    key="diagnosis_uploader"
)

# 处理上传文件，若存在则更新session_state并清除旧诊断结果
if uploaded_files:
    if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] != uploaded_files:
        st.session_state['uploaded_files'] = uploaded_files
        with st.spinner('正在处理数据...'):
            df_all = process_uploaded_files(uploaded_files)
            if df_all is not None and not df_all.empty:
                st.session_state['df_all'] = df_all
                # 清除旧诊断结果
                st.session_state.pop('comp_tbl', None)
                st.session_state.pop('day_tbl', None)
                st.session_state.pop('day_points', None)
                st.success(f"数据加载成功，共 {len(df_all)} 条记录，{df_all['组件'].nunique()} 个组件。")
            else:
                st.session_state.pop('df_all', None)
                st.error("未从上传文件中提取到有效数据，请检查文件格式。")
else:
    # 若未上传文件，清空可能存在的旧数据
    st.session_state.pop('df_all', None)
    st.session_state.pop('comp_tbl', None)
    st.session_state.pop('day_tbl', None)
    st.session_state.pop('day_points', None)
    st.info("请上传数据文件。")

# 原始数据预览区域（始终显示，无数据时提示）
# st.subheader("2. 查看原始数据")
with st.expander("点击展开原始数据表格（前1000行）", expanded=False):
    if 'df_all' in st.session_state and st.session_state['df_all'] is not None:
        st.dataframe(st.session_state['df_all'].head(1000), use_container_width=True)
    else:
        st.warning("暂无数据，请先上传文件。")

# 运行诊断按钮（无数据时禁用）
# st.subheader("3. 运行诊断")
if st.button("运行诊断", type="primary", disabled=('df_all' not in st.session_state)):
    with st.spinner("正在执行诊断，请稍候..."):
        df_all = st.session_state['df_all']
        dev = add_peer_deviation(df_all)
        day_points, day_tbl = build_day_table(dev)
        comp_tbl = diagnose(dev, day_tbl)
        model, if_scores = train_or_load_isoforest(day_points, CACHE_DIR)
        day_points["IF分数"] = np.nan
        if if_scores is not None:
            day_points.loc[if_scores.index, "IF分数"] = if_scores.values

        st.session_state['comp_tbl'] = comp_tbl
        st.session_state['day_tbl'] = day_tbl
        st.session_state['day_points'] = day_points
        st.success("诊断完成！")

if 'df_all' not in st.session_state:
    st.info("请先上传数据后再运行诊断。")

# 诊断结果区域（始终显示三个标签页，无结果时提示）
# st.subheader("4. 诊断结果")
tab1, tab2, tab3 = st.tabs(["组件级诊断", "组件×天画像", "偏差明细"])

with tab1:
    if 'comp_tbl' in st.session_state:
        comp_tbl = st.session_state['comp_tbl']
        st.dataframe(comp_tbl, use_container_width=True)
        csv = comp_tbl.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下载组件级诊断 CSV",
            data=csv,
            file_name="诊断结果_组件级.csv",
            mime="text/csv"
        )
    else:
        st.info("暂无诊断结果，请先运行诊断。")

with tab2:
    if 'day_tbl' in st.session_state:
        day_tbl = st.session_state['day_tbl']
        st.dataframe(day_tbl, use_container_width=True)
        csv = day_tbl.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下载组件×天画像 CSV",
            data=csv,
            file_name="组件x天画像.csv",
            mime="text/csv"
        )
    else:
        st.info("暂无诊断结果，请先运行诊断。")

with tab3:
    if 'day_points' in st.session_state:
        day_points = st.session_state['day_points']
        st.write(f"共 {len(day_points)} 行，显示前 5000 行")
        st.dataframe(day_points.head(5000), use_container_width=True)
        csv = day_points.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下载偏差明细 CSV（全部）",
            data=csv,
            file_name="偏差明细.csv",
            mime="text/csv"
        )
    else:
        st.info("暂无诊断结果，请先运行诊断。")
