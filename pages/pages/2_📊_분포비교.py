import streamlit as st
import pandas as pd
import plotly.graph_objects as go

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="분포 비교", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("ta_20260601093156.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        df.columns[0]: '날짜', df.columns[1]: '지점',
        df.columns[2]: '평균기온', df.columns[3]: '최저기온', df.columns[4]: '최고기온'
    })
    df['날짜'] = pd.to_datetime(df['날짜'].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['연도'] = df['날짜'].dt.year
    for col in ['평균기온', '최저기온', '최고기온']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['평균기온'])
    return df

# ─────────────────────────────────────────
st.title("📊 기준 연도 전후 기온 분포 비교")
st.markdown("---")

try:
    df = load_data()

    st.sidebar.header("⚙️ 설정")
    split_year = st.sidebar.slider(
        "기준 연도", min_value=1950, max_value=2000, value=1970, step=1
    )

    before_vals = df[df['연도'] < split_year]['평균기온'].dropna()
    after_vals  = df[df['연도'] >= split_year]['평균기온'].dropna()

    # 지표
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{split_year}년 이전 평균",
                f"{before_vals.mean():.2f}°C")
    col2.metric(f"{split_year}년 이후 평균",
                f"{after_vals.mean():.2f}°C")
    col3.metric("상승폭",
                f"+{after_vals.mean() - before_vals.mean():.2f}°C",
                delta=f"+{after_vals.mean() - before_vals.mean():.2f}°C")

    col_a, col_b = st.columns(2)

    # 박스플롯
    with col_a:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=before_vals, name=f'{split_year}년 이전',
            marker_color='steelblue', boxmean=True, notched=True
        ))
        fig_box.add_trace(go.Box(
            y=after_vals, name=f'{split_year}년 이후',
            marker_color='tomato', boxmean=True, notched=True
        ))
        fig_box.update_layout(
            title='기온 분포 비교 (박스플롯)',
            yaxis_title='평균기온 (°C)', height=450
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # 히스토그램
    with col_b:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=before_vals, name=f'{split_year}년 이전',
            marker_color='steelblue', opacity=0.6,
            nbinsx=60, histnorm='probability density'
        ))
        fig_hist.add_trace(go.Histogram(
            x=after_vals, name=f'{split_year}년 이후',
            marker_color='tomato', opacity=0.6,
            nbinsx=60, histnorm='probability density'
        ))
        fig_hist.add_vline(
            x=float(before_vals.mean()),
            line_color='steelblue', line_dash='dash', line_width=2,
            annotation_text=f"이전: {before_vals.mean():.1f}°C"
        )
        fig_hist.add_vline(
            x=float(after_vals.mean()),
            line_color='tomato', line_dash='dash', line_width=2,
            annotation_text=f"이후: {after_vals.mean():.1f}°C"
        )
        fig_hist.update_layout(
            title='기온 분포 비교 (히스토그램)',
            xaxis_title='평균기온 (°C)',
            yaxis_title='밀도', barmode='overlay', height=450
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # t-검정
    if SCIPY_OK:
        t_stat, p_val = stats.ttest_ind(before_vals, after_vals)
        p_str = "p < 0.001 ✅" if p_val < 0.001 else f"p = {p_val:.4f}"
        st.success(f"""
        📊 **통계 검정 (독립 t-검정)**
        - t-통계량: **{t_stat:.4f}**
        - p-값: **{p_str}**
        - → 두 기간의 기온 차이는 **통계적으로 매우 유의미**합니다.
        """)

except Exception as e:
    st.error(f"오류: {e}")
    st.exception(e)
