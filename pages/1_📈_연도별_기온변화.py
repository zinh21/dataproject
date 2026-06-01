import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="연도별 기온 변화", page_icon="📈", layout="wide")

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

@st.cache_data
def get_yearly(df):
    return df.groupby('연도').agg(
        평균기온=('평균기온', 'mean'),
        최저기온=('최저기온', 'mean'),
        최고기온=('최고기온', 'mean')
    ).reset_index()

# ─────────────────────────────────────────
st.title("📈 연도별 기온 변화")
st.markdown("---")

try:
    df     = load_data()
    yearly = get_yearly(df)

    # 사이드바
    st.sidebar.header("⚙️ 설정")
    split_year = st.sidebar.slider(
        "기준 연도", min_value=1950, max_value=2000, value=1970, step=1
    )
    show_trend = st.sidebar.checkbox("추세선 표시", value=True)
    show_ma    = st.sidebar.checkbox("10년 이동평균선", value=True)
    show_minmax = st.sidebar.checkbox("최고/최저기온 표시", value=False)

    before = yearly[yearly['연도'] < split_year]['평균기온'].mean()
    after  = yearly[yearly['연도'] >= split_year]['평균기온'].mean()
    diff   = after - before

    # 지표
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{split_year}년 이전 평균", f"{before:.2f}°C")
    col2.metric(f"{split_year}년 이후 평균", f"{after:.2f}°C")
    col3.metric("상승폭", f"+{diff:.2f}°C", delta=f"+{diff:.2f}°C")

    # 그래프
    before_df = yearly[yearly['연도'] < split_year]
    after_df  = yearly[yearly['연도'] >= split_year]

    fig = go.Figure()

    # 최고/최저 영역
    if show_minmax:
        fig.add_trace(go.Scatter(
            x=yearly['연도'], y=yearly['최고기온'],
            mode='lines', name='최고기온',
            line=dict(color='orange', width=1),
            opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=yearly['연도'], y=yearly['최저기온'],
            mode='lines', name='최저기온',
            line=dict(color='lightblue', width=1),
            opacity=0.5,
            fill='tonexty', fillcolor='rgba(173,216,230,0.1)'
        ))

    # 이전/이후 라인
    fig.add_trace(go.Scatter(
        x=before_df['연도'], y=before_df['평균기온'],
        mode='lines', name=f'{split_year}년 이전',
        line=dict(color='steelblue', width=1.5), opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=after_df['연도'], y=after_df['평균기온'],
        mode='lines', name=f'{split_year}년 이후',
        line=dict(color='tomato', width=1.5), opacity=0.8
    ))

    # 이동평균
    if show_ma:
        ys = yearly.sort_values('연도')
        ma = ys['평균기온'].rolling(window=10, center=True).mean()
        fig.add_trace(go.Scatter(
            x=ys['연도'], y=ma,
            mode='lines', name='10년 이동평균',
            line=dict(color='black', width=2.5, dash='dash')
        ))

    # 추세선
    if show_trend and SCIPY_OK:
        x_v = yearly['연도'].values
        y_v = yearly['평균기온'].values
        mask = ~np.isnan(y_v)
        slope, intercept, *_ = stats.linregress(x_v[mask], y_v[mask])
        fig.add_trace(go.Scatter(
            x=x_v, y=slope * x_v + intercept,
            mode='lines',
            name=f'추세선 ({slope*10:.3f}°C/10년)',
            line=dict(color='darkgreen', width=2, dash='dot')
        ))

    # 기준선
    fig.add_vline(x=split_year, line_color='orange', line_dash='dash',
                  line_width=2,
                  annotation_text=f"기준: {split_year}년",
                  annotation_position="top right")
    fig.add_hline(y=before, line_color='steelblue', line_dash='dot',
                  line_width=1,
                  annotation_text=f"이전 평균: {before:.2f}°C",
                  annotation_position="left")
    fig.add_hline(y=after, line_color='tomato', line_dash='dot',
                  line_width=1,
                  annotation_text=f"이후 평균: {after:.2f}°C",
                  annotation_position="right")

    fig.update_layout(
        title=f'서울 연도별 평균기온 변화 (기준: {split_year}년)',
        xaxis_title='연도', yaxis_title='평균기온 (°C)',
        hovermode='x unified', height=550,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
    💡 **분석 결과**: {split_year}년을 기준으로
    이전 평균 **{before:.2f}°C** → 이후 평균 **{after:.2f}°C** 로
    **{diff:.2f}°C 상승**하였습니다.
    """)

except Exception as e:
    st.error(f"오류: {e}")
    st.exception(e)
