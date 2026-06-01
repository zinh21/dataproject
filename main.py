import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# scipy 안전 임포트
try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="서울 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        "ta_20260601093156.csv",
        encoding="utf-8-sig"
    )
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        df.columns[0]: '날짜',
        df.columns[1]: '지점',
        df.columns[2]: '평균기온',
        df.columns[3]: '최저기온',
        df.columns[4]: '최고기온'
    })
    df['날짜'] = df['날짜'].astype(str).str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['연도'] = df['날짜'].dt.year
    df['월']   = df['날짜'].dt.month
    for col in ['평균기온', '최저기온', '최고기온']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['평균기온'])
    return df

@st.cache_data
def get_yearly(df):
    yearly = df.groupby('연도').agg(
        평균기온=('평균기온', 'mean'),
        최저기온=('최저기온', 'mean'),
        최고기온=('최고기온', 'mean')
    ).reset_index()
    return yearly

# ─────────────────────────────────────────
# 기온 → 색상 함수 (matplotlib 없이!)
# ─────────────────────────────────────────
def temp_to_color(val, vmin, vmax):
    """파란색(낮은 기온) ~ 빨간색(높은 기온)"""
    if pd.isna(val):
        return 'background-color: white'
    ratio = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    ratio = max(0.0, min(1.0, ratio))
    # 파랑 → 흰색 → 빨강
    if ratio < 0.5:
        r = int(255 * (ratio * 2))
        g = int(255 * (ratio * 2))
        b = 255
    else:
        r = 255
        g = int(255 * (1 - (ratio - 0.5) * 2))
        b = int(255 * (1 - (ratio - 0.5) * 2))
    return f'background-color: rgb({r},{g},{b})'

def color_temp_column(series):
    vmin = series.min()
    vmax = series.max()
    return [temp_to_color(v, vmin, vmax) for v in series]

# ─────────────────────────────────────────
# 차트: plotly 사용 (matplotlib 대체!)
# ─────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ─────────────────────────────────────────
# UI 시작
# ─────────────────────────────────────────
st.title("🌡️ 서울 기온 변화 분석 (1907~2026)")
st.markdown("#### 1970년대 전후로 기온이 급격히 상승했는지 확인해봅니다.")
st.markdown("---")

try:
    df     = load_data()
    yearly = get_yearly(df)

    # ── 사이드바 ──────────────────────────
    st.sidebar.header("⚙️ 분석 설정")
    split_year = st.sidebar.slider(
        "기준 연도 (전/후 비교)",
        min_value=1950, max_value=2000,
        value=1970, step=1
    )
    show_trend = st.sidebar.checkbox("추세선 표시", value=True)
    show_ma    = st.sidebar.checkbox("10년 이동평균선 표시", value=True)

    # ── 핵심 지표 ─────────────────────────
    before = yearly[yearly['연도'] < split_year]['평균기온'].mean()
    after  = yearly[yearly['연도'] >= split_year]['평균기온'].mean()
    diff   = after - before

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{split_year}년 이전 평균기온", f"{before:.2f}°C")
    col2.metric(f"{split_year}년 이후 평균기온", f"{after:.2f}°C")
    col3.metric("기온 상승폭", f"+{diff:.2f}°C", delta=f"+{diff:.2f}°C")
    col4.metric("전체 데이터 기간",
                f"{int(yearly['연도'].min())}~{int(yearly['연도'].max())}")

    st.markdown("---")

    # ── 탭 구성 ───────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 연도별 기온 변화",
        "📊 전후 비교 분포",
        "🔥 10년 단위 평균",
        "📋 통계 요약"
    ])

    # ══════════════════════════════════════
    # TAB 1: 연도별 기온 변화 (Plotly)
    # ══════════════════════════════════════
    with tab1:
        st.subheader("📈 연도별 평균기온 변화")

        if not PLOTLY_OK:
            st.error("plotly가 설치되지 않았습니다. requirements.txt를 확인하세요.")
        else:
            fig = go.Figure()

            before_df = yearly[yearly['연도'] < split_year]
            after_df  = yearly[yearly['연도'] >= split_year]

            # 이전 구간
            fig.add_trace(go.Scatter(
                x=before_df['연도'], y=before_df['평균기온'],
                mode='lines', name=f'{split_year}년 이전',
                line=dict(color='steelblue', width=1.5),
                opacity=0.8
            ))

            # 이후 구간
            fig.add_trace(go.Scatter(
                x=after_df['연도'], y=after_df['평균기온'],
                mode='lines', name=f'{split_year}년 이후',
                line=dict(color='tomato', width=1.5),
                opacity=0.8
            ))

            # 10년 이동평균
            if show_ma:
                yearly_sorted = yearly.sort_values('연도')
                ma = yearly_sorted['평균기온'].rolling(window=10, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=yearly_sorted['연도'], y=ma,
                    mode='lines', name='10년 이동평균',
                    line=dict(color='black', width=2.5, dash='dash')
                ))

            # 추세선
            if show_trend and SCIPY_OK:
                x_vals = yearly['연도'].values
                y_vals = yearly['평균기온'].values
                mask   = ~np.isnan(y_vals)
                slope, intercept, r, p, _ = stats.linregress(
                    x_vals[mask], y_vals[mask]
                )
                trend = slope * x_vals + intercept
                fig.add_trace(go.Scatter(
                    x=x_vals, y=trend,
                    mode='lines', name=f'추세선 ({slope*10:.3f}°C/10년)',
                    line=dict(color='darkgreen', width=2, dash='dot')
                ))

            # 기준선
            fig.add_vline(
                x=split_year, line_color='orange',
                line_dash='dash', line_width=2,
                annotation_text=f"기준: {split_year}년",
                annotation_position="top right"
            )
            fig.add_hline(
                y=before, line_color='steelblue',
                line_dash='dot', line_width=1,
                annotation_text=f"이전 평균: {before:.2f}°C",
                annotation_position="left"
            )
            fig.add_hline(
                y=after, line_color='tomato',
                line_dash='dot', line_width=1,
                annotation_text=f"이후 평균: {after:.2f}°C",
                annotation_position="right"
            )

            fig.update_layout(
                title='서울 연도별 평균기온 변화',
                xaxis_title='연도',
                yaxis_title='평균기온 (°C)',
                hovermode='x unified',
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        💡 **분석 결과**: {split_year}년을 기준으로
        이전 평균 **{before:.2f}°C** → 이후 평균 **{after:.2f}°C** 로
        **{diff:.2f}°C 상승**하였습니다.
        """)

    # ══════════════════════════════════════
    # TAB 2: 분포 비교 (Plotly)
    # ══════════════════════════════════════
    with tab2:
        st.subheader("📊 기준 연도 전후 기온 분포 비교")

        before_vals = df[df['연도'] < split_year]['평균기온'].dropna()
        after_vals  = df[df['연도'] >= split_year]['평균기온'].dropna()

        if not PLOTLY_OK:
            st.error("plotly가 설치되지 않았습니다.")
        else:
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
                    yaxis_title='평균기온 (°C)',
                    height=450
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
                    annotation_text=f"이전 평균: {before_vals.mean():.1f}°C"
                )
                fig_hist.add_vline(
                    x=float(after_vals.mean()),
                    line_color='tomato', line_dash='dash', line_width=2,
                    annotation_text=f"이후 평균: {after_vals.mean():.1f}°C"
                )
                fig_hist.update_layout(
                    title='기온 분포 비교 (히스토그램)',
                    xaxis_title='평균기온 (°C)',
                    yaxis_title='밀도',
                    barmode='overlay',
                    height=450
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

    # ══════════════════════════════════════
    # TAB 3: 10년 단위 평균 (Plotly)
    # ══════════════════════════════════════
    with tab3:
        st.subheader("🔥 10년 단위 평균기온")

        yearly2 = yearly.copy()
        yearly2['decade']     = (yearly2['연도'] // 10 * 10)
        yearly2['decade_str'] = yearly2['decade'].astype(str) + '년대'
        decade_avg = (
            yearly2.groupby(['decade', 'decade_str'])['평균기온']
            .mean().reset_index().sort_values('decade')
        )

        if not PLOTLY_OK:
            st.error("plotly가 설치되지 않았습니다.")
        else:
            colors = [
                'tomato' if d >= split_year else 'steelblue'
                for d in decade_avg['decade']
            ]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=decade_avg['decade_str'],
                y=decade_avg['평균기온'],
                marker_color=colors,
                opacity=0.85,
                text=[f"{v:.1f}°C" for v in decade_avg['평균기온']],
                textposition='outside'
            ))
            fig_bar.add_hline(
                y=before, line_color='steelblue',
                line_dash='dash', line_width=1.5,
                annotation_text=f"이전 평균 {before:.2f}°C",
                annotation_position="left"
            )
            fig_bar.add_hline(
                y=after, line_color='tomato',
                line_dash='dash', line_width=1.5,
                annotation_text=f"이후 평균 {after:.2f}°C",
                annotation_position="right"
            )
            fig_bar.update_layout(
                title='10년 단위 평균기온',
                xaxis_title='연대',
                yaxis_title='평균기온 (°C)',
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 4: 통계 요약
    # ══════════════════════════════════════
    with tab4:
        st.subheader("📋 기간별 통계 요약")

        before_df_stats = df[df['연도'] < split_year]
        after_df_stats  = df[df['연도'] >= split_year]

        def summary(d, label):
            return {
                '기간':           label,
                '데이터 수':      f"{len(d):,}일",
                '평균기온 평균':  f"{d['평균기온'].mean():.2f}°C",
                '최솟값':         f"{d['평균기온'].min():.1f}°C",
                '최댓값':         f"{d['평균기온'].max():.1f}°C",
                '표준편차':       f"{d['평균기온'].std():.2f}°C",
                '최고기온 평균':  f"{d['최고기온'].mean():.2f}°C",
                '최저기온 평균':  f"{d['최저기온'].mean():.2f}°C",
            }

        stats_df = pd.DataFrame([
            summary(before_df_stats, f"{split_year}년 이전"),
            summary(after_df_stats,  f"{split_year}년 이후")
        ])
        st.dataframe(stats_df.set_index('기간'), use_container_width=True)

        st.markdown("---")
        st.subheader("📅 연도별 데이터 테이블")

        # ✅ background_gradient 제거 → 직접 색상 함수 사용
        display_df = yearly.rename(columns={
            '평균기온': '평균기온(°C)',
            '최저기온': '최저기온(°C)',
            '최고기온': '최고기온(°C)'
        })
        styled = display_df.style.apply(
            color_temp_column, subset=['평균기온(°C)']
        )
        st.dataframe(styled, use_container_width=True, height=400)

except FileNotFoundError:
    st.error("⚠️ CSV 파일을 찾을 수 없습니다. 같은 폴더에 파일을 넣어주세요!")
except Exception as e:
    st.error(f"⚠️ 오류 발생: {e}")
    st.exception(e)
