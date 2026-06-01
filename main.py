import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# matplotlib 안전 임포트
# ─────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')  # ← 반드시 pyplot보다 먼저!
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    MATPLOTLIB_OK = True
except Exception as e:
    MATPLOTLIB_OK = False
    st.error(f"matplotlib 로드 실패: {e}")

try:
    from scipy import stats
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="서울 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# ─────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────
if MATPLOTLIB_OK:
    plt.rcParams['axes.unicode_minus'] = False
    try:
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        korean_fonts = [f for f in font_list if any(
            k in f.lower() for k in ['nanum', 'malgun', 'gothic', 'gulim', 'dotum']
        )]
        if korean_fonts:
            font_prop = fm.FontProperties(fname=korean_fonts[0])
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    except Exception:
        plt.rcParams['font.family'] = 'DejaVu Sans'

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
    df['월'] = df['날짜'].dt.month
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
# UI 시작
# ─────────────────────────────────────────
st.title("🌡️ 서울 기온 변화 분석 (1907~2026)")
st.markdown("#### 1970년대 전후로 기온이 급격히 상승했는지 확인해봅니다.")
st.markdown("---")

try:
    df = load_data()
    yearly = get_yearly(df)

    # ── 사이드바 ──────────────────────────
    st.sidebar.header("⚙️ 분석 설정")
    split_year = st.sidebar.slider(
        "기준 연도 (전/후 비교)",
        min_value=1950, max_value=2000,
        value=1970, step=1
    )
    show_trend = st.sidebar.checkbox("추세선 표시", value=True)
    show_ma = st.sidebar.checkbox("10년 이동평균선 표시", value=True)

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
        "📊 전후 비교 박스플롯",
        "🔥 10년 단위 평균",
        "📋 통계 요약"
    ])

    # ══════════════════════════════════════
    # TAB 1: 연도별 기온 변화
    # ══════════════════════════════════════
    with tab1:
        st.subheader("📈 연도별 평균기온 변화")

        if not MATPLOTLIB_OK:
            st.error("matplotlib를 불러올 수 없습니다.")
        else:
            fig, ax = plt.subplots(figsize=(14, 6))

            before_df = yearly[yearly['연도'] < split_year]
            after_df  = yearly[yearly['연도'] >= split_year]

            ax.plot(before_df['연도'], before_df['평균기온'],
                    color='steelblue', linewidth=1.2,
                    alpha=0.7, label=f'{split_year}년 이전')
            ax.plot(after_df['연도'], after_df['평균기온'],
                    color='tomato', linewidth=1.2,
                    alpha=0.7, label=f'{split_year}년 이후')

            if show_ma:
                yearly_sorted = yearly.sort_values('연도')
                ma = yearly_sorted['평균기온'].rolling(window=10, center=True).mean()
                ax.plot(yearly_sorted['연도'], ma,
                        color='black', linewidth=2.5,
                        linestyle='--', label='10년 이동평균')

            if show_trend and SCIPY_OK:
                x = yearly['연도'].values
                y = yearly['평균기온'].values
                mask = ~np.isnan(y)
                slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
                trend = slope * x + intercept
                ax.plot(x, trend, color='darkgreen',
                        linewidth=2, linestyle=':',
                        label=f'추세선 (기울기={slope*10:.3f}°C/10년)')

            ax.axvline(x=split_year, color='orange',
                       linewidth=2, linestyle='--',
                       label=f'기준: {split_year}년')
            ax.axhline(y=before, color='steelblue',
                       linewidth=1, linestyle=':', alpha=0.5)
            ax.axhline(y=after, color='tomato',
                       linewidth=1, linestyle=':', alpha=0.5)

            ax.fill_between(before_df['연도'], before_df['평균기온'],
                            before, alpha=0.08, color='steelblue')
            ax.fill_between(after_df['연도'], after_df['평균기온'],
                            after, alpha=0.08, color='tomato')

            ax.set_xlabel('연도', fontsize=12)
            ax.set_ylabel('평균기온 (degC)', fontsize=12)
            ax.set_title('서울 연도별 평균기온 변화', fontsize=15, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.info(f"""
        💡 **분석 결과**: {split_year}년을 기준으로 이전 평균 **{before:.2f}°C** →
        이후 평균 **{after:.2f}°C** 로 **{diff:.2f}°C 상승**하였습니다.
        """)

    # ══════════════════════════════════════
    # TAB 2: 박스플롯
    # ══════════════════════════════════════
    with tab2:
        st.subheader("📊 기준 연도 전후 기온 분포 비교")

        before_vals = df[df['연도'] < split_year]['평균기온'].dropna()
        after_vals  = df[df['연도'] >= split_year]['평균기온'].dropna()

        if not MATPLOTLIB_OK:
            st.error("matplotlib를 불러올 수 없습니다.")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            bp = axes[0].boxplot(
                [before_vals, after_vals],
                labels=[f'{split_year}년 이전', f'{split_year}년 이후'],
                patch_artist=True,
                notch=True
            )
            bp['boxes'][0].set_facecolor('steelblue')
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor('tomato')
            bp['boxes'][1].set_alpha(0.6)
            axes[0].set_ylabel('평균기온 (degC)', fontsize=12)
            axes[0].set_title('기온 분포 비교 (박스플롯)', fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3)

            axes[1].hist(before_vals, bins=50, alpha=0.6,
                         color='steelblue', label=f'{split_year}년 이전', density=True)
            axes[1].hist(after_vals, bins=50, alpha=0.6,
                         color='tomato', label=f'{split_year}년 이후', density=True)
            axes[1].axvline(before_vals.mean(), color='steelblue',
                            linewidth=2, linestyle='--',
                            label=f'이전 평균: {before_vals.mean():.1f}')
            axes[1].axvline(after_vals.mean(), color='tomato',
                            linewidth=2, linestyle='--',
                            label=f'이후 평균: {after_vals.mean():.1f}')
            axes[1].set_xlabel('평균기온 (degC)', fontsize=12)
            axes[1].set_ylabel('밀도', fontsize=12)
            axes[1].set_title('기온 분포 비교 (히스토그램)', fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if SCIPY_OK:
            t_stat, p_val = stats.ttest_ind(before_vals, after_vals)
            if p_val < 0.001:
                p_str = "p < 0.001 (매우 유의미한 차이! ✅)"
            else:
                p_str = f"p = {p_val:.4f}"
            st.success(f"""
            📊 **통계 검정 (독립 t-검정)**
            - t-통계량: **{t_stat:.4f}**
            - p-값: **{p_str}**
            - → 두 기간의 기온 차이는 **통계적으로 유의미**합니다.
            """)

    # ══════════════════════════════════════
    # TAB 3: 10년 단위 평균
    # ══════════════════════════════════════
    with tab3:
        st.subheader("🔥 10년 단위 평균기온")

        yearly2 = yearly.copy()
        yearly2['decade'] = (yearly2['연도'] // 10 * 10).astype(str) + '년대'
        decade_avg = yearly2.groupby('decade')['평균기온'].mean().reset_index()
        decade_avg = decade_avg.sort_values('decade')

        if not MATPLOTLIB_OK:
            st.error("matplotlib를 불러올 수 없습니다.")
        else:
            from matplotlib.patches import Patch

            fig, ax = plt.subplots(figsize=(14, 6))
            colors = ['tomato' if int(d[:4]) >= split_year else 'steelblue'
                      for d in decade_avg['decade']]

            bars = ax.bar(decade_avg['decade'], decade_avg['평균기온'],
                          color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)

            for bar, val in zip(bars, decade_avg['평균기온']):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.05,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

            legend_elements = [
                Patch(facecolor='steelblue', alpha=0.8, label=f'{split_year}년 이전'),
                Patch(facecolor='tomato', alpha=0.8, label=f'{split_year}년 이후'),
                plt.Line2D([0], [0], color='steelblue', linestyle='--',
                           label=f'이전 평균: {before:.2f}'),
                plt.Line2D([0], [0], color='tomato', linestyle='--',
                           label=f'이후 평균: {after:.2f}')
            ]

            ax.axhline(y=before, color='steelblue', linewidth=1.5,
                       linestyle='--', alpha=0.7)
            ax.axhline(y=after, color='tomato', linewidth=1.5,
                       linestyle='--', alpha=0.7)

            ax.set_xlabel('연대', fontsize=12)
            ax.set_ylabel('평균기온 (degC)', fontsize=12)
            ax.set_title('10년 단위 평균기온', fontsize=15, fontweight='bold')
            ax.legend(handles=legend_elements, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ══════════════════════════════════════
    # TAB 4: 통계 요약
    # ══════════════════════════════════════
    with tab4:
        st.subheader("📋 기간별 통계 요약")

        before_df_stats = df[df['연도'] < split_year]
        after_df_stats  = df[df['연도'] >= split_year]

        def summary(d, label):
            return {
                '기간': label,
                '데이터 수': f"{len(d):,}일",
                '평균기온 평균': f"{d['평균기온'].mean():.2f}°C",
                '평균기온 최솟값': f"{d['평균기온'].min():.1f}°C",
                '평균기온 최댓값': f"{d['평균기온'].max():.1f}°C",
                '평균기온 표준편차': f"{d['평균기온'].std():.2f}°C",
                '최고기온 평균': f"{d['최고기온'].mean():.2f}°C",
                '최저기온 평균': f"{d['최저기온'].mean():.2f}°C",
            }

        stats_df = pd.DataFrame([
            summary(before_df_stats, f"{split_year}년 이전"),
            summary(after_df_stats,  f"{split_year}년 이후")
        ])
        st.dataframe(stats_df.set_index('기간'), use_container_width=True)

        st.markdown("---")
        st.subheader("📅 연도별 데이터 테이블")
        st.dataframe(
            yearly.rename(columns={
                '평균기온': '평균기온(°C)',
                '최저기온': '최저기온(°C)',
                '최고기온': '최고기온(°C)'
            }).style.background_gradient(subset=['평균기온(°C)'], cmap='RdYlBu_r'),
            use_container_width=True,
            height=400
        )

except FileNotFoundError:
    st.error("⚠️ CSV 파일을 찾을 수 없습니다. 같은 폴더에 파일을 넣어주세요!")
except Exception as e:
    st.error(f"⚠️ 오류 발생: {e}")
    st.exception(e)
