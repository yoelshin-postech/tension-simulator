import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import os

# =========================================================
# 1. Page Config & CSS
# =========================================================
st.set_page_config(layout="wide", page_title="AI Tension Simulator")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 440px !important;
        }
        section[data-testid="stSidebar"] input {
            width: 100% !important;
        }
        div[data-testid="stDataFrame"] * {
            font-size: 16px !important;
        }
        div[data-testid="column"] {
            padding-left: 22px !important;
            padding-right: 22px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if platform.system() == "Darwin":
    rc("font", family="AppleGothic")
    plt.rcParams["axes.unicode_minus"] = False
elif platform.system() == "Windows":
    font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf"
    ).get_name()
    rc("font", family=font_name)
    plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cpu")
MODEL_DIR = "models"

# =========================================================
# 2. Model Definition
# =========================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attention(x), dim=1)
        return torch.sum(w * x, dim=1)


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = self.attention(out)
        return self.fc(ctx)


# =========================================================
# 3. Utils
# =========================================================
@st.cache_resource
def load_scalers(suffix):
    scaler_x = joblib.load(os.path.join(MODEL_DIR, "scaler_x.pkl"))
    scalers_y = {
        t: joblib.load(os.path.join(MODEL_DIR, f"scaler_y_{t}_{suffix}.pkl"))
        for t in ["TENSION1", "TENSION2", "TENSION3", "TENSION4"]
    }
    return scaler_x, scalers_y


@st.cache_resource
def load_model(target, suffix, input_dim):
    model = AttentionLSTM(input_dim=input_dim)
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, f"model_{target}_{suffix}.pth"),
            map_location=device
        )
    )
    model.eval()
    return model


def detect_transient_spikes_summary(df, targets, bin_size=50, threshold=100):
    logs = []
    for start in range(0, len(df), bin_size):
        end = min(start + bin_size, len(df))
        seg = df.iloc[start:end]

        detected = []
        for t in targets:
            v = seg[t].values
            if len(v) < 2:
                continue
            swing = v.max() - v.min()
            displacement = abs(v[-1] - v[0])
            if swing > threshold and displacement < swing * 0.6:
                detected.append(t)

        if detected:
            logs.append(
                f"⚠️ **{start}-{end} 구간**: 급격한 변동 감지 ({', '.join(detected)})"
            )
    return logs


def predict_logic(
    model, df, predict_idx, features, target,
    user_inputs, scaler_x, scaler_y, seq_len=15
):
    hist = df.iloc[predict_idx - seq_len:predict_idx].copy()

    for f in features:
        hist.iloc[-1, hist.columns.get_loc(f)] = user_inputs[f]

    x_scaled = scaler_x.transform(hist[features].values)
    y_scaled = scaler_y.transform(hist[[target]].values)

    seq = np.hstack([x_scaled, y_scaled])
    seq_t = torch.FloatTensor(seq).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(seq_t).numpy()

    return scaler_y.inverse_transform(pred_scaled)[0, 0]

# =========================================================
# 4. Sidebar
# =========================================================
st.sidebar.header("데이터 입력")
uploaded = st.sidebar.file_uploader(
    "시뮬레이션에 사용할 데이터 파일 (xlsx)", type="xlsx"
)
if uploaded is None:
    st.stop()

df = pd.read_excel(uploaded)
if "TIMESTAMP" in df.columns:
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

features = [
    "UNCOILER_SPEED", "UNCOILER_CURRENT", "3ROLL_SPEED",
    "SLITTER_SPEED", "RECOILER_SPEED",
    "LINE_SPEED_SET_VALUE", "LINE_SPEED_VALUE",
    "UNCOILER_WIDTH", "UNCOILER_ACTUAL_DIA", "RECOILER_ACTUAL_DIA"
]
targets = ["TENSION1", "TENSION2", "TENSION3", "TENSION4"]

st.sidebar.markdown("---")
st.sidebar.header("예측 모델 선택")
model_opt = st.sidebar.radio("예측 Horizon", ["1초 예측", "5초 예측"])
suffix = "1sec" if model_opt == "1초 예측" else "5step"
step_ahead = 1 if suffix == "1sec" else 5

st.sidebar.markdown("---")
st.sidebar.header("시점 범위 선택")
idx_min, idx_max = st.sidebar.slider(
    "Index (시간 정보를 인덱스로 환산)",
    min_value=15,
    max_value=len(df) - 1,
    value=(150, 300)
)
predict_idx = idx_max

st.sidebar.markdown("---")
st.sidebar.header("변수 제어")

base_row = df.iloc[predict_idx - 1]
user_inputs = {}
c1, c2 = st.sidebar.columns(2)

for i, f in enumerate(features):
    with (c1 if i % 2 == 0 else c2):
        user_inputs[f] = st.number_input(
            f, value=float(base_row[f]), format="%.2f"
        )

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
run_sim = st.sidebar.button("▶ 시뮬레이션 실행", type="primary")

# =========================================================
# 5. Main Result
# =========================================================
st.title("🏭 AI Tension Simulator Dashboard")

if run_sim:
    scaler_x, scalers_y = load_scalers(suffix)

    results = []
    for t in targets:
        model = load_model(t, suffix, len(features) + 1)
        pred = predict_logic(
            model, df, predict_idx, features,
            t, user_inputs, scaler_x, scalers_y[t]
        )
        actual = df.iloc[predict_idx][t]
        results.append({
            "Target": t,
            "Actual": f"{actual:.1f}",
            "Predicted": f"{pred:.1f}",
            "Diff": f"{pred - actual:+.1f}"
        })

    left_col, spacer, right_col = st.columns([1.3, 0.2, 1.7])

    with left_col:
        st.subheader("📊 시뮬레이션 결과")
        st.dataframe(
            pd.DataFrame(results),
            hide_index=True,
            width="stretch"
        )

    with right_col:
        st.subheader("⚠️ 이상 패턴 감지 로그")
        logs = detect_transient_spikes_summary(df, targets)
        if logs:
            for log in logs:
                st.markdown(log)
        else:
            st.success("✅ Spike 패턴 없음")


    # =====================================================
    # 6. Trend Visualization
    # =====================================================
    st.subheader("📈 상세 트렌드")

    # ✅ 실제 데이터는 선택 범위까지만
    hist_df = df.iloc[idx_min:idx_max + 1]

    chart_cols = st.columns(2)

    for i, r in enumerate(results):
        target = r["Target"]
        actual = float(r["Actual"])
        pred = float(r["Predicted"])

        with chart_cols[i % 2]:
            fig, ax = plt.subplots(figsize=(10, 4))

            # ---------------------------------
            # 1) 실제 Tension (회색)
            # ---------------------------------
            ax.plot(
                hist_df.index,
                hist_df[target],
                color="#333333",
                linewidth=1.2,
                label="tension"
            )

            # ---------------------------------
            # 2) 예측 Path
            # ---------------------------------
            # 예측 시작/끝 시점
            start_x = predict_idx
            end_x = predict_idx + step_ahead

            if step_ahead == 1:
                # ---- 1초 예측: 단순 연결 ----
                ax.plot(
                    [start_x, end_x],
                    [actual, pred],
                    color="#FF6B6B",
                    linestyle="-",
                    linewidth=2.2,
                    alpha=0.85,
                    label="Prediction Path"
                )
            else:
                # ---- 5초 예측: 보간된 trajectory ----
                pred_x = np.arange(start_x, end_x + 1)
                pred_y = np.linspace(actual, pred, len(pred_x))

                ax.plot(
                    pred_x,
                    pred_y,
                    color="#FF6B6B",
                    linestyle="-",
                    linewidth=2.2,
                    alpha=0.85,
                    label="Prediction Path"
                )

            # ---------------------------------
            # 3) 예측 결과 포인트
            # ---------------------------------
            ax.scatter(
                end_x,
                pred,
                color="#FF6B6B",
                s=90,
                alpha=0.9,
                edgecolors="white",
                linewidth=1.5,
                zorder=5,
                label="AI Prediction"
            )

            # ---------------------------------
            # 4) 스타일
            # ---------------------------------
            ax.set_title(target, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper left")
            ax.set_xlabel("Time Step (Index)")

            st.pyplot(fig)


else:
    st.info("👈 좌측에서 시뮬레이션 실행 버튼을 눌러주세요.")
