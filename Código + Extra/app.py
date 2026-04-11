
import json, tempfile, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import wfdb
import neurokit2 as nk
import torch
import torch.nn as nn

BASE_URL  = "https://physionet.org/files/ecg-arrhythmia/1.0.0"
LEADS_12  = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
CACHE_DIR = Path(tempfile.gettempdir()) / "ecg_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN   = 1000

CLASSES = ["Sinus Bradycardia", "Sinus Rhythm",
           "Atrial Fibrillation", "Sinus Tachycardia"]

# ── Modelo CNN ─────────────────────────────────────────────────────────────────
class ECG_CNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(32),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "ecg_cnn_best.pt"
    m = ECG_CNN(n_classes=4)
    m.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    m.eval()
    return m

@st.cache_data(show_spinner=False)
def load_tree():
    path = Path("SHA256SUMS.txt")
    if not path.exists():
        urllib.request.urlretrieve(f"{BASE_URL}/SHA256SUMS.txt", path)
    tree = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            p = parts[1]
            if not p.startswith("WFDBRecords/") or not p.endswith(".hea"): continue
            segs = p.split("/")
            if len(segs) != 4: continue
            _, folder, subfolder, fname = segs
            tree[folder][subfolder].append(fname.replace(".hea",""))
    return {f: {sf: sorted(r) for sf, r in sorted(s.items())}
            for f, s in sorted(tree.items())}

@st.cache_data(show_spinner=False)
def fetch_record(folder, subfolder, record):
    record_dir = CACHE_DIR / folder / subfolder
    record_dir.mkdir(parents=True, exist_ok=True)
    rel = f"WFDBRecords/{folder}/{subfolder}/{record}"
    for ext in [".hea", ".mat"]:
        local = record_dir / f"{record}{ext}"
        if not local.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{rel}{ext}", local)
    return wfdb.rdrecord(str(record_dir / record))

@st.cache_data(show_spinner=False)
def detect_r_peaks(signal_full, fs):
    signal_clean = nk.ecg_clean(signal_full, sampling_rate=fs)
    _, info = nk.ecg_peaks(signal_clean, sampling_rate=fs)
    return info["ECG_R_Peaks"]

def classify_record(rec):
    """Extrae derivada II, normaliza y clasifica con la CNN."""
    avail = rec.sig_name
    lead  = next((l for l in ["II","I","III"] if l in avail), avail[0])
    idx   = avail.index(lead)
    sig   = rec.p_signal[:, idx].astype(np.float32)
    sig   = sig[~np.isnan(sig)]
    if len(sig) >= SEQ_LEN:
        sig = sig[:SEQ_LEN]
    else:
        sig = np.pad(sig, (0, SEQ_LEN - len(sig)))
    std = sig.std()
    if std > 0:
        sig = (sig - sig.mean()) / std
    tensor = torch.tensor(sig).unsqueeze(0).unsqueeze(0)  # (1,1,SEQ_LEN)
    model  = load_model()
    with torch.no_grad():
        logits = model(tensor)[0]
        probs  = torch.softmax(logits, dim=0).numpy()
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], probs

# ── Layout ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ECG Viewer", page_icon="🫀", layout="wide")
st.title("🫀 Visualizador de Electrocardiograma")

TREE  = load_tree()

with st.sidebar:
    st.markdown("### 📂 Selección de registro")
    folder    = st.selectbox("Carpeta", sorted(TREE.keys()))
    subfolder = st.selectbox("Subcarpeta", sorted(TREE[folder].keys()))
    record    = st.selectbox("Registro", TREE[folder][subfolder])
    load_btn  = st.button("⚡ Cargar registro", use_container_width=True)

if load_btn:
    with st.spinner("Descargando..."):
        rec = fetch_record(folder, subfolder, record)
    st.session_state.rec    = rec
    st.session_state.record = record

rec = st.session_state.get("rec", None)
if rec is None:
    st.info("👈 Selecciona un registro y pulsa Cargar.")
    st.stop()

record = st.session_state.get("record", "")

# ── Métricas básicas ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registro", record)
c2.metric("Frecuencia", f"{rec.fs} Hz")
c3.metric("Duración", f"{rec.sig_len/rec.fs:.1f} s")
c4.metric("Derivaciones", rec.n_sig)

# ── Clasificación CNN ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🧠 Clasificación CNN 1D")

pred_class, probs = classify_record(rec)

col_c1, col_c2 = st.columns([1, 2])
with col_c1:
    color = {
        "Sinus Bradycardia":  "inverse",
        "Sinus Rhythm":       "normal",
        "Atrial Fibrillation":"off",
        "Sinus Tachycardia":  "inverse",
    }.get(pred_class, "normal")

    if pred_class == "Sinus Rhythm":
        st.success(f"✅ **{pred_class}**")
    elif pred_class == "Atrial Fibrillation":
        st.error(f"🚨 **{pred_class}**")
    else:
        st.warning(f"⚠️ **{pred_class}**")

with col_c2:
    # Barra de probabilidades
    prob_df = pd.DataFrame({
        "Clase":       CLASSES,
        "Probabilidad": [round(float(p), 3) for p in probs],
    }).sort_values("Probabilidad", ascending=True)

    fig_prob = go.Figure(go.Bar(
        x=prob_df["Probabilidad"], y=prob_df["Clase"],
        orientation="h",
        marker_color=["#00f5d4" if c == pred_class else "#2a3050"
                      for c in prob_df["Clase"]],
        text=[f"{p:.1%}" for p in prob_df["Probabilidad"]],
        textposition="outside",
    ))
    fig_prob.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#c8ccd4"),
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        height=180, margin=dict(l=10, r=60, t=10, b=10),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

# ── Controles visualización ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Visualización")
    leads_sel  = st.multiselect("Derivaciones", LEADS_12, default=LEADS_12)
    zoom_start = st.number_input("Inicio (s)", 0.0, float(rec.sig_len/rec.fs), 0.0, 0.5)
    zoom_end   = st.number_input("Fin (s)", 0.0, float(rec.sig_len/rec.fs),
                                  min(10.0, rec.sig_len/rec.fs), 0.5)

if not leads_sel or zoom_end <= zoom_start:
    st.warning("Ajusta el rango o selecciona al menos una derivación.")
    st.stop()

# ── Variables comunes ──────────────────────────────────────────────────────────
fs    = rec.fs
i0    = int(zoom_start * fs)
i1    = int(zoom_end * fs)
t     = np.arange(i0, i1) / fs
sig   = rec.p_signal
avail = rec.sig_name

leads_plot = [l for l in leads_sel if l in avail]
n = len(leads_plot)

colors = ["#00f5d4","#f72585","#4cc9f0","#fee440",
          "#fb5607","#8338ec","#06d6a0","#ff006e",
          "#3a86ff","#ffbe0b","#43aa8b","#f94144"]

SMALL_T  = 0.04
LARGE_T  = 0.20
SMALL_MV = 0.10
LARGE_MV = 0.50
OFFSET   = 1.8
y_min = -LARGE_MV
y_max = (n - 1) * OFFSET + LARGE_MV

# ── Detección picos R y HR ─────────────────────────────────────────────────────
LEAD_PRIORITY = ["II", "I", "III", "aVF", "V5", "V6"]
lead_hr = next((l for l in LEAD_PRIORITY if l in avail), avail[0])
idx_hr  = avail.index(lead_hr)
r_peaks = detect_r_peaks(sig[:, idx_hr], fs)

if len(r_peaks) >= 2:
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr      = np.mean(rr_intervals)
    heart_rate   = round(60 / mean_rr, 1)
else:
    heart_rate = None
    mean_rr    = None

st.markdown("---")
st.markdown("#### 💓 Frecuencia Cardíaca")
col_hr1, col_hr2, col_hr3, col_hr4 = st.columns(4)
col_hr1.metric("Derivada usada", lead_hr)
if heart_rate is not None:
    col_hr2.metric("Frecuencia cardíaca", f"{heart_rate} lpm")
    col_hr3.metric("Intervalo RR medio", f"{mean_rr*1000:.1f} ms")
    col_hr4.metric("Picos R detectados", len(r_peaks))
    if heart_rate < 60:
        st.warning(f"⚠️ **Bradicardia** — FC {heart_rate} lpm por debajo de 60 lpm")
    elif heart_rate > 100:
        st.error(f"🚨 **Taquicardia** — FC {heart_rate} lpm por encima de 100 lpm")
    else:
        st.success(f"✅ **FC normal** — {heart_rate} lpm (rango 60–100 lpm)")
else:
    st.error("❌ No se detectaron suficientes picos R")

st.markdown("---")

# ── Grilla papel ECG ───────────────────────────────────────────────────────────
shapes = []
t_start = round(zoom_start / SMALL_T) * SMALL_T
t_ticks = np.arange(t_start, zoom_end, SMALL_T)
for tv in t_ticks:
    is_large = abs(round(tv / LARGE_T) * LARGE_T - tv) < 1e-6
    shapes.append(dict(type="line", x0=tv, x1=tv, y0=y_min, y1=y_max,
        line=dict(color="#e8a0a0" if is_large else "#f5c5c5",
                  width=0.8 if is_large else 0.3), layer="below"))

for i in range(n):
    center   = (n - 1 - i) * OFFSET
    mv_ticks = np.arange(center - OFFSET/2, center + OFFSET/2 + SMALL_MV, SMALL_MV)
    for mv in mv_ticks:
        is_large = abs(round((mv - center) / LARGE_MV) * LARGE_MV - (mv - center)) < 1e-6
        shapes.append(dict(type="line", x0=zoom_start, x1=zoom_end, y0=mv, y1=mv,
            line=dict(color="#e8a0a0" if is_large else "#f5c5c5",
                      width=0.8 if is_large else 0.3), layer="below"))

# ── Trazados ECG ───────────────────────────────────────────────────────────────
fig = go.Figure()
for i, lead in enumerate(leads_plot):
    idx   = avail.index(lead)
    y_raw = sig[i0:i1, idx]
    y_off = y_raw - np.nanmean(y_raw) + (n - 1 - i) * OFFSET
    fig.add_trace(go.Scatter(
        x=t, y=y_off, mode="lines", name=lead,
        line=dict(width=1.2, color=colors[i % len(colors)]),
        hovertemplate=f"<b>{lead}</b><br>t=%{{x:.3f}} s<br>%{{customdata:.3f}} mV<extra></extra>",
        customdata=y_raw,
    ))
    fig.add_annotation(x=zoom_start, y=(n - 1 - i) * OFFSET,
        text=f"<b>{lead}</b>", showarrow=False,
        xanchor="right", xshift=-6,
        font=dict(color=colors[i % len(colors)], size=11))

# ── Picos R ────────────────────────────────────────────────────────────────────
r_peaks_zoom = r_peaks[(r_peaks >= i0) & (r_peaks < i1)]
if lead_hr in leads_plot and len(r_peaks_zoom) > 0:
    i_lead    = leads_plot.index(lead_hr)
    y_raw_hr  = sig[i0:i1, idx_hr]
    offset_hr = (n - 1 - i_lead) * OFFSET
    r_times   = r_peaks_zoom / fs
    r_values  = sig[r_peaks_zoom, idx_hr] - np.nanmean(y_raw_hr) + offset_hr
    fig.add_trace(go.Scatter(
        x=r_times, y=r_values, mode="markers+text", name="Picos R",
        marker=dict(symbol="triangle-down", size=10, color="#ffbe0b",
                    line=dict(width=1, color="#ff8800")),
        text=["R"] * len(r_times), textposition="top center",
        textfont=dict(color="#ffbe0b", size=9),
        hovertemplate="<b>Pico R</b><br>t=%{x:.3f} s<extra></extra>",
    ))
elif lead_hr not in leads_plot:
    st.info(f"ℹ️ Derivada **{lead_hr}** no está en el plot — agrégala para ver picos R.")

# ── Calibración ────────────────────────────────────────────────────────────────
cal_x = zoom_start + LARGE_T * 0.3
cal_y = y_max - LARGE_MV * 1.5
fig.add_shape(type="line", x0=cal_x, x1=cal_x, y0=cal_y, y1=cal_y + 1.0,
              line=dict(color="#ffffff", width=2))
fig.add_annotation(x=cal_x, y=cal_y + 0.5, text="1 mV",
                   showarrow=False, xanchor="left", xshift=6,
                   font=dict(color="#ffffff", size=9))

fig.update_layout(
    paper_bgcolor="#1a0a0a", plot_bgcolor="#1a0a0a",
    font=dict(color="#c8ccd4"), shapes=shapes,
    xaxis=dict(title="Tiempo (s)",
               tickvals=np.arange(round(zoom_start/LARGE_T)*LARGE_T,
                                  zoom_end+LARGE_T, LARGE_T).tolist(),
               tickformat=".2f", showgrid=False, zeroline=False, color="#aaaaaa"),
    yaxis=dict(showgrid=False, zeroline=False, tickvals=[], range=[y_min, y_max]),
    height=max(400, 80 * n), hovermode="x unified",
    margin=dict(l=80, r=20, t=30, b=50),
    legend=dict(bgcolor="rgba(26,10,10,0.7)", bordercolor="#3a1a1a", borderwidth=1),
)
st.plotly_chart(fig, use_container_width=True)

# ── Estadísticas ───────────────────────────────────────────────────────────────
with st.expander("📊 Estadísticas de señal"):
    rows = []
    for lead in leads_plot:
        idx = avail.index(lead)
        s   = sig[i0:i1, idx]
        rows.append({"Derivación": lead,
                     "Media (mV)": f"{np.nanmean(s):.4f}",
                     "Std (mV)":   f"{np.nanstd(s):.4f}",
                     "Min (mV)":   f"{np.nanmin(s):.4f}",
                     "Max (mV)":   f"{np.nanmax(s):.4f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
