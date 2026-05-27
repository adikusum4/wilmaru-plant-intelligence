"""
Wilmar Plant Intelligence — Streamlit Dashboard
Deploy gratis: streamlit.io/cloud
Run lokal: streamlit run dashboard/app.py
"""
import streamlit as st, pandas as pd, plotly.graph_objects as go
import plotly.express as px, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Wilmar Plant Intelligence", page_icon="🏭", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.kpi{background:#f8f9fa;border-radius:10px;padding:1rem;border-left:4px solid}
.kpi-red{border-color:#e53935}.kpi-amber{border-color:#fb8c00}.kpi-green{border-color:#43a047}
.kpi-val{font-size:1.8rem;font-weight:700;margin:0}.kpi-lbl{font-size:.8rem;color:#666;margin:0}
</style>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏭 Wilmar Plant Intelligence")
st.markdown("**Multi-Agent AI System** — Warehouse · Predictive Maintenance · Production Quality")
st.divider()

# ── Check data ────────────────────────────────────────────────────────────────
DATA_OK = os.path.exists("data/simulated/machine_sensor_logs.csv")
if not DATA_OK:
    st.error("⚠️ Data simulasi belum ada. Jalankan dulu: `python generate_data.py`")
    st.code("python generate_data.py")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_all():
    sensors = pd.read_csv("data/simulated/machine_sensor_logs.csv", parse_dates=["timestamp"])
    inventory = pd.read_csv("data/simulated/warehouse_inventory.csv")
    quality = pd.read_csv("data/simulated/production_quality.csv")
    return sensors, inventory, quality

sensors, inventory, quality = load_all()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🤖 AI Query", "🔧 Maintenance", "📦 Warehouse", "🎯 Quality", "🔗 Cross-Domain"])

# ─── TAB 1: AI QUERY ─────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Tanya Plant Intelligence AI")
    st.caption("Ketik pertanyaan dalam Bahasa Indonesia atau English")

    examples = [
        "Mesin mana yang akan rusak dalam 7 hari ke depan?",
        "Spare part apa yang perlu di-reorder sekarang?",
        "Analisis kualitas produksi minggu ini dan temukan anomalinya",
        "Jika CENTRIFUGE-01 breakdown besok, apa dampaknya ke produksi?",
        "Buat ringkasan kesehatan pabrik hari ini",
    ]
    st.markdown("**Contoh query:**")
    ecols = st.columns(len(examples))
    chosen = None
    for i, (col, ex) in enumerate(zip(ecols, examples)):
        with col:
            if st.button(ex[:35]+"…" if len(ex)>35 else ex, key=f"ex{i}", use_container_width=True):
                chosen = ex

    query = st.text_input("Pertanyaan kamu:", value=chosen or "", placeholder="Contoh: Mesin mana yang perlu maintenance minggu ini?")

    if st.button("🚀 Analyze", type="primary") and query:
        with st.spinner("🤖 Multi-agent AI sedang menganalisis..."):
            try:
                from core.orchestrator import run as run_orchestrator
                answer = run_orchestrator(query)
                st.success("✅ Analisis selesai")
                st.markdown("### 📋 Executive Brief")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Pastikan GROQ_API_KEY sudah diset di .env (daftar gratis di console.groq.com)")

# ─── TAB 2: MAINTENANCE ──────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🔧 Machine Health Monitor")
    latest_sensors = sensors.sort_values("timestamp").groupby("machine_id").last().reset_index()

    # KPI row
    critical_count = (latest_sensors["status"].isin(["CRITICAL","FAILED"])).sum()
    warning_count  = (latest_sensors["status"] == "WARNING").sum()
    avg_health     = latest_sensors["health_score"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Mesin", len(latest_sensors))
    k2.metric("Critical/Failed", int(critical_count), delta=None)
    k3.metric("Warning", int(warning_count))
    k4.metric("Avg Health Score", f"{avg_health:.1f}%")

    # Health bar chart
    fig = px.bar(latest_sensors.sort_values("health_score"),
                 x="machine_id", y="health_score", color="status",
                 color_discrete_map={"NORMAL":"#43a047","WARNING":"#fb8c00","CRITICAL":"#e53935","FAILED":"#b71c1c"},
                 title="Health Score per Mesin", labels={"health_score":"Health Score (%)","machine_id":"Machine ID"})
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical threshold")
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Vibration time series
    st.markdown("#### Vibration Trend (semua mesin, 30 hari)")
    recent = sensors[sensors["timestamp"] >= sensors["timestamp"].max() - pd.Timedelta(days=30)]
    fig2 = px.line(recent, x="timestamp", y="vibration_mm_s", color="machine_id",
                   title="Vibration (mm/s) — nilai tinggi = potensi kerusakan")
    fig2.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ─── TAB 3: WAREHOUSE ────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 📦 Inventory Dashboard")
    today_inv = inventory[inventory["date"] == inventory["date"].max()]

    critical_stock = (today_inv["status"] == "CRITICAL").sum()
    low_stock      = (today_inv["status"] == "LOW").sum()
    total_value    = today_inv["stock_value_myr"].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("SKU Total", len(today_inv))
    k2.metric("Stock Critical", int(critical_stock))
    k3.metric("Total Stock Value", f"MYR {total_value:,.0f}")

    # Stock status
    fig = px.bar(today_inv.sort_values("stock_value_myr", ascending=False),
                 x="sku_id", y="stock_qty", color="status",
                 color_discrete_map={"OK":"#43a047","LOW":"#fb8c00","CRITICAL":"#e53935"},
                 title="Stock Qty per SKU")
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0), xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Expiry risk
    expiry_risk = today_inv[today_inv["days_to_expiry"] <= 60].sort_values("days_to_expiry")
    if len(expiry_risk):
        st.markdown("#### ⏰ Expiry Risk (≤60 hari)")
        st.dataframe(expiry_risk[["sku_id","sku_name","stock_qty","unit","days_to_expiry","status"]].reset_index(drop=True))

# ─── TAB 4: QUALITY ──────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🎯 Production Quality")
    recent_q = quality.tail(30)

    avg_oee    = recent_q["oee_pct"].mean()
    avg_reject = recent_q["reject_rate_pct"].mean()
    fail_count = (recent_q["quality_status"] == "FAIL").sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Avg OEE (30d)", f"{avg_oee:.1f}%")
    k2.metric("Avg Reject Rate", f"{avg_reject:.2f}%")
    k3.metric("Quality Fails", int(fail_count))

    fig = px.line(quality, x="date", y="oee_pct", color="production_line",
                  title="OEE % per Production Line (90 hari)")
    fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Target OEE 85%")
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(recent_q, x="ffa_pct", y="reject_rate_pct", color="production_line",
                      size="actual_output_mt", title="FFA% vs Reject Rate (30 hari terakhir)",
                      labels={"ffa_pct":"FFA (%)","reject_rate_pct":"Reject Rate (%)"})
    fig2.add_vline(x=0.20, line_dash="dash", line_color="red", annotation_text="Max FFA spec")
    fig2.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ─── TAB 5: CROSS-DOMAIN ────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🔗 Cross-Domain Intelligence")
    st.markdown("Menghubungkan kondisi mesin, stok spare part, dan dampak ke produksi secara otomatis.")
    if st.button("🔍 Jalankan Analisis Cross-Domain", type="primary"):
        with st.spinner("Menganalisis korelasi antar modul..."):
            try:
                from core.cross_domain import analyze_maintenance_warehouse_impact
                result = analyze_maintenance_warehouse_impact()
                if result["urgent_count"] == 0:
                    st.success("✅ Tidak ada mesin urgent dalam 7 hari. Operasi normal.")
                else:
                    st.warning(f"⚠️ {result['urgent_count']} mesin perlu perhatian")
                    for ins in result["insights"]:
                        with st.expander(f"🔧 {ins['machine_id']} — {ins['urgency']} ({ins['rul_hours']}h RUL)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Spare Parts:**")
                                for sp in ins["spare_parts_check"]:
                                    icon = "✅" if "OK" in sp else "❌"
                                    st.markdown(f"{icon} {sp}")
                            with col2:
                                st.markdown(f"**Est. Downtime Loss:** {ins['production_loss_est_mt']} MT")
                                st.markdown(f"**Parts Available:** {'✅ Yes' if ins['all_parts_available'] else '❌ No'}")
                            st.info(f"💡 **Rekomendasi:** {ins['recommendation']}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Correlation heatmap
    st.markdown("#### Korelasi Sensor vs Quality Parameters")
    from agents.quality.root_cause import correlate_machine_quality
    try:
        corr_data = correlate_machine_quality(sensors, quality)
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            fig = go.Figure(go.Bar(
                x=corr_df["machine_id"],
                y=corr_df["corr_vibration_to_reject"].abs(),
                marker_color=["#e53935" if v > 0.5 else "#fb8c00" if v > 0.3 else "#43a047"
                              for v in corr_df["corr_vibration_to_reject"].abs()],
                name="Vibration → Reject Rate correlation"
            ))
            fig.update_layout(title="Kekuatan Korelasi: Vibrasi Mesin → Reject Rate Produksi",
                              yaxis_title="Absolute Correlation", height=300,
                              margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
    except Exception: pass

st.divider()
st.caption("🏭 Wilmar Plant Intelligence | Portfolio Project | Streamlit + LangGraph + scikit-learn + Groq (Free) | Not affiliated with Wilmar International Ltd.")
