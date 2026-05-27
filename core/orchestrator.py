"""
Plant Orchestrator Agent (CEO)
================================
Terima natural language query → route ke modul yang tepat
→ jalankan agents → syntheis dengan LLM → return executive brief.

Framework: LangGraph state machine.
"""
from typing import TypedDict, List
from dotenv import load_dotenv
load_dotenv()

class PlantState(TypedDict):
    query: str
    modules_needed: List[str]
    warehouse_report: str
    maintenance_report: str
    quality_report: str
    cross_domain_report: str
    final_answer: str
    error: str

# ── Routing ──────────────────────────────────────────────────────────────────
def route_query(state: PlantState) -> PlantState:
    q = state["query"].lower()
    mods = []
    if any(w in q for w in ["stok","stock","inventory","gudang","warehouse","reorder","expiry","expired","bahan"]):
        mods.append("warehouse")
    if any(w in q for w in ["mesin","machine","rusak","breakdown","sensor","vibration","maintenance","rul","repair","service","perbaikan"]):
        mods.append("maintenance")
    if any(w in q for w in ["kualitas","quality","defect","reject","ffa","oee","produksi","yield","output","anomali"]):
        mods.append("quality")
    if any(w in q for w in ["cross","gabung","impact","dampak","spare","part","hubungan","correlation","semua","all"]):
        mods = ["warehouse","maintenance","quality","cross"]
    if not mods:
        mods = ["maintenance","quality"]   # default: operational focus
    print(f"🎯 Orchestrator → modules: {mods}")
    return {**state, "modules_needed": mods}

# ── Module runners ────────────────────────────────────────────────────────────
def run_warehouse(state: PlantState) -> PlantState:
    if "warehouse" not in state.get("modules_needed", []): return state
    try:
        from agents.warehouse.reorder   import run_reorder_agent
        from agents.warehouse.expiry_risk import run_expiry_agent
        report = run_reorder_agent() + "\n\n" + run_expiry_agent()
        return {**state, "warehouse_report": report}
    except Exception as e:
        return {**state, "warehouse_report": f"Warehouse error: {e}"}

def run_maintenance(state: PlantState) -> PlantState:
    if "maintenance" not in state.get("modules_needed", []): return state
    try:
        from agents.maintenance.sensor_monitor import run_sensor_monitor
        from agents.maintenance.rul_predictor  import run_rul_analysis
        from agents.maintenance.schedule       import run_schedule_agent
        report = run_sensor_monitor() + "\n\n" + run_rul_analysis() + "\n\n" + run_schedule_agent()
        return {**state, "maintenance_report": report}
    except Exception as e:
        return {**state, "maintenance_report": f"Maintenance error: {e}"}

def run_quality(state: PlantState) -> PlantState:
    if "quality" not in state.get("modules_needed", []): return state
    try:
        from agents.quality.anomaly_detect import run_anomaly_detection
        from agents.quality.root_cause     import run_root_cause
        from agents.quality.yield_forecast import run_yield_forecast
        report = run_anomaly_detection() + "\n\n" + run_root_cause() + "\n\n" + run_yield_forecast()
        return {**state, "quality_report": report}
    except Exception as e:
        return {**state, "quality_report": f"Quality error: {e}"}

def run_cross(state: PlantState) -> PlantState:
    if "cross" not in state.get("modules_needed", []): return state
    try:
        from core.cross_domain import run_cross_domain_analysis
        return {**state, "cross_domain_report": run_cross_domain_analysis()}
    except Exception as e:
        return {**state, "cross_domain_report": f"Cross-domain error: {e}"}

def synthesize(state: PlantState) -> PlantState:
    """LLM merangkum semua report menjadi executive brief."""
    parts = [r for r in [
        state.get("maintenance_report",""),
        state.get("warehouse_report",""),
        state.get("quality_report",""),
        state.get("cross_domain_report",""),
    ] if r]
    all_data = "\n\n".join(parts)

    prompt = f"""Kamu adalah Plant Intelligence AI advisor untuk pabrik pengolahan kelapa sawit (CPO/PKO/Olein) skala besar.

Pertanyaan dari Plant Manager: {state['query']}

Data dari sistem AI:
{all_data[:6000]}

Berikan jawaban dalam format:
1. JAWABAN LANGSUNG (2-3 kalimat)
2. TEMUAN KRITIS (bullet, max 4 poin)
3. REKOMENDASI AKSI (spesifik, dengan timeframe)
4. RISIKO JIKA TIDAK DITINDAK

Gunakan angka dan fakta dari data. Bahasa profesional tapi ringkas."""

    try:
        from core.llm_client import get_llm
        resp = get_llm("groq").invoke(prompt)
        return {**state, "final_answer": resp.content}
    except Exception as e:
        return {**state, "final_answer": f"[LLM offline — raw data below]\n\n{all_data}"}

# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(PlantState)
    g.add_node("route",       route_query)
    g.add_node("warehouse",   run_warehouse)
    g.add_node("maintenance", run_maintenance)
    g.add_node("quality",     run_quality)
    g.add_node("cross",       run_cross)
    g.add_node("synthesize",  synthesize)
    g.set_entry_point("route")
    for node in ["warehouse","maintenance","quality","cross"]:
        g.add_edge("route", node)
        g.add_edge(node, "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()

def run(query: str) -> str:
    graph = build_graph()
    initial = PlantState(query=query, modules_needed=[], warehouse_report="",
                         maintenance_report="", quality_report="",
                         cross_domain_report="", final_answer="", error="")
    result = graph.invoke(initial)
    return result["final_answer"]
