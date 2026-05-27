"""
Wilmar Plant Intelligence — CLI
Usage:
  python main.py --setup          # Generate data simulasi
  python main.py --demo           # Demo semua modul
  python main.py -q "..."         # Query ke Plant AI
  python main.py --module maint   # Jalankan modul spesifik
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
except ImportError:
    class console:
        @staticmethod
        def print(x, **kw): print(x)
    class Panel:
        def __new__(cls, txt, **kw): return txt

DEMO_QUERIES = [
    "Mesin mana yang akan rusak dalam 7 hari ke depan?",
    "Ada item warehouse yang perlu di-reorder segera?",
    "Tunjukkan anomali kualitas produksi 30 hari terakhir",
    "Jika CENTRIFUGE-01 rusak besok, stok spare part cukup?",
]

def cmd_setup():
    console.print("[bold]Generating simulated datasets...[/bold]")
    import subprocess
    result = subprocess.run([sys.executable, "generate_data.py"], capture_output=False)
    if result.returncode == 0:
        console.print("[green]✅ Data siap![/green]")
    else:
        console.print("[red]❌ Error saat generate data[/red]")

def cmd_query(query: str):
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", title="🏭 Plant Intelligence AI"))
    console.print("[dim]Menjalankan multi-agent analysis...[/dim]\n")
    try:
        from core.orchestrator import run
        answer = run(query)
        console.print(Panel(answer, title="📋 Executive Brief", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Pastikan: 1) python generate_data.py sudah dijalankan  2) .env berisi GROQ_API_KEY[/yellow]")

def cmd_module(module: str):
    dispatch = {
        "maint":    lambda: __import__("agents.maintenance.sensor_monitor", fromlist=["run_sensor_monitor"]).run_sensor_monitor(),
        "rul":      lambda: __import__("agents.maintenance.rul_predictor", fromlist=["run_rul_analysis"]).run_rul_analysis(),
        "schedule": lambda: __import__("agents.maintenance.schedule", fromlist=["run_schedule_agent"]).run_schedule_agent(),
        "warehouse":lambda: __import__("agents.warehouse.reorder", fromlist=["run_reorder_agent"]).run_reorder_agent(),
        "expiry":   lambda: __import__("agents.warehouse.expiry_risk", fromlist=["run_expiry_agent"]).run_expiry_agent(),
        "quality":  lambda: __import__("agents.quality.anomaly_detect", fromlist=["run_anomaly_detection"]).run_anomaly_detection(),
        "cross":    lambda: __import__("core.cross_domain", fromlist=["run_cross_domain_analysis"]).run_cross_domain_analysis(),
    }
    fn = dispatch.get(module)
    if not fn:
        console.print(f"[red]Module tidak dikenal: {module}[/red]")
        console.print(f"Pilihan: {', '.join(dispatch.keys())}")
        return
    try:
        console.print(fn())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

def cmd_demo():
    console.print("[bold green]🏭 WILMAR PLANT INTELLIGENCE — DEMO[/bold green]\n")
    for i, q in enumerate(DEMO_QUERIES, 1):
        console.print(f"[bold]Demo {i}/{len(DEMO_QUERIES)}:[/bold] {q}")
        cmd_query(q)
        print()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Wilmar Plant Intelligence CLI")
    p.add_argument("--setup",    action="store_true",  help="Generate dataset simulasi")
    p.add_argument("--demo",     action="store_true",  help="Jalankan semua demo")
    p.add_argument("--module",   type=str,             help="Jalankan modul: maint/rul/schedule/warehouse/expiry/quality/cross")
    p.add_argument("-q","--query", type=str,           help="Query ke Plant AI")
    args = p.parse_args()

    if args.setup:    cmd_setup()
    elif args.demo:   cmd_demo()
    elif args.module: cmd_module(args.module)
    elif args.query:  cmd_query(args.query)
    else:
        console.print("[bold]🏭 Wilmar Plant Intelligence[/bold]")
        console.print("Pertama kali? Jalankan: [cyan]python main.py --setup[/cyan]")
        console.print("Demo:                   [cyan]python main.py --demo[/cyan]")
        console.print("Query:                  [cyan]python main.py -q 'mesin apa yang kritis?'[/cyan]")
        console.print("Dashboard:              [cyan]streamlit run dashboard/app.py[/cyan]")
