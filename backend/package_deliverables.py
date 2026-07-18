import os
import shutil
from pathlib import Path

def main():
    workspace_dir = Path("c:/Users/Admin/Downloads/e2ee-trust-simulator")
    backend_dir = workspace_dir / "backend"
    
    delivery_dir = workspace_dir / "publication_deliverables"
    shutil.rmtree(delivery_dir, ignore_errors=True)
    
    # Create structure
    for sub in ["figures", "tables", "raw_logs", "simulation_logs"]:
        (delivery_dir / sub).mkdir(parents=True, exist_ok=True)
        
    print("Packaging deliverables...")
    
    # 1. Copy Figures
    figures_src = backend_dir / "results" / "figures"
    if figures_src.exists():
        for f in figures_src.iterdir():
            if f.is_file():
                shutil.copy(f, delivery_dir / "figures" / f.name)
                
    # 2. Copy Tables
    tables_src = backend_dir / "results" / "tables"
    if tables_src.exists():
        for t in tables_src.iterdir():
            if t.is_file():
                shutil.copy(t, delivery_dir / "tables" / t.name)
                
    # 3. Copy Raw Logs
    data_src = backend_dir / "data"
    if data_src.exists():
        for l in data_src.iterdir():
            if l.is_file() and l.suffix in [".json", ".csv"]:
                shutil.copy(l, delivery_dir / "raw_logs" / l.name)
                
    # 4. Copy Simulation Seed Logs
    results_src = backend_dir / "results"
    if results_src.exists():
        for s in results_src.iterdir():
            if s.is_file() and s.suffix == ".log":
                shutil.copy(s, delivery_dir / "simulation_logs" / s.name)
                
    print(f"Successfully packaged deliverables in: {delivery_dir}")

if __name__ == "__main__":
    main()
