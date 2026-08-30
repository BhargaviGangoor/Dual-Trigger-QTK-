import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation import results

def main():
    print("==================================================")
    print("  DUAL-TRIGGER QTK REPRODUCIBILITY MASTER RUNNER  ")
    print("==================================================")
    
    # Execute full empirical experiment and evaluation suite
    results.main()
    
    print("\n==================================================")
    print(" ALL EXPERIMENTS, TABLES & FIGURES READY IN results/")
    print("==================================================")

if __name__ == "__main__":
    main()
