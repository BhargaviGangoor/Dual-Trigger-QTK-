import os
import sys

# Add backend to path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.append(backend_dir)

from backend.evaluation import results
from backend import package_deliverables

def main():
    print("==================================================")
    print("STARTING IEEE PAPER EXPERIMENT PIPELINE (ROOT)")
    print("==================================================")
    
    # Run the updated consistent evaluation pipeline
    results.main()
    
    # Package all deliverables to publication_deliverables/
    package_deliverables.main()
    
    print("\n==================================================")
    print("ROOT EXPERIMENT PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()
