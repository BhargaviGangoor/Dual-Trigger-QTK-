import sys
import os

# Adjust path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_trust import (
    test_fsm_transitions, 
    test_adaptive_trust_decay, 
    test_hmm_initialization, 
    test_decision_fusion,
    test_shamir_secret_sharing,
    test_device_relationship_graph,
    test_qtk_trigger
)

def run():
    print("==================================================")
    print("Running E2EE Trust Simulator Algorithm Unit Tests")
    print("==================================================")
    
    try:
        print("1. Running FSM Lifecycle Transition Checks... ", end="")
        test_fsm_transitions()
        print("PASSED")
        
        print("2. Running Adaptive Trust Decay Math Validation... ", end="")
        test_adaptive_trust_decay()
        print("PASSED")
        
        print("3. Running HMM Model Parameter Initializations... ", end="")
        test_hmm_initialization()
        print("PASSED")
        
        print("4. Running Multi-Model Decision Fusion & XAI Explainer... ", end="")
        test_decision_fusion()
        print("PASSED")
        
        print("5. Running Shamir (t,m) Secret Sharing Math Validation... ", end="")
        test_shamir_secret_sharing()
        print("PASSED")
        
        print("6. Running Decay-Weighted GCN & Graph-LSTM... ", end="")
        test_device_relationship_graph()
        print("PASSED")
        
        print("7. Running Behavior-Aware QTK Trigger Decisions... ", end="")
        test_qtk_trigger()
        print("PASSED")
        
        print("\nAll tests completed successfully! (7/7 passed)")
        print("==================================================")
        sys.exit(0)
    except AssertionError as e:
        print("FAILED")
        print(f"\nAssertion Error during validation: {e}")
        sys.exit(1)
    except Exception as e:
        print("ERROR")
        print(f"\nUnexpected error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
