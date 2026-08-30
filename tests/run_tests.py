import unittest
import os
import sys

def main():
    print("==================================================")
    print("      RUNNING DUAL-TRIGGER QTK TEST SUITE        ")
    print("==================================================")

    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("==================================================")
    if result.wasSuccessful():
        print("ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
