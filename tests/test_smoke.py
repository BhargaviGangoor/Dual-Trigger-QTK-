import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.smoke_test import run_smoke_test

class TestSmokeInvariants(unittest.TestCase):
    def test_all_five_invariants(self):
        passed = run_smoke_test(seed=42)
        self.assertTrue(passed, "Smoke test failed one or more critical invariants.")

if __name__ == "__main__":
    unittest.main()
