"""
Runs all 50 human-validated questions through the pipeline and saves results.
Usage: python -m georgia_ev_intelligence.scripts.run_eval
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from georgia_ev_intelligence.evaluation import evaluator


def main():
    results = evaluator.run_all(verbose=True)
    out_path = evaluator.save_results(results)
    evaluator.print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
