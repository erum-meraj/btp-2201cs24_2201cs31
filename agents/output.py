# agents/output_agent.py
"""

OutputAgent:
 - formats & prints the chosen placement and diagnostics
 - can also write JSON to disk for downstream consumption

"""

import json
from typing import Dict, Any, List, Optional

class OutputAgent:
    def __init__(self, sink_file: Optional[str] = None):
        self.sink_file = sink_file

    def present(self, scored_results: List[Dict[str, Any]], top_k: int = 1) -> Dict[str, Any]:
        """
        Prints a human-friendly summary and returns the top result dict.
        If sink_file is set, writes full scored_results as JSON to disk.
        """
        if not scored_results:
            print("No results to present.")
            return {}

        print("Top results (best first):")
        for i, r in enumerate(scored_results[:top_k]):
            print(f"#{i+1} cost={r['cost']}")
            print(" placement:", r["placement"])

        best = scored_results[0]
        summary = {
            "best_placement": best["placement"],
            "best_cost": best["cost"],
            "all": scored_results
        }

        if self.sink_file:
            try:
                with open(self.sink_file, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"Wrote results to {self.sink_file}")
            except Exception as e:
                print("Failed to write sink file:", e)

        return summary
