# Historical aggregators

These utilities preserve earlier ingestion policies for inspecting old runs.
The maintained build uses `../build_multicondition_visualizer_data.py` and the
committed grouped inputs; these scripts are not interchangeable replacements.

| Script | Historical policy |
| --- | --- |
| `aggregate_condition_a_grouped_json.py` | CSV-based GTT aggregation, success filtering, newest-file selection, and legacy model aliases |
| `aggregate_condition_b_grouped_json.py` | GTTQ CSVs with JSON fallback, parsed-answer counting, and legacy Grok aliases |

Supply your original data and write to a separate output. In particular,
Grok-2 → Grok-4.20 and GPT-5.4 Pro → GPT-5.4 mappings are historical conventions,
not evidence that those endpoints are equivalent. Inspect route metadata before
combining runs. The B utility now aligns Grok identifiers with its reference
model order instead of silently discarding those cells.

For the trial-JSON policy that generated the included snapshots, see
`../build_grouped_condition_json_from_results.py` and the
[data provenance guide](../../../data/README.md).
