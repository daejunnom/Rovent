#!/usr/bin/env python3
"""Compare independently produced CPU projection evidence from two builds.

Run `fixed_view_tensor_projection_evidence` with `--nocapture` in each build
and save each full stdout. The case generator replays fixed examples; this is
not a game, neural inference, model compatibility, or source-SHA attestation.
"""
import argparse
import json
import sys

MARKER = "ROVENT_PROJECTION_AUDIT_V1 "
CASES_MARKER = "ROVENT_PROJECTION_CASES_V1 "
COMPLETE_MARKER = "ROVENT_PROJECTION_COMPLETE_V1 "


def load(path):
    cases = {}
    declared = None
    completed = None
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            index = line.find(CASES_MARKER)
            if index >= 0:
                if declared is not None:
                    raise ValueError("duplicate fixture declaration")
                declared = json.loads(line[index + len(CASES_MARKER) :])
                continue
            index = line.find(COMPLETE_MARKER)
            if index >= 0:
                if completed is not None:
                    raise ValueError("duplicate fixture completion")
                completed = json.loads(line[index + len(COMPLETE_MARKER) :])
                continue
            index = line.find(MARKER)
            if index < 0:
                continue
            value = json.loads(line[index + len(MARKER) :])
            if not isinstance(value, dict) or set(value) != {"case", "view", "seals", "tensors"}:
                raise ValueError("invalid evidence fields")
            name = value["case"]
            if not isinstance(name, str) or not name or name in cases:
                raise ValueError("missing or duplicate case identity")
            cases[name] = value
    if (not isinstance(declared, list) or not declared
            or any(not isinstance(name, str) or not name for name in declared)
            or len(set(declared)) != len(declared)
            or completed != declared or list(cases) != declared):
        raise ValueError("fixture declaration, cases, or completion is missing or inconsistent")
    return cases


def first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return f"{path}: types differ"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: field sets differ"
        for key in sorted(left):
            result = first_difference(left[key], right[key], f"{path}.{key}")
            if result:
                return result
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths differ ({len(left)} vs {len(right)})"
        for index, (a, b) in enumerate(zip(left, right)):
            result = first_difference(a, b, f"{path}[{index}]")
            if result:
                return result
    elif left != right:
        return f"{path}: values differ"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_stdout")
    parser.add_argument("candidate_stdout")
    parser.add_argument("--baseline-exit-code", type=int, required=True)
    parser.add_argument("--candidate-exit-code", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.baseline_exit_code != 0 or args.candidate_exit_code != 0:
            raise ValueError("a recorded test execution did not exit successfully")
        baseline, candidate = load(args.baseline_stdout), load(args.candidate_stdout)
        if baseline.keys() != candidate.keys():
            raise ValueError("case sets differ between builds")
        rows = []
        for name in sorted(baseline):
            left, right = baseline[name], candidate[name]
            view_difference = first_difference(left["view"], right["view"], "view")
            seal_difference = first_difference(left["seals"], right["seals"], "seals")
            # A different selected view is a separate outcome. Comparing its
            # tensors would not be a fixed-view encoder equivalence test.
            tensor_difference = None if view_difference or seal_difference else first_difference(
                left["tensors"], right["tensors"], "tensors"
            )
            rows.append({
                "case": name,
                "selected_view": "different" if view_difference else "equal",
                "view_difference": view_difference,
                "seals": "different" if seal_difference else "equal",
                "seal_difference": seal_difference,
                "fixed_view_tensors": "not_comparable" if view_difference or seal_difference else (
                    "different" if tensor_difference else "equal"
                ),
                "tensor_difference": tensor_difference,
            })
        passed = all(row["fixed_view_tensors"] == "equal" for row in rows)
        print(json.dumps({"schema": "rovent-projection-comparison/1", "cases": rows,
                          "passed": passed,
                          "scope": "selected proposer examples only; no model or ORT execution",
                          "source_binary_identity": "not_checked",
                          "independent_executions": "not_attested",
                          "fresh_warm_onnx_inputs": "not_observed"},
                         ensure_ascii=False, sort_keys=True))
        return 0 if passed else 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error), "passed": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
