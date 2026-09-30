"""Small labelled smoke evaluation using only the C++ CLI for inference/search."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("collection")
    parser.add_argument("report", type=Path)
    parser.add_argument("--port", type=int, default=6333)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("Report already exists; choose a new output file")

    def run(*command):
        completed = subprocess.run([str(args.executable), str(args.bundle), str(args.port), args.collection, *map(str, command)], check=True, capture_output=True, encoding="utf-8")
        return json.loads(completed.stdout)

    # Use a fresh dedicated collection. No deletes or resets are performed.
    import urllib.request
    import urllib.error
    url = f"http://127.0.0.1:{args.port}/collections/{args.collection}"
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    else:
        raise SystemExit("Evaluation collection already exists; choose a new name")
    initialized = run("init")
    cases = [("bus", "reference-0.png", "a photo of a bus"), ("fruit", "reference-1.png", "a photo of fruits"), ("baboon", "reference-2.png", "a photo of a baboon")]
    for item_id, filename, _ in cases:
        run("index", item_id, args.bundle / filename)
    # Idempotent upsert of the same ID must not create another point.
    run("index", cases[0][0], args.bundle / cases[0][1])
    rows = []
    for item_id, filename, query in cases:
        for mode, value in [("text", query), ("image", args.bundle / filename)]:
            result = run(mode, value, 3)
            hits = result["results"]
            rows.append({"mode": mode, "query": str(value), "expected": item_id, "recall_at_1": int(bool(hits) and hits[0]["id"] == item_id), "results": hits})
    result = {"collection": args.collection, "space_id": initialized["space_id"], "dataset_size": len(cases), "warning": "Three local labelled examples only; image queries are exact self-retrieval. Not a generalisation benchmark.", "queries": rows}
    for mode in ("text", "image"):
        selected = [row for row in rows if row["mode"] == mode]
        result[f"{mode}_recall_at_1"] = sum(row["recall_at_1"] for row in selected) / len(selected)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "queries"}, indent=2))


if __name__ == "__main__":
    main()
