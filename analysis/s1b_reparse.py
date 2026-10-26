"""S1-B (exploratory): offline re-parse of the recorded RQ1 census of
Llama-3.1-8B and Granite-3.1-8B. Those serves returned many tool calls as text in the final answer
(Granite `<tool_call>[...]`, Llama `{"name": ..., "parameters": ...}`), so the harness never executed them.
Here every JSON object in `final` that names a function and carries arguments is appended to the trace as
if it had been issued, and the proximity level (v2.1) is recomputed. The calls never ran, so the oracle
verdict (security) and the trigger rate are unchanged; only the level of failed runs can move.
This is a cross-check for the re-served rerun (S1-A), not a replacement for it.
python3 s1b_reparse.py -> runs/px/s1b_reparse.json
"""
import collections, json, os, statistics as st, sys
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import engagefuzz as ef  # noqa: E402

MODELS = ["llama31-8b-local", "granite31-8b-local"]
DEC = json.JSONDecoder()


def calls_in_text(text):
    """Every JSON object (also inside a JSON list) with a function name and an argument dict."""
    out, i = [], 0
    while i < len(text):
        if text[i] in "{[":
            try:
                obj, j = DEC.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            for o in (obj if isinstance(obj, list) else [obj]):
                if not isinstance(o, dict):
                    continue
                f = o.get("name") or o.get("function")
                a = o.get("arguments", o.get("parameters", o.get("args")))
                if isinstance(f, dict):                     # {"function": {"name":..., "arguments":...}}
                    f, a = f.get("name"), f.get("arguments", a)
                if isinstance(f, str) and isinstance(a, dict):
                    out.append({"f": f, "a": a})
            i = j
        else:
            i += 1
    return out


def units(rows, key):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["suite"], r["ut"], r["it"], r["defense"])].append(r)
    xs, ys = [], []
    for v in cells.values():
        fa = [x for x in v if not x["security"]]
        if fa:
            xs.append(st.mean(x[key] for x in fa)); ys.append(st.mean(bool(x["security"]) for x in v))
    return xs, ys


def main():
    out = {}
    for mk in MODELS:
        R = [json.loads(l) for l in open(os.path.join(HERE, "..", "runs", "rq1", f"census_{mk}.jsonl"))]
        R = [r for r in R if r.get("err") is None and r.get("trace") is not None]
        moved, n_text = collections.Counter(), 0
        for r in R:
            r["_old"] = ef.prox_v2.feats(r)["level21"]
            extra = calls_in_text(r.get("final") or "")
            n_text += bool(extra)
            r2 = dict(r, trace=list(r["trace"]) + extra)
            r["_new"] = ef.prox_v2.feats(r2)["level21"]
            if r["_new"] != r["_old"]:
                moved[f"{r['_old']}->{r['_new']}"] += 1
        res = {"runs": len(R), "runs_with_text_calls": n_text, "level_moves": dict(moved)}
        for key in ("_old", "_new"):
            x, y = units(R, key)
            rho, p2 = spearmanr(x, y)
            res[key.strip("_")] = dict(n=len(x), rho=round(float(rho), 3),
                                       p_one_sided=float(p2 / 2 if rho > 0 else 1 - p2 / 2))
        out[mk] = res
        print(mk, json.dumps(res))
    json.dump(out, open(os.path.join(HERE, "..", "runs", "px", "s1b_reparse.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
