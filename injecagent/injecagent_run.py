"""InjecAgent transfer run for RQ4 (pre-registered plan.md §8). Drives a served model in
InjecAgent's own ReAct prompted-agent format, reuses InjecAgent's faithful step-1 evaluator, and
applies the EngageFuzz proximity ladder to each run.

python3 injecagent_run.py --model llama31-8b-local --base-url http://127.0.0.1:8023/v1\
    --n-cases 100 --reps 10 --out ../runs/rq4/injecagent_llama.jsonl [--cap 16] [--temp 1.0]
API models use --base-url $LLM_GATEWAY_URL/v1 (you set LLM_GATEWAY_URL) and --api-key-env LLM_GATEWAY_KEY.
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import openai

os.environ.setdefault("OPENAI_API_KEY", "none")  # src/utils builds a client at import
IA = os.environ.get("INJECAGENT_DIR", "InjecAgent")  # checkout of https://github.com/uiuc-kang-lab/InjecAgent
sys.path.insert(0, IA)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from src.prompts.agent_prompts import PROMPT_DICT  # noqa: E402
from src.utils import get_tool_dict  # noqa: E402
from src.output_parsing import evaluate_output_prompted, output_parser  # noqa: E402
import injecagent_prox as prox  # noqa: E402

SYS, USER = PROMPT_DICT["hwchase17_react"]


def build_prompt(item, tool_dict):
    names = [item["User Tool"]] + item["Attacker Tools"]
    descs = json.dumps([tool_dict[t] for t in names], indent=True)
    scratch = (f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}"
               f"\nObservation: {item['Tool Response']}\n")
    return USER.format(available_tool_descriptions=descs, input=item["User Instruction"],
                       available_tool_names=names, agent_scratchpad=scratch)


def parse_trace(output_text):
    """Trace of every tool action in the ReAct output (f, a), via InjecAgent's own parser.
    This lets the ladder score levels 2 and 3, not just success (level 4)."""
    tr = []
    try:
        info = output_parser(output_text)
    except Exception:
        return tr
    i = 0
    while i < len(info):
        k, v = info[i]
        if k == "Action":
            args = {}
            if i + 1 < len(info) and info[i + 1][0] == "Action Input":
                raw = info[i + 1][1]
                try:
                    args = json.loads(raw)
                except Exception:
                    args = {"_raw": raw}
                if not isinstance(args, dict):
                    args = {"_raw": str(args)}
            tr.append({"f": v.strip(), "a": args})
        i += 1
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--n-cases", type=int, default=100)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.chdir(IA)  # get_tool_dict reads data/tools.json relative to cwd
    key = os.environ[a.api_key_env] if a.api_key_env else "none"
    client = openai.OpenAI(base_url=a.base_url, api_key=key, timeout=180, max_retries=3)
    tool_dict = get_tool_dict()
    cases = []
    for fn in ("test_cases_ds_base.json", "test_cases_ds_enhanced.json"):  # 1088 data-stealing (plan §8)
        cases += json.load(open(os.path.join(IA, "data", fn)))
    # pre-registered: cases with no extractable attacker-unique value are excluded and counted
    usable = [c for c in cases if prox.attacker_values(c)]
    excluded = len(cases) - len(usable)
    random.Random(0).shuffle(usable)
    sample = usable[:a.n_cases]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if r.get("err") is None:
                done.add((r["idx"], r["rep"]))
    jobs = [(i, c, k) for i, c in enumerate(sample) for k in range(a.reps) if (i, k) not in done]
    print(f"ds cases usable={len(usable)} excluded_no_value={excluded} sample={len(sample)} jobs={len(jobs)}", flush=True)
    lk = threading.Lock()

    def go(job):
        i, c, k = job
        rec = dict(idx=i, rep=k, model=a.model, suite="injecagent_ds",
                   ut=c["User Tool"], it=c["Attacker Tools"][0], err=None)
        try:
            out = client.chat.completions.create(model=a.model, temperature=a.temp,
                                                 messages=[{"role": "system", "content": SYS},
                                                           {"role": "user", "content": build_prompt(c, tool_dict)}])
            txt = out.choices[0].message.content or ""
            parsed = evaluate_output_prompted(txt, c["Attacker Tools"][0], c["Attacker Tools"],
                                              c["User Tool"], c["Attacker Instruction"])
            succ = parsed.get("eval") == "succ"
            tr = parse_trace(txt)
            rec.update(security=succ, eval=parsed.get("eval"), trace=tr,
                       level_ia=prox.level(c, tr, succ), final=txt)
        except Exception as e:
            rec["err"] = repr(e)[:200]
        with lk, open(a.out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    with ThreadPoolExecutor(a.cap) as ex:
        list(ex.map(go, jobs))
    print("done", flush=True)


if __name__ == "__main__":
    main()
