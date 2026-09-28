"""Replay a captured Junie session against a running Splash server.

Port of mlx-vlm's ``research/junie-replay/replay.py``; prints the same
per-request lines and final mean line so results compare directly. Posts
every request body in ``requests/`` (in filename order, ``warm_up.json``
first and not counted) to ``/v1/chat/completions`` and reports:

  - KV cached: cached prompt tokens / total prompt tokens
  - prefill speed over the NEW (uncached) tokens
  - generation speed
  - DFlash acceptance (accepted / drafted, tokens/round)

Splash differs from mlx-vlm in where the numbers live. ``timings.prompt_n``
is the whole prompt, so new tokens come from ``metrics.prefill.tokens``.
Draft counts are not per-response; they are the delta of the cumulative
``/status`` counters around each request, which is exact because requests
run one at a time (do not replay while other clients use the server). A
round is one decode batch. Splash has no n-gram lookup, so that column is
absent. The captured ``stop`` is dropped from requests with tools, which
Splash does not accept together.

Usage (server must be running, e.g. via ``splash serve`` or start_splash.sh):
  python dev/benchmarks/junie_replay/replay.py [--url http://127.0.0.1:8000]
      [--model MODEL] [--reasoning-effort low] [--requests DIR]

Without --model, the served model from /v1/models replaces the captured one,
since Splash rejects unknown model names. Non-'none' --reasoning-effort also
sets enable_thinking=true. A server with an API key needs it in
SPLASH_API_KEY.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def request(url, path, body=None, timeout=3600):
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("SPLASH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{url}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code} from server: {exc.read().decode()}")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach {url}: {exc.reason}")


def served_model(url):
    models = request(url, "/v1/models", timeout=10).get("data") or []
    if not models:
        sys.exit("server lists no models")
    return models[0]["id"]


def draft_counters(url):
    """Cumulative (drafted, accepted, rounds) from /status."""
    status = request(url, "/status", timeout=30)
    metrics = status.get("metrics") or {}
    scheduler = status.get("scheduler") or {}
    return (
        int(metrics.get("drafted_tokens") or 0),
        int(metrics.get("accepted_draft_tokens") or 0),
        int(scheduler.get("decode_batches") or 0),
    )


def post(url, path, model, reasoning_effort=None):
    with open(path) as f:
        body = json.load(f)
    body["stream"] = False
    body["model"] = model
    # Junie sends a legacy "</COMMAND>" stop next to native tools; Splash
    # rejects stop with tools, and tool calls end generation on their own.
    if body.get("tools"):
        body.pop("stop", None)
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
        if reasoning_effort != "none":
            body["enable_thinking"] = True
    before = draft_counters(url)
    response = request(url, "/v1/chat/completions", body)
    after = draft_counters(url)
    response["draft"] = tuple(b - a for a, b in zip(before, after))
    return response


def prefill_stats(response):
    """(prompt tokens, cached tokens, new tokens, prefill ms) of a response."""
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    metrics = response.get("metrics") or {}
    prompt = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    cached = int(timings.get("cache_n") or 0)
    new = (metrics.get("prefill") or {}).get("tokens")
    new = int(new) if new is not None else max(0, prompt - cached)
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    return prompt, cached, new, prompt_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--model",
        default=None,
        help="Model name for all requests (default: the served model)",
    )
    ap.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "xhigh"],
        help="Set reasoning_effort in all requests. Non-'none' values also enable thinking.",
    )
    ap.add_argument(
        "--requests",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests"),
    )
    args = ap.parse_args()
    url = args.url.rstrip("/")
    model = args.model or served_model(url)

    files = sorted(
        os.path.join(args.requests, f)
        for f in os.listdir(args.requests)
        if f.endswith(".json")
    )
    if not files:
        sys.exit(f"no request files in {args.requests}")

    # Inject warm_up.json as the first request to prime the KV cache with the
    # cached prompt (uses the "before commit" system prompt). Not counted in stats.
    warm_up = os.path.join(args.requests, "warm_up.json")
    if os.path.exists(warm_up):
        print(f"warming up with {os.path.basename(warm_up)}...", flush=True)
        response = post(url, warm_up, model, args.reasoning_effort)
        prompt, cached, new, prompt_ms = prefill_stats(response)
        prefill_tps = new / (prompt_ms / 1000.0) if prompt_ms > 0 else 0.0
        print(
            f"{os.path.basename(warm_up)}: cached={cached}/{prompt} "
            f"({100.0 * cached / prompt if prompt else 0:.0f}%) | "
            f"prefill_new={new}tok @{prefill_tps:.0f}tok/s "
            f"({prompt_ms / 1000.0:.1f}s) | not counted in means",
            flush=True,
        )
        files = [f for f in files if os.path.basename(f) != "warm_up.json"]

    totals = {
        "prompt": 0, "cached": 0, "new": 0, "prompt_ms": 0.0,
        "gen": 0, "decode": 0, "gen_ms": 0.0,
        "draft_n": 0, "draft_acc": 0, "rounds": 0,
    }

    for path in files:
        response = post(url, path, model, args.reasoning_effort)
        usage = response.get("usage") or {}
        timings = response.get("timings") or {}
        metrics = response.get("metrics") or {}
        prompt, cached, new, prompt_ms = prefill_stats(response)
        gen = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
        gen_tps = float(timings.get("predicted_per_second") or 0.0)
        gen_ms = float(timings.get("predicted_ms") or 0.0)
        # predicted_ms starts at the first emission, which can carry a whole
        # speculative block; the rate counts only the tokens after it.
        decode = int((metrics.get("decode") or {}).get("tokens") or 0)
        prefill_tps = new / (prompt_ms / 1000.0) if prompt_ms > 0 else 0.0

        draft_n, draft_acc, rounds = response["draft"]
        line = (
            f"{os.path.basename(path)}: cached={cached}/{prompt} "
            f"({100.0 * cached / prompt if prompt else 0:.0f}%) | "
            f"prefill_new={new}tok @{prefill_tps:.0f}tok/s | "
            f"decode={gen}tok @{gen_tps:.1f}tok/s"
        )
        if draft_n:
            line += (
                f" | accept={100.0 * draft_acc / draft_n:.0f}% "
                f"({(draft_acc + rounds) / rounds if rounds else 0:.2f}tok/round)"
            )
            totals["draft_n"] += draft_n
            totals["draft_acc"] += draft_acc
            totals["rounds"] += rounds
        print(line, flush=True)

        totals["prompt"] += prompt
        totals["cached"] += cached
        totals["new"] += new
        totals["prompt_ms"] += prompt_ms
        totals["gen"] += gen
        totals["decode"] += decode
        totals["gen_ms"] += gen_ms

    print("-" * 72)
    cached_pct = 100.0 * totals["cached"] / totals["prompt"] if totals["prompt"] else 0
    prefill_tps = (
        totals["new"] / (totals["prompt_ms"] / 1000.0) if totals["prompt_ms"] else 0
    )
    gen_tps = totals["decode"] / (totals["gen_ms"] / 1000.0) if totals["gen_ms"] else 0
    accept_pct = (
        100.0 * totals["draft_acc"] / totals["draft_n"] if totals["draft_n"] else 0
    )
    tok_per_round = (
        (totals["draft_acc"] + totals["rounds"]) / totals["rounds"]
        if totals["rounds"]
        else 0
    )
    print(
        f"mean over {len(files)} requests: KV cached {cached_pct:.0f}% "
        f"({totals['cached']}/{totals['prompt']}) | "
        f"prefill {prefill_tps:.0f}tok/s over {totals['new']} new tokens "
        f"({totals['prompt_ms'] / 1000.0:.1f}s) | "
        f"generation {gen_tps:.1f}tok/s ({totals['gen']} tokens, "
        f"{totals['gen_ms'] / 1000.0:.1f}s) | "
        f"accept {accept_pct:.0f}% of drafted ({tok_per_round:.2f}tok/round)"
    )


if __name__ == "__main__":
    main()
