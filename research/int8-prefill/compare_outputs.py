#!/usr/bin/env python3
"""Greedy completions of the Junie replay requests, for A/B of prefill paths.

  compare_outputs.py capture OUT.json [--port 8000] [--max-tokens 256]
  compare_outputs.py compare A.json B.json

Each request is sent with temperature 0, a fixed seed and a token cap, and
the response message (content and tool calls) and usage are recorded.
`compare` reports, per request, whether the outputs are identical and the
first differing character; the total prefix agreement is the headline.
"""
import argparse
import glob
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REQUESTS = os.path.join(HERE, "..", "..", "dev", "benchmarks", "junie_replay", "requests")


def served_model(base):
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


def capture(args):
    base = f"http://127.0.0.1:{args.port}"
    model = served_model(base)
    results = {}
    for path in sorted(glob.glob(os.path.join(REQUESTS, "*.json"))):
        name = os.path.basename(path)
        body = json.load(open(path))
        body["model"] = model
        body["stream"] = False
        body["temperature"] = 0
        body["seed"] = 1
        body["max_tokens"] = args.max_tokens
        body.pop("stop", None)  # Splash rejects stop together with tools
        request = urllib.request.Request(
            f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as error:
            print(f"{name}: HTTP {error.code}: {error.read().decode()[:500]}", file=sys.stderr)
            raise
        message = reply["choices"][0]["message"]
        text = ""
        for field in ("reasoning_content", "reasoning"):
            if message.get(field):
                text += f"<think>{message[field]}</think>"
        text += message.get("content") or ""
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            text += f"\n<tool_call {function.get('name')}>{function.get('arguments')}"
        results[name] = {"text": text, "usage": reply.get("usage"),
                         "finish_reason": reply["choices"][0].get("finish_reason")}
        print(f"{name}: {len(text)} chars, finish={results[name]['finish_reason']}", flush=True)
    json.dump(results, open(args.output, "w"), indent=1)


def compare(args):
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    identical = 0
    total_chars = 0
    agreed_chars = 0
    for name in sorted(set(a) & set(b)):
        ta, tb = a[name]["text"], b[name]["text"]
        prefix = 0
        while prefix < min(len(ta), len(tb)) and ta[prefix] == tb[prefix]:
            prefix += 1
        same = ta == tb
        identical += same
        total_chars += max(len(ta), len(tb))
        agreed_chars += prefix
        status = "identical" if same else f"diverge at char {prefix} of {len(ta)}/{len(tb)}"
        print(f"{name}: {status}")
        if not same:
            print(f"   A: ...{ta[max(0, prefix - 40):prefix + 60]!r}")
            print(f"   B: ...{tb[max(0, prefix - 40):prefix + 60]!r}")
    print(f"identical outputs: {identical}/{len(set(a) & set(b))}; "
          f"agreeing prefix: {agreed_chars}/{total_chars} chars ({100.0 * agreed_chars / max(total_chars, 1):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("output")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--max-tokens", type=int, default=256)
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    else:
        compare(args)


if __name__ == "__main__":
    sys.exit(main())
