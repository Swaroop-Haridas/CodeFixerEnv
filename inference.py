import os
import json
import time
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

DIFFICULTIES = ["easy", "medium", "hard"]
MAX_STEPS    = 5
TEMPERATURE  = 0.2
MAX_TOKENS   = 800

client = OpenAI(
    base_url=API_BASE_URL if "huggingface" in API_BASE_URL else "https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an expert Python debugger.
You will receive a buggy Python function and a description of what it should do.
Your job is to return the corrected function.

You MUST reply with ONLY a valid JSON object — no markdown, no explanation, no preamble:
{
  "action_type": "fix",
  "action_content": "<full corrected Python function here>"
}

Rules:
- action_type must be exactly one of: fix, explain, give_up
- action_content must contain the complete corrected function (not a diff or snippet)
- Do NOT wrap action_content in backticks or markdown
- Do NOT include any text outside the JSON object
"""

def call_env(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    url = f"{API_BASE_URL}/{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=payload, timeout=30)
    else:
        resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()
def build_prompt(obs: dict) -> str:
    history_lines = ""
    if obs["history"]:
        history_lines = "\n\nPrevious attempts:\n" + "\n".join(
            f"  Step {h['step']}: {h['feedback']}" for h in obs["history"]
        )

    return (
        f"Task: {obs['context']}\n\n"
        f"Buggy code:\n{obs['input']}"
        f"{history_lines}\n\n"
        f"You are on step {obs['step_number'] + 1} of {obs['max_steps']}. Fix the code."
    )

def call_model(prompt: str) -> tuple[str, str]:
    """Call the model, parse JSON, return (action_type, action_content)."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = completion.choices[0].message.content or ""
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("action_type", "give_up"), parsed.get("action_content", "")
    except Exception as exc:
        print(f"  [MODEL ERROR] {exc} — giving up this step")
        return "give_up", ""

def run_episode(difficulty: str) -> dict:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Task: {difficulty.upper()}")
    print(sep)

    obs = call_env("reset", "POST", {"difficulty": difficulty})
    print(f"  Context : {obs['context'][:90]}...")
    print(f"  Buggy code preview:\n{''.join(obs['input'].splitlines(keepends=True)[:4])}  ...")

    total_reward  = 0.0
    final_score   = 0.0
    steps_taken   = 0
    outcome       = "timeout"

    for step_num in range(1, MAX_STEPS + 1):
        prompt = build_prompt(obs)
        action_type, action_content = call_model(prompt)
        print(f"\n  Step {step_num}: action_type = {action_type}")

        result = call_env("step", "POST", {
            "action_type":    action_type,
            "action_content": action_content,
        })

        reward_val = result["reward"]["value"]
        done       = result["done"]
        info       = result["info"]
        obs        = result["observation"]

        total_reward += reward_val
        steps_taken   = step_num
        outcome       = info.get("outcome", "unknown")

        if "grader_score" in info:
            final_score = info["grader_score"]
            gd = info.get("grade_detail", {})
            print(
                f"  Reward: {reward_val:+.4f} | "
                f"Grader: {final_score:.4f} | "
                f"Tests: {gd.get('tests_passed','?')}/{gd.get('tests_total','?')}"
            )
        else:
            print(f"  Reward: {reward_val:+.4f}")

        if done:
            print(f"  Outcome: {outcome}")
            break

    return {
        "difficulty":         difficulty,
        "steps_taken":        steps_taken,
        "total_reward":       round(total_reward, 4),
        "final_grader_score": final_score,
        "outcome":            outcome,
    }

def main():
    print("CodeFixerEnv — Baseline Inference")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Server : {API_BASE_URL}")

    try:
        h = call_env("health")
        print(f"  Server : {h['status'].upper()} — {h.get('environment', '')}")
    except Exception as e:
        print(f"  ERROR: Cannot reach server at {API_BASE_URL}\n  {e}")
        return

    start = time.time()
    results = [run_episode(d) for d in DIFFICULTIES]
    elapsed = round(time.time() - start, 1)

    sep = "=" * 62
    print(f"\n{sep}")
    print("  RESULTS")
    print(sep)
    for r in results:
        print(
            f"  {r['difficulty'].upper():8s} | "
            f"Steps: {r['steps_taken']} | "
            f"Reward: {r['total_reward']:+.4f} | "
            f"Score: {r['final_grader_score']:.4f} | "
            f"{r['outcome']}"
        )

    avg = round(sum(r["final_grader_score"] for r in results) / len(results), 4)
    print(f"\n  Average grader score : {avg:.4f}")
    print(f"  Total runtime        : {elapsed}s")
    print(sep)

    output = {"results": results, "average_score": avg, "runtime_seconds": elapsed}
    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Saved → inference_results.json")

if __name__ == "__main__":
    main()