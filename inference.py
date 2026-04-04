"""
inference.py — Baseline inference script for CodeFixerEnv.

Environment variables (as per official spec):
  API_BASE_URL      — base URL of the running server (has default)
  MODEL_NAME        — model identifier (has default)
  HF_TOKEN          — Hugging Face API key (NO default — must be set by user)
  LOCAL_IMAGE_NAME  — optional, only if using from_docker_image()
"""

import os
import json
import requests
from openai import OpenAI

API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")   # NO default — required by spec
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

DIFFICULTIES = ["easy", "medium", "hard"]
MAX_STEPS    = 5

SYSTEM_PROMPT = """You are an expert Python debugger.
You will receive a buggy Python function and a description of what it should do.
Fix the code and return ONLY a valid JSON object with no markdown, no explanation:
{
  "action_type": "fix",
  "action_content": "<full corrected Python function>"
}
action_type must be one of: fix, explain, give_up
action_content must be the complete corrected function, not a diff.
"""

def call_env(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    url = f"{API_BASE_URL.rstrip('/')}/{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=payload, timeout=30)
    else:
        resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

def call_model(obs: dict) -> tuple[str, str]:
    history_str = ""
    if obs["history"]:
        history_str = "\n\nPrevious attempts:\n" + "\n".join(
            f"  Step {h['step']}: {h['feedback']}" for h in obs["history"]
        )

    user_prompt = (
        f"Task: {obs['context']}\n\n"
        f"Buggy code:\n{obs['input']}"
        f"{history_str}\n\n"
        f"Step {obs['step_number'] + 1} of {obs['max_steps']}. Fix the code."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        raw = completion.choices[0].message.content or ""
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("action_type", "give_up"), parsed.get("action_content", "")
    except Exception as exc:
        print(f"[MODEL ERROR] {exc}")
        return "give_up", ""

def run_episode(difficulty: str) -> dict:
    print(f"START difficulty={difficulty}")

    obs      = call_env("reset", "POST", {"difficulty": difficulty})
    total_reward = 0.0
    final_score  = 0.0
    outcome      = "timeout"

    for step_num in range(1, MAX_STEPS + 1):
        action_type, action_content = call_model(obs)

        result      = call_env("step", "POST", {
            "action_type":    action_type,
            "action_content": action_content,
        })
        reward_val  = result["reward"]["value"]
        done        = result["done"]
        info        = result["info"]
        obs         = result["observation"]
        total_reward += reward_val
        outcome      = info.get("outcome", "unknown")

        grader_score = info.get("grader_score", None)
        if grader_score is not None:
            final_score = grader_score

        print(
            f"STEP difficulty={difficulty} step={step_num} "
            f"action={action_type} reward={reward_val:.4f} "
            f"score={final_score:.4f} done={done}"
        )

        if done:
            break

    print(
        f"END difficulty={difficulty} "
        f"total_reward={total_reward:.4f} "
        f"final_score={final_score:.4f} "
        f"outcome={outcome}"
    )

    return {
        "difficulty":         difficulty,
        "total_reward":       round(total_reward, 4),
        "final_grader_score": final_score,
        "outcome":            outcome,
    }

def main():
    print(f"[INFO] API_BASE_URL={API_BASE_URL}")
    print(f"[INFO] MODEL_NAME={MODEL_NAME}")

    try:
        h = call_env("health")
        print(f"[INFO] Server status: {h['status']}")
    except Exception as e:
        print(f"[ERROR] Cannot reach server: {e}")
        return

    results = [run_episode(d) for d in DIFFICULTIES]

    avg = round(sum(r["final_grader_score"] for r in results) / len(results), 4)
    print(f"[SUMMARY] average_score={avg}")

    with open("inference_results.json", "w") as f:
        json.dump({"results": results, "average_score": avg}, f, indent=2)
    print("[INFO] Saved inference_results.json")

if __name__ == "__main__":
    main()