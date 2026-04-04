import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.models import Observation, Action, Reward
from tasks.tasks import TASKS

class CodeFixerEnv:
    MAX_STEPS: int = 5

    def __init__(self) -> None:
        self._task        = None
        self._grader      = None
        self._difficulty  = None
        self._history     = []
        self._step_number = 0
        self._done        = False
        self._best_score  = 0.0
        self._cumulative_reward = 0.0

    def reset(self, difficulty: str = "easy") -> Observation:
        if difficulty not in TASKS:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. Valid choices: {sorted(TASKS.keys())}"
            )

        task_def, grader = TASKS[difficulty]
        self._task        = task_def
        self._grader      = grader
        self._difficulty  = difficulty
        self._history     = []
        self._step_number = 0
        self._done        = False
        self._best_score  = 0.0
        self._cumulative_reward = 0.0

        return Observation(
            input=task_def["buggy_code"],
            context=task_def["context"],
            history=[],
            step_number=0,
            max_steps=self.MAX_STEPS,
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is over. Call reset() to start a new episode.")
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._step_number += 1
        info = {
            "task_id":    self._task["id"],
            "difficulty": self._difficulty,
            "step":       self._step_number,
        }

        raw_value       = 0.0
        grader_score    = None
        improvement     = None
        efficiency_bonus = 0.0
        penalty         = 0.0
        feedback        = ""

        if action.type == "give_up":
            penalty   = -0.10
            raw_value = penalty
            self._done = True
            feedback  = "Agent chose to give up."
            info["outcome"] = "gave_up"

        elif action.type == "explain":
            penalty   = -0.05
            raw_value = penalty
            feedback  = (
                f"Hint: re-read the loop bounds carefully. "
                f"Task goal: {self._task['context']}"
            )
            info["outcome"] = "explained"

        elif action.type == "fix":
            grader_score, grade_detail = self._grader(action.content)
            info["grade_detail"]  = grade_detail
            info["grader_score"]  = grader_score

            improvement = grader_score - self._best_score

            if improvement > 0:
                raw_value = improvement
                self._best_score = grader_score
            else:
                penalty   = -0.05
                raw_value = penalty

            if grader_score >= 1.0 and self._step_number <= 2:
                efficiency_bonus = 0.20
                raw_value += efficiency_bonus
                info["efficiency_bonus"] = True

            if grader_score >= 1.0:
                self._done = True
                feedback   = "All test cases passed — perfect fix!"
                info["outcome"] = "solved"
            else:
                passed = grade_detail.get("tests_passed", "?")
                total  = grade_detail.get("tests_total",  "?")
                feedback = (
                    f"Score {grader_score:.2f} — "
                    f"{passed}/{total} tests passed. Keep refining."
                )
                info["outcome"] = "partial"

        raw_value = round(raw_value, 4)
        self._cumulative_reward = round(self._cumulative_reward + raw_value, 4)

        if self._step_number >= self.MAX_STEPS and not self._done:
            self._done = True
            info["outcome"] = info.get("outcome", "timeout")

        reward = Reward(
            value            = raw_value,
            grader_score     = grader_score,
            improvement      = round(improvement, 4) if improvement is not None else None,
            efficiency_bonus = efficiency_bonus,
            penalty          = penalty,
            reason           = feedback,
        )

        self._history.append({
            "step":           self._step_number,
            "action_type":    action.type,
            "action_content": (
                action.content[:120] + "..."
                if len(action.content) > 120 else action.content
            ),
            "reward":   raw_value,
            "feedback": feedback,
        })

        next_obs = Observation(
            input       = self._task["buggy_code"],
            context     = self._task["context"],
            history     = list(self._history),
            step_number = self._step_number,
            max_steps   = self.MAX_STEPS,
        )
        return next_obs, reward, self._done, info

    def state(self) -> dict:
        return {
            "task_id":           self._task["id"] if self._task else None,
            "difficulty":        self._difficulty,
            "step_number":       self._step_number,
            "max_steps":         self.MAX_STEPS,
            "done":              self._done,
            "best_score":        self._best_score,
            "cumulative_reward": self._cumulative_reward,
            "history_length":    len(self._history),
        }
