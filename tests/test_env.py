import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env.environment import CodeFixerEnv
from env.models import Action, Reward
from tasks.tasks import (
    grade_easy, grade_medium, grade_hard,
    TASK_EASY, TASK_MEDIUM, TASK_HARD,
)

def test_reset_easy():
    env = CodeFixerEnv()
    obs = env.reset("easy")
    assert obs.step_number == 0
    assert obs.history == []
    assert "sum_range" in obs.input

def test_reset_medium():
    env = CodeFixerEnv()
    obs = env.reset("medium")
    assert "flatten" in obs.input

def test_reset_hard():
    env = CodeFixerEnv()
    obs = env.reset("hard")
    assert "longest_unique" in obs.input

def test_invalid_difficulty():
    env = CodeFixerEnv()
    with pytest.raises(ValueError):
        env.reset("impossible")

def test_step_before_reset_raises():
    env = CodeFixerEnv()
    with pytest.raises(RuntimeError):
        env.step(Action(type="fix", content="def f(): pass"))

def test_step_after_done_raises():
    env = CodeFixerEnv()
    env.reset("easy")
    env.step(Action(type="give_up", content=""))
    with pytest.raises(RuntimeError):
        env.step(Action(type="fix", content="def f(): pass"))

def test_explain_action():
    env = CodeFixerEnv()
    env.reset("easy")
    obs, reward, done, info = env.step(Action(type="explain", content=""))
    assert reward.value == -0.05
    assert reward.penalty == -0.05
    assert not done
    assert obs.step_number == 1
    assert len(obs.history) == 1
    assert info["outcome"] == "explained"

def test_give_up_action():
    env = CodeFixerEnv()
    env.reset("easy")
    obs, reward, done, info = env.step(Action(type="give_up", content=""))
    assert reward.value == -0.10
    assert done
    assert info["outcome"] == "gave_up"

def test_max_steps_terminates():
    env = CodeFixerEnv()
    env.reset("easy")
    for _ in range(CodeFixerEnv.MAX_STEPS):
        if not env._done:
            env.step(Action(type="fix", content="def sum_range(n): return 0"))
    assert env._done

def test_partial_fix_gives_partial_reward():
    env = CodeFixerEnv()
    env.reset("easy")
    code = "def sum_range(n):\n    return sum(range(1, n+1))"
    _, reward, _, info = env.step(Action(type="fix", content=code))
    assert info["grader_score"] > 0.0

def test_perfect_fix_easy_solves_episode():
    env = CodeFixerEnv()
    env.reset("easy")
    code = "def sum_range(n):\n    return sum(range(1, n + 1))"
    obs, reward, done, info = env.step(Action(type="fix", content=code))
    assert info["grader_score"] == 1.0
    assert done
    assert reward.value > 0
    assert info["outcome"] == "solved"

def test_efficiency_bonus_on_first_step():
    env = CodeFixerEnv()
    env.reset("easy")
    code = "def sum_range(n):\n    return sum(range(1, n + 1))"
    _, reward, _, info = env.step(Action(type="fix", content=code))
    assert reward.efficiency_bonus == 0.20
    assert info.get("efficiency_bonus") is True

def test_no_efficiency_bonus_after_step_2():
    env = CodeFixerEnv()
    env.reset("easy")
    wrong = "def sum_range(n): return 0"
    env.step(Action(type="fix", content=wrong))
    env.step(Action(type="fix", content=wrong))
    perfect = "def sum_range(n):\n    return sum(range(1, n + 1))"
    _, reward, _, _ = env.step(Action(type="fix", content=perfect))
    assert reward.efficiency_bonus == 0.0

def test_state_returns_correct_fields():
    env = CodeFixerEnv()
    env.reset("medium")
    s = env.state()
    assert s["difficulty"] == "medium"
    assert s["done"] is False
    assert s["step_number"] == 0
    assert "best_score" in s

def test_history_accumulates_correctly():
    env = CodeFixerEnv()
    env.reset("easy")
    env.step(Action(type="explain", content=""))
    obs, _, _, _ = env.step(Action(type="fix", content="def sum_range(n): return 0"))
    assert len(obs.history) == 2
    assert obs.history[0]["action_type"] == "explain"
    assert obs.history[1]["action_type"] == "fix"

def test_reward_is_pydantic_model():
    env = CodeFixerEnv()
    env.reset("easy")
    _, reward, _, _ = env.step(Action(type="explain", content=""))
    assert isinstance(reward, Reward)
    assert hasattr(reward, "value")
    assert hasattr(reward, "reason")


def test_grade_easy_perfect():
    score, _ = grade_easy("def sum_range(n):\n    return sum(range(1, n+1))")
    assert score == 1.0

def test_grade_easy_syntax_error_returns_zero():
    score, detail = grade_easy("def sum_range(n)\n    pass")
    assert score == 0.0
    assert detail["syntax_ok"] is False

def test_grade_medium_canonical_solution():
    score, _ = grade_medium(TASK_MEDIUM["canonical_solution"])
    assert score >= 0.95

def test_grade_hard_brute_force_penalised():
    score, detail = grade_hard(TASK_HARD["buggy_code"])
    assert detail["likely_linear"] is False