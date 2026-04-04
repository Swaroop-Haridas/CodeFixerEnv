from pydantic import BaseModel, Field
from typing import Literal, List, Optional


class Observation(BaseModel):
    input: str = Field(
        ...,
        description="The buggy Python code snippet the agent must fix"
    )
    context: str = Field(
        ...,
        description="Plain-English description of what the correct code should do"
    )
    history: List[dict] = Field(
        default_factory=list,
        description="Ordered list of previous {step, action_type, action_content, reward, feedback} dicts"
    )
    step_number: int = Field(
        default=0,
        description="How many steps have elapsed since reset (0 = initial)"
    )
    max_steps: int = Field(
        default=5,
        description="Maximum steps allowed before the episode auto-terminates"
    )

class Action(BaseModel):
    type: Literal["fix", "explain", "give_up"] = Field(
        ...,
        description="Action type: 'fix' submits code, 'explain' requests a hint, 'give_up' ends episode"
    )
    content: str = Field(
        ...,
        description="Corrected code (fix), clarification question (explain), or empty string (give_up)"
    )

class Reward(BaseModel):
    value: float = Field(
        ...,
        description="Net scalar reward for this step"
    )
    grader_score: Optional[float] = Field(
        default=None,
        description="Raw grader output 0.0–1.0 (only set on 'fix' actions)"
    )
    improvement: Optional[float] = Field(
        default=None,
        description="Score delta vs. previous best (only set on 'fix' actions)"
    )
    efficiency_bonus: float = Field(
        default=0.0,
        description="+0.2 if solved in ≤ 2 steps, else 0.0"
    )
    penalty: float = Field(
        default=0.0,
        description="Negative component from redundancy, explain cost, or give_up"
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of how this reward was computed"
    )


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool = Field(..., description="True when the episode has ended")
    info: dict = Field(
        default_factory=dict,
        description="Auxiliary diagnostics: task_id, difficulty, grader detail, outcome"
    )