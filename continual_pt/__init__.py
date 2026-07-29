"""Web-grounded autonomous continual post-training."""

__all__ = ["ContinualLearningLoop", "LearningGoal"]


def __getattr__(name):
    # Keep goal configuration usable on a CPU-only machine; importing the
    # training runtime requires torch/transformers and is intentionally lazy.
    if name == "ContinualLearningLoop":
        from .loop import ContinualLearningLoop
        return ContinualLearningLoop
    if name == "LearningGoal":
        from .schema import LearningGoal
        return LearningGoal
    raise AttributeError(name)
