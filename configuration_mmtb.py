from transformers import PretrainedConfig


class MMTBConfig(PretrainedConfig):
    model_type = "treat-mmtb-2026"

    def __init__(self, task=1, mode=None, **kwargs):
        super().__init__(**kwargs)
        if task not in (1, 2):
            raise ValueError("task must be 1 or 2")
        mode = mode or ("seg" if task == 1 else "cls")
        if mode not in (("seg", "cls") if task == 1 else ("cls",)):
            raise ValueError(f"Unsupported mode {mode!r} for task {task}")
        self.task = task
        self.mode = mode
