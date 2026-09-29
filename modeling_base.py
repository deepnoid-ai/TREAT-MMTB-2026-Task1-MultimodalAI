import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel
from transformers.utils.hub import cached_file

from .configuration_mmtb import MMTBConfig


class MMTBBaseModel(PreTrainedModel):
    config_class = MMTBConfig
    base_model_prefix = "models"

    def __init__(self, config):
        super().__init__(config)
        if config.task != self.task:
            raise ValueError(f"Expected task {self.task}, received task {config.task}")
        self.models = nn.ModuleList(self.build_models(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None,
                        cache_dir=None, force_download=False, local_files_only=False,
                        token=None, revision="main", **kwargs):
        if model_args:
            raise TypeError("Positional model arguments are not supported")
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        for key in ("_from_auto", "_from_pipeline", "trust_remote_code", "adapter_kwargs"):
            kwargs.pop(key, None)
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, cache_dir=cache_dir, force_download=force_download,
                local_files_only=local_files_only, token=token, revision=revision, subfolder=subfolder,
            )
        config = copy.deepcopy(config)
        if "mode" in kwargs:
            config.mode = kwargs.pop("mode")
        device = kwargs.pop("device", "cpu")
        dtype = kwargs.pop("dtype", None)
        legacy_dtype = kwargs.pop("torch_dtype", None)
        dtype = dtype if dtype is not None else legacy_dtype
        if dtype is None or dtype == "auto":
            dtype = getattr(config, "dtype", None) or torch.float32
        if kwargs:
            raise TypeError(f"Unsupported loading arguments: {', '.join(sorted(kwargs))}; use device= or model.to() for placement")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if dtype in (None, "auto"):
            dtype = torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            raise ValueError("dtype must be a floating-point torch dtype or 'auto'")
        if config.task != cls.task:
            raise ValueError(f"Expected task {cls.task}, received task {config.task}")
        indices = cls.checkpoint_indices(config)
        resolved_revision = getattr(config, "_commit_hash", None) or commit_hash or revision
        paths = [cached_file(
            pretrained_model_name_or_path, f"weights/model_{index}.safetensors",
            cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only,
            token=token, revision=resolved_revision, subfolder=subfolder,
        ) for index in indices]
        model = cls(config)
        for net, path in zip(model.models, paths):
            net.load_state_dict(load_file(path), strict=True)
            net.to(device=device, dtype=dtype)
        model.eval()
        return model

    def save_pretrained(self, save_directory, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported saving arguments: {', '.join(sorted(kwargs))}")
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / "weights"
        weights.mkdir(exist_ok=True)
        expected = {f"model_{index}.safetensors" for index in self.checkpoint_indices(self.config)}
        if any(path.name not in expected for path in weights.iterdir()):
            raise ValueError("Save to a fresh directory or one containing only the selected checkpoint files")
        entries = []
        for index, net in zip(self.checkpoint_indices(self.config), self.models):
            target = weights / f"model_{index}.safetensors"
            state = {name: tensor.detach().cpu().contiguous() for name, tensor in net.state_dict().items()}
            save_file(state, str(target), metadata={"format": "pt"})
            digest = hashlib.sha256()
            with target.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            entries.append({"name": f"model_{index}", "file": f"weights/{target.name}",
                            "bytes": target.stat().st_size, "sha256": digest.hexdigest()})
        config = copy.deepcopy(self.config)
        config.auto_map = {"AutoConfig": "configuration_mmtb.MMTBConfig", "AutoModel": "modeling_mmtb.MMTBModel"}
        config.architectures = ["MMTBModel"]
        config.dtype = self.dtype
        config.save_pretrained(directory)
        for filename in ("configuration_mmtb.py", "modeling_base.py", "modeling_mmtb.py", "predict.py"):
            source = Path(__file__).resolve().parent / filename
            target = directory / filename
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        (directory / "model_manifest.json").write_text(json.dumps({"task": f"task{self.task}", "models": entries}, indent=2) + "\n")
