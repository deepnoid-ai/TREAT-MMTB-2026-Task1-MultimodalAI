import csv
from pathlib import Path

import torch

from .configuration_mmtb import MMTBConfig
from .modeling_base import MMTBBaseModel
from .predict import ClsModel, SegModel, Task1Model, discover_cases, save_mask


class MMTBModel(MMTBBaseModel):
    config_class = MMTBConfig
    task = 1

    @staticmethod
    def checkpoint_indices(config):
        if config.mode not in ("seg", "cls"):
            raise ValueError("mode must be 'seg' or 'cls'")
        return range(5 if config.mode == "seg" else 1)

    @classmethod
    def build_models(cls, config):
        return [ClsModel(config.cls_spec) if index == 0 else SegModel(config.seg_spec) for index in cls.checkpoint_indices(config)]

    @torch.inference_mode()
    def predict(self, input_path, output_dir=None):
        cases = discover_cases(input_path)
        output = Path(output_dir) if output_dir is not None else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
        pipeline = Task1Model(self.models[0], list(self.models[1:]), self.config.seg_spec, self.config.cls_spec)
        results = []
        for case_id, path in cases:
            mask, cavity = pipeline.predict(path)
            result = {"our_id": case_id, "cavity": cavity}
            if output is None:
                result["mask"] = mask
            if output is not None and mask is not None:
                target = output / f"{case_id}.nii.gz"
                save_mask(mask, path, target)
                result["mask_path"] = str(target)
            results.append(result)
        if output is not None:
            with (output / "prediction.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["our_id", "cavity"])
                writer.writerows((result["our_id"], result["cavity"]) for result in results)
        return results

    def forward(self, input_path, output_dir=None):
        return self.predict(input_path, output_dir=output_dir)
