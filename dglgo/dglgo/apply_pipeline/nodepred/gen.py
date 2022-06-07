import ruamel.yaml
import torch
import typer

from copy import deepcopy
from jinja2 import Template
from pathlib import Path
from typing import Optional

from ...utils.factory import ApplyPipelineFactory, PipelineBase, DataFactory, NodeModelFactory
from ...utils.yaml_dump import deep_convert_dict, merge_comment

@ApplyPipelineFactory.register("nodepred")
class ApplyNodepredPipeline(PipelineBase):

    def __init__(self):
        self.pipeline = {
            "name": "nodepred",
            "mode": "apply"
        }
        self.pipeline_name = "nodepred"

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        cls.user_cfg_cls = UserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter("nodepred").get_dataset_enum() = typer.Option(None, help="input data name"),
            cfg: Optional[str] = typer.Option(None, help="output configuration file path"),
            cpt: str = typer.Option(..., help="input checkpoint file path")
        ):
            # Training configuration
            train_cfg = torch.load(cpt)["cfg"]
            if data is None:
                print("data is not specified, use the training dataset")
                data = train_cfg["data_name"]
            if cfg is None:
                cfg = "_".join(["apply", "nodepred", data, train_cfg["model_name"]]) + ".yaml"

            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline": self.pipeline,
                "device": train_cfg["device"],
                "data": {"name": data},
                "cpt_path": cpt
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            output_cfg = deep_convert_dict(output_cfg)
            comment_dict = {
                "device": "Torch device name, e.g., cpu or cuda or cuda:0",
                "cpt_path": "Path to the checkpoint file"
            }
            comment_dict = merge_comment(output_cfg, comment_dict)

            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print("Configuration file is generated at {}".format(Path(cfg).absolute()))

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        # Check validation
        cls.setup_user_cfg_cls()
        cls.user_cfg_cls(**user_cfg_dict)

        # Training configuration
        train_cfg = torch.load(user_cfg_dict["cpt_path"])["cfg"]

        # Dict for code rendering
        render_cfg = deepcopy(user_cfg_dict)
        model_name = train_cfg["model_name"]
        model_code = NodeModelFactory.get_source_code(model_name)
        render_cfg["model_code"] = model_code
        render_cfg["model_class_name"] = NodeModelFactory.get_model_class_name(model_name)
        render_cfg.update(DataFactory.get_generated_code_dict(train_cfg["data_name"]))

        # Dict for defining cfg in the rendered code
        generated_user_cfg = deepcopy(user_cfg_dict)
        generated_user_cfg["data"].pop("name")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg["model"] = train_cfg["model"]

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict

        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "nodepred.jinja-py", "r") as f:
            template = Template(f.read())

        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline for inference"
