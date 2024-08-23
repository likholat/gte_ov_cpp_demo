from pathlib import Path

from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.openvino import main_export
from transformers import AutoConfig, AutoTokenizer
from openvino_tokenizers import convert_tokenizer
from openvino import save_model, Core, serialize
from optimum.intel import OVWeightQuantizationConfig, OVConfig

import argparse

parser = argparse.ArgumentParser(description='Download and convert a model.')
parser.add_argument('--model_id', '-m', type=str, default="Alibaba-NLP/gte-large-en-v1.5")
parser.add_argument('--quantize', '-q', default=True, action=argparse.BooleanOptionalAction, help='Whether to apply weight quantization.')
parser.add_argument("--static", default=False, action=argparse.BooleanOptionalAction, help='Whether to reshape model to static.')
parser.add_argument('--shape', nargs='+', type=int, default=[1, 1024], help='Shapes for static model.')

args = parser.parse_args()
model_id = args.model_id

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
custom_export_configs = {"model": BertOnnxConfig(config, task="feature-extraction")}

ov_config = OVConfig(quantization_config=OVWeightQuantizationConfig()) if args.quantize else None

main_export(
    model_name_or_path=model_id,
    custom_export_configs=custom_export_configs,
    library_name="transformers",
    output=Path("gte-large-ov"),
    task="feature-extraction",
    trust_remote_code=True,
    ov_config=ov_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
ov_tokenizer = convert_tokenizer(tokenizer, with_detokenizer=False)
save_model(ov_tokenizer, Path("gte-large-ov/") / "openvino_tokenizer.xml")

if args.static:
    assert len(args.shape) == 2, "Model should have 2 inpus shapes"

    core = Core()
    model = core.read_model("gte-large-ov/openvino_model.xml")
    model.reshape({model.input(0): args.shape, model.input(1): args.shape, model.input(2): args.shape})
    serialize(model, Path("gte-large-ov/static/") / "openvino_model.xml")


