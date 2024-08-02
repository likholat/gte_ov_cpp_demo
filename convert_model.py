from pathlib import Path

from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.openvino import main_export
from transformers import AutoConfig, AutoTokenizer
from openvino_tokenizers import convert_tokenizer
from openvino import save_model

model_id = "Alibaba-NLP/gte-large-en-v1.5"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
custom_export_configs = {"model": BertOnnxConfig(config, task="feature-extraction")}

main_export(
    model_name_or_path=model_id,
    custom_export_configs=custom_export_configs,
    library_name="transformers",
    output=Path("gte-large-ov"),
    task="feature-extraction",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
ov_tokenizer = convert_tokenizer(tokenizer, with_detokenizer=False)

tokenizer_dir = Path("gte-large-ov/")
save_model(ov_tokenizer, tokenizer_dir / "openvino_tokenizer.xml")
