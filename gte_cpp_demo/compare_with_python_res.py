# Requires transformers>=4.36.0
# Adapted from https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5
# 
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Download and convert a model.')
parser.add_argument('--model_id', '-m', type=str, default="Alibaba-NLP/gte-large-en-v1.5")
parser.add_argument('--input_prompt', '-i', type=str, default="how to implement quick sort in python?")
args = parser.parse_args()


input_texts = [args.input_prompt]
tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

# comment out either AutoModel or OVModel line and compare inference results between PyTorch and OpenVINO
model = OVModelForFeatureExtraction.from_pretrained("gte-large-ov", trust_remote_code=True)

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt"
)

outputs = model(**batch_dict)

python_data = outputs.last_hidden_state.numpy().flatten()

cpp_data = np.loadtxt("./build/cpp_res.txt")

if len(cpp_data) - len(python_data) == 0:
    print("Output tensors have the same sizes")
else:
    print(f"Different output tensor sizes!!!\nPython output len: {len(python_data)}, cpp output len: {len(cpp_data)}")

print(f"Accuracy: {max(abs(python_data-cpp_data))}")
