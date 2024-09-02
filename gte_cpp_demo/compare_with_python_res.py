# Requires transformers>=4.36.0
# Adapted from https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5
# 
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
from scipy.spatial import distance
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compare C++ demo result with Python Optimum output.')
parser.add_argument('--model_id', '-m', type=str, help="model path", default="Alibaba-NLP/gte-large-en-v1.5")
parser.add_argument('--cpp_res', '-cpp', type=str, help="path to .txt file with cpp demo output", default="./cpp_res_CPU.txt")
args = parser.parse_args()

with open('input_prompt.txt', 'r') as file:
    input_text = [file.read().rstrip()]
print("Input prompt: ", input_text[0])

tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

# comment out either AutoModel or OVModel line and compare inference results between PyTorch and OpenVINO
model = OVModelForFeatureExtraction.from_pretrained("gte-large-ov", trust_remote_code=True)

# Tokenize the input texts
batch_dict = tokenizer(
    input_text, max_length=8192, padding=True, truncation=True, return_tensors="pt"
)

outputs = model(**batch_dict)

python_data = outputs.last_hidden_state.numpy().flatten()

cpp_data = np.loadtxt(args.cpp_res)

if len(cpp_data) - len(python_data) == 0:
    print("Output tensors have the same sizes")
else:
    print(f"Different output tensor sizes!!!\nPython output len: {len(python_data)}, cpp output len: {len(cpp_data)}")

print("Cosine similarity : ", distance.cosine(python_data, cpp_data))
print("Mean Squared Error: ", np.square(np.subtract(python_data, cpp_data)).mean())
