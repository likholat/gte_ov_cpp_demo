1. Model conversion:

```
pip install -r requirements.txt
python convert_model.py
```

2. Install OpenVino for C++ and OpenVino Tokenizer for C++:

OpenVino for C++: https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_3_0&OP_SYSTEM=LINUX&DISTRIBUTION=ARCHIVE
OpenVino Tokenizer for C++: https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/ov-tokenizers.html#c-installation

- NOTE: Make sure that the versions of OpenVino and OV Tokenizer packages are the same

Initialize OpenVino environment variables:
```
source <ov_install_dir>/setupvars.sh
```

3. Build and run C++ sample:

```
mkdir build && cd build
cmake .. && cmake --build .
./gte_sample
```

4. Compare results with Python code:

```
cd ..
python compare_with_python_res.py
```

Expected output:
```
Compiling the model to CPU ...
Output tensors have the same sizes
Accuracy: 4.886474609300251e-05
```
