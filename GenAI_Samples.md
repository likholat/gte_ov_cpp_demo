# How to build and run OpenVino GenAI samples

This tutorial shows how to build and run OpenVino GenAI Chat Sample.

### 1. Install OpenVIno GenAI
Download and unzip archive: https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html#windows

### 2. Initialize OpenVino environment variables

```console
cd openvino_genai_2024.3.0.0\openvino_genai_windows_2024.3.0.0_x86_64
setupvars.bat
```

### 3. Download model

```console
cd openvino_genai_windows_2024.3.0.0_x86_64\samples\cpp\chat_sample
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

### 4. Build samples

```console
cd openvino_genai_windows_2024.3.0.0_x86_64\samples\cpp
build_samples_msvc.bat
```

You can find output path at the build log:

```console
-- Build files have been written to: C:/Users/user/Documents/Intel/OpenVINO/openvino_cpp_samples_build
```

### 5. Run sample

```console
C:\Users\user\Documents\Intel\OpenVINO\openvino_cpp_samples_build\intel64\Release\chat_sample.exe TinyLlama-1.1B-Chat-v1.0
```

Expected output:

```console
question:
what is the capital of China?
The capital of China is Beijing.
----------
question:
```
