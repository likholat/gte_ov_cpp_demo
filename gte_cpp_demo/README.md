# GTE with OpenVINO C++ API demo

This demo shows how to run "Alibaba-NLP/gte-large-en-v1.5" model with OpenVINO C++ API.

## 1. Download and convert the model

```console
pip install -r requirements.txt
python convert_model.py
```

## 2. Install OpenVINO for C++

### 1. Create an Intel folder in the C:\Program Files (x86)\ directory. 

```
mkdir "C:\Program Files (x86)\Intel"
```

- Skip this step if the folder already exists.

### 2. Download OpenVINO Runtime

Download the OpenVINO Runtime archive file for Windows to your local Downloads folder

```console
cd <user_home>/Downloads
curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.3/windows/w_openvino_toolkit_windows_2024.3.0.16041.1e3b88e4e3f_x86_64.zip --output openvino_2024.3.0.zip
```

### 3. Extract the archive file

Use your favorite tool to extract the archive file, rename the extracted folder, and move it to the C:\Program Files (x86)\Intel directory. To do this step using command-line, run the following commands in the command prompt window you opened:

```console
tar -xf openvino_2024.3.0.zip
ren w_openvino_toolkit_windows_2024.3.0.16041.1e3b88e4e3f_x86_64 openvino_2024.3.0
move openvino_2024.3.0 "C:\Program Files (x86)\Intel"
```

### 4. For simplicity, it is useful to create a symbolic link.

Open a command prompt window as administrator (see Step 1 for how to do this) and run the following commands:

```compile
cd C:\Program Files (x86)\Intel
mklink /D openvino_2024 openvino_2024.3.0
```

### 5. Configure the Environment

Open the __Command Prompt__, and run the setupvars.bat batch file to temporarily set your environment variables. If your <INSTALL_DIR> is not C:\Program Files (x86)\Intel\openvino_2024, use the correct directory instead.

```console
"C:\Program Files (x86)\Intel\openvino_2024\setupvars.bat"
```

- __Important__: You need to run the command for each new Command Prompt window.

## 3. Install OpenVINO Tokenizer for C++:

### 1. Download OpenVINO Tokenizers prebuild libraries.

```console
cd <user_home>/Downloads
curl -L https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/2024.3.0.0/openvino_tokenizers_windows_2024.3.0.0_x86_64.zip --output openvino_tokenizers_2024.3.0.zip
```

- __Important__: To ensure compatibility, the first three numbers of the OpenVINO Tokenizers version should match the OpenVINO version and OS.

OpenVINO Tokenizer for C++: https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/ov-tokenizers.html#c-installation

### 2. Extract OpenVINO Tokenizers archive into the OpenVINO installation directory.

```console
tar -xf openvino_tokenizers_2024.3.0.zip
move openvino_tokenizers_2024 <openvino_dir>\runtime\bin\intel64\Release\
```


## 4. Build and run C++ sample:

```
mkdir build && cd build
cmake .. && cmake --build .
./gte_sample
```

## 5. Compare results with Python code:

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
