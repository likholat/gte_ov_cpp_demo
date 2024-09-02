# GTE with OpenVINO C++ API demo

This demo shows how to run "Alibaba-NLP/gte-large-en-v1.5" model with OpenVINO C++ API.

## 1. Download and convert the model

```console
pip install -r requirements.txt
python convert_model.py
```

- __For NPU__: To create static model use:
    ```console
    python convert_model.py --static
    ```

## 2. Install OpenVINO for C++

### 1. Create an Intel folder in the C:\Program Files (x86)\ directory. 

```
mkdir "C:\Program Files (x86)\Intel"
```

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

## 3. Install OpenVINO Tokenizer for C++:

### 1. Download OpenVINO Tokenizers prebuild libraries.

```console
cd <user_home>/Downloads
curl -L https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/2024.3.0.0/openvino_tokenizers_windows_2024.3.0.0_x86_64.zip --output openvino_tokenizers_2024.3.0.zip
```

- __Important__: To ensure compatibility, the first three numbers of the OpenVINO Tokenizers version should match the OpenVINO version and OS.


### 2. Extract OpenVINO Tokenizers archive into the OpenVINO installation directory.

```console
tar -xf openvino_tokenizers_2024.3.0.zip
move runtime\bin\intel64\Release\* "C:\Program Files (x86)\Intel\openvino_2024.3.0\runtime\bin\intel64\Release"
```

## 4. Configure the Environment

Open the __Command Prompt__, and run the setupvars.bat batch file to temporarily set your environment variables. If your <INSTALL_DIR> is not C:\Program Files (x86)\Intel\openvino_2024, use the correct directory instead.

```console
"C:\Program Files (x86)\Intel\openvino_2024.3.0\setupvars.bat"
```

- __Important__: You need to run the command for each new Command Prompt window.

## 5. Build and run C++ sample:

Build the demo:
```console
cd  gte_ov_cpp_demo\gte_cpp_demo
mkdir build && cd build
cmake .. && cmake --build . —-config Release
```

Demo usage:
```console
"Release\gte_sample.exe" <path_to_embedding_model> <path_to_tokenizer_model> <device_name> <num_of_iterations>
```

- `path_to_embedding_model`: path to embedding model `.xml` file
- `path_to_tokenizer_model`: path to tokenizer model `.xml` file
- `device_name`: `CPU`, `GPU` or `NPU`
- `num_of_iterations`: `num_of_iterations=0` to generate model output only, `num_of_iterations>0` to run in benchmark mode

Run the demo on CPU:
```console
"Release\gte_sample.exe" ../gte-large-ov/openvino_model.xml ../gte-large-ov/openvino_tokenizer.xml CPU 0
```

Run the demo on NPU:
```console
"Release\gte_sample.exe" ../gte-large-ov/static/openvino_model.xml ../gte-large-ov/openvino_tokenizer.xml NPU 0
```

- default input prompt used in Python script: `how to implement quick sort in python?`

As the result demo will produce `gte_ov_cpp_demo\gte_cpp_demo\cpp_res_<device>.txt` file.

## 6. Compare results with Python code:

To compare C++ demo __CPU__ result with Python Optimum CPU output:
```console
cd ..
python compare_with_python_res.py -cpp ./cpp_res_CPU.txt
```

Expected output:
```
Cosine similarity :  2.7778948030743322e-08
Mean Squared Error:  2.612026866194886e-12
```

To compare C++ demo __GPU__ result with Python Optimum CPU output:
```console
python compare_with_python_res.py -cpp ./cpp_res_GPU.txt
```

Expected output:
```
Cosine similarity :  4.6911470351518325e-05
Mean Squared Error:  5.341443669158994e-05
```

To compare C++ demo __NPU__ result with Python Optimum CPU output:
```console
python compare_with_python_res.py -cpp ./cpp_res_NPU.txt
```

Expected output:
```
Cosine similarity :  3.3489112992723946e-06
Mean Squared Error:  3.788536779638944e-06
```
