# LLM C++ Samples

## 1. Preparation

### (1) Git Clone
```
$ git clone https://github.com/koba-jon/LLM_cpp.git
$ cd LLM_cpp
```

### (2) Download PyTorch C++
Please select the environment to use as follows on PyTorch official. <br>
PyTorch official : https://pytorch.org/ <br>
***
PyTorch Build : Preview (Nightly) <br>
Your OS : Linux <br>
Package : LibTorch <br>
Language : C++ / Java <br>
Run this Command : Download here (cxx11 ABI) <br>
CUDA 12.6 : https://download.pytorch.org/libtorch/nightly/cu126/libtorch-shared-with-deps-latest.zip <br>
CUDA 12.8 : https://download.pytorch.org/libtorch/nightly/cu128/libtorch-shared-with-deps-latest.zip <br>
CUDA 13.0 : https://download.pytorch.org/libtorch/nightly/cu130/libtorch-shared-with-deps-latest.zip <br>
CPU : https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip <br>
***

### (3) Install Boost
~~~
$ sudo apt install libboost-dev libboost-all-dev
~~~

### (4) Install tokenizers
```
$ git clone https://github.com/mlc-ai/tokenizers-cpp.git
$ cd tokenizers-cpp
$ git submodule update --init --recursive
$ cd ..
```

### (5) Install Rust
```
$ curl https://sh.rustup.rs -sSf | sh -s -- -y
$ source $HOME/.cargo/env
$ rustup default stable
$ cargo --version
```

## 2. Execution (Example: GPT-2)

```
$ cd GPT-2
```
### (1) Install dictionary
```
$ mkdir dist
$ cd dist
$ wget https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/resolve/main/tokenizer.json
$ cd ..
```

### (2) Build
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ cd ..
```

### (3) Training
```
$ sh scripts/train.sh
```

### (4) Test
```
$ sh scripts/test.sh
```

### (5) Prediction
```
$ sh scripts/predict.sh
```

### (6) Question Answering
```
$ sh scripts/question.sh
```
