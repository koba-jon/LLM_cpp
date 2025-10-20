# LLM_cpp
LLM written in C++


```
git clone https://github.com/koba-jon/LLM_cpp.git
cd LLM_cpp
```
```
git clone https://github.com/mlc-ai/tokenizers-cpp.git
cd tokenizers-cpp
git submodule update --init --recursivecd
cd ..
```

```
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default stable
cargo --version
```


```
cd GPT-2
mkdir build
cd build
cmake ..
make -j4
cd ..
```

```
mkdir dist
cd dist
wget https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/resolve/main/tokenizer.json
cd ..
```
