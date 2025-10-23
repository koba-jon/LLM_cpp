#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
#include <tokenizers_cpp.h>
// For Original Header
#include "datasets.hpp"

namespace fs = std::filesystem;
using torch::indexing::Slice;
using tokenizers::Tokenizer;


// -----------------------------------------------
// namespace{datasets} -> function{collect}
// -----------------------------------------------
void datasets::collect(const std::string root, std::vector<std::string> &paths){
    fs::path ROOT(root);
    for (auto &p : fs::directory_iterator(ROOT)){
        if (!fs::is_directory(p)){
            std::stringstream rpath, fname;
            rpath << p.path().string();
            fname << p.path().filename().string();
            paths.push_back(rpath.str());
        }
        else{
            std::stringstream subsub;
            subsub << p.path().filename().string();
            datasets::collect(root + '/' + subsub.str(), paths);
        }
    }
    return;
}

void datasets::collect(const std::string root, const std::string sub, std::vector<std::string> &paths, std::vector<std::string> &fnames){
    fs::path ROOT(root);
    for (auto &p : fs::directory_iterator(ROOT)){
        if (!fs::is_directory(p)){
            std::stringstream rpath, fname;
            rpath << p.path().string();
            fname << p.path().filename().string();
            paths.push_back(rpath.str());
            fnames.push_back(sub + fname.str());
        }
        else{
            std::stringstream subsub;
            subsub << p.path().filename().string();
            datasets::collect(root + '/' + subsub.str(), sub + subsub.str() + '/', paths, fnames);
        }
    }
    return;
}


// -----------------------------------------------
// namespace{datasets} -> function{Text_Loader}
// -----------------------------------------------
torch::Tensor datasets::Text_Loader(const std::string &path, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer, const long int &sequence, const long int &stride, const int &endoftext, const int &padding){

    std::ifstream ifs;
    std::istreambuf_iterator<char> it, last;
    std::string str;
    std::vector<int> ids_int;
    std::vector<int64_t> ids;
    torch::Tensor data;

    // Get Data
    ifs.open(path);
    it = std::istreambuf_iterator<char>(ifs);
    str = std::string(it, last);
    if (!str.empty() && str.back() == '\n') str.pop_back();
    ids_int = tokenizer->Encode(str);
    ids_int.push_back(endoftext);
    while(ids_int.size() <= (size_t)sequence) ids_int.push_back(padding);
    while((ids_int.size() - sequence - 1) % stride != 0) ids_int.push_back(padding);
    ids = std::vector<int64_t>(ids_int.size());
    for (size_t i = 0; i < ids_int.size(); i++) ids.at(i) = ids_int.at(i);
    ifs.close();

    // Get Tensor
    data = torch::from_blob(ids.data(), {(long int)ids.size()}, torch::kLong).clone();

    return data;

}


// -----------------------------------------------
// namespace{datasets} -> function{Text_Loader}
// -----------------------------------------------
torch::Tensor datasets::Text_Loader_Predict(const std::string &path, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer){

    std::ifstream ifs;
    std::istreambuf_iterator<char> it, last;
    std::string str;
    std::vector<int> ids_int;
    std::vector<int64_t> ids;
    torch::Tensor data;

    // Get Data
    ifs.open(path);
    it = std::istreambuf_iterator<char>(ifs);
    str = std::string(it, last);
    if (!str.empty() && str.back() == '\n') str.pop_back();
    ids_int = tokenizer->Encode(str);
    ids = std::vector<int64_t>(ids_int.size());
    for (size_t i = 0; i < ids_int.size(); i++) ids.at(i) = ids_int.at(i);
    ifs.close();

    // Get Tensor
    data = torch::from_blob(ids.data(), {(long int)ids.size()}, torch::kLong).clone();

    return data;

}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{TextFolder} -> constructor
// -------------------------------------------------------------------------
datasets::TextFolder::TextFolder(const std::string &root, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer_, const long int &sequence_, const long int &stride_, const int &endoftext_, const int &padding_){
    datasets::collect(root, this->paths);
    std::sort(this->paths.begin(), this->paths.end());
    this->tokenizer = tokenizer_;
    this->sequence = sequence_;
    this->stride = stride_;
    this->endoftext = endoftext_;
    this->padding = padding_;
    for (size_t i = 0; i < this->paths.size(); i++){
        torch::Tensor text = datasets::Text_Loader(this->paths.at(i), this->tokenizer, this->sequence, this->stride, this->endoftext, this->padding);
        this->texts.push_back(text);
        for (long int j = 0; j < text.numel() - this->sequence; j += stride){
            this->paths_idx.push_back(i);
            this->offset_idx.push_back(j);
        }
    }
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{TextFolder} -> function{get}
// -------------------------------------------------------------------------
void datasets::TextFolder::get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor> &data){
    torch::Tensor text_in = this->texts[this->paths_idx.at(idx)].index({Slice(this->offset_idx.at(idx), this->offset_idx.at(idx) + this->sequence)}).contiguous();
    torch::Tensor text_out = this->texts[this->paths_idx.at(idx)].index({Slice(this->offset_idx.at(idx) + 1, this->offset_idx.at(idx) + 1 + this->sequence)}).contiguous();
    data = {text_in.detach().clone(), text_out.detach().clone()};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{TextFolder} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::TextFolder::size(){
    return this->paths_idx.size();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{TextFolderPredictWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::TextFolderPredictWithPaths::TextFolderPredictWithPaths(const std::string &root, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer_){
    datasets::collect(root, "", this->paths, this->fnames);
    std::sort(this->paths.begin(), this->paths.end());
    std::sort(this->fnames.begin(), this->fnames.end());
    this->tokenizer = tokenizer_;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{TextFolderPredictWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::TextFolderPredictWithPaths::get(const size_t idx, std::tuple<torch::Tensor, std::string> &data){
    torch::Tensor text = datasets::Text_Loader_Predict(this->paths.at(idx), this->tokenizer);
    std::string fname = this->fnames.at(idx);
    data = {text.detach().clone(), fname};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{TextFolderPredictWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::TextFolderPredictWithPaths::size(){
    return this->fnames.size();
}
