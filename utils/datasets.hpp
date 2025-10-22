#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <string>
#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>
#include <tokenizers_cpp.h>

using tokenizers::Tokenizer;


// -----------------------
// namespace{datasets}
// -----------------------
namespace datasets{

    // Function Prototype
    void collect(const std::string root, std::vector<std::string> &paths);
    void collect(const std::string root, const std::string sub, std::vector<std::string> &paths, std::vector<std::string> &fnames);
    torch::Tensor Text_Loader(const std::string &path, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer, const long int &sequence, const long int &stride, const int &endoftext, const int &padding);
    torch::Tensor Text_Loader_Predict(const std::string &path, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer);

    // ------------------------------------------
    // namespace{datasets} -> class{TextFolder}
    // ------------------------------------------
    class TextFolder{
    private:
        std::vector<std::string> paths;
        std::shared_ptr<tokenizers::Tokenizer> tokenizer;
        long int sequence, stride;
        std::vector<size_t> paths_idx, offset_idx;
        int endoftext, padding;
    public:
        TextFolder(){}
        TextFolder(const std::string &root, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer_, const long int &sequence_, const long int &stride_, const int &endoftext_, const int &padding_);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor> &data);
        size_t size();
    };

    // ----------------------------------------------------------
    // namespace{datasets} -> class{TextFolderPredictWithPaths}
    // ----------------------------------------------------------
    class TextFolderPredictWithPaths{
    private:
        std::vector<std::string> paths, fnames;
        std::shared_ptr<tokenizers::Tokenizer> tokenizer;
    public:
        TextFolderPredictWithPaths(){}
        TextFolderPredictWithPaths(const std::string &root, const std::shared_ptr<tokenizers::Tokenizer> &tokenizer_);
        void get(const size_t idx, std::tuple<torch::Tensor, std::string> &data);
        size_t size();
    };

}



#endif