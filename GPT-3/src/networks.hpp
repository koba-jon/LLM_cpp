#ifndef NETWORKS_HPP
#define NETWORKS_HPP

// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);


// -------------------------------------------------
// struct{FeedForwardImpl}(nn::Module)
// -------------------------------------------------
struct FeedForwardImpl : nn::Module{
private:
    nn::Sequential layers;
public:
    FeedForwardImpl(){}
    FeedForwardImpl(const size_t emb_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(FeedForward);


// -------------------------------------------------
// struct{MultiHeadAttentionImpl}(nn::Module)
// -------------------------------------------------
struct MultiHeadAttentionImpl : nn::Module{
private:
    long int n_heads, head_dim;
    nn::Linear W_key{nullptr}, W_query{nullptr}, W_value{nullptr}, out_proj{nullptr};
    nn::Dropout dropout{nullptr};
    torch::Tensor mask;
public:
    MultiHeadAttentionImpl(){}
    MultiHeadAttentionImpl(const long int d_in, const long int d_out, const long int sequence, const float droprate, const long int n_heads_, const bool qkv_bias);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MultiHeadAttention);


// -------------------------------------------------
// struct{TransformerBlockImpl}(nn::Module)
// -------------------------------------------------
struct TransformerBlockImpl : nn::Module{
private:
    MultiHeadAttention attn;
    FeedForward ff;
    nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    nn::Dropout drop_shortcut{nullptr};
public:
    TransformerBlockImpl(){}
    TransformerBlockImpl(const long int emb_dim, const long int sequence, const float droprate, const long int n_heads, const bool qkv_bias);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TransformerBlock);


// -------------------------------------------------
// struct{GPT3Impl}(nn::Module)
// -------------------------------------------------
struct GPT3Impl : nn::Module{
private:
    nn::Embedding token_emb{nullptr}, pos_emb{nullptr};
    nn::Dropout drop_emb{nullptr};
    nn::Sequential transformer;
    nn::LayerNorm final_norm{nullptr};
    nn::Linear out_head{nullptr};
public:
    GPT3Impl(){}
    GPT3Impl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(GPT3);


#endif