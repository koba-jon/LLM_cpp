#include <vector>
#include <limits>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
using torch::indexing::Slice;


// ----------------------------------------------------------------------
// struct{FeedForwardImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
FeedForwardImpl::FeedForwardImpl(const size_t emb_dim){
    this->layers = nn::Sequential(
        nn::Linear(emb_dim, 4 * emb_dim),
        nn::GELU(),
        nn::Linear(4 * emb_dim, emb_dim)
    );
    register_module("layers", this->layers);
}


// ----------------------------------------------------------------------
// struct{FeedForwardImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor FeedForwardImpl::forward(torch::Tensor x){
    return this->layers->forward(x);
}


// ----------------------------------------------------------------------
// struct{MultiHeadAttentionImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MultiHeadAttentionImpl::MultiHeadAttentionImpl(const long int d_in, const long int d_out, const long int sequence, const float droprate, const long int n_heads_, const bool qkv_bias){

    this->n_heads = n_heads_;
    this->head_dim = d_out / n_heads;

    this->W_key = register_module("W_key", nn::Linear(nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    this->W_query = register_module("W_query", nn::Linear(nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    this->W_value = register_module("W_value", nn::Linear(nn::LinearOptions(d_in, d_out).bias(qkv_bias)));

    this->out_proj = register_module("out_proj", nn::Linear(nn::LinearOptions(d_out, d_out)));
    this->dropout = register_module("dropout", nn::Dropout(droprate));
    this->mask = register_buffer("mask", torch::triu(torch::ones({sequence, sequence}), /*diagonal=*/1).to(torch::kBool));

}


// ----------------------------------------------------------------------
// struct{MultiHeadAttentionImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x){

    torch::Tensor keys, queries, values, attn_scores, mask_bool, attn_weights, context_vec;

    keys = this->W_key->forward(x);  // {N,S,DI} ==> {N,S,DO}
    keys = keys.view({x.size(0), x.size(1), this->n_heads, this->head_dim}).transpose(1, 2);  // {N,H,S,HD}

    queries = this->W_query->forward(x);  // {N,S,DI} ==> {N,S,DO}
    queries = queries.view({x.size(0), x.size(1), this->n_heads, this->head_dim}).transpose(1, 2);  // {N,H,S,HD}

    values = this->W_value->forward(x);  // {N,S,DI} ==> {N,S,DO}
    values = values.view({x.size(0), x.size(1), this->n_heads, this->head_dim}).transpose(1, 2);  // {N,H,S,HD}

    attn_scores = queries.matmul(keys.transpose(2, 3));  // {N,H,S,S}
    mask_bool = this->mask.index({Slice(torch::indexing::None, x.size(1)), Slice(torch::indexing::None, x.size(1))});  // {S,S}
    attn_scores = attn_scores.masked_fill(mask_bool, -std::numeric_limits<float>::infinity());  // {N,H,S,S}
    attn_weights = torch::softmax((attn_scores / std::sqrt(keys.size(3))), -1);  // {N,H,S,S}
    attn_weights = this->dropout->forward(attn_weights);  // {N,H,S,S}
    context_vec = attn_weights.matmul(values).transpose(1, 2);  // {N,S,H,HD}
    context_vec = context_vec.contiguous().view({x.size(0), x.size(1), -1});  // {N,S,DO}
    context_vec = this->out_proj->forward(context_vec);  // {N,S,DO}

    return context_vec;

}


// ----------------------------------------------------------------------
// struct{TransformerBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
TransformerBlockImpl::TransformerBlockImpl(const long int emb_dim, const long int sequence, const float droprate, const long int n_heads, const bool qkv_bias){
    this->attn = register_module("attn", MultiHeadAttention(emb_dim, emb_dim, sequence, droprate, n_heads, qkv_bias));
    this->ff = register_module("ff", FeedForward(emb_dim));
    this->norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({emb_dim})));
    this->norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({emb_dim})));
    this->drop_shortcut = register_module("drop_shortcut", nn::Dropout(droprate));
}


// ----------------------------------------------------------------------
// struct{TransformerBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor TransformerBlockImpl::forward(torch::Tensor x){

    torch::Tensor shortcut;

    shortcut = x;
    x = this->norm1->forward(x);
    x = this->attn->forward(x);
    x = this->drop_shortcut->forward(x);
    x = x + shortcut;

    shortcut = x;
    x = this->norm2->forward(x);
    x = this->ff->forward(x);
    x = this->drop_shortcut->forward(x);
    x = x + shortcut;

    return x;

}


// ----------------------------------------------------------------------
// struct{GPT2Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
GPT2Impl::GPT2Impl(po::variables_map &vm){

    this->token_emb = register_module("token_emb", nn::Embedding(vm["vocab_size"].as<size_t>(), vm["emb_dim"].as<size_t>()));
    this->pos_emb = register_module("pos_emb", nn::Embedding(vm["sequence"].as<size_t>(), vm["emb_dim"].as<size_t>()));
    this->drop_emb = register_module("drop_emb", nn::Dropout(vm["droprate"].as<float>()));

    for (size_t i = 0; i < vm["n_layers"].as<size_t>(); i++){
        this->transformer->push_back(TransformerBlock(vm["emb_dim"].as<size_t>(), vm["sequence"].as<size_t>(), vm["droprate"].as<float>(), vm["n_heads"].as<size_t>(), vm["qkv_bias"].as<bool>()));
    }
    register_module("transformer", this->transformer);

    this->final_norm = register_module("final_norm", nn::LayerNorm(nn::LayerNormOptions({(long int)vm["emb_dim"].as<size_t>()})));
    this->out_head = register_module("out_head", nn::Linear(nn::LinearOptions(vm["emb_dim"].as<size_t>(), vm["vocab_size"].as<size_t>()).bias(false)));
    
}


// ----------------------------------------------------------------------
// struct{GPT2Impl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor GPT2Impl::forward(torch::Tensor x){

    torch::Tensor token_embeds, pos_embeds, out;

    token_embeds = this->token_emb->forward(x);
    pos_embeds = this->pos_emb->forward(torch::arange(x.size(1)).to(x.device()));
    x = token_embeds + pos_embeds;
    x = this->drop_emb->forward(x);
    x = this->transformer->forward(x);
    x = this->final_norm->forward(x);
    out = this->out_head->forward(x);

    return out;

}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}

