#include <iostream>
#include <string>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(int ignore_index){
    this->criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().ignore_index(ignore_index).reduction(torch::kMean));
}


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor input, torch::Tensor target){
    input = input.view({-1, input.size(2)});
    target = target.view({-1});
    return criterion(input, target);
}
