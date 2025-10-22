#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <omp.h>
// For Original Header
#include "datasets.hpp"
#include "dataloader.hpp"


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolder} -> constructor
// --------------------------------------------------------------------
DataLoader::TextFolder::TextFolder(datasets::TextFolder &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_, const bool pin_memory_, const bool drop_last_){

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;
    this->pin_memory = pin_memory_;
    this->drop_last = drop_last_;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++){
        this->idx.at(i) = i;
    }

    this->count = 0;
    if (this->drop_last){
        this->count_max = std::floor((float)this->size / (float)this->batch_size);
        if ((this->count_max == 0) && (this->size > 0)){
            this->count_max = 1;
        }
    }
    else{
        this->count_max = std::ceil((float)this->size / (float)this->batch_size);
    }

    this->mt.seed(std::rand());

}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolder} -> operator
// --------------------------------------------------------------------
bool DataLoader::TextFolder::operator()(std::tuple<torch::Tensor, torch::Tensor> &data){

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, data2, tensor;
    std::tuple<torch::Tensor, torch::Tensor> *data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle){
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if(this->count == this->count_max){
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, torch::Tensor>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0){
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else{
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]).unsqueeze(0);
    data2 = std::get<1>(data_before[0]).unsqueeze(0);
    for (i = 1; i < mini_batch_size; i++){
        data1 = torch::cat({data1, std::get<0>(data_before[i]).unsqueeze(0)}, /*dim=*/0);  // {i,D} + {1,D} ===> {i+1,D}
        data2 = torch::cat({data2, std::get<1>(data_before[i]).unsqueeze(0)}, /*dim=*/0);  // {i,D} + {1,D} ===> {i+1,D}
    }
    data1 = data1.contiguous().detach().clone();
    data2 = data2.contiguous().detach().clone();
    
    // (4) Pin
    if (this->pin_memory){
        data1 = data1.pin_memory();
        data2 = data2.pin_memory();
    }

    // Post Processing
    this->count++;
    data = {data1, data2};  // {N,D} (data), {N} (fnames)
    delete[] data_before;

    // End Processing
    return true;
    
}


// --------------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolder} -> function{reset}
// --------------------------------------------------------------------------
void DataLoader::TextFolder::reset(){
    this->count = 0;
    return;
}


// ---------------------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolder} -> function{get_count_max}
// ---------------------------------------------------------------------------------
size_t DataLoader::TextFolder::get_count_max(){
    return this->count_max;
}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolderPredictWithPaths} -> constructor
// --------------------------------------------------------------------
DataLoader::TextFolderPredictWithPaths::TextFolderPredictWithPaths(datasets::TextFolderPredictWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_, const bool pin_memory_, const bool drop_last_){

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;
    this->pin_memory = pin_memory_;
    this->drop_last = drop_last_;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++){
        this->idx.at(i) = i;
    }

    this->count = 0;
    if (this->drop_last){
        this->count_max = std::floor((float)this->size / (float)this->batch_size);
        if ((this->count_max == 0) && (this->size > 0)){
            this->count_max = 1;
        }
    }
    else{
        this->count_max = std::ceil((float)this->size / (float)this->batch_size);
    }

    this->mt.seed(std::rand());

}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolderPredictWithPaths} -> operator
// --------------------------------------------------------------------
bool DataLoader::TextFolderPredictWithPaths::operator()(std::tuple<torch::Tensor, std::vector<std::string>> &data){

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, tensor;
    std::vector<std::string> data2;
    std::tuple<torch::Tensor, std::string> *data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle){
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if(this->count == this->count_max){
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, std::string>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0){
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else{
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]).unsqueeze(0);
    data2.push_back(std::get<1>(data_before[0]));
    for (i = 1; i < mini_batch_size; i++){
        data1 = torch::cat({data1, std::get<0>(data_before[i]).unsqueeze(0)}, /*dim=*/0);  // {i,D} + {1,D} ===> {i+1,D}
        data2.push_back(std::get<1>(data_before[i]));
    }
    data1 = data1.contiguous().detach().clone();
    
    // (4) Pin
    if (this->pin_memory){
        data1 = data1.pin_memory();
    }

    // Post Processing
    this->count++;
    data = {data1, data2};  // {N,D} (data), {N} (fnames)
    delete[] data_before;

    // End Processing
    return true;
    
}


// --------------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolderPredictWithPaths} -> function{reset}
// --------------------------------------------------------------------------
void DataLoader::TextFolderPredictWithPaths::reset(){
    this->count = 0;
    return;
}


// ---------------------------------------------------------------------------------
// namespace{DataLoader} -> class{TextFolderPredictWithPaths} -> function{get_count_max}
// ---------------------------------------------------------------------------------
size_t DataLoader::TextFolderPredictWithPaths::get_count_max(){
    return this->count_max;
}
