#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <chrono>                      // std::chrono
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
#include <tokenizers_cpp.h>            // Tokenizer
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // GPT2
#include "datasets.hpp"                // datasets::TextFolder
#include "dataloader.hpp"              // DataLoader::TextFolder

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;
using tokenizers::Tokenizer;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, GPT2 &model, std::shared_ptr<tokenizers::Tokenizer> &tokenizer){

    // (0) Initialization and Declaration
    float ave_loss;
    double seconds, ave_time;
    size_t i;
    std::string path, result_dir;
    std::string dataroot;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor> data;
    torch::Tensor input, output, gt, loss;
    datasets::TextFolder dataset;
    DataLoader::TextFolder dataloader;

    // (1) Get Test Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_dir"].as<std::string>();
    dataset = datasets::TextFolder(dataroot, tokenizer, vm["sequence"].as<size_t>(), vm["stride"].as<size_t>(), vm["endoftext"].as<int>(), vm["padding"].as<int>());
    dataloader = DataLoader::TextFolder(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test data : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = Loss();

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_time = 0.0;
    i = 0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        input = std::get<0>(data).to(device);
        gt = std::get<1>(data).to(device);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        output = model->forward(input);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(output, gt);
        
        ave_loss += loss.item<float>();
        ave_time += seconds;

        std::cout << '<' << i << "> loss:" << loss.item<float>() << std::endl;
        ofs << '<' << i << "> loss:" << loss.item<float>() << std::endl;
        i++;

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> " << "loss:" << ave_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> " << "loss:" << ave_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
