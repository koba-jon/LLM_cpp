#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <utility>                     // std::pair
#include <tuple>                       // std::tuple
#include <algorithm>                   // std::min
// For External Library
#include <torch/torch.h>               // torch
#include <tokenizers_cpp.h>            // Tokenizer
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // GPT3
#include "datasets.hpp"                // datasets::TextFolderPredictWithPaths
#include "dataloader.hpp"              // DataLoader::TextFolderPredictWithPaths

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;
using torch::indexing::Slice;
using tokenizers::Tokenizer;


// ---------------------
// Prediction Function
// ---------------------
void predict(po::variables_map &vm, torch::Device &device, GPT3 &model, std::shared_ptr<tokenizers::Tokenizer> &tokenizer){

    // (0) Initialization and Declaration
    std::string path, result_dir;
    std::string dataroot;
    std::ofstream ofs;
    int id;
    std::vector<int> ids;
    std::string text;
    std::tuple<torch::Tensor, std::vector<std::string>> data;
    torch::Tensor input, output, topk_logits, topk_indices, masked, probs, next_id;
    datasets::TextFolderPredictWithPaths dataset;
    DataLoader::TextFolderPredictWithPaths dataloader;

    // (1) Get Prediction Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["predict_dir"].as<std::string>();
    dataset = datasets::TextFolderPredictWithPaths(dataroot, tokenizer);
    dataloader = DataLoader::TextFolderPredictWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total prediction data : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["predict_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (3) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["predict_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    while (dataloader(data)){

        input = std::get<0>(data).to(device);
        ofs.open(result_dir + "/" + std::get<1>(data)[0], std::ios::out);

        ids = std::vector<int>(input.numel());
        for (long int i = 0; i < input.numel(); i++){
            ids[i] = input.index({0, i}).item<int>();
        }
        text = tokenizer->Decode(ids);
        std::cout << text << std::flush;
        ofs << text << std::flush;

        for (size_t i = 0; i < vm["predict_token"].as<size_t>(); i++){

            if ((size_t)input.size(1) > vm["sequence"].as<size_t>()){
                input = input.index({Slice(), Slice(-vm["sequence"].as<size_t>(), torch::indexing::None)});
            }

            output = model->forward(input);  // {1,S} ===> {1,S,V}
            output = output.index({Slice(), -1, Slice()});  // {1,S,V} ===> {1,V}
            output = output / vm["temperature"].as<float>();  // {1,V}
            std::tie(topk_logits, topk_indices) = torch::topk(output, std::min(output.size(1), (long int)vm["topk"].as<size_t>()), /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
            masked = torch::full_like(output, -std::numeric_limits<float>::infinity());  // {1,V}
            masked.scatter_(-1, topk_indices, topk_logits);
            probs = torch::softmax(masked, -1);  // {1,V}
            next_id = torch::multinomial(probs, 1);  // {1,V} ===> {1,1}

            id = next_id.index({0, 0}).item<int>();
            if (id == vm["endoftext"].as<int>()) break;
            text = tokenizer->Decode(std::vector<int>{id});

            std::cout << text << std::flush;
            ofs << text << std::flush;

            input = torch::cat({input, next_id}, 1);

        }
        std::cout << std::endl;
        ofs << std::endl;

        ofs.close();

    }

    // End Processing
    return;

}