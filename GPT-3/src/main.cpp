#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand
// For External Library
#include <torch/torch.h>               // torch
#include <tokenizers_cpp.h>            // Tokenizer
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // GPT3

// Define Namespace and class
namespace fs = std::filesystem;
namespace po = boost::program_options;
using tokenizers::Tokenizer;

// Function Prototype
void train(po::variables_map &vm, torch::Device &device, GPT3 &model, std::shared_ptr<tokenizers::Tokenizer> &tokenizer);
void test(po::variables_map &vm, torch::Device &device, GPT3 &model, std::shared_ptr<tokenizers::Tokenizer> &tokenizer);
void predict(po::variables_map &vm, torch::Device &device, GPT3 &model, std::shared_ptr<tokenizers::Tokenizer> &tokenizer);
void question(po::variables_map &vm, torch::Device &device, GPT3 &model, std::shared_ptr<tokenizers::Tokenizer> &tokenizer);
torch::Device Set_Device(po::variables_map &vm);
std::string LoadBytesFromFile(const std::string& path);
template <typename T> void Set_Model_Params(po::variables_map &vm, T &model, const std::string name);
void Set_Options(po::variables_map &vm, int argc, const char *argv[], po::options_description &args, const std::string mode);


// -----------------------------------
// 0. Argument Function
// -----------------------------------
po::options_description parse_arguments(){

    po::options_description args("Options", 200, 30);
    
    args.add_options()

        // (1) Define for General Parameter
        ("help", "produce help message")
        ("dataset", po::value<std::string>(), "dataset name")
        ("tokenizer", po::value<std::string>()->default_value("dist/tokenizer.json"), "tokenizer file name")
        ("vocab_size", po::value<size_t>()->default_value(50277), "vocabulary size")
        ("sequence", po::value<size_t>()->default_value(2048), "maximum sequence length")
        ("stride", po::value<size_t>()->default_value(1), "stride of text sequence")
        ("endoftext", po::value<int>()->default_value(0), "id of <|endoftext|>")
        ("padding", po::value<int>()->default_value(1), "id of <|padding|>")
        ("temperature", po::value<float>()->default_value(0.7), "sampling temperature for prediction")
        ("topk", po::value<size_t>()->default_value(50), "top-k for prediction")
        ("gpu_id", po::value<int>()->default_value(0), "cuda device : 'x=-1' is cpu device")
        ("seed_random", po::value<bool>()->default_value(false), "whether to make the seed of random number in a random")
        ("seed", po::value<int>()->default_value(0), "seed of random number")

        // (2) Define for Training
        ("train", po::value<bool>()->default_value(false), "training mode on/off")
        ("train_dir", po::value<std::string>()->default_value("train"), "training data directory : ./datasets/<dataset>/<train_dir>/<data files>")
        ("epochs", po::value<size_t>()->default_value(200), "training total epoch")
        ("batch_size", po::value<size_t>()->default_value(8), "training batch size")
        ("train_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning")
        ("save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer")

        // (3) Define for Validation
        ("valid", po::value<bool>()->default_value(false), "validation mode on/off")
        ("valid_dir", po::value<std::string>()->default_value("valid"), "validation data directory : ./datasets/<dataset>/<valid_dir>/<data files>")
        ("valid_batch_size", po::value<size_t>()->default_value(1), "validation batch size")
        ("valid_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch")

        // (4) Define for Test
        ("test", po::value<bool>()->default_value(false), "test mode on/off")
        ("test_dir", po::value<std::string>()->default_value("test"), "test data directory : ./datasets/<dataset>/<test_dir>/<data files>")
        ("test_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for testing")
        ("test_result_dir", po::value<std::string>()->default_value("test_result"), "test result directory : ./<test_result_dir>")

        // (5) Define for Prediction
        ("predict", po::value<bool>()->default_value(false), "prediction mode on/off")
        ("predict_dir", po::value<std::string>()->default_value("predict"), "prediction data directory : ./datasets/<dataset>/<predict_dir>/<data files>")
        ("predict_token", po::value<size_t>()->default_value(3000), "the number of token for prediction")
        ("predict_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for prediction")
        ("predict_result_dir", po::value<std::string>()->default_value("predict_result"), "prediction result directory : ./<predict_result_dir>")

        // (6) Define for Question
        ("question", po::value<bool>()->default_value(false), "question mode on/off")
        ("question_token", po::value<size_t>()->default_value(1000), "the number of token for question")
        ("question_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for question")
        ("question_result_dir", po::value<std::string>()->default_value("question_result"), "question result directory : ./<question_result_dir>")

        // (7) Define for Network Parameter
        ("lr", po::value<float>()->default_value(2e-5), "learning rate")
        ("beta1", po::value<float>()->default_value(0.9), "beta 1 in Adam of optimizer method")
        ("beta2", po::value<float>()->default_value(0.999), "beta 2 in Adam of optimizer method")
        ("emb_dim", po::value<size_t>()->default_value(12288), "embedding feature dimensions")
        ("n_heads", po::value<size_t>()->default_value(96), "the number of heads")
        ("n_layers", po::value<size_t>()->default_value(96), "the number of layers")
        ("droprate", po::value<float>()->default_value(0.0), "the rate of dropout")
        ("qkv_bias", po::value<bool>()->default_value(true), "qkv bias")

    ;
    
    // End Processing
    return args;
}


// -----------------------------------
// 1. Main Function
// -----------------------------------
int main(int argc, const char *argv[]){

    // (1) Extract Arguments
    po::options_description args = parse_arguments();
    po::variables_map vm{};
    po::store(po::parse_command_line(argc, argv, args), vm);
    po::notify(vm);
    if (vm.empty() || vm.count("help")){
        std::cout << args << std::endl;
        return 1;
    }
    
    // (2) Select Device
    torch::Device device = Set_Device(vm);
    std::cout << "using device = " << device << std::endl;

    // (3) Set Seed
    if (vm["seed_random"].as<bool>()){
        std::random_device rd;
        std::srand(rd());
        torch::manual_seed(std::rand());
        torch::globalContext().setDeterministicCuDNN(false);
        torch::globalContext().setBenchmarkCuDNN(true);
    }
    else{
        std::srand(vm["seed"].as<int>());
        torch::manual_seed(std::rand());
        torch::globalContext().setDeterministicCuDNN(true);
        torch::globalContext().setBenchmarkCuDNN(false);
    }

    // (4) Set tokenizer
    auto blob = LoadBytesFromFile(vm["tokenizer"].as<std::string>());
    std::shared_ptr<tokenizers::Tokenizer> tokenizer = Tokenizer::FromBlobJSON(blob);
    
    // (5) Define Network
    GPT3 gpt3(vm);
    gpt3->to(device);
    
    // (6) Make Directories
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>();
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(vm, gpt3, "GPT-3");

    // (8.1) Training Phase
    if (vm["train"].as<bool>()){
        Set_Options(vm, argc, argv, args, "train");
        train(vm, device, gpt3, tokenizer);
    }

    // (8.2) Test Phase
    if (vm["test"].as<bool>()){
        Set_Options(vm, argc, argv, args, "test");
        test(vm, device, gpt3, tokenizer);
    }

    // (8.3) Prediction Phase
    if (vm["predict"].as<bool>()){
        Set_Options(vm, argc, argv, args, "predict");
        predict(vm, device, gpt3, tokenizer);
    }

    // (8.4) Question Phase
    if (vm["question"].as<bool>()){
        Set_Options(vm, argc, argv, args, "question");
        question(vm, device, gpt3, tokenizer);
    }

    // End Processing
    return 0;

}


// -----------------------------------
// 2. Device Setting Function
// -----------------------------------
torch::Device Set_Device(po::variables_map &vm){

    // (1) GPU Type
    int gpu_id = vm["gpu_id"].as<int>();
    if (torch::cuda::is_available() && gpu_id>=0){
        torch::Device device(torch::kCUDA, gpu_id);
        return device;
    }

    // (2) CPU Type
    torch::Device device(torch::kCPU);
    return device;

}


// -----------------------------------
// 3. Loading File Function
// -----------------------------------
std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << path << std::endl;
        std::exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}


// -----------------------------------
// 4. Model Parameters Setting Function
// -----------------------------------
template <typename T>
void Set_Model_Params(po::variables_map &vm, T &model, const std::string name){

    // (1) Make Directory
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>() + "/model_params/";
    fs::create_directories(dir);

    // (2.1) File Open
    std::string fname = dir + name + ".txt";
    std::ofstream ofs(fname);

    // (2.2) Calculation of Parameters
    size_t num_params = 0;
    for (auto param : model->parameters()){
        num_params += param.numel();
    }
    ofs << "Total number of parameters : " << (float)num_params/1e6f << "M" << std::endl << std::endl;
    ofs << model << std::endl;

    // (2.3) File Close
    ofs.close();

    // End Processing
    return;

}


// -----------------------------------
// 5. Options Setting Function
// -----------------------------------
void Set_Options(po::variables_map &vm, int argc, const char *argv[], po::options_description &args, const std::string mode){

    // (1) Make Directory
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>() + "/options/";
    fs::create_directories(dir);

    // (2) Terminal Output
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << args << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // (3.1) File Open
    std::string fname = dir + mode + ".txt";
    std::ofstream ofs(fname, std::ios::app);

    // (3.2) Arguments Output
    ofs << "--------------------------------------------" << std::endl;
    ofs << "Command Line Arguments:" << std::endl;
    for (int i = 1; i < argc; i++){
        if (i % 2 == 1){
            ofs << "  " << argv[i] << '\t' << std::flush;
        }
        else{
            ofs << argv[i] << std::endl;
        }
    }
    ofs << "--------------------------------------------" << std::endl;
    ofs << args << std::endl;
    ofs << "--------------------------------------------" << std::endl << std::endl;

    // (3.3) File Close
    ofs.close();

    // End Processing
    return;

}
