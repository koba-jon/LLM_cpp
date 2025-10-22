#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "visualizer.hpp"

// Define Namespace
namespace fs = std::filesystem;


// ----------------------------------------------------------
// namespace{visualizer} -> class{graph} -> constructor
// ----------------------------------------------------------
visualizer::graph::graph(const std::string dir_, const std::string gname_, const std::vector<std::string> label_){
    this->flag = false;
    this->dir = dir_;
    this->data_dir = this->dir + "/data";
    this->gname = gname_;
    this->graph_fname= this->dir + '/' + this->gname + ".png";
    this->data_fname= this->data_dir + '/' + this->gname + ".dat";
    this->label = label_;
    fs::create_directories(this->dir);
    fs::create_directories(this->data_dir);
}


// ----------------------------------------------------------
// namespace{visualizer} -> class{graph} -> function{plot}
// ----------------------------------------------------------
void visualizer::graph::plot(const float base, const std::vector<float> value){

    // (1) Value Output
    std::ofstream ofs(this->data_fname, std::ios::app);
    ofs << base << std::flush;
    for (auto &v : value){
        ofs << ' ' << v << std::flush;
    }
    ofs << std::endl;
    ofs.close();

    // (2) Graph Output
    if (this->flag){
        FILE* gp;
        gp = popen("gnuplot -persist", "w");
        fprintf(gp, "set terminal png\n");
        fprintf(gp, "set output '%s'\n", this->graph_fname.c_str());
        fprintf(gp, "plot ");
        for (size_t i = 0; i < this->label.size() - 1; i++){
            fprintf(gp, "'%s' using 1:%zu ti '%s' with lines,", this->data_fname.c_str(), i + 2, this->label.at(i).c_str());
        }
        fprintf(gp, "'%s' using 1:%zu ti '%s' with lines\n", this->data_fname.c_str(), this->label.size() + 1, this->label.at(this->label.size() - 1).c_str());
        pclose(gp);
    }

    // (3) Setting for after the Second Time
    this->flag = true;

    // End Processing
    return;

}