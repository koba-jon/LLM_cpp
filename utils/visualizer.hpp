#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include <string>
#include <tuple>
#include <vector>
#include <utility>
// For External Library
#include <torch/torch.h>


// -----------------------------------
// namespace{visualizer}
// -----------------------------------
namespace visualizer{
    
    // -----------------------------------
    // namespace{visualizer} -> class{graph}
    // -----------------------------------
    class graph{
    private:
        bool flag;
        std::string dir, data_dir;
        std::string gname;
        std::string graph_fname, data_fname;
        std::vector<std::string> label;
    public:
        graph(){}
        graph(const std::string dir_, const std::string gname_, const std::vector<std::string> label_);
        void plot(const float base, const std::vector<float> value);
    };

}

#endif