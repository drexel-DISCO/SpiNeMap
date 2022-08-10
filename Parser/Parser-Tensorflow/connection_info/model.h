#ifndef __MODEL_H__
#define __MODEL_H__

#include <boost/multiprecision/cpp_int.hpp> 
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>      // std::pair
#include <vector>

#include "hdf5.h"

#include <boost/property_tree/ptree.hpp>

namespace NCC
{
namespace NCC_FrontEnd
{

#define MAX_NAME 1024
class Model
{
  protected:
    // TODO, capture more information from json file.
    class Layer
    {
      public:
        // TODO, more layer type:
        // https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
	// https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
	// https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/
        // Batch-normalization: https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network
        enum class Layer_Type : int
        {
            Input, // the input layer
            Conv2D, // convolution layer
            Activation,
            Padding,
            BatchNormalization,
            Dropout,
            MaxPooling2D, // max pooling layer
            AveragePooling2D,
            GlobalPooling2D,
            Flatten, // flatten layer
            Dense, // dense (fully-connected) layer
            Ignore,
            Concatenate,
            Add,
            MAX
        }layer_type = Layer_Type::MAX;

        enum Cost : int
        {
            MAC         = 1,
            DIVIDE      = 2,
            COMPARE     = 1,
            ADDSUB      = 1,
            MULT        = 2,
            INIT        = 1, //For Padding layers
            MAX         = 0
        }cost = Cost::MAX;

        // Lacking:
        // ZeroPadding2D, Concatenate for DenseNet

        Layer() {}
        Layer(std::string &_name, Layer_Type &_type) : name(_name), layer_type(_type) {}

        void setStrides(std::vector<unsigned> &_strides)
        {
            strides = _strides;
        }
        void setKernelSize(std::vector<unsigned> &_kernel_sz)
        {
            kernel_sz = _kernel_sz;
        }
        void setBeta(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            beta_dims = dims;
            beta = data;
        }
        void setGamma(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            gamma_dims = dims;
            gamma = data;
        }
        void setMovingMean(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            moving_mean_dims = dims;
            moving_mean = data;
        }
        void setMovingVariance(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            moving_variance_dims = dims;
            moving_variance = data;
        }
        void setOutputDim(std::vector<unsigned> &_dims)
        {
            output_dims = _dims;
        }
        void setDepth(uint64_t _depth) { depth = _depth; }
        int getDepth() { return depth; }

        struct Computations{
            int num_MAC       = 0;
            int num_Div       = 0;
            int num_Compare   = 0;
            int num_AddSub    = 0;
            int num_Mult      = 0;
            int num_Init      = 0;
        } comp;

        std::string name; // Name of the layer

        std::vector<std::string> inbound_layers;
        std::vector<std::string> outbound_layers;
        uint64_t depth = -1;

        // weights/biases for CONV2D/Dense
        // std::vector<unsigned> w_dims; // dims of the weights
        // std::vector<float> weights; // all the weight
        // std::vector<unsigned> b_dims; // dims of the bias
        // std::vector<float> bias; // all the bias

        // Padding type of the layer, used for CONV2D
        enum class Padding_Type : int
        {
            same,
            valid
        }padding_type = Padding_Type::valid;
        // strides, used for CONV2D/MaxPooling/AveragePooling
        std::vector<unsigned> strides;
        // kernel size, used for CONV2D
        std::vector<unsigned> kernel_sz;
        std::vector<unsigned> padding;
        // Num filters for CONV2D
        unsigned num_filter;

        // TODO, need to extract more information
        // For batch-normalization
        std::vector<unsigned> beta_dims;
        std::vector<float> beta;
        std::vector<unsigned> gamma_dims;
        std::vector<float> gamma;
        std::vector<unsigned> moving_mean_dims;
        std::vector<float> moving_mean;
        std::vector<unsigned> moving_variance_dims;
        std::vector<float> moving_variance;

        // The output of the layer, including the dimension and the output neurons IDs.
        std::vector<unsigned> output_dims = {0, 0, 0}; // dimension of output
        std::vector<uint64_t> output_neuron_ids;
        uint64_t num_out_tok; // For sdf representation
        uint64_t compute_time=0;
    };

    // Model - Architecture
    class Architecture
    {
      protected:
        std::vector<Layer> layers;
        int maxDepth;

      protected:
        struct ConnEntry
        {
            ConnEntry(uint64_t _id, float _w)
            {
                out_neurons_ids.push_back(_id);
            }

            std::vector<uint64_t> out_neurons_ids;
        };
        std::unordered_map<uint64_t, ConnEntry> connections;

        void layerOutput();

        std::ofstream conns_output;

      public:
        Architecture() {}

        void addLayer(std::string &_name, Layer::Layer_Type &_type)
        {
            layers.emplace_back(_name, _type);
        }

        Layer& getLayer(const std::string &name)
        {
            for (auto &layer : layers)
            {
                if (layer.name == name) { return layer; }
            }
            std::cout << name << "\n";
            std::cerr << "Error: layer is not found.\n";
            exit(0);
        }

        void printLayerConns(std::string &out_root);
        void printSdfRep(std::string &out_root);

        int getmaxLayerDepth() { return maxDepth; };
        void setMaxLayerDepth();

        void printLayers() // Only used for small network debuggings.
        {
            for (auto &layer : layers)
            {
                auto name = layer.name;
                auto type = layer.layer_type;

                std::cout << "Layer name: " << name << "; ";
                if (type == Layer::Layer_Type::Input) 
                { std::cout << "Layer type: Input"; }
		        else if (type == Layer::Layer_Type::Conv2D) 
                { std::cout << "Layer type: Conv2D"; }
                else if (type == Layer::Layer_Type::Activation) 
                { std::cout << "Layer type: Activation"; }
                else if (type == Layer::Layer_Type::BatchNormalization) 
                { std::cout << "Layer type: BatchNormalization"; }
                else if (type == Layer::Layer_Type::Dropout) 
                { std::cout << "Layer type: Dropout"; }
                else if (type == Layer::Layer_Type::MaxPooling2D)
                { std::cout << "Layer type: MaxPooling2D"; }
                else if (type == Layer::Layer_Type::AveragePooling2D)
                { std::cout << "Layer type: AveragePooling2D"; }
                else if (type == Layer::Layer_Type::Flatten) 
                { std::cout << "Layer type: Flatten"; }
                else if (type == Layer::Layer_Type::Dense) 
                { std::cout << "Layer type: Dense"; }
                else if (type == Layer::Layer_Type::Ignore) 
                { std::cout << "Layer type: Ignore"; }
                else { std::cerr << "Error: unsupported layer type\n"; exit(0); }
                std::cout << "\n";

                auto &output_dims = layer.output_dims;
                std::cout << "Output shape: ";
                for (auto dim : output_dims) { std::cout << dim << " "; }
                std::cout << "\n";
            }
        }
        void labelLayerWithDepth(uint64_t starting_depth, std::set<std::string>&);
        void outputLayerDepthIR(const std::string&);
        void printLayerConnDepth(const std::string&);
        std::pair<uint64_t, uint64_t> getIrregularMetric();
    };

    Architecture arch;

  public:
    Model(std::string &arch_file, std::string &weight_file)
    {
        loadArch(arch_file);
    }

    Model(std::string &arch_file)
    {
        loadArch(arch_file);
    }

    void printLayers() { arch.printLayers(); }
    
    void printSdfRep(std::string &out_root) {arch.printSdfRep(out_root); }
    void printLayerConns(std::string &out_root) {arch.printLayerConns(out_root); }
    void outputLayerDepthIR(std::string &out_file) {arch.outputLayerDepthIR(out_file);}
    std::pair<uint64_t, uint64_t> getIrregularMetric() { return arch.getIrregularMetric();}
    uint64_t getMaxDepth() {return (uint64_t)arch.getmaxLayerDepth(); }

  protected:
    void loadArch(std::string &arch_file);
};
}
}

#endif
