#ifndef __MODEL_H__
#define __MODEL_H__

#include <boost/multiprecision/cpp_int.hpp> 
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>      // std::pair
#include <vector>
#include <set>

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
            GlobalAveragePooling2D,
            Flatten, // flatten layer
            Dense, // dense (fully-connected) layer
            Ignore,
            Concatenate,
            Add,
            ZeroPadding2D,
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
        } cost = Cost::MAX;

        // Lacking:
        // ZeroPadding2D, Concatenate for DenseNet

        Layer() {}
        Layer(std::string &_name, Layer_Type &_type) : name(_name), layer_type(_type) {}

        void setWeights(std::vector<unsigned> &_w_dims,
                        std::vector<float> &_weights)
        {
            w_dims = _w_dims;
            weights = _weights;
        }
        void setBiases(std::vector<unsigned> &_b_dims,
                       std::vector<float> &_bias)
        {
            b_dims = _b_dims;
            bias = _bias;
        }
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
        std::vector<unsigned> w_dims; // dims of the weights
        std::vector<float> weights; // all the weight
        std::vector<unsigned> b_dims; // dims of the bias
        std::vector<float> bias; // all the bias

        // Padding type of the layer, used for CONV2D
        enum class Padding_Type : int
        {
            same,
            valid
        }padding_type = Padding_Type::valid;
        // strides, used for CONV2D/MaxPooling/AveragePooling
        std::vector<unsigned> strides;
        //kernel size, used for CONV2D
        std::vector<unsigned> kernel_sz;
        std::vector<unsigned> padding;
        //Num filters for CONV2D
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
        uint64_t compute_time=0; //For sdf repre
    };

    // Model - Architecture
    class Architecture
    {
      protected:
        std::vector<Layer> layers;

      protected:
        struct ConnEntry
        {
            ConnEntry(uint64_t _id, float _w, std::string& _str)
            {
                out_neurons_ids.push_back(_id);
                out_layer_name.push_back(_str);
                weights.push_back(_w);
            }

            std::vector<uint64_t> out_neurons_ids;
            std::vector<std::string> out_layer_name;
            std::vector<float> weights;
        };
        std::unordered_map<uint64_t, ConnEntry> connections;

        void connToConv(unsigned, unsigned);
        void connToConvPadding(unsigned, unsigned);
        void connToAct(unsigned, unsigned);
        void connToNorm(unsigned, unsigned);
        void connToDrop(unsigned, unsigned);
        void connToPool(unsigned, unsigned);
        void connToFlat(unsigned, unsigned);
        void connToDense(unsigned, unsigned);
        void connToAdd(unsigned, unsigned);
        void connToPadding(unsigned, unsigned);

        void layerOutput();

        std::ofstream conns_output;
        std::ofstream weights_output;

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
            std::cerr << "Error: "<< name<< " layer is not found.\n";
            exit(0);
        }

        void connector();
        void connectLayers();

        void printConns(std::string &out_root);
        void printLayerConns(std::string &out_root);
        void printSdfRep(std::string &out_root);

        void setOutRoot(std::string &out_root);

        void printLayers() // Only used for small network debuggings.
        {
            for (auto &layer : layers)
            {
                auto name = layer.name;
                auto type = layer.layer_type;
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
                else if (type == Layer::Layer_Type::Concatenate) 
                { std::cout << "Layer type: Concatenate"; }
                else if (type == Layer::Layer_Type::Add) 
                { std::cout << "Layer type: Add"; }
                else if (type == Layer::Layer_Type::Dense) 
                { std::cout << "Layer type: Dense"; }
                else if (type == Layer::Layer_Type::Ignore) 
                { std::cout << "Layer type: Ignore"; }
                else { std::cerr << "Error: unsupported layer type\n"; exit(0); }
                std::cout << "\n";
                /*
                std::cout << "Dimension: ";
                auto &w_dims = layer.w_dims;
                auto &weights = layer.weights;

                for (auto dim : w_dims) { std::cout << dim << " "; }
                std::cout << "\n";

                unsigned i = 0;
                for (auto weight : weights)
                {
                    std::cout << weight << " ";
                    if ((i + 1) % w_dims[w_dims.size() - 1] == 0)
                    {
                        std::cout << "\n";
                    }
                    i++;
                }

                auto &strides = layer.strides;
                std::cout << "Strides: ";
                for (auto stride : strides) { std::cout << stride << " "; }
                std::cout << "\n";
*/
                auto &output_dims = layer.output_dims;
                std::cout << "Output shape: ";
                for (auto dim : output_dims) { std::cout << dim << " "; }
                std::cout << "\n";
                
                auto &w_dims = layer.w_dims;
                auto &weights = layer.weights;
                auto &b_dims = layer.b_dims;
                auto &bias = layer.bias;
                if (weights.size())
                {
                    std::cout << "Weights dim (" << weights.size() << "): ";
                    for (auto dim : w_dims) { std::cout << dim << " "; }
                    std::cout << "\n";
                }

                if (bias.size())
                {
                    std::cout << "Bias dim (" << bias.size() << "): ";
                    for (auto dim : b_dims) { std::cout << dim << " "; }
                    std::cout << "\n";
                }

                std::cout << "Total params: " 
                          << weights.size() + bias.size() << "\n";

                std::cout << "\n";
                // TODO, print neuron ID range
                // auto &out_neuro_ids = layer.output_neuron_ids;
                // std::cout << "Output neuron id range: "
                //           << out_neuro_ids[0] << " -> " 
                //           << out_neuro_ids[out_neuro_ids.size() - 1] 
                //           << "\n\n";
/*
                auto &out_neuro_ids = layer.output_neuron_ids;
                std::cout << "Output neuron id: ";
                std::cout << "\n";
                for (int k = 0; k < output_dims[2]; k++)
                {
                    for (int i = 0; i < output_dims[0]; i++)
                    {
                        for (int j = 0; j < output_dims[1]; j++)
                        {
                            std::cout << out_neuro_ids[
                                k * output_dims[0] * output_dims[1] + 
                                i * output_dims[1] + j] << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
	
                std::cout << "\n\n";
*/
                // exit(0);
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
        #ifdef NEURON // ab3586
        loadArchNeuron(arch_file);
        #else
        loadArch(arch_file);
        #endif
        if (weight_file != "") {
            loadWeights(weight_file);
        }
    }

    Model(std::string &arch_file)
    {
        #ifdef NEURON // ab3586
        loadArchNeuron(arch_file);
        #else
        loadArch(arch_file);
        #endif
        
    }

    void printLayers() { arch.printLayers(); }

    void connector() { arch.connector(); } 

    void printConns(std::string &out_root) { arch.printConns(out_root); }
    void printSdfRep(std::string &out_root) {arch.printSdfRep(out_root); }
    void printLayerConns(std::string &out_root) {arch.printLayerConns(out_root); }
    void outputLayerDepthIR(std::string &out_file) {arch.outputLayerDepthIR(out_file);}
    std::pair<uint64_t, uint64_t> getIrregularMetric() { return arch.getIrregularMetric();}

    void setOutRoot(std::string &out_root) 
    { arch.setOutRoot(out_root); }

  protected:
    void loadArchNeuron(std::string &arch_file); // ab3586:populate the data structures to extract neuron and connection information.
    void loadArch(std::string &arch_file);
    void loadArch2(std::string &arch_file); //Shihao's boost::ptree version
    void loadWeights(std::string &weight_file);

  protected:
    void scanGroup(hid_t);
    void extrWeights(hid_t);
};
}
}

#endif
