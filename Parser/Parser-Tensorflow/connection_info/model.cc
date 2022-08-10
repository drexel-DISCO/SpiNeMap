#include "model.h"

// boost library to parse json architecture file
#include <boost/foreach.hpp>
#include "nlohmann/json.hpp"

#include <sstream>

#include <boost/filesystem.hpp>

namespace NCC
{
namespace NCC_FrontEnd
{

using json = nlohmann::json;

void Model::loadArch(std::string &arch_file_path)
{
    try
    {
        std::ifstream arch_file(arch_file_path);
        json js;
        arch_file >> js;
        auto json_layers = js["config"]["layers"];

        unsigned layer_counter = 0;
        std::unordered_map<std::string, std::vector<std::string>> concat_input;

        // Initial processing of layers
        for (auto& layer: json_layers) {
            if (layer_counter == 0)
            {
                std::vector<std::string> input_shape;
                std::vector<unsigned> output_dims;
                for (auto &cell: layer["config"]["batch_input_shape"])
                {
                    if (cell == nlohmann::detail::value_t::null) cell = 0; 
                    output_dims.push_back(cell);
                }

                output_dims.erase(output_dims.begin()); //Delete the first null (see json for more details)
               
                std::string name = "input";
                Layer::Layer_Type layer_type = Layer::Layer_Type::Input;
                arch.addLayer(name, layer_type);
                arch.getLayer(name).setOutputDim(output_dims);

                layer_counter++;
            }


            std::string class_name = layer["class_name"];
            std::string name = layer["config"]["name"];

            Layer::Layer_Type layer_type = Layer::Layer_Type::MAX;
            if (class_name == "InputLayer") { layer_type = Layer::Layer_Type::Input; }
            else if (class_name == "Conv2D" || class_name == "QConv2D") { layer_type = Layer::Layer_Type::Conv2D; }
            else if (class_name == "Activation" || class_name == "QActivation") { layer_type = Layer::Layer_Type::Activation; }
            else if (class_name == "BatchNormalization") {layer_type = Layer::Layer_Type::BatchNormalization; }
            else if (class_name == "Dropout") { layer_type = Layer::Layer_Type::Dropout; }
            else if (class_name == "MaxPooling2D") { layer_type = Layer::Layer_Type::MaxPooling2D; }
            else if (class_name == "AveragePooling2D") { layer_type = Layer::Layer_Type::AveragePooling2D; }
            else if (class_name == "Flatten") { layer_type = Layer::Layer_Type::Flatten; }
            else if (class_name == "Dense" || class_name == "QDense") { layer_type = Layer::Layer_Type::Dense; }
            else if (class_name == "ZeroPadding2D") {layer_type = Layer::Layer_Type::Padding; }
            else if (class_name == "Concatenate") {layer_type = Layer::Layer_Type::Concatenate; }
            else if (class_name == "GlobalAveragePooling2D" || class_name == "GlobalMaxPooling2D")
            {layer_type = Layer::Layer_Type::GlobalPooling2D;}
            else if (class_name == "Add") {layer_type = Layer::Layer_Type::Add; }
            else { std::cerr << "Error: Unsupported layer type.\n"; exit(0); }

            if (class_name != "InputLayer")
            {
                arch.addLayer(name, layer_type);
            }
            else if (class_name == "InputLayer")
            {
                // The input layer is explicitly specified, we need to change its name here.
                std::string default_name = "input";
                arch.getLayer(default_name).name = name; // When input is explicitly mentioned.
            }

            if (class_name == "Conv2D" || class_name == "MaxPooling2D" || 
                class_name == "AveragePooling2D" || class_name == "QConv2D")
            {
                // get padding type
                std::string padding_type = layer["config"]["padding"];
                if (padding_type == "same")
                {
                    arch.getLayer(name).padding_type = Layer::Padding_Type::same;
                }

                // get strides information
                std::vector<unsigned> strides;

                for (auto &cell : layer["config"]["strides"]) {
                    strides.push_back(cell);
                }
                arch.getLayer(name).setStrides(strides);
            }

            if (class_name == "Conv2D" || class_name == "QConv2D") { 
                //get kernel size information
                std::vector<unsigned> kernel_size;
                for (auto &dim : layer["config"]["kernel_size"]) {
                    kernel_size.push_back(dim);
                }
                arch.getLayer(name).setKernelSize(kernel_size);
                arch.getLayer(name).num_filter = layer["config"]["filters"];
            }

            if (class_name == "MaxPooling2D" || class_name == "AveragePooling2D")
            {
                //purely for the sdf representation
                std::vector<unsigned> kernel_size;
                for (auto &dim : layer["config"]["pool_size"]) {
                    kernel_size.push_back(dim);
                }
                arch.getLayer(name).setKernelSize(kernel_size);
            }

            if (class_name == "ZeroPadding2D")
            {
                auto &padding= arch.getLayer(name).padding;
                int pad_pix_per_dim = 0;
                for (auto pad_dims : layer["config"]["padding"]) {
                    for (int num_pad_pix : pad_dims) {
                       // for (int num_pad_pix : dim) {
                            pad_pix_per_dim += num_pad_pix;
                        //}
                    }
                    padding.push_back(pad_pix_per_dim);
                    pad_pix_per_dim = 0;
                }
            }

            if (class_name == "Concatenate") {
                std::vector<std::string> concat_input_layers;
                for (auto& in_layer_info : layer["inbound_nodes"][0]) {  
                    std::string in_layer_s = (in_layer_info)[0];
                    
                    //concat has input concat
                    if (arch.getLayer(in_layer_s).layer_type == 
                            Layer::Layer_Type::Concatenate)
                    {
                        if (auto iter = concat_input.find(in_layer_s);
                                     iter != concat_input.end())
                        {
                            for (auto& l : (*iter).second) {
                                concat_input_layers.push_back(l);
                            }
                               
                        } else { //concat not registered
                            throw "Concatenate layers not in definition order";
                        }
                    } else { //concat has input conv + conv
                        concat_input_layers.push_back(in_layer_s);
                    }
                }
                concat_input.insert({name, concat_input_layers});
            }

            if (class_name == "Add") {
                continue;
            }

            layer_counter++;
            // TODO, more information to extract, such as activation method...
            // TODO, support more types of layers for more modern networks such as transformers
        }

        // Processing layers' connectivity
        for (auto& layer: json_layers) {
            std::string layer_name = layer["name"];
            
            if (arch.getLayer(layer_name).layer_type == Layer::Layer_Type::Concatenate) {
                continue;
            }

            for (auto& in_layer_info : layer["inbound_nodes"][0]) {  
                std::string in_layer_s = (in_layer_info)[0];

                //input layer is of type concat
                if (arch.getLayer(in_layer_s).layer_type == Layer::Layer_Type::Concatenate)
                {
                    if (auto iter = concat_input.find(in_layer_s);
                                     iter != concat_input.end())
                    {
                        for (auto& l : (*iter).second) {
                            arch.getLayer(layer_name).inbound_layers.push_back(l);
                            arch.getLayer(l).outbound_layers.push_back(layer_name);
                        }
                    }
                } 
                else //input layer is not concat
                {
                    arch.getLayer(in_layer_s).outbound_layers.push_back(layer_name);
                    arch.getLayer(layer_name).inbound_layers.push_back(in_layer_s);
                }

            }
        }

        //Processing layers' sdf representation
        for (auto& layer: json_layers) 
        {
            std::string layer_name = layer["name"];
            auto& layer_obj = arch.getLayer(layer_name);
            std::vector<unsigned>& output_dims = layer_obj.output_dims;
            std::vector<unsigned> input_dims(3, 0);

            if (layer_obj.layer_type == Layer::Layer_Type::Input)
            {
                input_dims = layer_obj.output_dims;
            } 
            else if (layer_obj.layer_type == Layer::Layer_Type::Add)
            {
                auto& inputLayers = layer_obj.inbound_layers;
                //All input layers should have the same output_dims
                output_dims[0] = arch.getLayer(inputLayers[0]).output_dims[0]; 
                output_dims[1] = arch.getLayer(inputLayers[0]).output_dims[1]; 
                output_dims[2] = arch.getLayer(inputLayers[0]).output_dims[2]; 
            }
            else
            {
                auto& inputLayers = layer_obj.inbound_layers;
                for (auto& inLayer : inputLayers)
                {
                    //All input layers should have the same output_dims[0, 1]
                    input_dims[0] = arch.getLayer(inLayer).output_dims[0]; 
                    input_dims[1] = arch.getLayer(inLayer).output_dims[1];
                    input_dims[2] += arch.getLayer(inLayer).output_dims[2];               
                }
                output_dims = input_dims;
            }

            if (layer_obj.layer_type == Layer::Layer_Type::Padding)
            {    
                int i = 0;
                
                for (auto pad_dim : layer_obj.padding) 
                {
                    output_dims[i] += pad_dim;
                    layer_obj.comp.num_Init += pad_dim;
                    i++;
                }
            } 
            else if (layer_obj.layer_type == Layer::Layer_Type::Conv2D ||
                    layer_obj.layer_type == Layer::Layer_Type::MaxPooling2D ||
                    layer_obj.layer_type == Layer::Layer_Type::AveragePooling2D)
            {
                for (int i=0; i < 2; i++) {
                    int k = layer_obj.kernel_sz[i];
                    int s = layer_obj.strides[i];
                    if (layer_obj.padding_type == Layer::Padding_Type::valid)
                        output_dims[i] = (float)((float)output_dims[i] - (float)k)/(float)s + 1;
                    else 
                        //layer_obj.padding_type == Layer::Padding_Type::same
                        output_dims[i] = ceil((float)output_dims[i]/(float)s);
                }

                if (layer_obj.layer_type == Layer::Layer_Type::Conv2D) 
                {
                    output_dims[2] = layer_obj.num_filter;
                    layer_obj.comp.num_MAC =  output_dims[2]              // n
                                            * input_dims[2]               // m
                                            * layer_obj.kernel_sz[0] 
                                            * layer_obj.kernel_sz[1]      // k^2
                                            * output_dims[0]              // new w
                                            * output_dims[1];             // new h 
                } 
                else 
                { // layer is Pooling2D
                    if (layer_obj.layer_type == Layer::Layer_Type::MaxPooling2D) 
                    {
                        layer_obj.comp.num_Compare =  output_dims[2]             // n
                                            * (layer_obj.kernel_sz[0] 
                                                * layer_obj.kernel_sz[1] - 1)  // k^2 -1
                                            * output_dims[0]                 // new w
                                            * output_dims[1];                // new h 
                    }
                    else if (layer_obj.layer_type == Layer::Layer_Type::AveragePooling2D) 
                    {
                        layer_obj.comp.num_AddSub =  output_dims[2]             // n
                                            * (layer_obj.kernel_sz[0] 
                                                * layer_obj.kernel_sz[1] - 1) // k^2 -1
                                            * output_dims[0]                 // new w
                                            * output_dims[1]                // new h
                                            +  1;           // (optional) Calculate kernel sz
                        
                        layer_obj.comp.num_Div =  output_dims[2]             // n
                                            * output_dims[0]                 // new w
                                            * output_dims[1];                // new h 
                    }
                }
            } 
            else if (layer_obj.layer_type == Layer::Layer_Type::BatchNormalization) 
            {
                // Compute gamma*(x - moving_mean)/(moving_variance + epsilon) + beta
                // moving mean, gamma, beta, (moving_variance + epsilon) are constant during inference
                uint64_t num_input = input_dims[0]*input_dims[1]*input_dims[2];

                layer_obj.comp.num_AddSub = num_input*2;                
                layer_obj.comp.num_Mult = num_input;
                layer_obj.comp.num_Div = num_input*1;
            } 
            else if (layer_obj.layer_type == Layer::Layer_Type::Activation)
            {
                //Assuming Activation used is relu = 1 compare with 0 per element
                layer_obj.comp.num_Compare = input_dims[0]*input_dims[1]*input_dims[2];
            } 
            else if (layer_obj.layer_type == Layer::Layer_Type::GlobalPooling2D) 
            {
                if (layer["class_name"] == "GlobalAveragePooling2D") {
                    layer_obj.comp.num_AddSub = (input_dims[0] * input_dims[1] - 1)
                                                * input_dims[2];     
                    //Each feature map (wxh pix) requires (wxh -1) 

                    layer_obj.comp.num_Div = input_dims[2];              
                }
                output_dims = {1, 1, input_dims[2]};
            } 
            else if (layer_obj.layer_type == Layer::Layer_Type::Dense) 
            {
                unsigned units = layer["config"]["units"];
                layer_obj.comp.num_MAC = input_dims[2] * units;
                output_dims = {1, 1, units};
            }
            else if (layer_obj.layer_type == Layer::Layer_Type::Flatten)
            {
                output_dims = {1, 1, input_dims[0]*input_dims[1]*input_dims[2]};
                layer_obj.comp.num_Init = input_dims[0]*input_dims[1]*input_dims[2];
            }
            else if (layer_obj.layer_type == Layer::Layer_Type::Add)
            {
                layer_obj.comp.num_AddSub = output_dims[0]*output_dims[1]*output_dims[2];
            }

            layer_obj.num_out_tok = output_dims[0]*output_dims[1]*output_dims[2];
            layer_obj.compute_time =  layer_obj.comp.num_AddSub  * Layer::Cost::ADDSUB 
                                    + layer_obj.comp.num_MAC     * Layer::Cost::MAC
                                    + layer_obj.comp.num_Compare * Layer::Cost::COMPARE
                                    + layer_obj.comp.num_Div     * Layer::Cost::DIVIDE
                                    + layer_obj.comp.num_Init    * Layer::Cost::INIT
                                    + layer_obj.comp.num_Mult    * Layer::Cost::MULT;

        }
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
        exit(0);
    }

    std::set<std::string> temp = {};
    arch.labelLayerWithDepth(0, temp);  
    arch.setMaxLayerDepth();
}

void Model::Architecture::printSdfRep(std::string &out_root) {
    std::string layer_conn_out_txt = out_root + ".sdf_connection.txt";
    std::ofstream conns_out(layer_conn_out_txt);

    for (int i = 0; i < layers.size(); i++) 
    {
        if (layers[i].layer_type != Layer::Layer_Type::Concatenate) 
        {
            for (auto& out_layer: layers[i].outbound_layers) 
            {
                conns_out << layers[i].name << " " 
                          << out_layer << " " 
                          << layers[i].num_out_tok
                          << "\n";
            } 
        }
    }

    std::string layer_com_out_txt = out_root + ".sdf_computation.txt";
    std::ofstream com_out(layer_com_out_txt); 

    for (int i = 0; i < layers.size(); i++) 
    {
        if (layers[i].layer_type != Layer::Layer_Type::Concatenate) 
        {
            com_out << layers[i].name             << " "
                    << layers[i].comp.num_AddSub  << " " 
                    << layers[i].comp.num_MAC     << " "
                    << layers[i].comp.num_Compare << " "
                    << layers[i].comp.num_Div     << " "
                    << layers[i].comp.num_Init    << " "
                    << layers[i].comp.num_Mult    << "\n";

        }
    } 
}


void Model::Architecture::printLayerConns(std::string &out_root) {
    std::string layer_conn_out_txt = out_root + ".layer_connection_info.txt";
    std::ofstream conns_out(layer_conn_out_txt);

    for (int i = 0; i < layers.size() - 1; i++) 
    {
        if (layers[i].layer_type != Layer::Layer_Type::Concatenate) 
        {
            conns_out << layers[i].name << " ";
            for (auto& out_layer: layers[i].outbound_layers) 
                conns_out << out_layer << " ";
            conns_out << "\n";
        }
    } 

    std::string layer_outshape_txt = out_root + ".layer_outputshape_info.txt";
    std::ofstream out_shape_f(layer_outshape_txt);
    
    for (int i = 0; i < layers.size() - 1; i++) 
    {
        auto name = layers[i].name;
        auto type = layers[i].layer_type;

        out_shape_f << "Layer name: " << name << "; ";
        if (type == Layer::Layer_Type::Input) 
        { out_shape_f << "Layer type: Input"; }
        else if (type == Layer::Layer_Type::Conv2D) 
        { out_shape_f << "Layer type: Conv2D"; }
        else if (type == Layer::Layer_Type::Activation) 
        { out_shape_f << "Layer type: Activation"; }
        else if (type == Layer::Layer_Type::BatchNormalization) 
        { out_shape_f << "Layer type: BatchNormalization"; }
        else if (type == Layer::Layer_Type::Dropout) 
        { out_shape_f << "Layer type: Dropout"; }
        else if (type == Layer::Layer_Type::MaxPooling2D) 
        { out_shape_f << "Layer type: MaxPooling2D"; }
        else if (type == Layer::Layer_Type::AveragePooling2D) 
        { out_shape_f << "Layer type: AveragePooling2D"; }
        else if (type == Layer::Layer_Type::GlobalPooling2D) 
        { out_shape_f << "Layer type: GlobalPooling2D"; }
        else if (type == Layer::Layer_Type::Flatten) 
        { out_shape_f << "Layer type: Flatten"; }
        else if (type == Layer::Layer_Type::Dense) 
        { out_shape_f << "Layer type: Dense"; }
        else if (type == Layer::Layer_Type::Padding) 
        { out_shape_f << "Layer type: Padding"; }
        else if (type == Layer::Layer_Type::Concatenate) 
        { out_shape_f << "Layer type: Concatenate"; }
        else if (type == Layer::Layer_Type::Add) 
        { out_shape_f << "Layer type: Add"; }
        else { std::cerr << "Error: unsupported layer type\n"; exit(0); }
        out_shape_f << "\n";

        auto &output_dims = layers[i].output_dims;
        out_shape_f << "Output shape: ";
        for (auto dim : output_dims) { out_shape_f << dim << " "; }
        out_shape_f << "; Num compute cycle: " << layers[i].compute_time << "\n\n";
    }
}

void Model::Architecture::labelLayerWithDepth(uint64_t starting_depth, std::set<std::string>& indepth) {
    std::set<std::string> outdepth = {};
    if (starting_depth == 0) { //indepth should be snn 
        for (auto& layer : layers) 
        {
            if ((layer.inbound_layers.size() == 0) && (layer.layer_type != Layer::Layer_Type::Concatenate))
            {
                layer.setDepth(1);
                outdepth.insert(layer.name);
            }   
        }
        labelLayerWithDepth(1, outdepth);
    } else if (indepth.size()) { // If not last depth   
        for (auto& layer_name : indepth)
        {
            for (auto& out_layer_name: getLayer(layer_name).outbound_layers) 
            {
                Layer& layer = getLayer(out_layer_name);
                layer.setDepth(std::max((int)starting_depth + 1, layer.getDepth()));
                if (layer.getDepth() == starting_depth + 1)                
                    outdepth.insert(out_layer_name);
            }
        }
        labelLayerWithDepth(starting_depth + 1, outdepth);
    } else {
        return;
    }
}

void Model::Architecture::setMaxLayerDepth()
{
    int tempMaxDepth = -1;
    for (auto &layer : layers)
    {
        tempMaxDepth = layer.getDepth() > tempMaxDepth ?
                       layer.getDepth() : tempMaxDepth;
    }
    maxDepth = tempMaxDepth;
}

void Model::Architecture::outputLayerDepthIR(const std::string& out_file)
{
    if (layers.size() == 0) {
        return;
    }

    std::fstream file;
    file.open(out_file, std::fstream::out);

    for (auto &layer : layers)
    {
        if (layer.layer_type != Layer::Layer_Type::Concatenate) 
        {
            file << layer.name << " ";
            file << layer.getDepth();
            file << "\n";            
        }
    }

    file.close();
    return;
}

std::pair<uint64_t, uint64_t> Model::Architecture::getIrregularMetric() {
    //Define irregular metric as (\sum_{i=0} n {\sum_{j=0} k_i {d_i - d_j}}) / num_connection
    //for networks that has n layers, layer i has k_i input connections
    uint64_t metric = 0;
    uint64_t num_connections = 0;
    for (auto& layer : layers) {
        for (auto& in_layer_name : layer.inbound_layers) {
            metric += (layer.getDepth() - getLayer(in_layer_name).getDepth() - 1);
            num_connections++;
        }
    }
    std::pair<uint64_t, uint64_t> returnVal = std::make_pair(metric, num_connections);
    return returnVal;
}

}
}
