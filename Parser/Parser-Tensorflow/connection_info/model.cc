#include "model.h"

// boost library to parse json architecture file
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "nlohmann/json.hpp"

#include <sstream>

// #include "protobuf/proto_graph/graph.pb.h"

#include <boost/filesystem.hpp>

#define DEBUG 0
#define CONNECTION 1

namespace NCC
{
namespace NCC_FrontEnd
{

using json = nlohmann::json;


void Model::Architecture::connectLayers() 
{
    int prev = 0;
}

void Model::Architecture::connector()
{
    // TODO : ab3586: Connection function currently connects layers sequentially. Look for in_bound layers. 
    int prev = 0;
    for (int i = 0; i < layers.size() - 1; i++)
    {
        // ab3586
        // find the inbound layer IDs
        for(auto& n_inbound : layers[i].inbound_layers)
        {   
            int j=0;
            std::string name =  n_inbound;
             
            auto& l_inbound = getLayer(name);    
             
            // find the layer id (index) for the layers ds 
            for(j=0; j<layers.size(); j++)
            {
                std::string l_name = layers[j].name;
                if(l_name.compare(name) == 0)
                {
                    break;
                }    
            }

            // Logs
#if DEBUG
            std::cout << "\nInbound layer of " << layers[i].name << " is: "<<  name << std::endl;
            std::cout << "\nLayer ID: " << i << " Inbound ID: "<<  j << std::endl;
#endif
            if (layers[i].layer_type == Layer::Layer_Type::Conv2D)
            {
                if (layers[i].padding_type == Layer::Padding_Type::same)
                {
                    //std::cout << "Conv with Padding" << std::endl;
                    connToConvPadding(j, i);
                }
                else
                {
                    //std::cout << "Conv without Padding" << std::endl;
                    connToConv(j, i);
                }
            }
            
            else if (layers[i].layer_type == Layer::Layer_Type::Activation)
            {
                connToAct(j, i);
            }
            else if (layers[i].layer_type == Layer::Layer_Type::BatchNormalization)
            {
                connToNorm(j, i);
            }
            else if (layers[i].layer_type == Layer::Layer_Type::Dropout)
            {
                //connToDrop(j, i);
            }
            
            else if (layers[i].layer_type == Layer::Layer_Type::BatchNormalization)
            {
            }
            else if (layers[i].layer_type == Layer::Layer_Type::MaxPooling2D \
                    || layers[i].layer_type == Layer::Layer_Type::AveragePooling2D \
                    || layers[i].layer_type == Layer::Layer_Type::GlobalPooling2D \
                    || layers[i].layer_type == Layer::Layer_Type::GlobalAveragePooling2D)
            {
                connToPool(j, i); 
            }
            else if (layers[i].layer_type == Layer::Layer_Type::Flatten)
            {
                connToFlat(j, i); 
            }
            else if (layers[i].layer_type == Layer::Layer_Type::Dense)
            {
                connToDense(j, i);
            }
            else if (layers[i].layer_type == Layer::Layer_Type::Ignore)
            {
                //Do nothing
            }
            else if (layers[i].layer_type == Layer::Layer_Type::Concatenate)
            {
            }
            else if (layers[i].layer_type == Layer::Layer_Type::Add)
            {
                connToAdd(j, i);
            }
            else if (layers[i].layer_type == Layer::Layer_Type::ZeroPadding2D)
            {
            	connToPadding(j,i);

            }
            else if (layers[i].layer_type == Layer::Layer_Type::MAX)
            {
                //connToDense(j, i);
            }
            else
            {   
                int type = int(layers[i].layer_type) ;
                std::cout << "ERROR: " << layers[i].name << " from "<< layers[j].name <<std::endl;
                std::cout << "ERROR: " << type << std::endl;
                std::cerr << "Error: unsupported connection type. \n";
                exit(0);
            }

        }
    }
}


void Model::Architecture::connToConv(unsigned cur_layer_id, 
                                     unsigned next_layer_id)
{

    auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;

    
    auto &conv_kernel_dims = layers[next_layer_id].w_dims;
    auto &conv_kernel_weights = layers[next_layer_id].weights;
    auto &conv_strides = layers[next_layer_id].strides;
    auto &conv_output_dims = layers[next_layer_id].output_dims;
    auto &conv_output_neuron_ids = layers[next_layer_id].output_neuron_ids;
    
#if DEBUG
    std::cout << "Current Neuron IDs Size: " << cur_neurons_ids.size()  << std::endl;
    std::cout << "Input Add: "<< &cur_neurons_ids << std::endl;
    std::cout << "Output Add: "<< &conv_output_neuron_ids << std::endl;
#endif

    // Important. We need to re-organize the conv kernel to be more memory-friendly
    // Original layout: row->col->dep->filter
    // New layer: filter->dep->row->col

    unsigned row_limit = conv_kernel_dims[0];
    unsigned col_limit = conv_kernel_dims[1];
    unsigned dep_limit = conv_kernel_dims[2];
    unsigned filter_limit = conv_kernel_dims[3];

    std::string out_layer_name = layers[next_layer_id].name;
    if(layers[cur_layer_id].layer_type == Layer::Layer_Type::ZeroPadding2D)
    {
    	out_layer_name = layers[cur_layer_id].inbound_layers[0];
    }
    std::vector<float> conv_kernel_weights_format(filter_limit * 
                                                  dep_limit * 
                                                  row_limit * 
                                                  col_limit, 0.0);

    for (unsigned row = 0; row < row_limit; row++)
    {
        for (unsigned col = 0; col < col_limit; col++)
        {
            for (unsigned dep = 0; dep < dep_limit; dep++)
            {
                for (unsigned filter = 0; filter < filter_limit; filter++)
                {
                    conv_kernel_weights_format[
                        filter * dep_limit * row_limit * col_limit + 
                        dep * row_limit * col_limit +
                        row * col_limit +
                        col] = 
                    conv_kernel_weights[
                        row * col_limit * dep_limit * filter_limit +
                        col * dep_limit * filter_limit +
                        dep * filter_limit +
                        filter];
                }
            }
        }
    }
    
    uint64_t conv_neuron_id_track = cur_neurons_ids[cur_neurons_ids.size() - 1] + 1;


    // std::cout << conv_neuron_id_track << "\n";

    unsigned conv_output_dims_x = 0;
    unsigned conv_output_dims_y = 0;
    // For each filter
    for (unsigned filter = 0; filter < conv_kernel_dims[3]; filter++)
    {
        conv_output_dims_x = 0;
        for (unsigned row = conv_kernel_dims[0] - 1; 
            row < cur_neurons_dims[0]; 
            row += conv_strides[0])
        {
            conv_output_dims_x++;

            conv_output_dims_y = 0;
            for (unsigned col = conv_kernel_dims[1] - 1; 
                col < cur_neurons_dims[1]; 
                col += conv_strides[1])
            {
                conv_output_dims_y++;

                // All neurons inside the current kernel
                unsigned starting_row = row + 1 - conv_kernel_dims[0];
                unsigned ending_row = row;
                unsigned starting_col = col + 1 - conv_kernel_dims[1];
                unsigned ending_col = col;

                // std::cout << starting_row << " " 
                //     << ending_row << " " 
                //     << starting_col << " " 
                //     << ending_col << "\n";
                for (unsigned k = 0; k < cur_neurons_dims[2]; k++)
                {
                    for (unsigned i = starting_row; i <= ending_row; i++)
                    {
                        for (unsigned j = starting_col; j <= ending_col; j++)
                        {
                            uint64_t cur_neuron_id = 
                                cur_neurons_ids[
                                k * cur_neurons_dims[0] * cur_neurons_dims[1] +
                                i * cur_neurons_dims[1] + j];

                            float weight = 
                                conv_kernel_weights_format[
                                    filter * conv_kernel_dims[2] * 
                                    conv_kernel_dims[0] * 
                                    conv_kernel_dims[1] +


                                    k * conv_kernel_dims[0] * 
	                            conv_kernel_dims[1] +

                                    (i - starting_row) * conv_kernel_dims[1] +
                                    (j - starting_col)];
                            // conn_txt << cur_neuron_id << " " 
			    //     << conv_neuron_id_track << " " 
                            //     << weight << "\n";
                            
                            // Record the connection information
                            if (auto iter = connections.find(cur_neuron_id);
                                     iter != connections.end())
                            {
                                (*iter).second.out_neurons_ids.push_back(
                                    conv_neuron_id_track);
                                (*iter).second.weights.push_back(weight);
                                (*iter).second.out_layer_name.push_back(
                                		out_layer_name);
                            }
                            else
                            {
                                connections.insert({cur_neuron_id, 
                                    {conv_neuron_id_track, weight, out_layer_name}});
                            }
                            
                            // std::cout << cur_neuron_id << " ";
                        }
                    }
                }
                // std::cout << "-> " << conv_neuron_id_track << "\n";
                conv_output_neuron_ids.push_back(conv_neuron_id_track);
                conv_neuron_id_track++;
                // std::cout << "\n";
            }
        }
    }
    // std::cout << "\n";
    conv_output_dims[0] = conv_output_dims_x;
    conv_output_dims[1] = conv_output_dims_y;
    conv_output_dims[2] = conv_kernel_dims[3];
}

void Model::Architecture::connToConvPadding(unsigned cur_layer_id, unsigned next_layer_id)
{


    auto &ori_neurons_dims = layers[cur_layer_id].output_dims;
    auto &ori_neurons_ids = layers[cur_layer_id].output_neuron_ids;

#if DEBUG
    std::cout << cur_layer_id << ":" << next_layer_id << std::endl;
    std::cout <<"Current Neuron IDs Size: " << ori_neurons_ids.size()  << std::endl;
#endif

    auto &conv_kernel_dims = layers[next_layer_id].w_dims;
    auto &conv_kernel_weights = layers[next_layer_id].weights;
    auto &conv_strides = layers[next_layer_id].strides;
    auto &conv_output_dims = layers[next_layer_id].output_dims;
    auto &conv_output_neuron_ids = layers[next_layer_id].output_neuron_ids;

#if DEBUG
    std::cout << "Input Add: "<< &ori_neurons_ids << std::endl;
    std::cout << "Output Add: "<< &conv_output_neuron_ids << std::endl;
    std::cout <<"NeuronID Size: "<< layers[next_layer_id].output_neuron_ids.size() << std::endl;
#endif
    // Important. We need to re-organize the conv kernel to be more memory-friendly
    // Original layout: row->col->dep->filter
    // New layer: filter->dep->row->col
    unsigned row_limit = conv_kernel_dims[0];
    unsigned col_limit = conv_kernel_dims[1];
    unsigned dep_limit = conv_kernel_dims[2];
    unsigned filter_limit = conv_kernel_dims[3];

    std::string out_layer_name = layers[next_layer_id].name;
    if(layers[cur_layer_id].layer_type == Layer::Layer_Type::ZeroPadding2D)
    {
    	out_layer_name = layers[cur_layer_id].inbound_layers[0];
    }


    std::vector<float> conv_kernel_weights_format(filter_limit * 
                                                  dep_limit * 
                                                  row_limit * 
                                                  col_limit, 0.0);

    for (unsigned row = 0; row < row_limit; row++)
    {
        for (unsigned col = 0; col < col_limit; col++)
        {
            for (unsigned dep = 0; dep < dep_limit; dep++)
            {
                for (unsigned filter = 0; filter < filter_limit; filter++)
                {
                    
                    conv_kernel_weights_format[
                        filter * dep_limit * row_limit * col_limit +
                        dep * row_limit * col_limit +
                        row * col_limit + col] =

                    conv_kernel_weights[
                        row * col_limit * dep_limit * filter_limit +
                        col * dep_limit * filter_limit +
                        dep * filter_limit +
                        filter];
                }
            }
        }
    }

    // Determine the number of paddings
    auto padding_to_row = 
        ((ori_neurons_dims[0] - 1) * 
          conv_strides[0] - ori_neurons_dims[0] + 
          conv_kernel_dims[0]) / 2;

    auto padding_to_col = 
        ((ori_neurons_dims[1] - 1) * 
          conv_strides[1] - ori_neurons_dims[1] + 
          conv_kernel_dims[1]) / 2;

    auto final_row_size = ori_neurons_dims[0] + 2 * padding_to_row;
    auto final_col_size = ori_neurons_dims[1] + 2 * padding_to_col;
    auto final_dep_size = ori_neurons_dims[2];

    std::vector<unsigned> final_neurons_dims{final_row_size, 
                                             final_col_size, 
                                             final_dep_size};

    std::vector<uint64_t> final_neurons_ids(final_row_size * 
                                            final_col_size * 
                                            final_dep_size, 0);
    std::vector<bool> final_neurons_ids_valid(final_row_size * 
                                              final_col_size * 
                                              final_dep_size, 0);
    for (unsigned dep = 0; dep < ori_neurons_dims[2]; dep++)
    {
        for (unsigned row = 0; row < ori_neurons_dims[0]; row++)
        {
            for (unsigned col = 0; col < ori_neurons_dims[1]; col++)
            {
                final_neurons_ids[
                    dep * final_row_size * final_col_size + 
                    (row + padding_to_row) * final_col_size + 
                    (col + padding_to_col)] = 

                ori_neurons_ids[
                    dep * ori_neurons_dims[0] * ori_neurons_dims[1] +
                    row * ori_neurons_dims[1] +
                    col];

                final_neurons_ids_valid[
                    dep * final_row_size * final_col_size + 
                    (row + padding_to_row) * final_col_size + 
                    (col + padding_to_col)] = 1;
            }
        }
    }

    uint64_t conv_neuron_id_track = 
        ori_neurons_ids[ori_neurons_ids.size() - 1] + 1;




    unsigned conv_output_dims_x = 0;
    unsigned conv_output_dims_y = 0;
    // For each filter
    for (unsigned filter = 0; 
         filter < conv_kernel_dims[3]; 
         filter++)
    {
        conv_output_dims_x = 0;
        for (unsigned row = conv_kernel_dims[0] - 1; 
             row < final_neurons_dims[0]; 
             row += conv_strides[0])
        {
            conv_output_dims_x++;

            conv_output_dims_y = 0;
            for (unsigned col = conv_kernel_dims[1] - 1; 
                 col < final_neurons_dims[1]; 
                 col += conv_strides[1])
            {
                conv_output_dims_y++;

                // All neurons inside the current kernel
                unsigned starting_row = row + 1 - conv_kernel_dims[0];
                unsigned ending_row = row;
                unsigned starting_col = col + 1 - conv_kernel_dims[1];
                unsigned ending_col = col;

                for (unsigned k = 0; k < final_neurons_dims[2]; k++)
                {
                    for (unsigned i = starting_row; i <= ending_row; i++)
                    {
                        for (unsigned j = starting_col; j <= ending_col; j++)
                        {
                            
                            if (final_neurons_ids_valid[
                                k * final_neurons_dims[0] * 
                                    final_neurons_dims[1] +
                                i * final_neurons_dims[1] + j])
                            {
                                uint64_t cur_neuron_id = final_neurons_ids[
                                    k * final_neurons_dims[0] * 
                                        final_neurons_dims[1] +
                                    i * final_neurons_dims[1] + j];

                                float weight =conv_kernel_weights_format[
                                    filter * conv_kernel_dims[2] * 
                                        conv_kernel_dims[0] * 
                                        conv_kernel_dims[1] +
                                    k * conv_kernel_dims[0] * 
                                        conv_kernel_dims[1] +
                                    (i - starting_row) * conv_kernel_dims[1] +
                                    (j - starting_col)];
                                
                                // Record the connection information
                                if (auto iter = connections.find(cur_neuron_id);
                                         iter != connections.end())
                                {
                                    (*iter).second.out_neurons_ids.push_back(
                                        conv_neuron_id_track);
                                    (*iter).second.weights.push_back(weight);
                                    (*iter).second.out_layer_name.push_back(out_layer_name);
                                }
                                else
                                {
                                    connections.insert({cur_neuron_id, 
                                                       {conv_neuron_id_track, 
                                                        weight, out_layer_name}});
                                }
                            }
                        }
                    }
                }
                conv_output_neuron_ids.push_back(conv_neuron_id_track);
                conv_neuron_id_track++;
            }
        }
    }
    conv_output_dims[0] = conv_output_dims_x;
    conv_output_dims[1] = conv_output_dims_y;
    conv_output_dims[2] = conv_kernel_dims[3];
}

void Model::Architecture::connToAct(unsigned cur_layer_id, unsigned next_layer_id)
{


	auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;

    auto &output_dims = layers[cur_layer_id].output_dims;
    auto &output_neuron_ids = layers[next_layer_id].output_neuron_ids;

    layers[next_layer_id].output_dims = layers[cur_layer_id].output_dims;
    std::string out_layer_name = layers[next_layer_id].name;

    uint64_t cur_layer_size = cur_neurons_ids.size();
    uint64_t out_neuron_id_track = cur_neurons_ids[cur_neurons_ids.size() - 1] + 1;

    uint64_t data_dim = 1;
    for (auto dim : cur_neurons_dims) { data_dim *= dim; }

    for (unsigned i = 0; i < cur_layer_size; i++)
    {
        for (unsigned j = 0; j < cur_layer_size; j++)
        {
            uint64_t cur_neuron_id = cur_neurons_ids[j];

            if(i==j)
            {
                if (auto iter = connections.find(cur_neuron_id);
                       iter != connections.end())
                {
                    (*iter).second.out_neurons_ids.push_back(out_neuron_id_track);
                    (*iter).second.weights.push_back(1);
                    (*iter).second.out_layer_name.push_back(out_layer_name);
                }
                else
                {
                    connections.insert({cur_neuron_id, {out_neuron_id_track, -1, out_layer_name}});
                }
            }
        }
        output_neuron_ids.push_back(out_neuron_id_track);
        out_neuron_id_track++;
    }
}

void Model::Architecture::connToPadding(unsigned cur_layer_id,
                                     unsigned next_layer_id)
{

    auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;

    // TODO: Assumption - maximum of only 3 dimensions will be processed.
    auto &padding_dims = layers[next_layer_id].output_dims;
    auto &output_neuron_ids = layers[next_layer_id].output_neuron_ids;

    padding_dims[0] = cur_neurons_dims[0] + layers[next_layer_id].padding[0];
    padding_dims[1] = cur_neurons_dims[1] + layers[next_layer_id].padding[1];
    padding_dims[2] = cur_neurons_dims[2];
    std::string out_layer_name = layers[next_layer_id].name;


    //output_neuron_ids
    uint64_t out_neuron_id_track = cur_neurons_ids[0];

    for (int k = 0; k < padding_dims[2]; k++)
    {
        for (int i = 0; i < padding_dims[0]; i++)
        {
            for (int j = 0; j < padding_dims[1]; j++)
            {
            	output_neuron_ids.push_back(out_neuron_id_track + ( k * padding_dims[0] * padding_dims[1] +
                                        i * padding_dims[1] + j));
            }
        }
    }

}

void Model::Architecture::connToNorm(unsigned cur_layer_id, unsigned next_layer_id)
{
    auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;


    auto &output_neuron_ids = layers[next_layer_id].output_neuron_ids;
    layers[next_layer_id].output_dims = layers[cur_layer_id].output_dims;
    uint64_t cur_layer_size = cur_neurons_ids.size();
    uint64_t out_neuron_id_track = cur_neurons_ids[cur_neurons_ids.size() - 1] + 1;
    std::string out_layer_name = layers[next_layer_id].name;

    uint64_t data_dim = 1;
    for (auto dim : cur_neurons_dims) { data_dim *= dim; }

    for (unsigned i = 0; i < cur_layer_size; i++)
    {
        for (unsigned j = 0; j < cur_layer_size; j++)
        {
            uint64_t cur_neuron_id = cur_neurons_ids[j];

            if(i==j)
            {
                if (auto iter = connections.find(cur_neuron_id);
                       iter != connections.end())
                {
                    (*iter).second.out_neurons_ids.push_back(out_neuron_id_track);
                    (*iter).second.weights.push_back(1);
                    (*iter).second.out_layer_name.push_back(out_layer_name);

                }
                else
                {
                    connections.insert({cur_neuron_id, {out_neuron_id_track, -1, out_layer_name}});
                }
            }
        }
        output_neuron_ids.push_back(out_neuron_id_track);
        out_neuron_id_track++;
    }
}

void Model::Architecture::connToDrop(unsigned cur_layer_id, unsigned next_layer_id)
{

}

void Model::Architecture::connToPool(unsigned cur_layer_id, unsigned next_layer_id)
{
    auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;

    auto &pool_kernel_dims = layers[next_layer_id].w_dims;
    auto &pool_strides = layers[next_layer_id].strides;
    auto &pool_output_dims = layers[next_layer_id].output_dims;
    auto &pool_output_neuron_ids = layers[next_layer_id].output_neuron_ids;

#if DEBUG
    std::cout << cur_layer_id << " : " << next_layer_id << std::endl;
    std::cout << "Input Add: "<< &cur_neurons_ids << std::endl;
    std::cout << "Output Add: "<<&pool_output_neuron_ids << std::endl;
#endif

    uint64_t pool_neuron_id_track = 
        cur_neurons_ids[cur_neurons_ids.size() - 1] + 1;

    unsigned pool_output_dims_x = 0;
    unsigned pool_output_dims_y = 0;

    pool_kernel_dims.push_back(cur_neurons_dims[2]);

    if (cur_neurons_dims[0] == 1){
    	pool_output_neuron_ids.push_back(pool_neuron_id_track);
    	return;}
    
    //if(pool_kernel_dims[3 == 0]) pool_kernel_dims[3] = 1;
    
    for (unsigned filter = 0; 
         filter < pool_kernel_dims[3]; 
         filter++)
    {
        pool_output_dims_x = 0;
        for (unsigned row = pool_kernel_dims[0] - 1; 
             row < cur_neurons_dims[0]; 
             row += pool_strides[0])
        {
            pool_output_dims_x++;

            pool_output_dims_y = 0;
            for (unsigned col = pool_kernel_dims[1] - 1; 
                 col < cur_neurons_dims[1]; 
                 col += pool_strides[1])
            {
                pool_output_dims_y++;

                // All neurons inside the current kernel
                unsigned starting_row = row + 1 - pool_kernel_dims[0];
                unsigned ending_row = row;
                unsigned starting_col = col + 1 - pool_kernel_dims[1];
                unsigned ending_col = col;

                for (unsigned i = starting_row; i <= ending_row; i++)
                {
                    for (unsigned j = starting_col; j <= ending_col; j++)
                    {
                        uint64_t cur_neuron_id = 
                            cur_neurons_ids[filter * cur_neurons_dims[0] * 
                                cur_neurons_dims[1] +
                            i * cur_neurons_dims[1] + j];
                        
                        // Record the connection information
                        if (auto iter = connections.find(cur_neuron_id);
                                 iter != connections.end())
                        {
                            (*iter).second.out_neurons_ids.push_back(
                                pool_neuron_id_track);
                            (*iter).second.weights.push_back(-1);
                            (*iter).second.out_layer_name.push_back(layers[next_layer_id].name);
                        }
                        else
                        {
                            connections.insert({cur_neuron_id, 
                                               {pool_neuron_id_track, 
                                                -1, layers[next_layer_id].name}});
                        }
                    }
                }
                pool_output_neuron_ids.push_back(pool_neuron_id_track);
                pool_neuron_id_track++;
            }
        }
    }

    pool_output_dims[0] = pool_output_dims_x;
    pool_output_dims[1] = pool_output_dims_y;
    pool_output_dims[2] = pool_kernel_dims[3];

}

void Model::Architecture::connToFlat(unsigned cur_layer_id, unsigned next_layer_id)
{
    auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;

    auto &output_dims = layers[next_layer_id].output_dims;
    auto &output_neuron_ids = layers[next_layer_id].output_neuron_ids;


    uint64_t out_neuron_id_track = cur_neurons_ids[cur_neurons_ids.size() - 1] + 1;

    uint64_t data_dim = 1;
    for (auto dim : cur_neurons_dims) { data_dim *= dim; }

    output_dims.push_back(data_dim);
    output_dims.push_back(1);
    output_dims.push_back(1);

    std::string out_layer_name = layers[next_layer_id].name;
    for (uint64_t i = 0; i < data_dim; i++)
    {
        uint64_t cur_neuron_id = cur_neurons_ids[i];
        
        if (auto iter = connections.find(cur_neuron_id);
               iter != connections.end())
        {
            (*iter).second.out_neurons_ids.push_back(out_neuron_id_track);
            (*iter).second.weights.push_back(-1);
            (*iter).second.out_layer_name.push_back(out_layer_name);
        }
        else
        {
            connections.insert({cur_neuron_id, {out_neuron_id_track, -1, out_layer_name}});
        }
        
        output_neuron_ids.push_back(out_neuron_id_track);
        out_neuron_id_track++;
    }
}

void Model::Architecture::connToDense(unsigned cur_layer_id, unsigned next_layer_id)
{
    auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;

    auto &dense_dims = layers[next_layer_id].w_dims;
    auto &dense_weights = layers[next_layer_id].weights;
    auto &output_dims = layers[next_layer_id].output_dims;
    auto &output_neuron_ids = layers[next_layer_id].output_neuron_ids;

    uint64_t out_neuron_id_track = cur_neurons_ids[cur_neurons_ids.size() - 1] + 1;
    std::string out_layer_name = layers[next_layer_id].name;

    uint64_t data_dim = 1;
    for (auto dim : cur_neurons_dims) { data_dim *= dim; }

    output_dims.push_back(dense_dims[1]);
    output_dims.push_back(1);
    output_dims.push_back(1);
    
    for (unsigned i = 0; i < dense_dims[1]; i++)
    {
        for (unsigned j = 0; j < dense_dims[0]; j++)
        {
            uint64_t cur_neuron_id = cur_neurons_ids[j];
            float weight = dense_weights[j * dense_dims[1] + i];
            
            if (auto iter = connections.find(cur_neuron_id);
                   iter != connections.end())
            {
                (*iter).second.out_neurons_ids.push_back(out_neuron_id_track);
                (*iter).second.weights.push_back(weight);
                (*iter).second.out_layer_name.push_back(out_layer_name);
            }
            else    
            {
                connections.insert({cur_neuron_id, {out_neuron_id_track, weight, out_layer_name}});
            }
        }
        output_neuron_ids.push_back(out_neuron_id_track);
        out_neuron_id_track++;
    }
}

void Model::Architecture::connToAdd(unsigned cur_layer_id, unsigned next_layer_id)
{
    auto &cur_neurons_dims = layers[cur_layer_id].output_dims;
    auto &cur_neurons_ids = layers[cur_layer_id].output_neuron_ids;

    auto &output_neuron_ids = layers[next_layer_id].output_neuron_ids;
    layers[next_layer_id].output_dims = layers[cur_layer_id].output_dims;
    std::string out_layer_name = layers[next_layer_id].name;

    uint64_t cur_layer_size = cur_neurons_ids.size();
    uint64_t out_neuron_id_track = cur_neurons_ids[cur_neurons_ids.size() - 1] + 1;

    uint64_t data_dim = 1;
    for (auto dim : cur_neurons_dims) { data_dim *= dim; }

    for (unsigned i = 0; i < cur_layer_size; i++)
    {
        for (unsigned j = 0; j < cur_layer_size; j++)
        {
            uint64_t cur_neuron_id = cur_neurons_ids[j];

            if(i==j)
            {
                if (auto iter = connections.find(cur_neuron_id);
                       iter != connections.end())
                {
                    (*iter).second.out_neurons_ids.push_back(out_neuron_id_track);
                    (*iter).second.weights.push_back(1);
                    (*iter).second.out_layer_name.push_back(out_layer_name);
                }
                else
                {
                    connections.insert({cur_neuron_id, {out_neuron_id_track, -1, out_layer_name}});
                }
            }
        }
        output_neuron_ids.push_back(out_neuron_id_track);
        out_neuron_id_track++;
    }
}
void Model::Architecture::setOutRoot(std::string &out_root)
{
    std::string conns_out_txt = out_root + "connection_info.txt";
    conns_output.open(conns_out_txt);

    std::string weights_out_txt = out_root + "weight_info.txt";
    weights_output.open(weights_out_txt);
}

void Model::Architecture::printConns(std::string &out_root)
{
    // Txt record
    std::string conns_out_txt = out_root + "lconnection_info.txt";
    std::ofstream conns_out(conns_out_txt);

    //std::string weights_out_txt = out_root + ".weight_info.txt";
    //std::ofstream weights_out(weights_out_txt);

    for (int i = 0; i < layers.size() - 1; i++)
    {
        auto &output_neurons = layers[i].output_neuron_ids;

        for (auto neuron : output_neurons)
        {            
            auto iter = connections.find(neuron);
            if (iter == connections.end()) { continue; }

            auto &out_neurons_ids = (*iter).second.out_neurons_ids;
            auto &weights = (*iter).second.weights;
            auto &layer_name = (*iter).second.out_layer_name;

           // weights_out << neuron << " ";
            conns_out << '(' <<neuron <<','<< layers[i].name << ')' << " ";
            for (unsigned j = 0; j < out_neurons_ids.size(); j++)
            {
              //  weights_out << weights[j] << " ";
                conns_out << "(" << out_neurons_ids[j] << "," << layer_name[j] << ")" << " ";
            }
            //weights_out << "\n";
            conns_out << "\n";
        }
    }
    //weights_out.close();
    conns_out.close();
}




// Call this  function for neurons information.


#ifdef NEURON
void Model::loadArchNeuron(std::string &arch_file_path)
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
                    //input_shape.push_back(cell.second.get_value<std::string>());
                    if (cell == nlohmann::detail::value_t::null) cell = 0; 
                    output_dims.push_back(cell);
                }
                
                //input_shape.erase(input_shape.begin());
                output_dims.erase(output_dims.begin()); //Delete the first null (see json for more details)
               
                std::string name = "input";
                Layer::Layer_Type layer_type = Layer::Layer_Type::Input;
                arch.addLayer(name, layer_type);
                arch.getLayer(name).setOutputDim(output_dims);
                
                auto &out_neuro_ids = arch.getLayer(name).output_neuron_ids;
                
                for (int k = 0; k < output_dims[2]; k++)
                {
                    for (int i = 0; i < output_dims[0]; i++)
                    {
                        for (int j = 0; j < output_dims[1]; j++)
                        {
                            out_neuro_ids.push_back(k * output_dims[0] * output_dims[1] +
                                                    i * output_dims[1] + j);
                        }
                    }
                }
                
                

                layer_counter++;
            }

            std::string class_name = layer["class_name"];
            std::string name = layer["config"]["name"];
            
            // ab3586
            #if 0
            std::cout << layer["name"] <<" : " << layer["class_name"] << std::endl;
            //std::cout <<"Layer Name: " << name << " Dimensions: " << out_neuro_ids.size()  << std::endl;
            #endif

            Layer::Layer_Type layer_type = Layer::Layer_Type::MAX;
            if (class_name == "InputLayer") { layer_type = Layer::Layer_Type::Input; }
            else if (class_name == "Conv2D" || class_name == "QConv2D") { layer_type = Layer::Layer_Type::Conv2D; }
            else if (class_name == "Activation" || class_name == "QActivation") {layer_type = Layer::Layer_Type::Activation; }
            else if (class_name == "BatchNormalization" || class_name == "QBatchNormalization") {layer_type = Layer::Layer_Type::BatchNormalization; }
            else if (class_name == "Dropout") { layer_type = Layer::Layer_Type::Dropout; }
            else if (class_name == "MaxPooling2D") { layer_type = Layer::Layer_Type::MaxPooling2D; }
            else if (class_name == "AveragePooling2D") { layer_type = Layer::Layer_Type::AveragePooling2D; }
            else if (class_name == "Flatten") { layer_type = Layer::Layer_Type::Flatten; }
            else if (class_name == "Dense" || class_name == "QDense") { layer_type = Layer::Layer_Type::Dense; }
            else if (class_name == "ZeroPadding2D") {layer_type = Layer::Layer_Type::ZeroPadding2D; }
            else if (class_name == "Concatenate") {layer_type = Layer::Layer_Type::Concatenate; }
            else if (class_name == "Add") {layer_type = Layer::Layer_Type::Add; }
            else if (class_name == "GlobalAveragePooling2D" || class_name == "GlobalMaxPooling2D") {layer_type = Layer::Layer_Type::GlobalPooling2D;}
            else { std::cerr << "Error: Unsupported layer type: " << class_name << std::endl; exit(0); }

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
                // We need pool_size since Conv2D's kernel size can be extracted from h5 file
                auto &pool_size = arch.getLayer(name).w_dims;
                
                for (auto &cell : layer["config"]["pool_size"]) {
                    pool_size.push_back(cell);
                }
                pool_size.push_back(1); // depth is 1

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

            layer_counter++;
            // TODO, more information to extract, such as activation method...
        }
        


        // Processing layers' connectivity
        for (auto& layer: json_layers) 
        {
            std::string layer_name = layer["name"];
        
            auto& ll = arch.getLayer(layer["name"]);

            //if(ll.layer_type == Layer::Layer_Type::Concatenate \
                  ||  ll.layer_type == Layer::Layer_Type::Dropout \
                   ||  ll.layer_type == Layer::Layer_Type::BatchNormalization\
                    || ll.layer_type == Layer::Layer_Type::Activation\
					|| ll.layer_type == Layer::Layer_Type::ZeroPadding2D)

            if(ll.layer_type == Layer::Layer_Type::Concatenate \
                   ||  ll.layer_type == Layer::Layer_Type::Dropout)
                  // ||  ll.layer_type == Layer::Layer_Type::BatchNormalization
                   // || ll.layer_type == Layer::Layer_Type::Activation\
					//|| ll.layer_type == Layer::Layer_Type::ZeroPadding2D)

            {
                continue;
			}

    
            for (auto& in_layer_info : layer["inbound_nodes"][0]) 
            {  
                std::string in_layer_s = (in_layer_info)[0];
               
                int new_layer = 1;

                //std::cout << layer_name << std::endl;          
                while(new_layer)
                {
                    auto& inLayer = arch.getLayer(in_layer_s);

                  //  if(inLayer.layer_type == Layer::Layer_Type::Concatenate \
                           ||  inLayer.layer_type == Layer::Layer_Type::Dropout \
                           ||  inLayer.layer_type == Layer::Layer_Type::BatchNormalization\
                            || inLayer.layer_type == Layer::Layer_Type::Activation\
							|| inLayer.layer_type == Layer::Layer_Type::ZeroPadding2D)

                    if(inLayer.layer_type == Layer::Layer_Type::Concatenate \
                           ||  inLayer.layer_type == Layer::Layer_Type::Dropout)
                           //||  inLayer.layer_type == Layer::Layer_Type::BatchNormalization
                           // || inLayer.layer_type == Layer::Layer_Type::Activation\
							//|| inLayer.layer_type == Layer::Layer_Type::ZeroPadding2D)

					{
                        
                        new_layer = 1; 
                        // find the layer info in the JSON file
                        for(auto& s_layer: json_layers)
                        {
                            std::string l_s = s_layer["name"];
                            if(l_s.compare(in_layer_s) == 0)
                            {
                                for(auto& inn_layer_info : s_layer["inbound_nodes"][0])
                                {
                                    std::string layer_s = (inn_layer_info)[0];
                                    new_layer = 0;

                                    auto& in_layer = arch.getLayer(layer_s);
                                    if(in_layer.layer_type == Layer::Layer_Type::Concatenate \
                                            || in_layer.layer_type == Layer::Layer_Type::Dropout )
                                           // || in_layer.layer_type == Layer::Layer_Type::BatchNormalization
                                         //   || in_layer.layer_type == Layer::Layer_Type::Activation\
										//	|| in_layer.layer_type == Layer::Layer_Type::ZeroPadding2D)

									{
                                        in_layer_s = layer_s;
                                        new_layer = 1;
                                    }
                                    else
                                    {
                                        arch.getLayer(layer_name).inbound_layers.push_back(layer_s);
                                        arch.getLayer(layer_s).outbound_layers.push_back(layer_name);
                                    }

                                }
                                break;
                            }

                        }
                       
                    }
                    else
                    {
                        arch.getLayer(layer_name).inbound_layers.push_back(in_layer_s);
                        arch.getLayer(in_layer_s).outbound_layers.push_back(layer_name);
                        new_layer = 0;
                        //std::cout << layer_name << " : "<< in_layer_s << std::endl;          
                    }
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
}
#endif

// Call function for Layer information extraction
#ifndef NEURON
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
            else if (class_name == "Activation" || class_name == "QActivation") {layer_type = Layer::Layer_Type::Ignore; }
            else if (class_name == "BatchNormalization") {layer_type = Layer::Layer_Type::Ignore; }
            else if (class_name == "Dropout") { layer_type = Layer::Layer_Type::Ignore; }
            else if (class_name == "MaxPooling2D") { layer_type = Layer::Layer_Type::MaxPooling2D; }
            else if (class_name == "AveragePooling2D") { layer_type = Layer::Layer_Type::AveragePooling2D; }
            else if (class_name == "Flatten") { layer_type = Layer::Layer_Type::Flatten; }
            else if (class_name == "Dense" || class_name == "QDense") { layer_type = Layer::Layer_Type::Dense; }
            else if (class_name == "ZeroPadding2D") {layer_type = Layer::Layer_Type::Ignore; }
            else if (class_name == "Concatenate") {layer_type = Layer::Layer_Type::Concatenate; }
            else if (class_name == "Add") {layer_type = Layer::Layer_Type::Add; }
            else if (class_name == "GlobalAveragePooling2D" || class_name == "GlobalMaxPooling2D") {layer_type = Layer::Layer_Type::GlobalPooling2D;}
            else { std::cerr << "Error: Unsupported layer type: " << class_name << std::endl; exit(0); }

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
                // We need pool_size since Conv2D's kernel size can be extracted from h5 file
                auto &pool_size = arch.getLayer(name).w_dims;
                
                for (auto &cell : layer["config"]["pool_size"]) {
                    pool_size.push_back(cell);
                }
                pool_size.push_back(1); // depth is 1

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

            layer_counter++;
            // TODO, more information to extract, such as activation method...
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
            printf("TESTCASE");
            std::string layer_name = layer["name"];
            auto& layer_obj = arch.getLayer(layer_name);
            std::vector<unsigned>& output_dims = layer_obj.output_dims;
            std::vector<unsigned> input_dims(3, 0);

            if (layer_obj.layer_type == Layer::Layer_Type::Input)
            {
                input_dims = layer_obj.output_dims;
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
}


void Model::Architecture::printSdfRep(std::string &out_root) {
    std::string layer_conn_out_txt = out_root + ".sdf_rep.txt";
    std::ofstream conns_out(layer_conn_out_txt);

    for (int i = 0; i < layers.size() - 1; i++) 
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
}

#endif

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
        else if (type == Layer::Layer_Type::GlobalAveragePooling2D) 
        { out_shape_f << "Layer type: GlobalAveragePooling2D"; }
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
        else if (type == Layer::Layer_Type::ZeroPadding2D) 
        { out_shape_f << "Layer type: ZeroPadding2D"; }
        else if (type == Layer::Layer_Type::Ignore) 
        { out_shape_f << "Ignored Layer"; }
        else { std::cerr << "Here Error: unsupported layer type: " << name << std::endl;  exit(0); }
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
//        for (auto& inneu_id : neuron.getInputNeuronIDList()) {
            metric += (layer.getDepth() - getLayer(in_layer_name).getDepth() - 1);
            num_connections++;
        }
    }
    std::pair<uint64_t, uint64_t> returnVal = std::make_pair(metric, num_connections);
    return returnVal;
}


void Model::loadWeights(std::string &weight_file)
{
    // Example on parsing H5 format
    hid_t file;
    hid_t gid; // group id
    herr_t status;

    // char model_path[MAX_NAME];

    // Open h5 model
    file = H5Fopen(weight_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    gid = H5Gopen(file, "/", H5P_DEFAULT); // open root
    scanGroup(gid);

    status = H5Fclose(file);
}

void Model::scanGroup(hid_t gid)
{
    ssize_t len;
    hsize_t nobj;
    herr_t err;
    int otype;
    hid_t grpid, dsid;
    char group_name[MAX_NAME];
    char memb_name[MAX_NAME];
    char ds_name[MAX_NAME];

    // Get number of objects in the group
    len = H5Iget_name(gid, group_name, MAX_NAME);
    err = H5Gget_num_objs(gid, &nobj);

    // Iterate over every object in the group
    for (int i = 0; i < nobj; i++)
    {
        // Get object type
        len = H5Gget_objname_by_idx(gid, (hsize_t)i, memb_name, (size_t)MAX_NAME);
        otype = H5Gget_objtype_by_idx(gid, (size_t)i);

        switch (otype)
        {
            // If it's a group, recurse over it
        case H5G_GROUP:
            grpid = H5Gopen(gid, memb_name, H5P_DEFAULT);
            scanGroup(grpid);
            H5Gclose(grpid);
            break;
            // If it's a dataset, that means group has a bias and kernel dataset
        case H5G_DATASET:
            dsid = H5Dopen(gid, memb_name, H5P_DEFAULT);
            H5Iget_name(dsid, ds_name, MAX_NAME);   
            // std::cout << ds_name << "\n";
            extrWeights(dsid);
            break;
        default:
            break;
        }
    }
}

void Model::extrWeights(hid_t id)
{
    hid_t datatype_id, space_id;
    herr_t status;
    hsize_t size;
    char ds_name[MAX_NAME];

    H5Iget_name(id, ds_name, MAX_NAME);
    space_id = H5Dget_space(id);
    datatype_id = H5Dget_type(id);

    // Get dataset dimensions to create buffer of same size
    const int ndims = H5Sget_simple_extent_ndims(space_id);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    // Calculating total 1D size
    unsigned data_size = 1;
    for (int i = 0; i < ndims; i++) { data_size *= dims[i]; }
    float *rdata = (float *)malloc(data_size * sizeof(float));
    status = H5Dread(id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    
    // Add information to the corres. layer
    std::stringstream full_name(ds_name);
    std::vector <std::string> tokens;
    std::string intermediate;

    while(getline(full_name, intermediate, '/'))
    {
        tokens.push_back(intermediate);
    }
    // The secondary last element indicates the layer name
    // TODO, I'm not sure if this is always true. Need to do more research
    //printf("Reached Exta Weights");
    
    if (tokens[tokens.size() - 1].find("kernel") != std::string::npos)
    {
    	// ab3586: changed tokens.size() -2

    	std::string name = "test";
    	if (tokens.size() == 6)
    	{    	// ab3586: changed tokens.size() -2
        	if (tokens.size() == 6)
        	{
        		name = tokens[tokens.size()-3]+'/'+tokens[tokens.size()-2];
        	}
        	else
        	{
        		name = tokens[tokens.size()-2];
        	}

    	}
    	else
    	{
    		name = tokens[tokens.size()-2];
    	}

        Layer &layer = arch.getLayer(name);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);
#if DEBUG
        //std::cout << "Layer name: " << tokens[tokens.size()-2] << std::endl;
        //std::cout << "Dims: " << dims << " " << dims + ndims<< std::endl;
#endif
        layer.setWeights(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("bias") != std::string::npos)
    {
    	std::string name = "test";
    	if (tokens.size() == 6)
    	{    	// ab3586: changed tokens.size() -2i
        	if (tokens.size() == 6)
        	{
        		name = tokens[tokens.size()-3]+'/'+tokens[tokens.size()-2];
        	}
        	else
        	{
        		name = tokens[tokens.size()-2];
        	}

    	}
    	else
    	{
    		name = tokens[tokens.size()-2];
    	}

        Layer &layer = arch.getLayer(name);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setBiases(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("beta") != std::string::npos)
    {
    	std::string name = "test";
    	if (tokens.size() == 6)
    	{    	// ab3586: changed tokens.size() -2i
        	if (tokens.size() == 6)
        	{
        		name = tokens[tokens.size()-3]+'/'+tokens[tokens.size()-2];
        	}
        	else
        	{
        		name = tokens[tokens.size()-2];
        	}

    	}
    	else
    	{
    		name = tokens[tokens.size()-2];
    	}
        Layer &layer = arch.getLayer(name);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setBeta(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("gamma") != std::string::npos)
    {
    	std::string name = "test";
    	if (tokens.size() == 6)
    	{    	// ab3586: changed tokens.size() -2i
        	if (tokens.size() == 6)
        	{
        		name = tokens[tokens.size()-3]+'/'+tokens[tokens.size()-2];
        	}
        	else
        	{
        		name = tokens[tokens.size()-2];
        	}

    	}
    	else
    	{
    		name = tokens[tokens.size()-2];
    	}
        Layer &layer = arch.getLayer(name);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setGamma(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("moving_mean") != std::string::npos)
    {
    	std::string name = "test";
    	if (tokens.size() == 6)
    	{    	// ab3586: changed tokens.size() -2i
        	if (tokens.size() == 6)
        	{
        		name = tokens[tokens.size()-3]+'/'+tokens[tokens.size()-2];
        	}
        	else
        	{
        		name = tokens[tokens.size()-2];
        	}

    	}
    	else
    	{
    		name = tokens[tokens.size()-2];
    	}
        Layer &layer = arch.getLayer(name);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setMovingMean(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("moving_variance") != std::string::npos)
    {
    	std::string name = "test";
    	if (tokens.size() == 6)
    	{    	// ab3586: changed tokens.size() -2i
        	if (tokens.size() == 6)
        	{
        		name = tokens[tokens.size()-3]+'/'+tokens[tokens.size()-2];
        	}
        	else
        	{
        		name = tokens[tokens.size()-2];
        	}

    	}
    	else
    	{
    		name = tokens[tokens.size()-2];
    	}
        Layer &layer = arch.getLayer(name);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setMovingVariance(dims_vec, rdata_vec);
    }
    free(rdata);
}

}
}
