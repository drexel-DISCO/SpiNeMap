#include <stdlib.h>

#include "model.h"


/* Calculate combinatorial n choose 2*/
uint64_t nC2(uint64_t n) {
    return n * (n-1) / 2;
}

int main(int argc, char *argv[])
{
    std::vector<std::string> args(argv, argv+argc);
    if ((argc > 4) || (argc < 2))
    {
        std::cout << "Incorrect Usage\n";
        std::cout << "Do this:\n";
        std::cout << "./ncc <arch_file> --layer\n";
        return 1;
    }
    std::string arch_file(argv[1]);
    std::string weight_file;

    if (argc == 4) {
        weight_file = args[2];
    } else {
        weight_file = "";
    }
    
    NCC::NCC_FrontEnd::Model model(arch_file, weight_file);
    
    if (args[2] == "--layer") {
        std::cout << "outputting layer connection stats\n";
        
        std::string out_root = arch_file.substr(0, arch_file.find(".json"));
        std::string outputIRFile = out_root + ".layer_depth.txt";
        
        model.printSdfRep(out_root);
        model.printLayerConns(out_root);
        
        std::pair<uint64_t, uint64_t> irr_metric = model.getIrregularMetric();
        uint64_t metric = std::get<0>(irr_metric);
        uint64_t num_connections = std::get<1>(irr_metric);
        uint64_t max_depth = model.getMaxDepth();
        
        std::cout << num_connections << ", " << metric << ", "  
                  << max_depth << ", "
                  << (float)metric/(float)num_connections << ", " 
                  << (float)metric/(float)nC2(max_depth) << std::endl;

    } 
}
