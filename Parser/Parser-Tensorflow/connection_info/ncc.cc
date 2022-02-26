#include <stdlib.h>

#include "model.h"
// #include "snn_converter/ann_to_snn.h"

int main(int argc, char *argv[])
{
    std::vector<std::string> args(argv, argv+argc);
    if ((argc > 4) || (argc < 2))
    {
        std::cout << "Incorrect Usage\n";
        std::cout << "Do this:\n";
        std::cout << "./ncc <arch_file> <weight_file>\n";
        std::cout << "or this:\n";
        std::cout << "./ncc <arch_file>\n";
        return 1;
    }
    std::string arch_file(argv[1]);
    std::string weight_file;

    if (argc == 3) {
        weight_file = args[2];
    } else {
        weight_file = "";
    }

    std::cout << "Loading Architecture File!" << std::endl;
    NCC::NCC_FrontEnd::Model model(arch_file, weight_file);
    std::cout << "Generating Connection Information!" << std::endl;
    model.connector();
    std::string out_root = arch_file.substr(0, arch_file.find(".json"));
    std::cout << "Printing Connection Information!" << std::endl;
    model.printConns(out_root);
    std::cout << "Output file location: " << "./" << out_root << std::endl;
}
