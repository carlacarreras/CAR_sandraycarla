#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include <future>
#include <omp.h>


Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0,0,0,-1); kernel.set(0,1,0,2); kernel.set(0,2,0,-2); kernel.set(0,3,0,2); kernel.set(0,4,0,-1);
    kernel.set(1,0,0,2); kernel.set(1,1,0,-6); kernel.set(1,2,0,8); kernel.set(1,3,0,-6); kernel.set(1,4,0,2);
    kernel.set(2,0,0,-2); kernel.set(2,1,0,8); kernel.set(2,2,0,-12); kernel.set(2,3,0,8); kernel.set(2,4,0,-2);
    kernel.set(3,0,0,2); kernel.set(3,1,0,-6); kernel.set(3,2,0,8); kernel.set(3,3,0,-6); kernel.set(3,4,0,2);
    kernel.set(4,0,0,-1); kernel.set(4,1,0,2); kernel.set(4,2,0,-2); kernel.set(4,3,0,2); kernel.set(4,4,0,-1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    return (size == 3) ? get_srm_3x3() : get_srm_5x5();
}


Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size) {
    auto begin= std::chrono::steady_clock::now();
    std::cout << "[SRM " << kernel_size << "x" << kernel_size << "] Iniciando..." << std::endl;

    Image<float> srm = image.to_grayscale().convert<float>();

    srm = srm.convolution(get_srm_kernel(kernel_size));
    srm = srm.abs().normalized();
    srm = srm * 255;
    Image<unsigned char> result = srm.convert<unsigned char>();

    auto end = std::chrono::steady_clock::now();
    std::cout << "[SRM " << kernel_size << "x" << kernel_size << "] Tiempo: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    return result;
}


Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "[DCT" << (invert ? " inversa" : " directa") << " " << block_size
              << "x" << block_size << "] Iniciando..." << std::endl;

    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);
    int num_blocks = (int)blocks.size();


    #pragma omp parallel for schedule(dynamic, 4) shared(blocks) default(none) firstprivate(block_size, invert, num_blocks)
    for (int i = 0; i < num_blocks; i++) {
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
            for (int k = 0; k < blocks[i].size / 2; k++)
                for (int l = 0; l < blocks[i].size / 2; l++)
                    dctBlock[k][l] = 0.0;
            dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }

    Image<unsigned char> result = grayscale.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout << "[DCT" << (invert ? " inversa" : " directa") << "] Tiempo: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    return result;
}
Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "[ELA quality=" << quality << "] Iniciando..." << std::endl;


    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp_ela.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp_ela.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>() * (-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();

    auto end = std::chrono::steady_clock::now();
    std::cout << "[ELA] Tiempo: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    return result;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        std::cerr << "Uso: ./detect_parallel <imagen>" << std::endl;
        exit(1);
    }

    int num_threads = omp_get_max_threads();
    if (argc >= 3) {
        num_threads = std::atoi(argv[2]);
        omp_set_num_threads(num_threads);
    }
    std::cout << "Hilos OpenMP configurados: " << num_threads << std::endl;

    int block_size = 8;

    auto t_global_start = std::chrono::steady_clock::now();

    Image<unsigned char> image = load_from_file(argv[1]);
    std::cout << "Imagen cargada: " << image.width << "x" << image.height
              << " (" << image.channels << " canales)" << std::endl;


    std::cout << "\n--- Lanzando procesos en paralelo (std::async) ---" << std::endl;

    auto future_srm3 = std::async(std::launch::async, compute_srm, std::cref(image), 3);
    auto future_srm5 = std::async(std::launch::async, compute_srm, std::cref(image), 5);
    auto future_ela  = std::async(std::launch::async, compute_ela,  std::cref(image), 90);
    auto future_dct_inv = std::async(std::launch::async, compute_dct, std::cref(image), block_size, true);
    auto future_dct_dir = std::async(std::launch::async, compute_dct, std::cref(image), block_size, false);

    save_to_file("srm_kernel_3x3.png", future_srm3.get());
    save_to_file("srm_kernel_5x5.png", future_srm5.get());
    save_to_file("ela.png",            future_ela.get());
    save_to_file("dct_invert.png",     future_dct_inv.get());
    save_to_file("dct_direct.png",     future_dct_dir.get());

    auto t_global_end = std::chrono::steady_clock::now();
    std::cout << "\n=== Tiempo total paralelo: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_global_end - t_global_start).count()
              << " ms ===" << std::endl;

    return 0;
}
