#ifndef CUDACUT_H
#define CUDACUT_H

#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include<opencv2/opencv.hpp>
#include <chrono>

#define WIDTH 80
#define HEIGHT 480
#define getMoment std::chrono::high_resolution_clock::now()
using namespace std;
using namespace cv;

#define gpuErrChk(call) {gpuError((call));}
inline void gpuError(cudaError_t call){
    const cudaError_t error= call;
    if(error != cudaSuccess){
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
inline void writeToFile(char * filename,int *data, int width, int height)
{
    printf("Writing to file ...\n");
    std::cout << filename << std::endl;
    FILE* file;
    file = fopen(filename, "w");
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            fprintf(file, "%d ", data[y * width + x]);
        }
        fprintf(file, "\n");
    }
//    fprintf(file, "%d ", data[width*height]);
//    fprintf(file, "%d ", data[width*height+1]);
    fclose(file);
}
class CudaCut
{
public:
    CudaCut();
    CudaCut(int row, int col, cv::Mat& img1, cv::Mat& img2);

public:
    bool push(int u);
    void relabel(int u);
    void preflow(int s);
    void updateReverseEdgeFlow(int i, int flow);

    void addEdge(int u, int v, int capacity);

    int getMaxFlow(int s, int t);
    void h_mem_init();
    void d_mem_init();

    int cudaMemoryInit();
    int cudaCutsInit();
    int cudaCutsSetupDataTerm(int *);
    int cudaCutsSetupSmoothTerm(int *);
    int cudaCutsSetupHCue(int *);
    int cudaCutsSetupVCue(int *);

    // This function constructs the graph on the device
    int cudaCutsSetupGraph();

    int setupGraphforTextureSynthesis();

    // This function calls the Cuda Cuts optimization algorithm and
    // bfs algorithm to assign a label to each pixel
    int cudaCutsNonAtomicOptimize();

    // This function calls 3 kernels which performs the push, pull and relabel operation
    void cudaCutsNonAtomic();

    // This finds which of the nodes are in source set and sink set
    void bfsLabeling();

    int cudaCutsAtomicOptimize();
    void cudaCutsAtomic();

    // This function assigns a label to each pixel and stores them in pixelLabel
    // array of size width * height
    int cudaCutsGetResult();
    // De-allocates all the memory allocated on the host and the device
    void cudaCutsFreeMem();
    // Functions calculates the total energy of the configuration
    int cudaCutsGetEnergy();
    int data_energy();
    int smooth_energy();

public:
    /*************************************************
     * n-edges and t-edges                          **
     * **********************************************/
    int width, height, graph_size, size_int, graph_size1;
    cv::Mat img1, img2, process_are, result;
    dim3 grid, block;

    int *d_left_weight, *d_right_weight, *d_down_weight, *d_up_weight;
    int *d_left_flow, *d_right_flow, *d_down_flow, *d_up_flow;
    int *d_excess_flow;
    int *d_relabel_mask;
    int *d_graph_height;
    int *d_push_state;
    int *d_height_backup;
    int *d_excess_flow_backup;
    int *d_visited; //for bfs
    bool *d_frontier; //for bfs
    int *d_m1, *d_m2, *d_process_area;
    int *d_pull_left, *d_pull_right, *d_pull_down, *d_pull_up, *d_graph_heightr, *d_graph_heightw;
    int *d_sink_weight;

    int *h_left_weight, *h_right_weight, *h_down_weight, *h_up_weight;
    int *h_left_flow, *h_right_flow, *h_down_flow, *h_up_flow;
    int *h_excess_flow;
    int *h_relabel_mask;
    int *h_graph_height;
    int *h_push_state;
    int *h_height_backup;
    int *h_excess_flow_backup;

    int *h_visited; // for bfs
    bool *h_frontier; // for bfs
    int *h_m1, *h_m2, *h_process_area;
    int *h_pull_left, *h_pull_right, *h_pull_down, *h_pull_up, *h_graph_heightr, *h_graph_heightw;
    int *h_sink_weight;

//    int *s_left_weight, *s_right_weight, *s_down_weight, *s_up_weight, *s_push_reser, *s_sink_weight;
//    int *d_pull_left, *d_pull_right, *d_pull_down, *d_pull_up;
//    int *h_push_reser, *h_sink_weight;
//    int *h_left_weight, *h_right_weight, *h_down_weight, *h_up_weight;


//    int *d_stochastic,  *d_stochastic_pixel ;

//    /*************************************************
//     * Energy parameters stored                     **
//     * **********************************************/

//    int *hcue, *vcue, *datacost, *smoothnesscost ;
//    int *dataTerm, *smoothTerm, *hCue, *vCue ;
//    int *dDataTerm, *dSmoothTerm, *dHcue, *dVcue, *dPixelLabel ;


//    /*************************************************
//     * Height and mask functions are stored         **
//     * **********************************************/

//    // 2 array for heights (double buffering to read and write
//    int *h_file_graph_height;

//    /*************************************************
//     * Grid and Block parameters                    **
//     * **********************************************/

//    int graph_size, size_int, width, height, graph_size1, width1, height1, depth, num_Labels;
//    int blocks_x, blocks_y, threads_x, threads_y, num_of_blocks, num_of_threads_per_block ;

//    /***************************************************
//     * Label of each pixel is stored in this function **
//     * *************************************************/

//    int *pixelLabel ;

//    bool *d_pixel_mask, h_over, *d_over, *h_pixel_mask ;
//    int *d_counter, *h_graph_height ;
//    int *h_reset_mem ;
//    int cueValues, deviceCheck, deviceCount ;

//    int *h_stochastic, *h_stochastic_pixel, *h_relabel_mask ;
//    int counter ;

};

#endif // CUDACUT_H
