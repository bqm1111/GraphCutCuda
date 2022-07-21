#ifndef CUDACUT_H
#define CUDACUT_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>

#define WIDTH 640
#define HEIGHT 480
#define threadPerBlock_x 16
#define threadPerBlock_y 8
#define OVERLAP_WIDTH 80
#define getMoment std::chrono::high_resolution_clock::now()
#define TimeCpu(end,start) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
using namespace std;
using namespace cv;

enum class Colors : unsigned char
{
    Gray,
    RGB
};

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
    CudaCut(int image_width, int image_height, int overlap_width);
    void graphCorrectionImage(unsigned char* data, int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight);

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
    void cudaWarmUp();

    // This function constructs the graph on the device
    int cudaCutsSetupGraph(cv::Mat& img1, cv::Mat& img2);

    int setupGraphforTextureSynthesis();

    // This function calls the Cuda Cuts optimization algorithm and
    // bfs algorithm to assign a label to each pixel
    int cudaCutsNonAtomicOptimize();

    // This function calls 3 kernels which performs the push, pull and relabel operation
    void cudaCutsNonAtomic(cv::Mat& result);

    // This finds which of the nodes are in source set and sink set
    void bfsLabeling();

    int cudaCutsAtomicOptimize(cv::Mat& result);
    void cudaCutsAtomic(int blockDimy, int number_loops, int backwardCycle,
                        int relabelCycle, int averageDistance, int stopPoint);

    // This function assigns a label to each pixel and stores them in pixelLabel
    // array of size width * height
    int cudaCutsGetResult();
    // De-allocates all the memory allocated on the host and the device
    void cudaCutsFreeMem();
    // Functions calculates the total energy of the configuration
    int cudaCutsGetEnergy();
    int data_energy();
    int smooth_energy();
    void globalRelabelCpu(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight,
                          bool *visited, int *h_graph_height, int *h_bfs_counter, int *d_excess_flow);
    void forwardBfsCpu(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight,
                                   bool *visited, int *h_graph_height, int *h_bfs_counter, int *d_excess_flow);
    int BfsCpuBackward(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight, bool *visited);
    int BfsCpuForward(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight, bool *visited, int *d_excess_flow);
    void getStitchingImage(cv::Mat& result, cv::Mat& result1, Colors color = Colors::Gray);
    //void selectPix(cv::Mat& result, cv::Mat& result1);
    void selectPix(cv::Mat& result, cv::Mat& result1, bool *visited, Colors color = Colors::Gray);
    void findMin(int * src, int * dst, int width, int height);
    void findMinCol(int * src, int * dst, int width, int height);
public:
    /*************************************************
     * n-edges and t-edges                          **
     * **********************************************/
    vector<int> vec1, vec2, vec3, vec4, vec5;
    int width, height, graph_size, size_int, graph_size1, image_width;
    cv::Mat img1, img2, process_are, result;
    dim3 grid, block;

    int *d_left_weight, *d_right_weight, *d_down_weight, *d_up_weight;
    int *d_left_weight1, *d_right_weight1, *d_down_weight1, *d_up_weight1;
    int *d_excess_flow;
    int *d_relabel_mask;
    int *d_graph_height;
    int *d_height_backup;
    int *d_visited; //for bfs
    bool *d_frontier; //for bfs

    int *d_m1, *d_m2, *d_process_area, *d_horizontal, *d_vertical;
    int *d_push_block_position;
    int *d_up_right_sum, *d_up_left_sum;
    int *d_down_right_sum, *d_down_left_sum;
    int *d_bfs_counter;
    int *d_min_block;
    int *d_min_col;

    int *h_left_weight, *h_right_weight, *h_down_weight, *h_up_weight;
    int *h_excess_flow;
    int *h_relabel_mask;
    int *h_graph_height;
    int *h_height_backup;

    int *h_visited; // for bfs
    bool *h_frontier; // for bfs
    bool *h_visited_backward, *h_visited_forward;


    unsigned char *h_m1, *h_m2;
    int *h_process_area, *h_horizontal, *h_vertical;
    int *h_push_block_position;
    int *h_up_right_sum, *h_up_left_sum;
    int *h_down_right_sum, *h_down_left_sum;
    int *h_bfs_counter;
    int *h_active_node;
    unsigned char* data_;
    int *h_average_active;

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
