#include "cudacut.h"
#include "CudaCut_kernel.cu"

CudaCut::CudaCut(int image_width, int image_height, int overlap_width)
    : height(image_height), width(overlap_width), image_width(image_width){
    graph_size = width*height;
    size_int = sizeof(int)*graph_size;


}
__global__ void warm_up_kernel(int width){
    unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tid = iy*width + ix;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}
void CudaCut::cudaWarmUp(){
    dim3 block(16, 16, 1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    warm_up_kernel<<<grid, block>>>(width);
    gpuErrChk(cudaDeviceSynchronize());
}

void CudaCut::h_mem_init()
{

    h_left_weight = (int*)malloc(size_int);
    h_right_weight = (int*)malloc(size_int);
    h_down_weight  = (int*)malloc(size_int);
    h_up_weight = (int*)malloc(size_int);
    h_graph_height  =  (int*)malloc(sizeof(int)*graph_size);

    h_excess_flow = (int*)malloc(sizeof(int)*graph_size);
    h_relabel_mask = (int*)malloc(size_int);
    h_height_backup = (int*)malloc(4*size_int);

    h_visited = (int*)malloc(sizeof(int)*graph_size);
    h_frontier = (bool*)malloc(sizeof(bool)*graph_size);
    h_visited_backward = (bool*)malloc(sizeof(bool)*graph_size);
    h_visited_forward = (bool*)malloc(sizeof(bool)*graph_size);

    h_m1 = (unsigned char *)malloc(sizeof(unsigned char)*graph_size);
    h_m2 = (unsigned char*)malloc(sizeof(unsigned char)*graph_size);
    h_process_area = (int*)malloc(size_int);
    h_horizontal = (int*)malloc(size_int + height*sizeof(int));
    h_vertical = (int*)malloc(size_int + width*sizeof(int));

    h_push_block_position = (int*)malloc(sizeof(int)*(5*height));
    h_up_right_sum = (int*)malloc(sizeof(int)*graph_size);
    h_up_left_sum = (int*)malloc(sizeof(int)*graph_size);
    h_down_right_sum = (int*)malloc(sizeof(int)*graph_size);
    h_down_left_sum = (int*)malloc(sizeof(int)*graph_size);
    h_bfs_counter = (int*)malloc(sizeof(int)*graph_size);
    h_active_node = (int*)malloc(sizeof(int)*6000);
    memset(h_active_node, 0, sizeof(int)*6000);



    // initial h_weight, h_flow from input

}

void CudaCut::d_mem_init()
{
//    gpuErrChk(cudaMalloc((void**)&d_left_weight, size_int));
//    gpuErrChk(cudaMalloc((void**)&d_right_weight, size_int));
//    gpuErrChk(cudaMalloc((void**)&d_down_weight, size_int));
//    gpuErrChk(cudaMalloc((void**)&d_up_weight, size_int));

    //gpuErrChk(cudaMalloc((void**)&d_graph_height, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_excess_flow, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_relabel_mask, size_int));
    gpuErrChk(cudaMalloc((void**)&d_height_backup, 4*size_int));

    gpuErrChk(cudaMalloc((void**)&d_visited, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_frontier, sizeof(bool)*graph_size));

    gpuErrChk(cudaMalloc((void**)&d_m1, size_int));
    gpuErrChk(cudaMalloc((void**)&d_m2, size_int));
    gpuErrChk(cudaMalloc((void**)&d_process_area, size_int));
    //gpuErrChk(cudaMalloc((void**)&d_horizontal, size_int + height*sizeof(int)));
    //gpuErrChk(cudaMalloc((void**)&d_vertical, size_int + width*sizeof(int)));

    gpuErrChk(cudaMalloc((void**)&d_push_block_position, sizeof(int)*(5*height)));
    gpuErrChk(cudaMalloc((void**)&d_up_right_sum, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_up_left_sum, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_down_right_sum, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_down_left_sum, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_bfs_counter, sizeof(int)*graph_size));

    cudaMallocManaged((void**)&d_left_weight, size_int);
    cudaMallocManaged((void**)&d_right_weight, size_int);
    cudaMallocManaged((void**)&d_down_weight, size_int);
    cudaMallocManaged((void**)&d_up_weight, size_int);
    cudaMallocManaged((void**)&d_horizontal, size_int + height*sizeof(int));
    cudaMallocManaged((void**)&d_vertical,  size_int + width*sizeof(int));
    cudaMallocManaged((void**)&d_graph_height, size_int);
    //cudaMallocManaged((void**)&d_relabel_mask, size_int);

}

__global__ void
setupGraph_kernel(int *d_horizontal, int *d_vertical, int *d_right_weight, int *d_left_weight, int *d_up_weight,
                  int *d_down_weight, int *d_excess_flow, int *d_push_block_position, int *d_graph_height,
                  int *d_relabel_mask, int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    d_right_weight[node_i] = d_horizontal[iy*(width+1) + ix+1];
    d_left_weight[node_i] = d_horizontal[iy*(width+1) + ix];

    d_down_weight[node_i] = d_vertical[(iy+1)*width + ix];
    d_up_weight[node_i] = d_vertical[node_i];
    d_excess_flow[node_i] = 0;
    d_graph_height[node_i] = width - ix - 1;
    d_relabel_mask[node_i] = 0;

    //blockIdx.x == 0? d_push_block_position[iy*5 + blockIdx.x] = 1 : d_push_block_position[iy*5 + blockIdx.x] = 0;
//    if(blockIdx.x == 0 && threadIdx.x == 1){
//        d_push_block_position[iy*5 + blockIdx.x] = 1;
//    }
//    if(blockIdx.x != 0 && threadIdx.x == 0){
//        d_push_block_position[iy*5 + blockIdx.x] = 0;
//    }

}

__global__ void
adjustGraph_kernel(int *d_excess_flow, int *d_left_weight, int *d_right_weight, int *d_down_weight, int*d_up_weight,
                   int *d_graph_height,int *d_up_right_sum, int *d_up_left_sum, int *d_down_right_sum, int *d_down_left_sum,
                   int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;

    if(ix == 0){
        //d_excess_flow[node_i] = -10000;
        d_right_weight[node_i] = 0;
        d_up_weight[node_i] = 0;
        d_down_weight[node_i] = 0;
        d_graph_height[node_i] = N;
    }
    if(ix == width-1){
        d_left_weight[node_i] = 0;
        d_up_weight[node_i] = 0;
        d_down_weight[node_i] = 0;
    }

    if(ix == 1){
        int tmp = d_left_weight[node_i];
        d_excess_flow[node_i] = tmp;
        d_left_weight[node_i] = tmp + tmp;
    }
//    __syncthreads();
//    d_up_right_sum[node_i] = d_right_weight[node_i] + d_up_weight[node_i];
//    d_up_left_sum[node_i] = d_left_weight[node_i] + d_up_weight[node_i];

//    d_down_right_sum[node_i] = d_right_weight[node_i] + d_down_weight[node_i];
//    d_down_left_sum[node_i] = d_left_weight[node_i] + d_down_weight[node_i];
//    if(ix % 20==0 && ix != 0){
//        int tmp = d_left_weight[node_i];
//        d_excess_flow[node_i] = tmp;
//        d_left_weight[node_i] = tmp + tmp;
//        d_right_weight[node_i - 1] = 0;
//    }

}
__global__ void
adjustGraph_kernel2(int *d_excess_flow, int *d_left_weight, int *d_right_weight, int *d_down_weight, int*d_up_weight,
                   int *d_graph_height, int *d_up_right_sum, int *d_up_left_sum, int *d_down_right_sum, int *d_down_left_sum,
                    int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    __shared__ int smem[20*8];
    int idx = (threadIdx.y *20) + threadIdx.x;
    ix > 0 && ix < width-1 ? smem[idx] = d_up_right_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();
    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }

        __syncthreads();
    }
    int min1 = smem[threadIdx.y * 20];
    __syncthreads();

    ix > 0 && ix < width-1 ? smem[idx] = d_up_left_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }

        __syncthreads();
    }

    int min2 = smem[threadIdx.y*20];
    __syncthreads();

    ix > 0 && ix < width-1 ? smem[idx] = d_down_right_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }

        __syncthreads();
    }

    int min3 = smem[threadIdx.y*20];
    __syncthreads();

    ix > 0 && ix < width-1 ? smem[idx] = d_down_left_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }

        __syncthreads();
    }

    int min4 = smem[threadIdx.y*20];
    __syncthreads();

    if(d_up_right_sum[node_i] == min1){
            d_right_weight[node_i] = 0;
            d_up_weight[node_i] = 0;
            d_excess_flow[node_i] = min1;
            //d_excess_flow[node_i - ix/2] = min1;
            //d_excess_flow[node_i + (width-ix)/2] = min1;
        }
    __syncthreads();

    if(d_up_left_sum[node_i] == min2){
        if(iy > 0)
            d_down_weight[node_i - width] = 0;
        d_right_weight[node_i - 1] = 0;
        //d_excess_flow[node_i] = min2;
    }

    if(d_down_right_sum[node_i] == min3){
            d_right_weight[node_i] = 0;
            d_down_weight[node_i] = 0;
            //d_excess_flow[node_i] = min3;
        }
    __syncthreads();

    if(d_down_left_sum[node_i] == min4){
        if(iy < height - 1)
            d_up_weight[node_i + width] = 0;
        d_right_weight[node_i - 1] = 0;
       //d_excess_flow[node_i] = min4;
    }

}

__global__ void
adjustGraph_kernel5(int *d_excess_flow, int *d_left_weight, int *d_right_weight, int *d_down_weight, int*d_up_weight,
                   int *d_graph_height, int *d_up_right_sum, int *d_up_left_sum, int *d_down_right_sum, int *d_down_left_sum,
                    int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int node_i = iy*width + ix;
        //printf("cond in push_block_kernel %d\n", cond);
        //int pos = d_push_block_position[iy*gridDim.x + blockIdx.x];
        //printf("pos of blockIdx.%d  %d\n", blockIdx.x, pos);
        //__shared__ int flow_block[10];
    __shared__ int smem[16*8];
    int idx = (threadIdx.y << 4) + threadIdx.x;
    ix > 0 && ix < width-1 ? smem[idx] = d_up_right_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();
//            if(x1 == 0 && y1 == 0 && blockIdx.y == 0){
//                for(int i = 0; i < 8; i++){
//                    for(int j = 0; j < 16; j++){
//                        printf("%d ", smem[(i+1)*18+j+1]);
//                    }
//                    printf("\n");
//                }
//            }
//            __syncthreads();
    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }

        __syncthreads();
    }
//            if(x1 == 0 && y1 == 0 && blockIdx.y == 0){
//                for(int i = 0; i < 8; i++){
//                    for(int j = 0; j < 16; j++){
//                        printf("%d ", smem[(i+1)*18+j+1]);
//                    }
//                    printf("\n");
//                }
//            }
//            __syncthreads();
    // min1
    int min = smem[threadIdx.y << 4];
    __syncthreads();
    if(d_up_right_sum[node_i] == min){
            d_right_weight[node_i] = 0;
            d_up_weight[node_i] = 0;
            d_excess_flow[node_i] = min;
        }
    //__syncthreads();

    ix > 0 && ix < width-1 ? smem[idx] = d_up_left_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }
        __syncthreads();
    }
    // min2
    min = smem[threadIdx.y << 4];
    __syncthreads();
    if(d_up_left_sum[node_i] == min){
        if(iy > 0)
            d_down_weight[node_i - width] = 0;
        d_right_weight[node_i - 1] = 0;
        //d_excess_flow[node_i] = min;
    }

    ix > 0 && ix < width-1 ? smem[idx] = d_down_right_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }
        __syncthreads();
    }
    // min3
    min = smem[threadIdx.y << 4];
    __syncthreads();
    if(d_down_right_sum[node_i] == min){
            d_right_weight[node_i] = 0;
            d_down_weight[node_i] = 0;
            //d_excess_flow[node_i] = min;
        }

    ix > 0 && ix < width-1 ? smem[idx] = d_down_left_sum[node_i] : smem[idx] = 1<<20;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

        }
        __syncthreads();
    }
    // min4
    min = smem[threadIdx.y << 4];
    __syncthreads();
    if(d_down_left_sum[node_i] == min){
        if(iy < height - 1)
            d_up_weight[node_i + width] = 0;
        d_right_weight[node_i - 1] = 0;
        //d_excess_flow[node_i] = min;
    }

}

int CudaCut::cudaCutsSetupGraph(cv::Mat& img1, cv::Mat& img2){
//    int x = 40;
//    int y = (40-x)/2;
    cv::Mat area1 (img1, cv::Rect(img1.cols-OVERLAP_WIDTH, 0, OVERLAP_WIDTH, img1.rows) );
    cv::Mat area2 (img2, cv::Rect(0, 0, OVERLAP_WIDTH, img2.rows) );

    cv::Mat m1, m2;
    area1.convertTo(m1,CV_8UC1);
    area2.convertTo(m2,CV_8UC1);
    int xoffset = img1.cols - OVERLAP_WIDTH;

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width - 1; x++)
        {
            uchar a0 = img1.at<uchar>(y, xoffset + x);
            uchar b0 = img2.at<uchar>(y, x);
            uchar cap0 = abs(a0 - b0);

            uchar a1 = img1.at<uchar>(y, xoffset + x+1);
            uchar b1 = img2.at<uchar>(y, x + 1);
            uchar cap1 = abs(a1 - b1);

            d_horizontal[y * (width+1) + x+1] = (int)(cap0 + cap1);
        }

        d_horizontal[y * (width+1) + width] = 0;
        d_horizontal[y * (width+1)] = 0;
    }

    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height - 1; y++)
        {
            uchar a0 = img1.at<uchar>(y, xoffset + x);
            uchar b0 = img2.at<uchar>(y, x);
            uchar cap0 = abs(a0 - b0);

            uchar a1 = img1.at<uchar>(y + 1, xoffset + x);
            uchar b1 = img2.at<uchar>(y + 1, x);
            uchar cap1 = abs(a1 - b1);
            d_vertical[(y+1) * width + x] = (int)(cap0 + cap1);
            //            vertical[y * graph.width + x] = 0;
        }
        d_vertical[(height) * width + x] = 0;
        d_vertical[x] = 0;
    }
    memcpy(h_m1, m1.ptr(0), sizeof(unsigned char)*graph_size);
    memcpy(h_m2, m2.ptr(0), sizeof(unsigned char)*graph_size);

    dim3 block(16, 8, 1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    dim3 block1(20, 8, 1);
    dim3 grid1((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    dim3 block2(1, 480, 1);
    dim3 grid2((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    setupGraph_kernel<<<grid, block>>>(d_horizontal, d_vertical, d_right_weight, d_left_weight, d_up_weight,
                      d_down_weight, d_excess_flow, d_push_block_position, d_graph_height,
                                       d_relabel_mask, width, height, graph_size);
    adjustGraph_kernel<<<grid, block>>>(d_excess_flow, d_left_weight,d_right_weight, d_down_weight, d_up_weight, d_graph_height,
                                        d_up_right_sum, d_up_left_sum, d_down_right_sum, d_down_left_sum, width, height, graph_size);
    //    gpuErrChk(cudaMemcpy(h_right_weight, d_right_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    //    gpuErrChk(cudaMemcpy(h_left_weight, d_left_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    //    gpuErrChk(cudaMemcpy(h_down_weight, d_down_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    //    gpuErrChk(cudaMemcpy(h_up_weight, d_up_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    cudaDeviceSynchronize();
//        gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));

//        gpuErrChk(cudaMemcpy(h_up_right_sum, d_up_right_sum, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        gpuErrChk(cudaMemcpy(h_up_left_sum, d_up_left_sum, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_right_weight_initial.txt", d_right_weight, width, height);
//        writeToFile("../variable/h_left_weight_initial.txt", d_left_weight, width, height);
//        writeToFile("../variable/h_down_weight_initial.txt", d_down_weight, width, height);
//        writeToFile("../variable/h_up_weight_initial.txt", d_up_weight, width, height);
//        writeToFile("../variable/h_excess_flow_initial.txt", h_excess_flow, width, height);
//        writeToFile("../variable/h_horizontal_initial.txt", d_horizontal, width+1, height);
//        writeToFile("../variable/h_vertical_initial.txt", d_vertical, width, height+1);
//        writeToFile("../variable/h_graph_height_initial.txt", d_graph_height, width, height);
//        writeToFile("../variable/h_up_right_sum_initial.txt", h_up_right_sum, width, height);
//        writeToFile("../variable/h_up_left_sum_initial.txt", h_up_left_sum, width, height);
//        while(getchar() != 32);
    //adjustGraph_kernel5<<<grid, block>>>(d_excess_flow, d_left_weight,d_right_weight, d_down_weight, d_up_weight, d_graph_height,
       //                                d_up_right_sum, d_up_left_sum, d_down_right_sum, d_down_left_sum, width, height, graph_size);
    cudaDeviceSynchronize();
    return 0;
}

int CudaCut::cudaCutsInit(){
    h_mem_init();
    d_mem_init();
    return 0;
}


void CudaCut::cudaCutsAtomic(int blockDimy, int number_loops){
    //size = 80*480
    dim3 block(16, blockDimy, 1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    int h_finished_count;

    int *d_finished_count;


    //cudaMallocManaged((void**)&d_finished_count, sizeof(int));
    gpuErrChk(cudaMalloc((void**)&d_finished_count, sizeof(int)));
    h_finished_count = 1;
    int counter = 0;
    int sum;
    while(h_finished_count != 0){



        push_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                                     d_excess_flow, d_graph_height, d_relabel_mask, d_height_backup,
                                     width, height, graph_size);
//        push_kernel2<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
//                                     d_excess_flow, d_graph_height, d_relabel_mask, d_height_backup,
//                                     width, height, graph_size);
        if((counter+1)%256 == 0){
            gpuErrChk(cudaDeviceSynchronize());

//            h_finish_bfs = 1;
//            std::cout << "stop 1" << std::endl;
//            for(int i = 0; i < graph_size; i++){
//                if((i+1)%width != 0)
//                    h_visited[i] = -1;
//                else
//                    h_visited[i] = 0;
//            }

//            gpuErrChk(cudaMemcpy(d_visited, h_visited, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
            auto start1 = getMoment;
            memset(h_bfs_counter, 0, sizeof(int)*graph_size);
            //std::cout << " stop 1 " << std::endl;

            for(int i = 0; i < graph_size; i++){
                if((i+1)%width != 0)
                    h_visited_backward[i] = false;
                else
                    h_visited_backward[i] = true;
            }
            globalRelabelCpu(d_right_weight, d_left_weight, d_down_weight, d_up_weight,
                             h_visited_backward, d_graph_height, h_bfs_counter);

//            while(h_finish_bfs != 0){
//                h_finish_bfs = 0;
//                //std::cout << "stop 7" << std::endl;
//                gpuErrChk(cudaMemcpy(d_finish_bfs, &h_finish_bfs, sizeof(int), cudaMemcpyHostToDevice));
//                gpuErrChk(cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice));

//                backward_bfs_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
//                            d_right_flow, d_legpuErrChk(cudaFree(d_down_left_sum));ft_flow, d_up_flow, d_down_flow, d_visited, d_frontier, d_graph_height,
//                            width, height, graph_size, d_finish_bfs, d_count);


//                gpuErrChk(cudaMemcpy(&h_finish_bfs, d_finish_bfs, sizeof(int), cudaMemcpyDeviceToHost));
//                //gpuErrChk(cudaDeviceSynchronize());
//                h_count++;

//            }
//            //gpuErrChk(cudaDeviceSynchronize());
            auto end1 = getMoment;
            std::cout << "Global Relabel Time = "<< TimeCpu(end1, start1)/1000.0 << "\n";
//            h_count = 0;


        }
        if(counter%2 == 0){
            //std::cout << " stop 4 " << std::endl;
            h_finished_count = 0;
            gpuErrChk(cudaMemcpy(d_finished_count, &h_finished_count, sizeof(int), cudaMemcpyHostToDevice));
            check_finished_condition<<<grid, block>>>(d_excess_flow, d_finished_count, width, height, graph_size);
            relabel_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                                                d_graph_height, d_relabel_mask, d_height_backup,
                                                d_excess_flow, width, height, graph_size);

            gpuErrChk(cudaMemcpy(&h_finished_count, d_finished_count, sizeof(int), cudaMemcpyDeviceToHost));
            //gpuErrChk(cudaDeviceSynchronize());
            //}
                //std::cout << "number active nodes " << *h_finished_count << std::endl;
                //h_active_node[counter/2] = counter;
                //h_active_node[counter/2 + 3000] =  *h_finished_count;
        }
        counter++;
        if(counter == number_loops)
            h_finished_count = 0;
    }
    gpuErrChk(cudaDeviceSynchronize());
    //vec1.push_back(t);
    //std::cout << "kernel Time = "<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << std::endl;
    cout << "couter = " << counter << endl;
    cudaFree(d_finished_count);
    //writeToFile("../variable/h_active_node.txt", h_active_node, 3000, 2);

//    gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    //gpuErrChk(cudaDeviceSynchronize());
//    sum = 0;
//    for(int i = 0; i < graph_size; i++){
//        if((i+1)%width == 0)
//            sum += h_excess_flow[i];
//    }


//    std::cout << "max flow: " << sum << std::endl;
//    std::cout << "final_excess_flow " << std::endl;
//    std::cout << "\n";


}

int CudaCut::cudaCutsAtomicOptimize(cv::Mat& result)
{
    //cudaCutsAtomic(result);
    //bfsLabeling();

    return 0 ;

}

void CudaCut::cudaCutsFreeMem()
{

    free(h_left_weight);
    free(h_right_weight);
    free(h_down_weight);
    free(h_up_weight);

    free(h_excess_flow);
    free(h_relabel_mask);
    free(h_graph_height);
    free(h_height_backup);
    free(h_visited);
    free(h_frontier);
    free(h_visited_backward);
    free(h_visited_forward);

    free(h_m1);
    free(h_m2);
    free(h_process_area);
    free(h_horizontal);
    free(h_vertical);
    free(h_push_block_position);
    free(h_up_right_sum);
    free(h_up_left_sum);
    free(h_down_right_sum);
    free(h_down_left_sum);
    free(h_bfs_counter);
    free(h_active_node);




    gpuErrChk(cudaFree(d_left_weight));
    gpuErrChk(cudaFree(d_right_weight));
    gpuErrChk(cudaFree(d_down_weight));
    gpuErrChk(cudaFree(d_up_weight));

    gpuErrChk(cudaFree(d_excess_flow));
    gpuErrChk(cudaFree(d_relabel_mask));
    gpuErrChk(cudaFree(d_graph_height));
    gpuErrChk(cudaFree(d_height_backup));
    gpuErrChk(cudaFree(d_visited));
    gpuErrChk(cudaFree(d_frontier));

    gpuErrChk(cudaFree(d_m1));
    gpuErrChk(cudaFree(d_m2));
    gpuErrChk(cudaFree(d_process_area));
    gpuErrChk(cudaFree(d_horizontal));
    gpuErrChk(cudaFree(d_vertical));
    gpuErrChk(cudaFree(d_push_block_position));
    gpuErrChk(cudaFree(d_up_right_sum));
    gpuErrChk(cudaFree(d_up_left_sum));
    gpuErrChk(cudaFree(d_down_right_sum));
    gpuErrChk(cudaFree(d_down_left_sum));
    gpuErrChk(cudaFree(d_bfs_counter));

}

