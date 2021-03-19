#include "cudacut.h"
#include "CudaCut_kernel.cu"

CudaCut::CudaCut(int row, int col, cv::Mat& img1, cv::Mat& img2)
    : height(row), width(col), img1(img1), img2(img2){
    graph_size = width*height;
    graph_size1 = graph_size + 2;
    size_int = sizeof(int)*graph_size;


}

void CudaCut::h_mem_init()
{

    h_left_weight = (int*)malloc(size_int);
    h_right_weight = (int*)malloc(size_int);
    h_down_weight  = (int*)malloc(size_int);
    h_up_weight = (int*)malloc(size_int);
    h_left_flow = (int*)malloc(size_int);
    h_right_flow = (int*)malloc(size_int);
    h_down_flow  = (int*)malloc(size_int);
    h_up_flow = (int*)malloc(size_int);
    h_graph_heightr  =  (int*)malloc(sizeof(int)*graph_size1);
    h_graph_heightw  =  (int*)malloc(sizeof(int)*graph_size1);
    h_graph_height  =  (int*)malloc(sizeof(int)*graph_size);

    h_excess_flow = (int*)malloc(sizeof(int)*graph_size1);
    h_relabel_mask = (int*)malloc(size_int);
    h_push_state = (int*)malloc(size_int);
    h_height_backup = (int*)malloc(4*size_int);
    h_excess_flow_backup = (int*)malloc(sizeof(int)*graph_size);

    h_visited = (int*)malloc(sizeof(int)*graph_size);
    h_frontier = (bool*)malloc(sizeof(bool)*graph_size);

    h_m1 = (unsigned char *)malloc(sizeof(unsigned char)*graph_size);
    h_m2 = (unsigned char*)malloc(sizeof(unsigned char)*graph_size);
    h_process_area = (int*)malloc(size_int);
    h_horizontal = (int*)malloc(size_int);
    h_vertical = (int*)malloc(size_int);

    h_pull_left = (int*)malloc(sizeof(int)*graph_size);
    h_pull_right = (int*)malloc(sizeof(int)*graph_size);
    h_pull_down = (int*)malloc(sizeof(int)*graph_size);
    h_pull_up = (int*)malloc(sizeof(int)*graph_size);

    h_sink_weight = (int*)malloc(sizeof(int)*graph_size);

    h_push_block_position = (int*)malloc(sizeof(int)*(5*height));



    // initial h_weight, h_flow from input

}

void CudaCut::d_mem_init()
{
    gpuErrChk(cudaMalloc((void**)&d_left_weight, size_int));
    gpuErrChk(cudaMalloc((void**)&d_right_weight, size_int));
    gpuErrChk(cudaMalloc((void**)&d_down_weight, size_int));
    gpuErrChk(cudaMalloc((void**)&d_up_weight, size_int));

    gpuErrChk(cudaMalloc((void**)&d_left_flow, size_int));
    gpuErrChk(cudaMalloc((void**)&d_right_flow, size_int));
    gpuErrChk(cudaMalloc((void**)&d_down_flow, size_int));
    gpuErrChk(cudaMalloc((void**)&d_up_flow, size_int));

    gpuErrChk(cudaMalloc((void**)&d_graph_height, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_excess_flow, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_relabel_mask, size_int));
    gpuErrChk(cudaMalloc((void**)&d_push_state, size_int));
    gpuErrChk(cudaMalloc((void**)&d_height_backup, 4*size_int));
    gpuErrChk(cudaMalloc((void**)&d_excess_flow_backup, sizeof(int)*graph_size));

    gpuErrChk(cudaMalloc((void**)&d_visited, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_frontier, sizeof(bool)*graph_size));

    gpuErrChk(cudaMalloc((void**)&d_m1, size_int));
    gpuErrChk(cudaMalloc((void**)&d_m2, size_int));
    gpuErrChk(cudaMalloc((void**)&d_process_area, size_int));
    gpuErrChk(cudaMalloc((void**)&d_horizontal, size_int));
    gpuErrChk(cudaMalloc((void**)&d_vertical, size_int));

    gpuErrChk(cudaMalloc((void**)&d_pull_left, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_pull_right, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_pull_down, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_pull_up, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_graph_heightr, sizeof(int)*graph_size1));
    gpuErrChk(cudaMalloc((void**)&d_graph_heightw, sizeof(int)*graph_size1));

    gpuErrChk(cudaMalloc((void**)&d_sink_weight, sizeof(int)*graph_size));

    gpuErrChk(cudaMalloc((void**)&d_push_block_position, sizeof(int)*(5*height)));

//    memset(h_graph_height, 0, sizeof(int)*graph_size1);
//    h_graph_height[graph_size] = graph_size1;
//    cout << "h_graph_height s " << h_graph_height[graph_size] << endl;
//    memset(h_excess_flow, 0, sizeof(int)*graph_size1);
//    memset(h_excess_flow_backup, 0, sizeof(int)*graph_size1);
//    memset(h_height_backup, 0, 4*size_int);
//    memset(h_relabel_mask, 0, size_int);

//    gpuErrChk(cudaMemcpy(d_graph_height, h_graph_height, sizeof(int) * graph_size1 , cudaMemcpyHostToDevice));
//    gpuErrChk(cudaMemcpy(d_excess_flow, h_excess_flow, sizeof(int) * graph_size1 , cudaMemcpyHostToDevice));
//    gpuErrChk(cudaMemcpy(d_excess_flow_backup, h_excess_flow_backup, sizeof(int) * graph_size1 , cudaMemcpyHostToDevice));
//    gpuErrChk(cudaMemcpy(d_height_backup, h_height_backup, 4*size_int , cudaMemcpyHostToDevice));
//    gpuErrChk(cudaMemcpy(d_relabel_mask, h_relabel_mask, size_int , cudaMemcpyHostToDevice));

    // initial for d_visited and d_frontier
//    for(int i = 0; i < graph_size; i++){
//        if((i+1)%width != 0)
//            h_visited[i] = -1;
//        else
//            h_visited[i] = 1;
//    }
//    //h_visited[0] = true;
//    gpuErrChk(cudaMemcpy(d_visited, h_visited, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
//    for(int i = 1; i < graph_size; i++){
//        if((i+1)%width != 0)
//            h_frontier[i] = false;
//        else
//            h_frontier[i] = true;
//    }
//    //h_frontier[0] = true;
//    gpuErrChk(cudaMemcpy(d_frontier, h_frontier, sizeof(bool) * graph_size , cudaMemcpyHostToDevice));

//    int x = 10000;
    for(int i = 0 ; i < graph_size ; i++)
        h_excess_flow_backup[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_excess_flow_backup, h_excess_flow_backup, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
//    //initial pull left
//    for(int i = 0 ; i < graph_size ; i++)
//        h_pull_left[i] = 0 ;

//    gpuErrChk(cudaMemcpy(d_pull_left, h_pull_left, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
//    //initial pull right
//    for(int i = 0 ; i < graph_size ; i++)
//        h_pull_right[i] = 0 ;

//    gpuErrChk(cudaMemcpy(d_pull_right, h_pull_right, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
//    //initial pull down
//    for(int i = 0 ; i < graph_size ; i++)
//        h_pull_down[i] = 0 ;

//    gpuErrChk(cudaMemcpy(d_pull_down, h_pull_down, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
//    //initial pull up
//    for(int i = 0 ; i < graph_size ; i++)
//        h_pull_up[i] = 0 ;

//    gpuErrChk(cudaMemcpy(d_pull_up, h_pull_up, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    for(int i = 0 ; i < 4*graph_size ; i++)
//        h_height_backup[i] = 0 ;

//    gpuErrChk(cudaMemcpy(d_height_backup, h_height_backup, 4*sizeof(int) * graph_size , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    for(int i = 0 ; i < graph_size ; i++)
//        h_push_state[i] = 0 ;

//    gpuErrChk(cudaMemcpy(d_push_state, h_push_state, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    for(int i = 0 ; i < graph_size ; i++)
        h_relabel_mask[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_relabel_mask, h_relabel_mask, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial right_weight
//    for(int i = 0 ; i < graph_size ; i++){
//        if(((i+1)%width) != 0 && i%width != 0)
//            h_right_weight[i] = 3;
//        else if(((i+1)%width) == 0)
//            h_right_weight[i] = 0;
//        else{
//            h_right_weight[i] = -10000;
//        }
//    }
//    gpuErrChk(cudaMemcpy(d_right_weight, h_right_weight, size_int , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    // initial left_weight
//    for(int i = 0 ; i < graph_size ; i++){
//        if((i%width) != 0 && (i-1)%width != 0)
//            h_left_weight[i] = 1;
//        else if(i%width == 0)
//            h_left_weight[i] = 0;
//        else
//            h_left_weight[i] = 4;
//        if(i == 1)
//            h_left_weight[i] = 501;
//    }
//    gpuErrChk(cudaMemcpy(d_left_weight, h_left_weight, size_int , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    // initial down_weight
//    for(int i = 0 ; i < graph_size ; i++){
//        if((i%width) == 0 || ((i+1)%width) == 0 || i >= graph_size-width)
//            h_down_weight[i] = 0;
//        else
//            h_down_weight[i] = 3;
//    }
//    gpuErrChk(cudaMemcpy(d_down_weight, h_down_weight, size_int , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    // initial up_weight
//    for(int i = 0 ; i < graph_size ; i++){

//        if((i%width) == 0 || ((i+1)%width) == 0 || i < width)
//            h_up_weight[i] = 0;
//        else
//            h_up_weight[i] = 1;
//    }
//    gpuErrChk(cudaMemcpy(d_up_weight, h_up_weight, size_int , cudaMemcpyHostToDevice));

    //gpuErrChk(cudaDeviceSynchronize());

    // initial right_flow
//    for(int i = 0 ; i < graph_size ; i++){
//        if(i != 0)
//            h_right_flow[i] = 0;
//        else
//            h_right_flow[i] = x;
//    }
//    gpuErrChk(cudaMemcpy(d_right_flow, h_right_flow, size_int , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    // initial left_flow
//    for(int i = 0 ; i < graph_size ; i++){
//        if(i != 1)
//            h_left_flow[i] = 0;
//        else
//            h_left_flow[i] = -x;
//    }
//    gpuErrChk(cudaMemcpy(d_left_flow, h_left_flow, size_int , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    // initial down_flow
//    for(int i = 0 ; i < graph_size ; i++){
//        if(i != 0)
//            h_down_flow[i] = 0;
//        else
//            h_down_flow[i] = 3;
//    }
//    gpuErrChk(cudaMemcpy(d_down_flow, h_down_flow, size_int , cudaMemcpyHostToDevice));
//    //gpuErrChk(cudaDeviceSynchronize());

//    // initial up_flow
//    for(int i = 0 ; i < graph_size ; i++){
//        if(i != width)
//            h_up_flow[i] = 0;
//        else
//            h_up_flow[i] = -3;
//    }
//    gpuErrChk(cudaMemcpy(d_up_flow, h_up_flow, size_int , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial graph_height
    for(int i = 0 ; i < graph_size ; i++){
        if(i%width != 0)
            h_graph_height[i] = 0;
        else
            h_graph_height[i] = graph_size;
    }
    gpuErrChk(cudaMemcpy(d_graph_height, h_graph_height, sizeof(int)*graph_size, cudaMemcpyHostToDevice));
//    for(int i = 0 ; i < graph_size1 ; i++){
//        if(i != graph_size)
//            h_graph_heightr[i] = 0;
//        else
//            h_graph_heightr[i] = graph_size1;
//    }
//    gpuErrChk(cudaMemcpy(d_graph_heightr, h_graph_heightr, sizeof(int)*graph_size1 , cudaMemcpyHostToDevice));

//    for(int i = 0 ; i < graph_size1 ; i++){
//        if(i != graph_size)
//            h_graph_heightw[i] = 0;
//        else
//            h_graph_heightw[i] = graph_size1;
//    }
//    gpuErrChk(cudaMemcpy(d_graph_heightw, h_graph_heightw, sizeof(int)*graph_size1 , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial excess_flow
//    for(int i = 0 ; i < graph_size ; i++){
//        if(((i-1)%width) != 0)
//            h_excess_flow[i] = 0;
//        else
//            h_excess_flow[i] = 3;
//        if(i == 1)
//            h_excess_flow[i] = 500;
//    }
//    gpuErrChk(cudaMemcpy(d_excess_flow, h_excess_flow, sizeof(int)*graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial relabel_mask
//    for(int i = 0 ; i < graph_size ; i++){
//        h_relabel_mask[i] = 0;
//    }
//    gpuErrChk(cudaMemcpy(d_relabel_mask, h_relabel_mask, size_int , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

//    for(int i = 0; i < graph_size; i++){
//        cout << h_right_weight[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_left_weight[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_down_weight[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_up_weight[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_right_flow[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_left_flow[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_down_flow[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_up_flow[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_graph_height[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_excess_flow[i] << " ";
//    }
//    cout << "\n";
//    for(int i = 0; i < graph_size; i++){
//        cout << h_relabel_mask[i] << " ";
//    }
//    cout << "\n";

}

__global__ void
setupGraph_kernel(int *d_horizontal, int *d_vertical, int *d_right_weight, int *d_left_weight, int *d_up_weight,
                  int *d_down_weight, int *d_excess_flow, int *d_push_block_position, int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;

    if(ix > 0 && iy > 0){
        d_right_weight[node_i] = d_horizontal[node_i];
        d_left_weight[node_i] = d_horizontal[node_i - 1];

        d_down_weight[node_i] = d_vertical[node_i];
        d_up_weight[node_i] = d_vertical[node_i - width];
        d_excess_flow[node_i] = 0;
    }
    else if(ix > 0 && iy == 0){
        d_right_weight[node_i] = d_horizontal[node_i];
        d_left_weight[node_i] = d_horizontal[node_i - 1];

        d_down_weight[node_i] = d_vertical[node_i];
        d_up_weight[node_i] = 0;
        d_excess_flow[node_i] = 0;
    }
    else if(ix == 0 && iy > 0){
        d_right_weight[node_i] = 0;
        d_left_weight[node_i] = 0;

        d_down_weight[node_i] = d_vertical[node_i];
        d_up_weight[node_i] = d_vertical[node_i - width];
        d_excess_flow[node_i] = 0;
    }
    else{
        d_right_weight[node_i] = d_horizontal[node_i];
        d_left_weight[node_i] = 0;

        d_down_weight[node_i] = d_vertical[node_i];
        d_up_weight[node_i] = 0;
        d_excess_flow[node_i] = 0;
    }

    //blockIdx.x == 0? d_push_block_position[iy*5 + blockIdx.x] = 1 : d_push_block_position[iy*5 + blockIdx.x] = 0;
    if(blockIdx.x == 0 && threadIdx.x == 1){
        d_push_block_position[iy*5 + blockIdx.x] = 1;
    }
    if(blockIdx.x != 0 && threadIdx.x == 0){
        d_push_block_position[iy*5 + blockIdx.x] = 0;
    }

}

__global__ void
adjustGraph_kernel(int *d_excess_flow, int *d_left_weight, int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;

    if((ix-1)%width != 0) return;
    int tmp = d_left_weight[node_i];
    d_excess_flow[node_i] = tmp;
    d_left_weight[node_i] = 2*tmp;
}

int CudaCut::cudaCutsSetupGraph(){
//    int x = 40;
//    int y = (40-x)/2;
    cv::Mat area1 (img1, cv::Rect(img1.cols-OVERLAP_WIDTH, 0, OVERLAP_WIDTH, img1.rows) );
    cv::Mat area2 (img2, cv::Rect(0, 0, OVERLAP_WIDTH, img2.rows) );

    cv::Mat m1, m2;
    area1.convertTo(m1,CV_8UC1);
    area2.convertTo(m2,CV_8UC1);
    int xoffset = img1.cols - OVERLAP_WIDTH;
    //cout << m1.rows << " " << m1.cols << " " << m2.rows << " " << m2.cols << endl;

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width - 1; x++)
        {
            uchar a0 = img1.at<uchar>(y, xoffset + x);
            uchar b0 = img2.at<uchar>(y, x);
            uchar cap0 = abs(a0 - b0);

            uchar a1 = img1.at<uchar>(y, xoffset + x + 1);
            uchar b1 = img2.at<uchar>(y, x + 1);
            uchar cap1 = abs(a1 - b1);

            h_horizontal[y * width + x] = (int)(cap0 + cap1);
            //            horizontal[y * graph.width + x] = 0;
        }

        h_horizontal[y * width + width - 1] = 0;
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
            h_vertical[y * width + x] = (int)(cap0 + cap1);
            //            vertical[y * graph.width + x] = 0;
        }
        h_vertical[(img1.rows - 1) * width + x] = 0;
    }
    memcpy(h_m1, m1.ptr(0), sizeof(unsigned char)*graph_size);
    memcpy(h_m2, m2.ptr(0), sizeof(unsigned char)*graph_size);
//    for(int i = 0; i < 480; i++){
//        for(int j = 0; j < 80; j++){
//            cout << h_m1[i*80+j] << " ";
//        }
//        cout << "\n";
//    }

    gpuErrChk(cudaMemcpy(d_horizontal, h_horizontal, size_int , cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_vertical, h_vertical, size_int , cudaMemcpyHostToDevice));
    dim3 block(16, 8, 1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    cout << "<<<grid " << grid.x << grid.y << " block " << block.x << block.y << ">>>" << endl;
    setupGraph_kernel<<<grid, block>>>(d_horizontal, d_vertical, d_right_weight, d_left_weight, d_up_weight,
                      d_down_weight, d_excess_flow, d_push_block_position, width, height, graph_size);
    adjustGraph_kernel<<<grid, block>>>(d_excess_flow, d_left_weight, width, height, graph_size);

    gpuErrChk(cudaMemcpy(h_right_weight, d_right_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_left_weight, d_left_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_down_weight, d_down_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_up_weight, d_up_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_push_block_position, d_push_block_position, sizeof(int)*5*height, cudaMemcpyDeviceToHost));
    writeToFile("../variable/h_right_weight.txt", h_right_weight, width, height);
    writeToFile("../variable/h_left_weight.txt", h_left_weight, width, height);
    writeToFile("../variable/h_down_weight.txt", h_down_weight, width, height);
    writeToFile("../variable/h_up_weight.txt", h_up_weight, width, height);
    writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
    writeToFile("../variable/h_horizontal.txt", h_horizontal, width, height);
    writeToFile("../variable/h_vertical.txt", h_vertical, width, height);
    writeToFile("../variable/h_push_block_position.txt", h_push_block_position, 5, height);
    while(getchar() != 32);
    return 0;
}

int CudaCut::cudaCutsInit(){
    h_mem_init();
    d_mem_init();
    return 0;
}


void CudaCut::cudaCutsAtomic(cv::Mat& result, cv::Mat& result1){
    //size = 80*480

    dim3 block(16, 8, 1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    printf("<<<grid(%d, %d), block(%d, %d)>>>", grid.x, grid.y, block.x, block.y);
    int h_finished_count;
    int h_relabel_count;
    int  h_finish_bfs;
    int h_count = 0;

    int *d_finished_count, *d_finish_bfs;
    int *d_count;
    int *d_relabel_count;


    gpuErrChk(cudaMalloc((void**)&d_finished_count, sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_relabel_count, sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_finish_bfs, sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_count, sizeof(int)));



    //gpuErrChk(cudaDeviceSynchronize());
//    gpuErrChk(cudaMemcpy(h_graph_height, d_graph_height, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    gpuErrChk(cudaMemcpy(h_right_weight, d_right_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    gpuErrChk(cudaMemcpy(h_left_weight, d_left_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    gpuErrChk(cudaMemcpy(h_down_weight, d_down_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    gpuErrChk(cudaMemcpy(h_up_weight, d_up_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//    writeToFile("../variable/h_graph_height.txt", h_graph_height, width, height);
//    writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
//    writeToFile("../variable/h_right_weight.txt", h_right_weight, width, height);
//    writeToFile("../variable/h_left_weight.txt", h_left_weight, width, height);
//    writeToFile("../variable/h_down_weight.txt", h_down_weight, width, height);
//    writeToFile("../variable/h_up_weight.txt", h_up_weight, width, height);
//    while(getchar() != 32);

    h_finished_count = 1;
    int counter = 0;
    int *d_counter;
    gpuErrChk(cudaMalloc((void**)&d_counter, sizeof(int)));
    int sum;
    while(h_finished_count != 0){

        //gpuErrChk(cudaDeviceSynchronize());
        gpuErrChk(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));
        //gpuErrChk(cudaDeviceSynchronize());

        // global relabeling
        if(counter%20 == 0){
            h_finish_bfs = 1;
            for(int i = 0; i < graph_size; i++){
                if((i+1)%width != 0)
                    h_visited[i] = -1;
                else
                    h_visited[i] = 0;
            }
            //h_visited[0] = true;
            gpuErrChk(cudaMemcpy(d_visited, h_visited, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
            for(int i = 0; i < graph_size; i++){
                if((i+1)%width != 0)
                    h_frontier[i] = false;
                else
                    h_frontier[i] = true;
            }
            //h_frontier[0] = true;
            gpuErrChk(cudaMemcpy(d_frontier, h_frontier, sizeof(bool) * graph_size , cudaMemcpyHostToDevice));


            while(h_finish_bfs != 0){
                h_finish_bfs = 0;
                gpuErrChk(cudaMemcpy(d_finish_bfs, &h_finish_bfs, sizeof(int), cudaMemcpyHostToDevice));
                gpuErrChk(cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice));

                backward_bfs_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                            d_right_flow, d_left_flow, d_up_flow, d_down_flow, d_visited, d_frontier, d_graph_height,
                            width, height, graph_size, d_finish_bfs, d_count);

                gpuErrChk(cudaMemcpy(&h_finish_bfs, d_finish_bfs, sizeof(int), cudaMemcpyDeviceToHost));
                h_count++;
            }
            h_count = 0;
//            gpuErrChk(cudaMemcpy(h_graph_height, d_graph_height, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//            gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//            gpuErrChk(cudaMemcpy(h_right_weight, d_right_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//            gpuErrChk(cudaMemcpy(h_left_weight, d_left_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//            gpuErrChk(cudaMemcpy(h_down_weight, d_down_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//            gpuErrChk(cudaMemcpy(h_up_weight, d_up_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//            writeToFile("../variable/h_graph_height.txt", h_graph_height, width, height);
//            writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
//            writeToFile("../variable/h_right_weight.txt", h_right_weight, width, height);
//            writeToFile("../variable/h_left_weight.txt", h_left_weight, width, height);
//            writeToFile("../variable/h_down_weight.txt", h_down_weight, width, height);
//            writeToFile("../variable/h_up_weight.txt", h_up_weight, width, height);
//            cout << "graph height" << endl;
//            for(int i = 0; i < 16; i++){
//                for(int j = 0; j < 16; j++){
//                    cout << h_graph_heightr[i*16+j] << " ";
//                }
//                cout << endl;
//            }
//            cout << h_graph_heightr[graph_size] << " " << h_graph_heightr[graph_size+1] << endl;
//            cout << "excess flow" << endl;
//            for(int i = 0; i < 16; i++){
//                for(int j = 0; j < 16; j++){
//                    cout << h_excess_flow[i*16+j] << " ";
//                }
//                cout << endl;
//            }
//            cout << h_excess_flow[graph_size] << " " << h_excess_flow[graph_size+1] << endl;
//            while(getchar() != 32);

        }
        // state on for node that has excess flow > 0
        //push_state_kernel<<<grid, block>>>(d_push_state, d_excess_flow, width, graph_size);
        //gpuErrChk(cudaDeviceSynchronize());

        // pushing and save offset of flow to mediate array (d_excess_flow_backup)
//        gpuErrChk(cudaDeviceSynchronize());
//        gpuErrChk(cudaMemcpy(h_graph_height, d_graph_height, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        gpuErrChk(cudaMemcpy(h_right_weight, d_right_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        gpuErrChk(cudaMemcpy(h_left_weight, d_left_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        gpuErrChk(cudaMemcpy(h_down_weight, d_down_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        gpuErrChk(cudaMemcpy(h_up_weight, d_up_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_graph_height.txt", h_graph_height, width, height);
//        writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
//        writeToFile("../variable/h_right_weight.txt", h_right_weight, width, height);
//        writeToFile("../variable/h_left_weight.txt", h_left_weight, width, height);
//        writeToFile("../variable/h_down_weight.txt", h_down_weight, width, height);
//        writeToFile("../variable/h_up_weight.txt", h_up_weight, width, height);
        //while(getchar() != 32);

        //auto start = getMoment;
        if(counter == 0 || counter == 1 || counter == 2 || counter == 3 || counter == 4){
            push_block_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                                         d_excess_flow, d_graph_height, d_relabel_mask, d_height_backup, d_excess_flow_backup,
                                         d_push_block_position, width, height, graph_size, d_counter);
        }
        else{
        push_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                                     d_excess_flow, d_graph_height, d_relabel_mask, d_height_backup, d_excess_flow_backup,
                                     width, height, graph_size);
        }
        gpuErrChk(cudaDeviceSynchronize());
        //if(counter%30 == 0){
        gpuErrChk(cudaMemcpy(h_graph_height, d_graph_height, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(h_right_weight, d_right_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(h_left_weight, d_left_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(h_down_weight, d_down_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(h_up_weight, d_up_weight, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        writeToFile("../variable/h_graph_height.txt", h_graph_height, width, height);
        writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
        writeToFile("../variable/h_right_weight.txt", h_right_weight, width, height);
        writeToFile("../variable/h_left_weight.txt", h_left_weight, width, height);
        writeToFile("../variable/h_down_weight.txt", h_down_weight, width, height);
        writeToFile("../variable/h_up_weight.txt", h_up_weight, width, height);
        //while(getchar() != 32);
        //}
        //auto end = getMoment;
        //cout << "push kernel time iteration " << counter << " " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microsecond" << endl;
//        gpuErrChk(cudaMemcpy(h_graph_height, d_graph_height, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_graph_height.txt", h_graph_height, width, height);
//        writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
        //while(getchar() != 32);
        //gpuErrChk(cudaDeviceSynchronize());

        // add offset flow
//        add_excess_flow_kernel<<<grid, block>>>(d_excess_flow, d_pull_right, d_pull_left,
//                                                d_pull_up, d_pull_down, d_right_weight, d_left_weight,
//                                                d_up_weight, d_down_weight, d_excess_flow_backup, width, graph_size);
        //gpuErrChk(cudaDeviceSynchronize());

        // get relabel_count
        gpuErrChk(cudaMemcpy(&h_relabel_count, d_relabel_count, sizeof(int), cudaMemcpyDeviceToHost));
        //gpuErrChk(cudaDeviceSynchronize());
        //std::cout << "h_relabel_count " << h_relabel_count << std::endl;
        if(counter%1 == 0){
            //if(h_relabel_count != 0){
                // relabel if relabel_count > 0
            h_finished_count = 0;
            //h_relabel_count = 0;
    //        //printf("stop\n");
            gpuErrChk(cudaMemcpy(d_finished_count, &h_finished_count, sizeof(int), cudaMemcpyHostToDevice));
                relabel_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                                                d_graph_height, d_relabel_mask, d_height_backup,
                                                d_excess_flow, d_excess_flow_backup,
                                                width, height, graph_size);
                check_finished_condition<<<grid, block>>>(d_excess_flow, d_finished_count, width, height, graph_size);
                gpuErrChk(cudaMemcpy(&h_finished_count, d_finished_count, sizeof(int), cudaMemcpyDeviceToHost));
                cout << "h_finish_count " << h_finished_count << endl;
                //gpuErrChk(cudaDeviceSynchronize());
            //}
        }
        gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        //gpuErrChk(cudaDeviceSynchronize());
//        sum = 0;
//        for(int i = 0; i < graph_size; i++){
//            if((i+1)%width == 0)
//                sum += h_excess_flow[i];
//        }
//        if(sum == 4253)
//            h_finished_count = 0;


//        gpuErrChk(cudaMemcpy(h_graph_heightr, d_graph_heightr, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_graph_heightr.txt", h_graph_height, width, height);
//        gpuErrChk(cudaMemcpy(h_graph_heightw, d_graph_heightw, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_graph_heightw.txt", h_graph_height, width, height);
////        gpuErrChk(cudaMemcpy(h_file_graph_height, d_graph_heightr, size_int, cudaMemcpyDeviceToHost));
////        writeToFile("../variable/read.txt", h_file_graph_height, width1, height1);
//        gpuErrChk(cudaMemcpy(h_right_flow, d_right_flow, size_int, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_right_flow.txt", h_right_flow, width, height);
//        gpuErrChk(cudaMemcpy(h_left_flow, d_left_flow, size_int, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_left_flow.txt", h_left_flow, width, height);
//        gpuErrChk(cudaMemcpy(h_down_flow, d_down_flow, size_int, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_down_flow.txt", h_down_flow, width, height);
//        gpuErrChk(cudaMemcpy(h_up_flow, d_up_flow, size_int, cudaMemcpyDeviceToHost));
//        writeToFile("../variable/h_up_flow.txt", h_up_flow, width, height);
        //            printData(h_file_graph_height, width1, height1);
        //printf("Counter = %d\n", counter);
        //while(getchar() != 32);
        //gpuErrChk(cudaDeviceSynchronize());
        //std::cout << "h_finished_count " << h_finished_count << std::endl;
        counter++;
        if(counter == 30)
            h_finished_count = 0;

    }
    cout << "cout = " << counter << endl;
    gpuErrChk(cudaFree(d_finished_count));
    gpuErrChk(cudaFree(d_relabel_count));
    gpuErrChk(cudaFree(d_counter));

    gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    //gpuErrChk(cudaDeviceSynchronize());
    sum = 0;
    for(int i = 0; i < graph_size; i++){
        if((i+1)%width == 0)
            sum += h_excess_flow[i];
    }
    std::cout << "max flow: " << sum << std::endl;
    std::cout << "final_excess_flow " << std::endl;
//    for(int i = 0; i< graph_size; i++){
//        std::cout << h_excess_flow[i] << " ";
//    }
    std::cout << "\n";

    // get result
    for(int i = 0; i < graph_size; i++){
        if((i)%width != 0)
            h_visited[i] = -1;
        else
            h_visited[i] = 0;
    }
    //h_visited[0] = true;
    gpuErrChk(cudaMemcpy(d_visited, h_visited, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    for(int i = 0; i < graph_size; i++){
        if((i)%width != 0)
            h_frontier[i] = false;
        else
            h_frontier[i] = true;
    }
    //h_frontier[0] = true;
    gpuErrChk(cudaMemcpy(d_frontier, h_frontier, sizeof(bool) * graph_size , cudaMemcpyHostToDevice));


    h_finish_bfs = 1;
//    int *d_finish_bfs;
    //gpuErrChk(cudaMalloc((void**)&d_finish_bfs, sizeof(int)));
//    int x = 40;
//    int y = (40-x)/2;
    cv::Mat tmp(result, cv::Rect(img1.cols-OVERLAP_WIDTH,0, OVERLAP_WIDTH, img1.rows));

    while(h_finish_bfs != 0){
        h_finish_bfs = 0;
        gpuErrChk(cudaMemcpy(d_finish_bfs, &h_finish_bfs, sizeof(int), cudaMemcpyHostToDevice));

        bfs_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                    d_right_flow, d_left_flow, d_up_flow, d_down_flow, d_visited, d_frontier,
                    width, height, graph_size, d_finish_bfs);

        gpuErrChk(cudaMemcpy(&h_finish_bfs, d_finish_bfs, sizeof(int), cudaMemcpyDeviceToHost));
    }
    gpuErrChk(cudaMemcpy(h_visited, d_visited, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    writeToFile("../variable/h_visited.txt", h_visited, width, height);

    std::cout << "set(s): ";

    int count = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(h_visited[i*width+j] == 0){
                tmp.at<uchar>(i,j) = h_m1[i*width+j];
                count++;
            }
            else
                tmp.at<uchar>(i,j) = h_m2[i*width+j];

        }
    }
    result.copyTo(result1);
    cv::Mat temp1(result1, cv::Rect(img1.cols-OVERLAP_WIDTH,0, OVERLAP_WIDTH, img1.rows));
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(j > 0 && j< width-1 && h_visited[i*width+j] != h_visited[i*width+j+1]){
                temp1.at<uchar>(i,j) = 255;
                temp1.at<uchar>(i,j+1) = 255;
                temp1.at<uchar>(i,j-1) = 255;

            }
            if(i > 0 && i< height-1 && h_visited[i*width+j] != h_visited[(i+1)*width+j]){
                temp1.at<uchar>(i,j) = 255;
                temp1.at<uchar>(i+1,j) = 255;
                temp1.at<uchar>(i-1,j) = 255;
            }

        }
    }
    std::cout << "count " << count << endl;

    gpuErrChk(cudaFree(d_finish_bfs));



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

    free(h_left_flow);
    free(h_right_flow);
    free(h_down_flow);
    free(h_up_flow);

    free(h_excess_flow);
    free(h_relabel_mask);
    free(h_graph_height);
    free(h_push_state);
    free(h_height_backup);
    free(h_excess_flow_backup);
    free(h_visited);
    free(h_frontier);

    free(h_m1);
    free(h_m2);
    free(h_process_area);
    free(h_horizontal);
    free(h_vertical);

    free(h_pull_left);
    free(h_pull_right);
    free(h_pull_down);
    free(h_pull_up);
    free(h_graph_heightr);
    free(h_graph_heightw);

    free(h_sink_weight);
    free(h_push_block_position);



    gpuErrChk(cudaFree(d_left_weight));
    gpuErrChk(cudaFree(d_right_weight));
    gpuErrChk(cudaFree(d_down_weight));
    gpuErrChk(cudaFree(d_up_weight));

    gpuErrChk(cudaFree(d_left_flow));
    gpuErrChk(cudaFree(d_right_flow));
    gpuErrChk(cudaFree(d_down_flow));
    gpuErrChk(cudaFree(d_up_flow));

    gpuErrChk(cudaFree(d_excess_flow));
    gpuErrChk(cudaFree(d_relabel_mask));
    gpuErrChk(cudaFree(d_graph_height));
    gpuErrChk(cudaFree(d_push_state));
    gpuErrChk(cudaFree(d_height_backup));
    gpuErrChk(cudaFree(d_excess_flow_backup));
    gpuErrChk(cudaFree(d_visited));
    gpuErrChk(cudaFree(d_frontier));

    gpuErrChk(cudaFree(d_m1));
    gpuErrChk(cudaFree(d_m2));
    gpuErrChk(cudaFree(d_process_area));
    gpuErrChk(cudaFree(d_horizontal));
    gpuErrChk(cudaFree(d_vertical));

    gpuErrChk(cudaFree(d_pull_left));
    gpuErrChk(cudaFree(d_pull_right));
    gpuErrChk(cudaFree(d_pull_down));
    gpuErrChk(cudaFree(d_pull_up));
    gpuErrChk(cudaFree(d_graph_heightr));
    gpuErrChk(cudaFree(d_graph_heightw));

    gpuErrChk(cudaFree(d_sink_weight));
    gpuErrChk(cudaFree(d_push_block_position));

}

