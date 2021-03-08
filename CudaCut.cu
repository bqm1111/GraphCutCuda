#include "cudacut.h"


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
    h_excess_flow_backup = (int*)malloc(sizeof(int)*graph_size1);

    h_visited = (int*)malloc(sizeof(int)*graph_size);
    h_frontier = (bool*)malloc(sizeof(bool)*graph_size);

    h_m1 = (int*)malloc(size_int);
    h_m2 = (int*)malloc(size_int);
    h_process_area = (int*)malloc(size_int);

    h_pull_left = (int*)malloc(sizeof(int)*graph_size);
    h_pull_right = (int*)malloc(sizeof(int)*graph_size);
    h_pull_down = (int*)malloc(sizeof(int)*graph_size);
    h_pull_up = (int*)malloc(sizeof(int)*graph_size);

    h_sink_weight = (int*)malloc(sizeof(int)*graph_size);



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
    gpuErrChk(cudaMalloc((void**)&d_excess_flow_backup, sizeof(int)*graph_size1));

    gpuErrChk(cudaMalloc((void**)&d_visited, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_frontier, sizeof(bool)*graph_size));

    gpuErrChk(cudaMalloc((void**)&d_m1, size_int));
    gpuErrChk(cudaMalloc((void**)&d_m2, size_int));
    gpuErrChk(cudaMalloc((void**)&d_process_area, size_int));

    gpuErrChk(cudaMalloc((void**)&d_pull_left, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_pull_right, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_pull_down, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_pull_up, sizeof(int)*graph_size));
    gpuErrChk(cudaMalloc((void**)&d_graph_heightr, sizeof(int)*graph_size1));
    gpuErrChk(cudaMalloc((void**)&d_graph_heightw, sizeof(int)*graph_size1));

    gpuErrChk(cudaMalloc((void**)&d_sink_weight, sizeof(int)*graph_size));

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

    int x = 10000;
    for(int i = 0 ; i < graph_size1 ; i++)
        h_excess_flow_backup[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_excess_flow_backup, h_excess_flow_backup, sizeof(int) * graph_size1 , cudaMemcpyHostToDevice));
    //initial pull left
    for(int i = 0 ; i < graph_size ; i++)
        h_pull_left[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_pull_left, h_pull_left, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //initial pull right
    for(int i = 0 ; i < graph_size ; i++)
        h_pull_right[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_pull_right, h_pull_right, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //initial pull down
    for(int i = 0 ; i < graph_size ; i++)
        h_pull_down[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_pull_down, h_pull_down, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //initial pull up
    for(int i = 0 ; i < graph_size ; i++)
        h_pull_up[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_pull_up, h_pull_up, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    for(int i = 0 ; i < 4*graph_size ; i++)
        h_height_backup[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_height_backup, h_height_backup, 4*sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    for(int i = 0 ; i < graph_size ; i++)
        h_push_state[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_push_state, h_push_state, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    for(int i = 0 ; i < graph_size ; i++)
        h_relabel_mask[i] = 0 ;

    gpuErrChk(cudaMemcpy(d_relabel_mask, h_relabel_mask, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial right_weight
    for(int i = 0 ; i < graph_size ; i++){
        if(((i+1)%width) != 0 && i%width != 0)
            h_right_weight[i] = 3;
        else if(((i+1)%width) == 0)
            h_right_weight[i] = 0;
        else{
            h_right_weight[i] = -10000;
        }
    }
    gpuErrChk(cudaMemcpy(d_right_weight, h_right_weight, size_int , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial left_weight
    for(int i = 0 ; i < graph_size ; i++){
        if((i%width) != 0 && (i-1)%width != 0)
            h_left_weight[i] = 1;
        else if(i%width == 0)
            h_left_weight[i] = 0;
        else
            h_left_weight[i] = 4;
        if(i == 1)
            h_left_weight[i] = 501;
    }
    gpuErrChk(cudaMemcpy(d_left_weight, h_left_weight, size_int , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial down_weight
    for(int i = 0 ; i < graph_size ; i++){
        if((i%width) == 0 || ((i+1)%width) == 0 || i >= graph_size-width)
            h_down_weight[i] = 0;
        else
            h_down_weight[i] = 3;
    }
    gpuErrChk(cudaMemcpy(d_down_weight, h_down_weight, size_int , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial up_weight
    for(int i = 0 ; i < graph_size ; i++){

        if((i%width) == 0 || ((i+1)%width) == 0 || i < width)
            h_up_weight[i] = 0;
        else
            h_up_weight[i] = 1;
    }
    gpuErrChk(cudaMemcpy(d_up_weight, h_up_weight, size_int , cudaMemcpyHostToDevice));

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
    for(int i = 0 ; i < graph_size1 ; i++){
        if(i != graph_size)
            h_graph_heightr[i] = 0;
        else
            h_graph_heightr[i] = graph_size1;
    }
    gpuErrChk(cudaMemcpy(d_graph_heightr, h_graph_heightr, sizeof(int)*graph_size1 , cudaMemcpyHostToDevice));

    for(int i = 0 ; i < graph_size1 ; i++){
        if(i != graph_size)
            h_graph_heightw[i] = 0;
        else
            h_graph_heightw[i] = graph_size1;
    }
    gpuErrChk(cudaMemcpy(d_graph_heightw, h_graph_heightw, sizeof(int)*graph_size1 , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial excess_flow
    for(int i = 0 ; i < graph_size ; i++){
        if(((i-1)%width) != 0)
            h_excess_flow[i] = 0;
        else
            h_excess_flow[i] = 3;
        if(i == 1)
            h_excess_flow[i] = 500;
    }
    gpuErrChk(cudaMemcpy(d_excess_flow, h_excess_flow, sizeof(int)*graph_size , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    // initial relabel_mask
    for(int i = 0 ; i < graph_size ; i++){
        h_relabel_mask[i] = 0;
    }
    gpuErrChk(cudaMemcpy(d_relabel_mask, h_relabel_mask, size_int , cudaMemcpyHostToDevice));
    //gpuErrChk(cudaDeviceSynchronize());

    for(int i = 0; i < graph_size; i++){
        cout << h_right_weight[i] << " ";
    }
    cout << "\n";
    for(int i = 0; i < graph_size; i++){
        cout << h_left_weight[i] << " ";
    }
    cout << "\n";
    for(int i = 0; i < graph_size; i++){
        cout << h_down_weight[i] << " ";
    }
    cout << "\n";
    for(int i = 0; i < graph_size; i++){
        cout << h_up_weight[i] << " ";
    }
    cout << "\n";
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
    cout << "\n";
    for(int i = 0; i < graph_size; i++){
        cout << h_graph_height[i] << " ";
    }
    cout << "\n";
    for(int i = 0; i < graph_size; i++){
        cout << h_excess_flow[i] << " ";
    }
    cout << "\n";
    for(int i = 0; i < graph_size; i++){
        cout << h_relabel_mask[i] << " ";
    }
    cout << "\n";

}

__global__ void
setupGraph_kernel(int *d_m1, int *d_m2, int *d_right_weight, int *d_left_weight, int *d_up_weight,
                  int *d_down_weight, int *d_right_flow, int *d_left_flow, int *d_up_flow, int *d_down_flow,
                  int *d_excess_flow,
                  int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    if(node_i >= N) return;


    if(ix+1 < width){
        d_right_weight[node_i] = abs(d_m1[node_i]-d_m2[node_i]) + abs(d_m1[iy*width + ix+1]-d_m2[iy*width + ix+1]);
        //printf("d_right_weight[%d] %d", node_i, d_right_weight[node_i]);
        d_right_flow[node_i] = 0;
    }
    else{
        d_right_weight[node_i] = 100000;
        //printf("d_right_weight[%d] %d", node_i, d_right_weight[node_i]);
        d_right_flow[node_i] = 0;
    }
    if(iy+1 < height){
        if(ix+1 >= width || ix-1 < 0)
            d_down_weight[node_i] = 0;
        else
            d_down_weight[node_i] = abs(d_m1[node_i]-d_m2[node_i]) + abs(d_m1[(iy+1)*width + ix]-d_m2[(iy+1)*width + ix]);

        d_down_flow[node_i] = 0;
    }
    else{
        d_down_weight[node_i] = 0;
        d_down_flow[node_i] = 0;
    }
    if(ix-1 >= 0){
        d_left_weight[node_i] = abs(d_m1[node_i]-d_m2[node_i]) + abs(d_m1[iy*width + ix-1]-d_m2[iy*width + ix-1]);
        d_left_flow[node_i] = 0;
    }
    else{
        d_left_weight[node_i] = 0;
        d_left_flow[node_i] = -300000;
        //printf("d_left_flow[%d] %d", node_i, d_left_flow[node_i]);
    }
    if(iy-1 >= 0){
        if(ix+1 >= width || ix-1 < 0)
            d_up_weight[node_i] = 0;
        else
            d_up_weight[node_i] = abs(d_m1[node_i]-d_m2[node_i]) + abs(d_m1[(iy-1)*width + ix]-d_m2[(iy-1)*width + ix]);
        d_up_flow[node_i] = 0;
    }
    else{
        d_up_weight[node_i] = 0;
        d_up_flow[node_i] = 0;
    }

    if(ix-1 < 0){
        d_excess_flow[node_i] = 100000;
        //printf("d_excess_flow[%d] %d", node_i, d_excess_flow[node_i]);
    }

}

int CudaCut::cudaCutsSetupGraph(){
    cv::Mat area1 (img1, cv::Rect(560, 0, 80, 480) );
    cv::Mat area2 (img2, cv::Rect(0, 0, 80, 480) );

    cv::Mat m1, m2;
    area1.convertTo(m1,CV_32SC1);
    area2.convertTo(m2,CV_32SC1);
    cout << m1.rows << " " << m1.cols << " " << m2.rows << " " << m2.cols << endl;
    memcpy(h_m1, m1.ptr(0), size_int);
    memcpy(h_m2, m2.ptr(0), size_int);
//    for(int i = 0; i < 480; i++){
//        for(int j = 0; j < 80; j++){
//            cout << h_m1[i*80+j] << " ";
//        }
//        cout << "\n";
//    }

    gpuErrChk(cudaMemcpy(d_m1, h_m1, size_int , cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_m2, h_m2, size_int , cudaMemcpyHostToDevice));
    dim3 block(16, 32, 1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    cout << "<<<grid " << grid.x << grid.y << " block " << block.x << block.y << ">>>" << endl;
    setupGraph_kernel<<<grid, block>>>(d_m1, d_m2, d_right_weight, d_left_weight, d_up_weight,
                      d_down_weight, d_right_flow, d_left_flow, d_up_flow, d_down_flow,
                      d_excess_flow,
                      width, height, graph_size);
    return 0;
}

int CudaCut::cudaCutsInit(){
    h_mem_init();
    d_mem_init();
    return 0;
}

// push kernel
inline __global__ void
push_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_pull_right, int *d_pull_left, int *d_pull_up, int *d_pull_down, int *d_excess_flow,
            int *d_graph_height, int *d_relabel_mask, int *d_push_state, int *d_height_backup, int *d_excess_flow_backup,
            int width, int height, int N, int *d_relabel_count){

    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = x1 + blockIdx.x*blockDim.x;
    int iy = y1 + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    //if(ix >= width) return;


    int *height_backup = d_height_backup + 4*node_i;
//    __shared__ int height_fn[180];

//    int temp_mult = __umul24(y1+1 , 18) + x1 + 1 ;

//    height_fn[temp_mult] = d_graph_height[node_i];

//    (threadIdx.x == 15 && ix < width - 1) ? height_fn[temp_mult + 1] =  (d_graph_height[node_i + 1]) : 0;
//    (threadIdx.x == 0  && ix > 0) ? height_fn[temp_mult - 1] = (d_graph_height[node_i - 1]) : 0;
//    (threadIdx.y == 7  && iy < height - 1) ? height_fn[temp_mult + 18] = (d_graph_height[node_i + width]) : 0;
//    (threadIdx.y == 0  && iy > 0) ? height_fn[temp_mult - 18] = (d_graph_height[node_i - width]) : 0;

////    (ix >= width - 1) ? height_fn[temp_mult + 1] =  0 : 0;
////    (ix <= 0) ? height_fn[temp_mult - 1] = N+2 : 0;
////    printf("height_fn block(%d, %d)\n", blockIdx.x, blockIdx.y);
////    printf("x1 = %d, y1 = %d\n", x1, y1);
////    printf("%d\t%d\t%d\t%d\t%d\t", height_fn[temp_mult], height_fn[temp_mult-1], height_fn[temp_mult+1], height_fn[temp_mult-4], height_fn[temp_mult+4]);

//    //int temp_mult2 = 340 + __umul24(y1,32) + x1 ;
//    height_backup[0] = height_fn[temp_mult + 1];
//    height_backup[1] = height_fn[temp_mult + 18];
//    height_backup[2] = height_fn[temp_mult - 1];
//    height_backup[3] = height_fn[temp_mult - 18];

//    int flow_push = d_excess_flow[node_i] ;
//    d_excess_flow[node_i] = 0;
//    int min_flow_pushed = 0, temp_weight = 0;
//    __syncthreads();
//    //if(node_i%width != 0 && (node_i+1)%width != 0){
//    if(ix != 0 && ix != width - 1){
//    // push left
//    //if(ix-1 >= 0){
//        if(flow_push > 0){
//            min_flow_pushed = flow_push ;
//            temp_weight = d_left_weight[node_i] ;
//            if(temp_weight > 0){
//                if(height_fn[temp_mult] == 1 + height_fn[temp_mult-1]){
//                (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
//                temp_weight = temp_weight - min_flow_pushed ;
//                d_left_weight[node_i] = temp_weight ;

//                d_right_weight[node_i - 1] += min_flow_pushed;
//                atomicAdd(&d_excess_flow[node_i - 1], min_flow_pushed);
//                //d_pull_left[node_i - 1] = min_flow_pushed ;
//                flow_push = flow_push - min_flow_pushed ;
//                //d_excess_flow[node_i] -= flow;
//                //atomicSub(&d_excess_flow[node_i], min_flow_pushed);
//                //flow_push -= min_flow_push;
//                //atomicAdd(&d_excess_flow[iy*width + ix-1], flow);
//                //atomicAdd(&d_excess_flow_backup[iy*width + ix-1], flow);
//                //d_excess_flow[iy*width + ix-1] += flow;
//                //d_left_flow[node_i] += flow;
//                //d_right_flow[iy*width + ix-1] -= flow;
//                //d_left_weight[node_i] -= flow;
//                //d_right_weight[iy*width + ix-1] += flow;
//                printf("pushing node %d to the left \n", node_i);
//                //d_relabel_mask[node_i] = 0;
//                }
//                else{
//                    d_relabel_mask[node_i] = 1;
//                    printf("in push left, node %d need relabel \n", node_i);
//                }
//            }

//        }
//    //}
////    else{
////        if(flow_push > 0){
////            min_flow_pushed = flow_push ;
////            temp_weight = d_left_weight[node_i] ;
////            if(temp_weight > 0){
////                if(height_fn[temp_mult] == 1 + height_fn[temp_mult-1]){
////                (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
////                temp_weight -= min_flow_pushed ;
////                d_left_weight[node_i] = temp_weight ;
////                atomicAdd(&d_pull_left[N], min_flow_pushed) ;
////                flow_push -= min_flow_pushed ;
////                //d_excess_flow[node_i] -= flow;
////                //atomicSub(&d_excess_flow[node_i], flow);
////                //flow_push -= min_flow_push;
////                //atomicAdd(&d_excess_flow[iy*width + ix-1], flow);
////                //atomicAdd(&d_excess_flow_backup[iy*width + ix-1], flow);
////                //d_excess_flow[iy*width + ix-1] += flow;
////                //d_left_flow[node_i] += flow;
////                //d_right_flow[iy*width + ix-1] -= flow;
////                //d_left_weight[node_i] -= flow;
////                //d_right_weight[iy*width + ix-1] += flow;
////                printf("pushing node %d to the S \n", node_i);
////                //d_relabel_mask[node_i] = 0;
////                }
////                else{
////                    d_relabel_mask[node_i] = 1;
////                    printf("in push S, node %d need relabel \n", node_i);
////                }
////            }
////        }
////    }
//    __threadfence();
//    // push to the right


//    //if(ix+1 < width){
//        if(flow_push > 0){
//            min_flow_pushed = flow_push ;
//            temp_weight = d_right_weight[node_i] ;
//            if(temp_weight > 0){
//                if(height_fn[temp_mult] == 1 + height_fn[temp_mult+1]){
//                (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
//                temp_weight -= min_flow_pushed ;
//                d_right_weight[node_i] = temp_weight ;

//                d_left_weight[node_i + 1] += min_flow_pushed;
//                atomicAdd(&d_excess_flow[node_i + 1], min_flow_pushed);
//                //d_pull_right[node_i + 1] = min_flow_pushed ;
//                flow_push -= min_flow_pushed ;
//                //flow = min(d_right_weight[node_i], d_excess_flow[node_i]);
//                //d_excess_flow[node_i] -= flow;
//                //atomicSub(&d_excess_flow[node_i], flow);
//                //atomicAdd(&d_excess_flow[iy*width + ix+1], flow);
//                //atomicAdd(&d_excess_flow_backup[iy*width + ix+1], flow);
//                //d_excess_flow[iy*width + ix+1] += flow;
//                //d_right_flow[node_i] += flow;
//                //d_left_flow[iy*width + ix+1] -= flow;
//                //d_right_weight[node_i] -= flow;
//                //d_left_weight[iy*width + ix+1] += flow;
//                //printf("pushing node %d to the right \n", node_i);
//                //d_relabel_mask[node_i] = 0;
//                }
//                else{
//                    d_relabel_mask[node_i] = 1;
//                    printf("in push right, node %d need relabel \n", node_i);
//                }
//            }

//        }
//    //}
////    else{
////        if(flow_push > 0){
////            min_flow_pushed = flow_push ;
////            temp_weight = d_right_weight[node_i] ;
////            if(temp_weight > 0){
////                if(height_fn[temp_mult] == 1 + height_fn[temp_mult+1]){
////                    (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
////                    temp_weight -= min_flow_pushed ;
////                    d_right_weight[node_i] = temp_weight ;
////                    atomicAdd(&d_pull_right[N+1], min_flow_pushed) ;
////                    flow_push -= min_flow_pushed ;
////                    //flow = min(d_right_weight[node_i], d_excess_flow[node_i]);
////                    //d_excess_flow[node_i] -= flow;
////                    //atomicSub(&d_excess_flow[node_i], flow);
////                    //atomicAdd(&d_excess_flow[iy*width + ix+1], flow);
////                    //atomicAdd(&d_excess_flow_backup[N+1], flow);
////                    //d_excess_flow[iy*width + ix+1] += flow;
////                    //d_right_flow[node_i] += flow;
////                    //d_left_flow[iy*width + ix+1] -= flow;
////                    //d_right_weight[node_i] -= flow;
////                    //atomicSub(&d_left_flow[iy*width + ix+1], flow);
////                    printf("pushing node %d to the T \n", node_i);
////                    printf("flow in push to T %d\n", min_flow_pushed);
////                    //d_relabel_mask[node_i] = 0;
////                }
////                else{
////                    d_relabel_mask[node_i] = 1;
////                    printf("in push T, node %d need relabel \n", node_i);
////                }
////            }
////        }
////    }
//    //__syncthreads();

//    // push to the down

//    if(iy+1 < height){
//        if(flow_push > 0){
//            min_flow_pushed = flow_push ;
//            temp_weight = d_down_weight[node_i] ;
//            if(temp_weight > 0){
//                if(height_fn[temp_mult] == 1 + height_fn[temp_mult+18]){
//                    (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
//                    temp_weight -= min_flow_pushed ;
//                    d_down_weight[node_i] = temp_weight ;

//                    d_up_weight[node_i + width] += min_flow_pushed;
//                    atomicAdd(&d_excess_flow[node_i + width], min_flow_pushed);
//                    //d_pull_down[node_i + width] = min_flow_pushed ;
//                    flow_push -= min_flow_pushed ;
//                    //flow = min(d_down_weight[node_i], d_excess_flow[node_i]);
//                    //d_excess_flow[node_i] -= flow;
//                    //atomicSub(&d_excess_flow[node_i], flow);
//                    //atomicAdd(&d_excess_flow[(iy+1)*width + ix], flow);
//                    //atomicAdd(&d_excess_flow_backup[(iy+1)*width + ix], flow);
//                    //d_excess_flow[(iy+1)*width + ix] += flow;
//                    //d_down_flow[node_i] += flow;
//                    //d_up_flow[(iy+1)*width + ix] -= flow;

//                    //d_down_weight[node_i] -= flow;
//                    //d_up_weight[(iy+1)*width + ix] += flow;
//                    //printf("pushing node %d to the down\n", node_i);
//                    //d_relabel_mask[node_i] = 0;
//                }
//                else{
//                    d_relabel_mask[node_i] = 1;
//                    printf("in push down, node %d need relabel \n", node_i);
//                }
//            }
//        }
//    }
//    //__syncthreads();


//    //__syncthreads();

//    //push up

//    if(iy-1 >= 0){
//        if(flow_push > 0){
//            min_flow_pushed = flow_push ;
//            temp_weight = d_up_weight[node_i] ;
//            if(temp_weight > 0){
//                if(height_fn[temp_mult] == 1 + height_fn[temp_mult-18]){
//                    (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
//                    temp_weight -= min_flow_pushed ;
//                    d_up_weight[node_i] = temp_weight ;

//                    d_down_weight[node_i - width] += min_flow_pushed;
//                    atomicAdd(&d_excess_flow[node_i - width], min_flow_pushed);
//                    //d_pull_up[node_i - width] = min_flow_pushed ;
//                    flow_push -= min_flow_pushed ;
//                    //flow = min(d_up_weight[node_i], d_excess_flow[node_i]);
//                    //d_excess_flow[node_i] -= flow;
//                    //atomicSub(&d_excess_flow[node_i], flow);
//                    //atomicAdd(&d_excess_flow[(iy-1)*width + ix], flow);
//                    //atomicAdd(&d_excess_flow_backup[(iy-1)*width + ix], flow);
//                    //d_excess_flow[(iy-1)*width + ix] += flow;
//                    //d_up_flow[node_i] += flow;
//                    //d_down_flow[(iy-1)*width + ix] -= flow;
//                    //d_up_weight[node_i] -= flow;
//                    //d_down_weight[(iy-1)*width + ix] += flow;
//                    //printf("pushing node %d to the up \n", node_i);
//                    //d_relabel_mask[node_i] = 0;
//                }
//                else{
//                    d_relabel_mask[node_i] = 1;
//                    //printf("in push up, node %d need relabel \n", node_i);
//                }
//            }
//        }
//    }
//    }

//    d_excess_flow[node_i] += flow_push;
        //__syncthreads();



        //d_push_state[node_i] = 0;
    int flow;

    if(ix != 0 && ix != width - 1){
        // push to the left
        if(d_excess_flow[node_i] > 0 && d_left_weight[node_i] > 0){
            if(d_graph_height[node_i] == 1 + d_graph_height[node_i - 1]){
                flow = min(d_left_weight[node_i], d_excess_flow[node_i]);
                //d_excess_flow[node_i] -= flow;
                d_left_weight[node_i] -= flow;
                d_right_weight[node_i - 1] += flow;
                atomicSub(&d_excess_flow[node_i], flow);
                atomicAdd(&d_excess_flow[node_i - 1], flow);
                //atomicAdd(&d_excess_flow_backup[iy*width + ix-1], flow);
                //d_excess_flow[iy*width + ix-1] += flow;
                //d_left_flow[node_i] += flow;
                //d_right_flow[iy*width + ix-1] -= flow;
                //atomicAdd(&d_left_flow[node_i], flow);
                //atomicSub(&d_right_flow[iy*width + ix-1], flow);
                //printf("pushing node %d to the left \n", node_i);
                //d_relabel_mask[node_i] = 0;
            }
            else{
                d_relabel_mask[node_i] = 1;
                //printf("in push left, node %d need relabel \n", node_i);
            }

        }
        //__threadfence();

        // push right
        if(d_excess_flow[node_i] > 0 && d_right_weight[node_i] > 0){
            if(d_graph_height[node_i] == 1 + d_graph_height[node_i + 1]){
                flow = min(d_right_weight[node_i], d_excess_flow[node_i]);
                d_right_weight[node_i] -= flow;
                d_left_weight[node_i + 1] += flow;
                atomicSub(&d_excess_flow[node_i], flow);
                atomicAdd(&d_excess_flow[node_i + 1], flow);
                //atomicAdd(&d_excess_flow_backup[iy*width + ix+1], flow);
                //d_excess_flow[iy*width + ix+1] += flow;
                //d_right_flow[node_i] += flow;
                //d_left_flow[iy*width + ix+1] -= flow;
                //atomicAdd(&d_right_flow[node_i], flow);
                //atomicSub(&d_left_flow[iy*width + ix+1], flow);
                //printf("pushing node %d to the right \n", node_i);
                //d_relabel_mask[node_i] = 0;
            }
            else{
                d_relabel_mask[node_i] = 1;
                //printf("in push right, node %d need relabel \n", node_i);
            }

        }
        //__threadfence();

        // push to the down
        if(iy+1 < height){
            if(d_excess_flow[node_i] > 0 && d_down_weight[node_i] > 0){
                if(d_graph_height[node_i] == 1 + d_graph_height[node_i + width]){
                    flow = min(d_down_weight[node_i], d_excess_flow[node_i]);
                    //d_excess_flow[node_i] -= flow;
                    d_down_weight[node_i] -= flow;
                    d_up_weight[node_i + width] += flow;
                    atomicSub(&d_excess_flow[node_i], flow);
                    atomicAdd(&d_excess_flow[node_i + width], flow);
                    //atomicAdd(&d_excess_flow[(iy+1)*width + ix], flow);
                    //atomicAdd(&d_excess_flow_backup[(iy+1)*width + ix], flow);
                    //d_excess_flow[(iy+1)*width + ix] += flow;
                    //d_down_flow[node_i] += flow;
                    //d_up_flow[(iy+1)*width + ix] -= flow;

                    //atomicAdd(&d_down_flow[node_i], flow);
                    //atomicSub(&d_up_flow[(iy+1)*width + ix], flow);
                    //printf("pushing node %d to the down\n", node_i);
                    //d_relabel_mask[node_i] = 0;
                }
                else{
                    d_relabel_mask[node_i] = 1;
                    //printf("in push down, node %d need relabel \n", node_i);
                }

            }
        }
        //__threadfence();

        //push up
        if(iy-1 >= 0){
            if(d_excess_flow[node_i] > 0 && d_up_weight[node_i] > 0){
                if(d_graph_height[node_i] == 1 + d_graph_height[node_i - width]){
                    flow = min(d_up_weight[node_i], d_excess_flow[node_i]);
                    //d_excess_flow[node_i] -= flow;
                    d_up_weight[node_i] -= flow;
                    d_down_weight[node_i - width] += flow;
                    atomicSub(&d_excess_flow[node_i], flow);
                    atomicAdd(&d_excess_flow[node_i - width], flow);
                    //atomicAdd(&d_excess_flow[(iy-1)*width + ix], flow);
                    //atomicAdd(&d_excess_flow_backup[(iy-1)*width + ix], flow);
                    //d_excess_flow[(iy-1)*width + ix] += flow;
                    //d_up_flow[node_i] += flow;
                    //d_down_flow[(iy-1)*width + ix] -= flow;
                    //atomicAdd(&d_up_flow[node_i], flow);
                    //atomicSub(&d_down_flow[(iy-1)*width + ix], flow);
                    //printf("pushing node %d to the up \n", node_i);
                    //d_relabel_mask[node_i] = 0;
                }
                else{
                    d_relabel_mask[node_i] = 1;
                    //printf("in push up, node %d need relabel \n", node_i);
                }

            }
        }
        //__threadfence();

        height_backup[0]=d_graph_height[node_i + 1];
        height_backup[2]=d_graph_height[node_i - 1];
        //ix+1 >= width?height_backup[0]=d_graph_height[N+1]:height_backup[0]=d_graph_height[iy*width + ix+1];
        iy+1 >= height?height_backup[1]=-1:height_backup[1]=d_graph_height[node_i + width];
        //ix-1 < 0?height_backup[2]=d_graph_height[N]:height_backup[2]=d_graph_height[iy*width + ix-1];
        iy-1 < 0?height_backup[3]=-1:height_backup[3]=d_graph_height[node_i - width];

        if(d_excess_flow[node_i] == 0){
            d_relabel_mask[node_i] = 0;
        }
    }




    // increase d_label_count if has node want to relabel


    //backup height

//    ix+1 >= width?height_backup[0]=-1:height_backup[0]=d_graph_height[iy*width + ix+1];
//    iy+1 >= height?height_backup[1]=-1:height_backup[1]=d_graph_height[(iy+1)*width + ix];
//    ix-1 < 0?height_backup[2]=-1:height_backup[2]=d_graph_height[iy*width + ix-1];
//    iy-1 < 0?height_backup[3]=-1:height_backup[3]=d_graph_height[(iy-1)*width + ix];


}

// check finish condition kernel
inline __global__ void
check_finished_condition(int *d_excess_flow, int *d_finished_count,int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    //if(ix >= width) return;
    if(ix != 0 && ix != width-1){
        if(d_excess_flow[node_i] > 0){
            *d_finished_count = 1;
            //printf("check finish condition, node %d has excessflow > 0\n", node_i);
        }
    }

    __syncthreads();
}

// relabel kernel
inline __global__ void
relabel_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
               int *d_right_flow, int *d_left_flow, int *d_up_flow, int *d_down_flow,
               int *d_graph_height, int *d_relabel_mask, int *d_height_backup, int *d_excess_flow, int *d_excess_flow_backup,
               int width, int height, int N){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    int mh = 1<<26;
    int *height_backup = d_height_backup + 4*node_i;

//    int flow = d_excess_flow_backup[node_i] + d_excess_flow[node_i];
//    d_excess_flow_backup[node_i] = 0;
//    d_excess_flow[node_i] = flow;
//    if(node_i == 0){
//        d_excess_flow[N] += d_excess_flow_backup[N];
//        d_excess_flow_backup[N] = 0;
//        printf("adding excess flow node 0, d_excessflow[N] %d\n", d_excess_flow[N]);
//    }
//    if(node_i == 1){
//        d_excess_flow[N+1] += d_excess_flow_backup[N+1];
//        d_excess_flow_backup[N+1] = 0;
//        printf("adding excess flow node 1, d_excessflow[N+1] %d\n", d_excess_flow[N+1]);
//    }


    // check right side
    //if(node_i > 0 && node_i < N-1){
//    if(flow <= 0 || (d_right_weight[node_i] <= 0 && d_left_weight[node_i] <= 0
//                     && d_down_weight[node_i] <= 0 && d_up_weight[node_i] <= 0))
//        d_relabel_mask[node_i] = 2;

//    flow > 0 && ((d_right_weight[node_i] > 0 && d_graph_height[node_i] == 1+height_backup[0])
//                 || (d_left_weight[node_i] > 0 && d_graph_height[node_i] == 1+height_backup[2])
//                 || (d_down_weight[node_i] > 0 && d_graph_height[node_i] == 1+height_backup[1])
//                 || (d_up_weight[node_i] > 0 && d_graph_height[node_i] == 1+height_backup[3]))? d_relabel_mask[node_i] = 1: d_relabel_mask[node_i] = 0;

//    if(d_relabel_mask[node_i] == 0){
//        int min_height = 1<<20 ;
//        (d_left_weight[node_i] > 0 && min_height > height_backup[2]) ? min_height = height_backup[2] : 0 ;
//        (d_right_weight[node_i] > 0 && min_height > height_backup[0]) ? min_height = height_backup[0] : 0 ;
//        (d_down_weight[node_i] > 0 && min_height > height_backup[1]) ? min_height = height_backup[1] : 0 ;
//        (d_up_weight[node_i] > 0 && min_height > height_backup[3]) ? min_height = height_backup[3] : 0 ;
//        if(min_height == 0)
//            printf("min_height = 0 at x - y: %d - %d\n", ix,iy);
//        d_graph_height[node_i] = min_height + 1 ;
//        d_relabel_mask[node_i] = 1;
//    }

//    __syncthreads();
        if(d_relabel_mask[node_i] != 0){
            if(d_right_weight[node_i] > 0 && height_backup[0] != -1){
                if(height_backup[0] < mh){
                    mh = height_backup[0];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }
        }
    //}

    __threadfence();
    __syncthreads();


    // check down side
    //if(node_i > 0 && node_i < N-1){
        if(d_relabel_mask[node_i] != 0){
            if(d_down_weight[node_i] > 0 && height_backup[1] != -1){
                if(height_backup[1] < mh){
                    mh = height_backup[1];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }
        }
    //}

    __threadfence();
    __syncthreads();

    // check left side
    //if(node_i > 0 && node_i < N-1){
        if(d_relabel_mask[node_i] != 0){
            if(d_left_weight[node_i] > 0 && height_backup[2] != -1){
                if(height_backup[2] < mh){
                    mh = height_backup[2];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }
        }
    //}


    __threadfence();
    __syncthreads();

    // check up side
    //if(node_i > 0 && node_i < N-1){
        if(d_relabel_mask[node_i] != 0){
            if(d_up_weight[node_i] > 0 && height_backup[3] != -1){
                if(height_backup[3] < mh){
                    mh = height_backup[3];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }
        }
    //}


    __threadfence();
    __syncthreads();
//    d_graph_height[node_i] = tmp;

    d_relabel_mask[node_i] = 0;
    __syncthreads();
}

// state on when node's excess flow > 0
// push state kernel
inline __global__ void
push_state_kernel(int *d_push_state, int *d_excess_flow, int width, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    //if(node_i > 0 && node_i < N-1){
        if(d_excess_flow[node_i] > 0){
            d_push_state[node_i] = 1;
            //printf("push state, node %d has excessflow > 0\n", node_i);
        }
    //}
    __syncthreads();
}


// add excess flow after push. This is a mediate step to avoid data race
// add excess flow kernel
inline __global__ void
add_excess_flow_kernel(int *d_excess_flow, int *d_pull_right, int *d_pull_left,
                       int *d_pull_up, int *d_pull_down, int *d_right_weight, int *d_left_weight,
                       int *d_up_weight, int *d_down_weight, int *d_excess_flow_backup, int width, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    //if((ix >= width && iy > 0) || (ix > width+1 && iy == 0)) return;
    int flow_left, flow_right, flow_up, flow_down, flow_push;
    //if(ix < width){
        flow_left = d_pull_right[node_i], flow_right = d_pull_left[node_i],
                flow_up = d_pull_down[node_i], flow_down = d_pull_up[node_i];
        d_pull_right[node_i] = 0;
        d_pull_left[node_i] = 0;
        d_pull_down[node_i] = 0;
        d_pull_up[node_i] = 0;
        flow_push = d_excess_flow[node_i];

        (flow_left > 0)? d_left_weight[node_i] += flow_left, flow_push += flow_left : 0;
        (flow_right > 0)? d_right_weight[node_i] += flow_right, flow_push += flow_right : 0;
        (flow_down > 0)? d_down_weight[node_i] += flow_down, flow_push += flow_down : 0;
        (flow_up > 0)? d_up_weight[node_i] += flow_up, flow_push += flow_up : 0;

        d_excess_flow[node_i] = flow_push;
    //}
//    else if(ix == width){
//        flow_right = d_pull_left[N];
//        d_pull_left[N] = 0;
//        d_excess_flow[N] += flow_right;
//        printf("adding excess flow node0, d_excessflow[N] %d\n", d_excess_flow[N]);
//    }
//    else{
//        flow_left = d_pull_right[N+1];
//        d_pull_right[N+1] = 0;
//        d_excess_flow[N+1] += flow_left;
//        printf("adding excess flow node0, d_excessflow[N+1] %d\n", d_excess_flow[N+1]);
//    }


    //d_excess_flow[node_i] += d_excess_flow_backup[node_i];
    //d_excess_flow_backup[node_i] = 0;

}

// bfs kernel
inline __global__ void
bfs_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_right_flow, int *d_left_flow, int *d_up_flow, int *d_down_flow,
            bool *d_visited, bool *d_frontier,
            int width, int height, int N, int *d_finish_bfs){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    if(d_frontier[node_i]){
        if(d_right_flow[node_i] != d_right_weight[node_i]){
            if(ix+1 < width){
                if(!(d_visited[iy*width + ix+1])){
                    d_visited[iy*width + ix+1] = true;
                    d_frontier[iy*width + ix+1] = true;
                }
            }
        }

        if(d_down_flow[node_i] != d_down_weight[node_i]){
            if(iy+1 < width){
                if(!(d_visited[(iy+1)*width + ix])){
                    d_visited[(iy+1)*width + ix] = true;
                    d_frontier[(iy+1)*width + ix] = true;
                }
            }
        }

        if(d_left_flow[node_i] != d_left_weight[node_i]){
            if(ix-1 >= 0){
                if(!(d_visited[iy*width + ix-1])){
                    d_visited[iy*width + ix-1] = true;
                    d_frontier[iy*width + ix-1] = true;
                }
            }
        }

        if(d_up_flow[node_i] != d_up_weight[node_i]){
            if(iy-1 < width){
                if(!(d_visited[(iy-1)*width + ix])){
                    d_visited[(iy-1)*width + ix] = true;
                    d_frontier[(iy-1)*width + ix] = true;
                }
            }
        }

        d_frontier[node_i] = false;
        *d_finish_bfs = 1;
    }


}

// backward bfs kernel
inline __global__ void
backward_bfs_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_right_flow, int *d_left_flow, int *d_up_flow, int *d_down_flow,
            int *d_visited, bool *d_frontier, int *d_graph_height,
            int width, int height, int N, int *d_finish_bfs){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;

    if(d_frontier[node_i]){
        if(d_right_weight[iy*width + ix-1] > 0){
            if(ix-1 >= 0){
                if(d_visited[iy*width + ix-1] == -1){
                    d_visited[iy*width + ix-1] = d_visited[node_i] + 1;
                    d_frontier[iy*width + ix-1] = true;
                    //printf("right-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
                }
            }
        }

        if(d_down_weight[(iy-1)*width + ix] > 0){
            if(iy-1 >= 0){
                if(!(d_visited[(iy-1)*width + ix])){
                    d_visited[(iy-1)*width + ix] = d_visited[node_i] + 1;
                    d_frontier[(iy-1)*width + ix] = true;
                    //printf("down-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
                }
            }
        }

        if(d_left_weight[iy*width + ix+1] > 0){
            if(ix+1 < width){
                if(!(d_visited[iy*width + ix+1])){
                    d_visited[iy*width + ix+1] = d_visited[node_i] + 1;
                    d_frontier[iy*width + ix+1] = true;
                    //printf("left-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
                }
            }
        }

        if(d_up_weight[(iy+1)*width + ix] > 0){
            if(iy+1 < height){
                if(!(d_visited[(iy+1)*width + ix])){
                    d_visited[(iy+1)*width + ix] = d_visited[node_i] + 1;
                    d_frontier[(iy+1)*width + ix] = true;
                    //printf("up-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
                }
            }
        }

        d_frontier[node_i] = false;
        *d_finish_bfs = 1;
        //d_graph_height[node_i] = d_visited[node_i];
    }
    if(ix != 0 && ix != width - 1){
    if(d_visited[node_i] != -1)
        d_graph_height[node_i] = d_visited[node_i];
    else
        d_graph_height[node_i] = N;
    }


}

inline __global__ void
kernel_push_nonatomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
                      int *g_excess_flow, int *g_pull_left, int *g_pull_right,
                      int *g_pull_down, int *g_pull_up, int *g_relabel_mask, int *g_graph_height,
                      int graph_size, int width, int height)
{
    int x1 = threadIdx.x ;
    int y1 = threadIdx.y ;
    int x  = blockIdx.x * blockDim.x + threadIdx.x ;
    int y  = blockIdx.y * blockDim.y + threadIdx.y ;
    int thid =  y * width + x ;

    __shared__ int height_fn[16];

    int temp_mult = __umul24(y1+1 , 4) + x1 + 1 ;

    height_fn[temp_mult] = g_graph_height[thid];

    (threadIdx.x == 1 && x < width - 1) ? height_fn[temp_mult + 1] =  (g_graph_height[thid + 1]) : 0;
    (threadIdx.x == 0  && x > 0) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
    (threadIdx.y == 1  && y < height - 1) ? height_fn[temp_mult + 4] = (g_graph_height[thid + width]) : 0;
    (threadIdx.y == 0  && y > 0) ? height_fn[temp_mult - 4] = (g_graph_height[thid - width]) : 0;

    (x >= width - 1) ? height_fn[temp_mult + 1] =  0 : 0;
    (x <= 0) ? height_fn[temp_mult - 1] = (g_graph_height[graph_size]) : 0;
    printf("height_fn block(%d, %d)\n", blockIdx.x, blockIdx.y);
    printf("x1 = %d, y1 = %d\n", x1, y1);
    printf("%d\t%d\t%d\t%d\t%d\t", height_fn[temp_mult], height_fn[temp_mult-1], height_fn[temp_mult+1], height_fn[temp_mult-4], height_fn[temp_mult+4]);

    //int temp_mult2 = 340 + __umul24(y1,32) + x1 ;
    int flow_push = g_excess_flow[thid] ;
    __syncthreads();
    if(g_relabel_mask[thid] == 1 && x <= width - 1 && x >= 0 && y <= height - 1 && y >= 0)
    {
        int  min_flow_pushed = 0 ;
        {
            int temp_weight = 0;

            /*****************************************
             *flow is pushed towards sink ***********
             ***************************************/

//            temp_weight = g_sink_weight[thid] ; // weight = capacity, excess_flow = flow_push
//            min_flow_pushed = flow_push ; //height_fn[temp_mult2] ;

//            if(temp_weight > 0 && flow_push  > 0 && height_fn[temp_mult] == 1)
//            {
//                (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
//                temp_weight = temp_weight - min_flow_pushed ;
//                g_sink_weight[thid] = temp_weight ;
//                flow_push = flow_push - min_flow_pushed ;
//            }

            /*****************************************
             * flow is pushed towards left edge ******
             ***************************************/
            min_flow_pushed = flow_push  ;
            temp_weight = g_left_weight[thid] ;

            if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 1] + 1)
            {
                (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
                temp_weight = temp_weight - min_flow_pushed ;
                g_left_weight[thid] = temp_weight ;
                if(x > 0){
                    g_pull_left[thid-1] = min_flow_pushed ;
                    printf("pushing left ... x = %d, y = %d \n",x, y);
                }
                else
                    atomicAdd(&g_excess_flow[graph_size],min_flow_pushed);
                flow_push = flow_push - min_flow_pushed ;

            }else g_pull_left[thid-1] = -1 ;

            /*****************************************
             *flow is pushed towards right edge *****
             ***************************************/

            min_flow_pushed = flow_push ;
            temp_weight = g_right_weight[thid] ;

            if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult + 1] + 1)
            {
                (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
                temp_weight = temp_weight - min_flow_pushed ;
                g_right_weight[thid] = temp_weight ;
                if(x < width -1){
                    g_pull_right[thid + 1] = min_flow_pushed ;
                    printf("pushing right ... x = %d, y = %d \n",x, y);
                }
                else
                    atomicAdd(&g_excess_flow[graph_size + 1], min_flow_pushed);
                flow_push = flow_push - min_flow_pushed ;
            }else g_pull_right[thid+1] = -1 ;

            /*****************************************
             * flow is pushed towards down edge  *****
             ***************************************/

            min_flow_pushed = flow_push ;
            temp_weight = g_down_weight[thid] ;

            if(temp_weight > 0 && flow_push  > 0 && height_fn[temp_mult] == height_fn[temp_mult + 4] + 1)
            {
                (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
                temp_weight = temp_weight - min_flow_pushed ;
                g_down_weight[thid] = temp_weight ;
                g_pull_down[thid + width] = min_flow_pushed ;
                flow_push = flow_push - min_flow_pushed ;
                printf("pushing down ... x = %d, y = %d \n",x, y);
            }else g_pull_down[thid + width] = -1 ;

            /*****************************************
             * flow is pushed towards up edge  *******
             ***************************************/

            min_flow_pushed = flow_push ;
            temp_weight = g_up_weight[thid] ;

            if(temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == height_fn[temp_mult - 4] + 1)
            {
                (temp_weight<flow_push) ? min_flow_pushed = temp_weight : 0 ;
                temp_weight = temp_weight - min_flow_pushed ;
                g_up_weight[thid] = temp_weight ;
                g_pull_up[thid - width] =  min_flow_pushed ;
                flow_push = flow_push - min_flow_pushed ;
                printf("pushing up ... x = %d, y = %d \n",x, y);
            }else g_pull_up[thid - width] = -1 ;
            g_excess_flow[thid] = flow_push ;
            //            if(flow_push!=0)
            //                printf("g_push_reser[%d] = %d\n", thid, flow_push);
        }
    }
}


inline __global__ void
krelabel_nonatomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
                   int *g_excess_flow, int *g_pull_left, int *g_pull_right, int *g_pull_down, int *g_pull_up,
                   int *g_relabel_mask, int* g_height_read, int* g_height_write, int graph_size,
                   int width, int height)
{
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = y * width + x;

    __shared__ int height_fn[16];

    int temp_mult = __umul24(y1+1 , 4) + x1 + 1 ;

    height_fn[temp_mult] = g_height_read[thid] ;

    (threadIdx.x == 1 && x < width - 1) ? height_fn[temp_mult + 1] =  (g_height_read[thid + 1]) : 0;
    (threadIdx.x == 0  && x > 0) ? height_fn[temp_mult - 1] = (g_height_read[thid - 1]) : 0;
    (threadIdx.y == 1  && y < height - 1) ? height_fn[temp_mult + 4] = (g_height_read[thid + width]) : 0;
    (threadIdx.y == 0  && y > 0) ? height_fn[temp_mult - 4] = (g_height_read[thid - width]) : 0;

    (x >= width - 1) ? height_fn[temp_mult + 1] =  0 : 0;
    (x <= 0) ? height_fn[temp_mult - 1] = (g_height_read[graph_size]) : 0;
    printf("height_fn block(%d, %d)\n", blockIdx.x, blockIdx.y);
    printf("x1 = %d, y1 = %d\n", x1, y1);
    printf("%d\t%d\t%d\t%d\t%d\t", height_fn[temp_mult], height_fn[temp_mult-1], height_fn[temp_mult+1], height_fn[temp_mult-4], height_fn[temp_mult+4]);

    __syncthreads();

    int flow_push = g_excess_flow[thid] ;


    if(x < width && x >= 0 && y < height && y >= 0)
    {
        int flow_left = g_pull_right[thid] , flow_right = g_pull_left[thid], flow_down = g_pull_up[thid], flow_up = g_pull_down[thid] ;

        g_pull_left[thid] = 0 ;
        g_pull_right[thid] = 0 ;
        g_pull_down[thid] = 0 ;
        g_pull_up[thid] = 0;

        // Flow <=0  means there is no push in a specific direction
        (flow_left > 0) ? (g_left_weight[thid] = g_left_weight[thid] + flow_left), flow_push += flow_left : 0 ;
        (flow_right > 0) ? (g_right_weight[thid] = g_right_weight[thid] + flow_right), flow_push += flow_right : 0 ;
        (flow_down > 0) ? (g_down_weight[thid] = g_down_weight[thid] + flow_down), flow_push += flow_down : 0 ;
        (flow_up > 0) ? (g_up_weight[thid] = g_up_weight[thid] + flow_up), flow_push  += flow_up : 0 ;

        g_excess_flow[thid] = flow_push ;

        if(flow_push <= 0 || (g_left_weight[thid] == 0 && g_right_weight[thid] == 0 && g_down_weight[thid] == 0 && g_up_weight[thid] == 0))
            g_relabel_mask[thid] = 2 ; // = 2 means inactive.
        else
        {
//            (flow_push > 0 && (flow_left == -1 && flow_right == -1 && flow_down == -1 && flow_up == -1 /*g_height_read[thid] != 1*/)) ? g_relabel_mask[thid] = 0 : g_relabel_mask[thid] = 1 ;
            (flow_push > 0 && (height_fn[temp_mult] <= height_fn[temp_mult - 1]&&
                               height_fn[temp_mult] <= height_fn[temp_mult + 1]&&
                               height_fn[temp_mult] <= height_fn[temp_mult + 4]&&
                               height_fn[temp_mult] <= height_fn[temp_mult - 4])) ? g_relabel_mask[thid] = 0 : 0 ;

        }

    }

    __syncthreads();
//    if(g_relabel_mask[thid] == 0)
//        printf("g_relabel_mask = 0 at x - y: %d - %d\n", x, y);
    if(/*thid < graph_size1 && */x <= width - 1  && x >= 0 && y <= height - 1  && y >= 0)
    {
//        if(g_sink_weight[thid] > 0)
//        {
//            g_height_write[thid] = 1 ;
////            if(x!=width -1)
////                printf("g_sink_weight = 0 in x - y: %d - %d\n", x, y);
//        }

        if(g_relabel_mask[thid] == 0)
        {
            int min_height = 1<<20 ;
            (g_left_weight[thid] > 0 && min_height > height_fn[temp_mult - 1]) ? min_height = height_fn[temp_mult - 1] : 0 ;
            (g_right_weight[thid] > 0 && min_height > height_fn[temp_mult + 1]) ? min_height = height_fn[temp_mult + 1] : 0 ;
            (g_down_weight[thid] > 0 && min_height > height_fn[temp_mult + 4]) ? min_height = height_fn[temp_mult + 4] : 0 ;
            (g_up_weight[thid] > 0 && min_height > height_fn[temp_mult - 4]) ? min_height = height_fn[temp_mult - 4] : 0 ;
            if(min_height == 0)
                printf("min_height = 0 at x - y: %d - %d\n", x,y);
            g_height_write[thid] = min_height + 1 ;
            g_relabel_mask[thid] = 1;
        }
    }
    __syncthreads();
    //    printf("g_height_write[%d] = %d\n", thid, g_height_write[thid]);
}
void CudaCut::cudaCutsAtomic(){
    //size = 80*480

    dim3 block(16, 8, 1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    printf("<<<grid(%d, %d), block(%d, %d)>>>", grid.x, grid.y, block.x, block.y);
    int h_finished_count;
    int h_relabel_count;
    int  h_finish_bfs;

    int *d_finished_count, *d_finish_bfs;

    int *d_relabel_count;


    gpuErrChk(cudaMalloc((void**)&d_finished_count, sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_relabel_count, sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_finish_bfs, sizeof(int)));



    //gpuErrChk(cudaDeviceSynchronize());

    h_finished_count = 1;
    int counter = 0;
    while(h_finished_count != 0){

        //gpuErrChk(cudaDeviceSynchronize());
        //gpuErrChk(cudaMemcpy(d_relabel_count, &h_relabel_count, sizeof(int), cudaMemcpyHostToDevice));
        //gpuErrChk(cudaDeviceSynchronize());

        // global relabeling
        if(counter%80 == 0){
            h_finish_bfs = 1;
            for(int i = 0; i < graph_size; i++){
                if((i+1)%width != 0)
                    h_visited[i] = -1;
                else
                    h_visited[i] = 0;
            }
            //h_visited[0] = true;
            gpuErrChk(cudaMemcpy(d_visited, h_visited, sizeof(int) * graph_size , cudaMemcpyHostToDevice));
            for(int i = 1; i < graph_size; i++){
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

                backward_bfs_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                            d_right_flow, d_left_flow, d_up_flow, d_down_flow, d_visited, d_frontier, d_graph_height,
                            width, height, graph_size, d_finish_bfs);

                gpuErrChk(cudaMemcpy(&h_finish_bfs, d_finish_bfs, sizeof(int), cudaMemcpyDeviceToHost));
            }
            gpuErrChk(cudaMemcpy(h_graph_height, d_graph_height, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
            gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
            writeToFile("../variable/h_graph_height.txt", h_graph_height, width, height);
            writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
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
            //while(getchar() != 32);

        }
        // state on for node that has excess flow > 0
        //push_state_kernel<<<grid, block>>>(d_push_state, d_excess_flow, width, graph_size);
        //gpuErrChk(cudaDeviceSynchronize());

        // pushing and save offset of flow to mediate array (d_excess_flow_backup)
        auto start = getMoment;
        push_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                                     d_pull_right, d_pull_left, d_pull_up, d_pull_down, d_excess_flow,
                                     d_graph_height, d_relabel_mask,d_push_state, d_height_backup, d_excess_flow_backup,
                                     width, height, graph_size, d_relabel_count);
        //gpuErrChk(cudaDeviceSynchronize());
        auto end = getMoment;
        cout << "push kernel time iteration " << counter << " " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microsecond" << endl;
        gpuErrChk(cudaMemcpy(h_graph_height, d_graph_height, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
        writeToFile("../variable/h_graph_height.txt", h_graph_height, width, height);
        writeToFile("../variable/h_excess_flow.txt", h_excess_flow, width, height);
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
        if(counter%2 == 0){
            //if(h_relabel_count != 0){
                // relabel if relabel_count > 0
            h_finished_count = 0;
            //h_relabel_count = 0;
    //        //printf("stop\n");
            gpuErrChk(cudaMemcpy(d_finished_count, &h_finished_count, sizeof(int), cudaMemcpyHostToDevice));
                relabel_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                                                d_right_flow, d_left_flow, d_up_flow, d_down_flow,
                                                d_graph_height, d_relabel_mask, d_height_backup, d_excess_flow, d_excess_flow_backup,
                                                width, height, graph_size);
                check_finished_condition<<<grid, block>>>(d_excess_flow, d_finished_count, width, height, graph_size);
                gpuErrChk(cudaMemcpy(&h_finished_count, d_finished_count, sizeof(int), cudaMemcpyDeviceToHost));
                //gpuErrChk(cudaDeviceSynchronize());
            //}
        }



        //gpuErrChk(cudaDeviceSynchronize());

        // get finished_count to check finish condition
//        if(counter % 2 == 0)
//        {

//            kernel_push_nonatomic<<<grid, block>>>(d_left_weight,d_right_weight, d_down_weight, d_up_weight,
//                                                            d_excess_flow,d_pull_left, d_pull_right, d_pull_down, d_pull_up,
//                                                            d_relabel_mask,d_graph_heightr, graph_size,width,height);


//            krelabel_nonatomic<<<grid, block>>>(d_left_weight,d_right_weight, d_down_weight, d_up_weight,
//                                                         d_excess_flow,d_pull_left, d_pull_right, d_pull_down, d_pull_up,
//                                                         d_relabel_mask, d_graph_heightr,d_graph_heightw, graph_size,width,height);

//        }
//        else
//        {
//            kernel_push_nonatomic<<<grid, block>>>(d_left_weight,d_right_weight, d_down_weight, d_up_weight,
//                                                            d_excess_flow,d_pull_left, d_pull_right, d_pull_down, d_pull_up,
//                                                            d_relabel_mask,d_graph_heightw, graph_size,width,height);


//            krelabel_nonatomic<<<grid, block>>>(d_left_weight,d_right_weight, d_down_weight, d_up_weight,
//                                                         d_excess_flow, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
//                                                         d_relabel_mask, d_graph_heightw,d_graph_heightr, graph_size,width,height);
//        }


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

    }
    cout << "cout = " << counter << endl;
    gpuErrChk(cudaFree(d_finished_count));
    gpuErrChk(cudaFree(d_relabel_count));

    gpuErrChk(cudaMemcpy(h_excess_flow, d_excess_flow, sizeof(int)*graph_size, cudaMemcpyDeviceToHost));
    //gpuErrChk(cudaDeviceSynchronize());
    int sum = 0;
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

//    int h_finish_bfs = 1;
//    int *d_finish_bfs;
//    gpuErrChk(cudaMalloc((void**)&d_finish_bfs, sizeof(int)));

//    while(h_finish_bfs != 0){
//        h_finish_bfs = 0;
//        gpuErrChk(cudaMemcpy(d_finish_bfs, &h_finish_bfs, sizeof(int), cudaMemcpyHostToDevice));

//        bfs_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
//                    d_right_flow, d_left_flow, d_up_flow, d_down_flow, d_visited, d_frontier,
//                    width, height, graph_size, d_finish_bfs);

//        gpuErrChk(cudaMemcpy(&h_finish_bfs, d_finish_bfs, sizeof(int), cudaMemcpyDeviceToHost));
//    }
//    gpuErrChk(cudaMemcpy(h_visited, d_visited, sizeof(bool)*graph_size, cudaMemcpyDeviceToHost));

//    std::cout << "set(s): ";
//    for(int i = 0; i < graph_size; i++){
//        if(h_visited[i])
//            std::cout << i << " ";
//    }
//    std::cout << "\n";

//    gpuErrChk(cudaFree(d_finish_bfs));



}

int CudaCut::cudaCutsAtomicOptimize()
{
    cudaCutsAtomic();
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

    free(h_pull_left);
    free(h_pull_right);
    free(h_pull_down);
    free(h_pull_up);
    free(h_graph_heightr);
    free(h_graph_heightw);

    free(h_sink_weight);



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

    gpuErrChk(cudaFree(d_pull_left));
    gpuErrChk(cudaFree(d_pull_right));
    gpuErrChk(cudaFree(d_pull_down));
    gpuErrChk(cudaFree(d_pull_up));
    gpuErrChk(cudaFree(d_graph_heightr));
    gpuErrChk(cudaFree(d_graph_heightw));

    gpuErrChk(cudaFree(d_sink_weight));

}

