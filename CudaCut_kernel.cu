#include "cudacut.h"


// push kernel
inline __global__ void
push_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_excess_flow, int *d_graph_height, int *d_relabel_mask, int *d_height_backup,
            int width, int height, int N){

    //int x1 = threadIdx.x;
    //int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    //if(ix >= width) return;

    int flow;
    //bool left_relabel = true, right_relabel = true, down_relabel = true, up_relabel = true;
    int *height_backup = d_height_backup + 4*node_i;

        //__syncthreads();



        //d_push_state[node_i] = 0;


    if(ix != 0 && ix != width - 1){
        // push to the left
        if(d_excess_flow[node_i] > 0 && d_left_weight[node_i] > 0){
            if(d_graph_height[node_i] == 1+ d_graph_height[node_i - 1]){
                flow = min(d_left_weight[node_i], d_excess_flow[node_i]);
                //d_excess_flow[node_i] -= flow;
                d_left_weight[node_i] -= flow;
                d_right_weight[node_i - 1] += flow;
                atomicSub(&d_excess_flow[node_i], flow);
                atomicAdd(&d_excess_flow[node_i - 1], flow);
                //left_relabel = false;
                //atomicAdd(&d_excess_flow_backup[node_i - 1], flow);
                //printf("pushing node %d to the left \n", node_i);
            }
//            else{
//                d_relabel_mask[node_i] = 1;
//                //printf("in push left, node %d need relabel \n", node_i);
//            }

        }


        // push right
        if(d_excess_flow[node_i] > 0 && d_right_weight[node_i] > 0){
            if(d_graph_height[node_i] == 1 + d_graph_height[node_i + 1]){
                flow = min(d_right_weight[node_i], d_excess_flow[node_i]);
                d_right_weight[node_i] -= flow;
                d_left_weight[node_i + 1] += flow;
                atomicSub(&d_excess_flow[node_i], flow);
                atomicAdd(&d_excess_flow[node_i + 1], flow);
                //right_relabel = false;
                //atomicAdd(&d_excess_flow_backup[node_i + 1], flow);
                //printf("pushing node %d to the right \n", node_i);
            }
//            else{
//                d_relabel_mask[node_i] = 1;
//                //printf("in push right, node %d need relabel \n", node_i);
//            }

        }
        //__threadfence();


        //__threadfence();

        // push to the down
        if(iy+1 < height){
            if(d_excess_flow[node_i] > 0 && d_down_weight[node_i] > 0){
                if(d_graph_height[node_i] == 1+ d_graph_height[node_i + width]){
                    flow = min(d_down_weight[node_i], d_excess_flow[node_i]);
                    //d_excess_flow[node_i] -= flow;
                    d_down_weight[node_i] -= flow;
                    d_up_weight[node_i + width] += flow;
                    atomicSub(&d_excess_flow[node_i], flow);
                    atomicAdd(&d_excess_flow[node_i + width], flow);
                    //down_relabel = false;
                    //atomicAdd(&d_excess_flow_backup[node_i + width], flow);
                    //printf("pushing node %d to the down\n", node_i);
                }
//                else{
//                    d_relabel_mask[node_i] = 1;
//                    //printf("in push down, node %d need relabel \n", node_i);
//                }

            }
        }
        //__threadfence();

        //push up
        if(iy-1 >= 0){
            if(d_excess_flow[node_i] > 0 && d_up_weight[node_i] > 0){
                if(d_graph_height[node_i] == 1+ d_graph_height[node_i - width]){
                    flow = min(d_up_weight[node_i], d_excess_flow[node_i]);
                    //d_excess_flow[node_i] -= flow;
                    d_up_weight[node_i] -= flow;
                    d_down_weight[node_i - width] += flow;
                    atomicSub(&d_excess_flow[node_i], flow);
                    atomicAdd(&d_excess_flow[node_i - width], flow);
                    //up_relabel = false;
                    //atomicAdd(&d_excess_flow_backup[node_i - width], flow);
                    //printf("pushing node %d to the up \n", node_i);
                }
//                else{
//                    d_relabel_mask[node_i] = 1;
//                    //printf("in push up, node %d need relabel \n", node_i);
//                }

            }
        }

        //__threadfence();

        /*right_relabel ? */height_backup[0]=d_graph_height[node_i + 1]/* : height_backup[0] = -1*/; // right
        /*left_relabel ? */height_backup[2]=d_graph_height[node_i - 1]/* : height_backup[2] = -1*/; // left
        //ix+1 >= width?height_backup[0]=d_graph_height[N+1]:height_backup[0]=d_graph_height[iy*width + ix+1];
        iy+1 < height ? height_backup[1]=d_graph_height[node_i + width] : height_backup[1] = -1;
        //ix-1 < 0?height_backup[2]=d_graph_height[N]:height_backup[2]=d_graph_height[iy*width + ix-1];
        iy-1 >= 0 ? height_backup[3]=d_graph_height[node_i - width] : height_backup[3] = -1;

        if(d_excess_flow[node_i] > 0){
            d_relabel_mask[node_i] = 1;
        }
    }




    // increase d_label_count if has node want to relabel


    //backup height

//    ix+1 >= width?height_backup[0]=-1:height_backup[0]=d_graph_height[iy*width + ix+1];
//    iy+1 >= height?height_backup[1]=-1:height_backup[1]=d_graph_height[(iy+1)*width + ix];
//    ix-1 < 0?height_backup[2]=-1:height_backup[2]=d_graph_height[iy*width + ix-1];
//    iy-1 < 0?height_backup[3]=-1:height_backup[3]=d_graph_height[(iy-1)*width + ix];


}
inline __global__ void
push_kernel2(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_excess_flow, int *d_graph_height, int *d_relabel_mask, int *d_height_backup,
            int width, int height, int N){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;

    int flow;
    int *height_backup = d_height_backup + 4*node_i;


    if(ix != 0 && ix != width - 1){
        // push right
        if(d_excess_flow[node_i] > 0 && d_right_weight[node_i] > 0){
            if(d_graph_height[node_i] == 1 + d_graph_height[node_i + 1]){
                flow = min(d_right_weight[node_i], d_excess_flow[node_i]);
                d_right_weight[node_i] -= flow;
                d_left_weight[node_i + 1] += flow;
                atomicSub(&d_excess_flow[node_i], flow);
                atomicAdd(&d_excess_flow[node_i + 1], flow);
            }

        }
        // push to the left
        if(d_excess_flow[node_i] > 0 && d_left_weight[node_i] > 0){
            if(d_graph_height[node_i] == 1+ d_graph_height[node_i - 1]){
                flow = min(d_left_weight[node_i], d_excess_flow[node_i]);
                //d_excess_flow[node_i] -= flow;
                d_left_weight[node_i] -= flow;
                d_right_weight[node_i - 1] += flow;
                atomicSub(&d_excess_flow[node_i], flow);
                atomicAdd(&d_excess_flow[node_i - 1], flow);
            }

        }
        //__threadfence();

        // push to the down
        if(iy+1 < height){
            if(d_excess_flow[node_i] > 0 && d_down_weight[node_i] > 0){
                if(d_graph_height[node_i] == 1+ d_graph_height[node_i + width]){
                    flow = min(d_down_weight[node_i], d_excess_flow[node_i]);
                    //d_excess_flow[node_i] -= flow;
                    d_down_weight[node_i] -= flow;
                    d_up_weight[node_i + width] += flow;
                    atomicSub(&d_excess_flow[node_i], flow);
                    atomicAdd(&d_excess_flow[node_i + width], flow);
                }
            }
        }


        //push up
        if(iy-1 >= 0){
            if(d_excess_flow[node_i] > 0 && d_up_weight[node_i] > 0){
                if(d_graph_height[node_i] == 1+ d_graph_height[node_i - width]){
                    flow = min(d_up_weight[node_i], d_excess_flow[node_i]);
                    //d_excess_flow[node_i] -= flow;
                    d_up_weight[node_i] -= flow;
                    d_down_weight[node_i - width] += flow;
                    atomicSub(&d_excess_flow[node_i], flow);
                    atomicAdd(&d_excess_flow[node_i - width], flow);
                }

            }
        }


        //__threadfence();

        /*right_relabel ? */height_backup[0]=d_graph_height[node_i + 1]/* : height_backup[0] = -1*/; // right
        /*left_relabel ? */height_backup[2]=d_graph_height[node_i - 1]/* : height_backup[2] = -1*/; // left
        //ix+1 >= width?height_backup[0]=d_graph_height[N+1]:height_backup[0]=d_graph_height[iy*width + ix+1];
        iy+1 < height ? height_backup[1]=d_graph_height[node_i + width] : height_backup[1] = -1;
        //ix-1 < 0?height_backup[2]=d_graph_height[N]:height_backup[2]=d_graph_height[iy*width + ix-1];
        iy-1 >= 0 ? height_backup[3]=d_graph_height[node_i - width] : height_backup[3] = -1;

        if(d_excess_flow[node_i] > 0){
            d_relabel_mask[node_i] = 1;
        }
    }

}
inline __global__ void
push_block_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_excess_flow, int *d_graph_height, int *d_relabel_mask, int *d_height_backup, int *d_excess_flow_backup,
            int *d_push_block_position, int width, int height, int N, int *counter){

    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = x1 + blockIdx.x*blockDim.x;
    int iy = y1 + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;

    // counter = 0,1,2,3,4 || 10,11,12,13,14
    //printf("counter in push_block_kernel %d, blockIdx.x %d \n", *counter, blockIdx.x);
    int cond = (*counter)%15;
    if(cond < 5){
        if(cond == blockIdx.x){
            //printf("cond in push_block_kernel %d\n", cond);
            int pos = d_push_block_position[iy*gridDim.x + blockIdx.x];
            //printf("pos of blockIdx.%d  %d\n", blockIdx.x, pos);
            __shared__ int flow_block[10];
            __shared__ int smem[180];
            int idx = (y1+1)*18 + x1+1;
            x1 >= pos?smem[idx] = d_right_weight[node_i] : smem[idx] = 1<<20;
            if(blockIdx.x == 4 && x1 == 15)
                smem[idx] = 1<<20;
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
            for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
                if(x1 < stride){
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

            if(x1 >= pos){
                if(d_excess_flow[node_i] > 0){
                    flow_block[y1] = min(d_excess_flow[node_i], smem[(y1+1)*18 + 1]);
                    atomicSub(&d_excess_flow[node_i], flow_block[y1]);
                    // notice
                    if(blockIdx.x != 4)
                        atomicAdd(&d_excess_flow[iy*width + blockDim.x*(blockIdx.x+1)], flow_block[y1]);
                    else
                        atomicAdd(&d_excess_flow[iy*width + blockDim.x*(blockIdx.x+1) - 1], flow_block[y1]);
                }
            }
            __syncthreads();
            // print flow_block
//            if(x1 == 0 && y1 == 0 && blockIdx.y == 0){
//                for(int j = 0; j < 8; j++){
//                    printf("%d ", flow_block[j]);
//                    }
//                 printf("\n");
//            }
//            __syncthreads();
            if(x1 >= pos){
                d_right_weight[node_i] -= flow_block[y1];
                if(d_right_weight[node_i] == 0){
                    d_push_block_position[iy*gridDim.x + blockIdx.x] = ix+1;
                }
            }

        }
        if(blockIdx.x < cond){
            int *height_backup = d_height_backup + 4*node_i;

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
                        //atomicAdd(&d_excess_flow_backup[node_i - 1], flow);
                        //printf("pushing node %d to the left \n", node_i);
                    }
                    else{
                        d_relabel_mask[node_i] = 1;
                        //printf("in push left, node %d need relabel \n", node_i);
                    }

                }

                // push right
                if(d_excess_flow[node_i] > 0 && d_right_weight[node_i] > 0){
                    if(d_graph_height[node_i] == 1 + d_graph_height[node_i + 1]){
                        flow = min(d_right_weight[node_i], d_excess_flow[node_i]);
                        d_right_weight[node_i] -= flow;
                        d_left_weight[node_i + 1] += flow;
                        atomicSub(&d_excess_flow[node_i], flow);
                        atomicAdd(&d_excess_flow[node_i + 1], flow);
                        //atomicAdd(&d_excess_flow_backup[node_i + 1], flow);
                        //printf("pushing node %d to the right \n", node_i);
                    }
                    else{
                        d_relabel_mask[node_i] = 1;
                        //printf("in push right, node %d need relabel \n", node_i);
                    }

                }
                //__threadfence();


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
                            //atomicAdd(&d_excess_flow_backup[node_i + width], flow);
                            //printf("pushing node %d to the down\n", node_i);
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
                            //atomicAdd(&d_excess_flow_backup[node_i - width], flow);
                            //printf("pushing node %d to the up \n", node_i);
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

        }
        //  __syncthreads();
        return;
    }
//    if(*counter == 1){
//        if(blockIdx.x == 1){
//            //printf("cond in push_block_kernel %d\n", cond);
//            int pos = d_push_block_position[iy*gridDim.x + blockIdx.x];
//            //printf("pos of blockIdx.%d  %d\n", blockIdx.x, pos);
//            __shared__ int flow_block[10];
//            __shared__ int smem[180];
//            int idx = (y1+1)*18 + x1+1;
//            x1 >= pos?smem[idx] = d_right_weight[node_i] : smem[idx] = 1<<20;
//            __syncthreads();
//            // print smem 1
//            if(x1 == 0 && y1 == 0 && blockIdx.y == 0){
//                for(int i = 0; i < 8; i++){
//                    for(int j = 0; j < 16; j++){
//                        printf("%d ", smem[(i+1)*18+j+1]);
//                    }
//                    printf("\n");
//                }
//            }
//            __syncthreads();
//            for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
//                if(x1 < stride){
//                    smem[idx]>smem[idx+stride]?smem[idx] = smem[idx+stride] : 0;

//                }

//                __syncthreads();
//            }

//            // print smem 2
//            if(x1 == 0 && y1 == 0 && blockIdx.y == 0){
//                for(int i = 0; i < 8; i++){
//                    for(int j = 0; j < 16; j++){
//                        printf("%d ", smem[(i+1)*18+j+1]);
//                    }
//                    printf("\n");
//                }
//            }
//            __syncthreads();


//            if(x1 == pos){
//                if(d_excess_flow[node_i] > 0){
//                    flow_block[y1] = min(d_excess_flow[node_i], smem[(y1+1)*18 + 1]);
//                    atomicSub(&d_excess_flow[node_i], flow_block[y1]);
//                    atomicAdd(&d_excess_flow[iy*width + blockDim.x*2], flow_block[y1]);
//                }
//            }

//            // print flow_block
//            if(x1 == 0 && y1 == 0 && blockIdx.y == 0){
//                for(int j = 0; j < 8; j++){
//                    printf("%d ", flow_block[j]);
//                    }
//                 printf("\n");
//            }
//            __syncthreads();

//            if(x1 >= pos){
//                d_right_weight[node_i] -= flow_block[y1];
//                if(d_right_weight[node_i] == 0){
//                    d_push_block_position[iy*gridDim.x + blockIdx.x] = ix+1;
//                }
//            }

//        }
//        __syncthreads();
//        return;
//    }


}

// check finish condition kernel
inline __global__ void
check_finished_condition(int *d_excess_flow, int *d_finished_count, int *d_graph_height, int width, int height, int N){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    //if(ix >= width) return;
    if(ix != 0 && ix != width-1){
        if(d_excess_flow[node_i] > 0){
            atomicAdd(d_finished_count,1);
//            if(d_graph_height[node_i] >= N)
//                d_excess_flow[node_i] = 0;
            //printf("check finish condition, node (%d %d) has excessflow > 0\n", ix, iy);
        }
    }

    __syncthreads();
}

// relabel kernel
inline __global__ void
relabel_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight,
               int *d_down_weight, int *d_graph_height, int *d_relabel_mask,
               int *d_height_backup, int *d_excess_flow, int width, int height, int N){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    int mh = 1<<26;
    int *height_backup = d_height_backup + 4*node_i;

    if(ix > 0 && ix < width - 1){
        if(d_relabel_mask[node_i] != 0){
            if(d_right_weight[node_i] > 0 && height_backup[0] != -1){
                if(height_backup[0] < mh){
                    mh = height_backup[0];
                    d_graph_height[node_i] = mh + 1;
                    //height_node_i[idx] = mh + 1;
                    //tmp = mh + 1;
                }
            }
            if(d_left_weight[node_i] > 0 && height_backup[2] != -1){
                if(height_backup[2] < mh){
                    mh = height_backup[2];
                    d_graph_height[node_i] = mh + 1;
                    //height_node_i[idx] = mh + 1;
                    //tmp = mh + 1;
                }
            }
            if(d_down_weight[node_i] > 0 && height_backup[1] != -1){
                if(height_backup[1] < mh){
                    mh = height_backup[1];
                    d_graph_height[node_i] = mh + 1;
                    //height_node_i[idx] = mh + 1;
                    //tmp = mh + 1;
                }
            }
            if(d_up_weight[node_i] > 0 && height_backup[3] != -1){
                if(height_backup[3] < mh){
                    mh = height_backup[3];
                    d_graph_height[node_i] = mh + 1;
                    //height_node_i[idx] = mh + 1;
                    //tmp = mh + 1;
                }
            }
//            __threadfence();
//            d_graph_height[node_i] = height_node_i[idx];

        }
    //}

    //__threadfence();
    //__syncthreads();


    // check down side
    //if(node_i > 0 && node_i < N-1){
//        if(d_relabel_mask[node_i] != 0){
//            if(d_down_weight[node_i] > 0 && height_backup[1] != -1){
//                if(height_backup[1] < mh){
//                    mh = height_backup[1];
//                    d_graph_height[node_i] = mh + 1;
//                    //tmp = mh + 1;
//                }
//            }
//        }
//    //}

//    __threadfence();
//    __syncthreads();

//    // check left side
//    //if(node_i > 0 && node_i < N-1){
//        if(d_relabel_mask[node_i] != 0){
//            if(d_left_weight[node_i] > 0 && height_backup[2] != -1){
//                if(height_backup[2] < mh){
//                    mh = height_backup[2];
//                    d_graph_height[node_i] = mh + 1;
//                    //tmp = mh + 1;
//                }
//            }
//        }
//    //}


//    __threadfence();
//    __syncthreads();

//    // check up side
//    //if(node_i > 0 && node_i < N-1){
//        if(d_relabel_mask[node_i] != 0){
//            if(d_up_weight[node_i] > 0 && height_backup[3] != -1){
//                if(height_backup[3] < mh){
//                    mh = height_backup[3];
//                    d_graph_height[node_i] = mh + 1;
//                    //tmp = mh + 1;
//                }
//            }
//        }
//    //}


//    __threadfence();
//    __syncthreads();
//    d_graph_height[node_i] = tmp;

    d_relabel_mask[node_i] = 0;
    //__syncthreads();
    }
}

// bfs kernel
inline __global__ void
bfs_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_right_flow, int *d_left_flow, int *d_up_flow, int *d_down_flow,
            int *d_visited, bool *d_frontier,
            int width, int height, int N, int *d_finish_bfs){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    if(d_frontier[node_i]){
        if(ix+1 < width && d_right_weight[node_i] > 0){
            if(d_visited[node_i + 1] == -1){
                d_visited[node_i + 1] = 0;
                d_frontier[node_i + 1] = true;
            }

        }

        if(iy+1 < height && d_down_weight[node_i] > 0){
            if(d_visited[node_i + width] == -1){
                d_visited[node_i + width] = 0;
                d_frontier[node_i + width] = true;
            }

        }

        if(iy >= 1 && d_up_weight[node_i] > 0){
                if(d_visited[node_i - width] == -1){
                    d_visited[node_i - width] = 0;
                    d_frontier[node_i - width] = true;
                }

        }

        if(ix >= 1 && d_left_weight[node_i] > 0){
            if(d_visited[node_i - 1] == -1){
                d_visited[node_i - 1] = 0;
                d_frontier[node_i - 1] = true;
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
            int width, int height, int N, int *d_finish_bfs, int *d_count){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;

//    if(d_frontier[node_i] == true && ix !=0 && ix != width-1)
//    {
//        int height = d_graph_height[node_i];
//        int height_l = 0, height_d = 0, height_u = 0 , height_r = 0 ;
//        height_r = g_graph_height[node_i+1] ;
//        height_l = g_graph_height[node_i-1] ;
//        iy >= 1 ? height_d = g_graph_height[node_i+width1] ;
//        height_u = g_graph_height[node_i-width1] ;
//        //            if(height_d !=0 |height_r !=0|height_u !=0|height_l !=0)
//        //                printf("g_counter - l - r - d - u: %d - %d - %d -%d -%d\n", *g_counter, height_l, height_r, height_d, height_u);
//        //            printf("g_counter - r - u - l - d: %d - %d - %d -%d -%d\n", *g_counter, g_right_weight[thid-1], g_up_weight[thid+width1], g_left_weight[thid+1], g_down_weight[thid-width1]);

//        if(((height_l == (*g_counter) && g_right_weight[thid-1] > 0)) ||
//                ((height_d == (*g_counter) && g_up_weight[thid+width1] > 0) ||
//                 (height_r == (*g_counter) && g_left_weight[thid+1] > 0) ||
//                 (height_u == (*g_counter) && g_down_weight[thid-width1] > 0)))
//        {
//            g_graph_height[thid] = (*g_counter) + 1 ;
//            g_pixel_mask[thid] = false ;
//            *g_over = true ;
//        }
//    }

    if(d_frontier[node_i] && d_visited[node_i] == *d_count){
        if(ix > 1 && d_right_weight[node_i - 1] > 0){
            if(d_visited[node_i - 1] == -1){
                d_visited[node_i - 1] = d_visited[node_i] + 1;
                d_frontier[node_i - 1] = true;
                //printf("right-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
            }

        }

        if(iy >= 1 && ix >= 1 && d_down_weight[node_i - width] > 0){
            if(d_visited[node_i - width] == -1){
                d_visited[node_i - width] = d_visited[node_i] + 1;
                d_frontier[node_i - width] = true;
                //printf("down-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
            }

        }

        if(iy < height-1 && ix >= 1 && d_up_weight[node_i + width] > 0){
            if(d_visited[node_i + width] == -1){
                d_visited[node_i + width] = d_visited[node_i] + 1;
                d_frontier[node_i + width] = true;
                //printf("up-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
            }

        }

        if(ix < width -1 && d_left_weight[node_i + 1] > 0){
            if(d_visited[node_i + 1] == -1){
                d_visited[node_i + 1] = d_visited[node_i] + 1;
                d_frontier[node_i + 1] = true;
                //printf("left-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
            }

        }



        d_frontier[node_i] = false;
        *d_finish_bfs = 1;
        d_graph_height[node_i] = d_visited[node_i];
    }
    if(ix != 0 && ix != width - 1){
    if(d_visited[node_i] == -1)
        d_graph_height[node_i] = N;
    }


}

inline __global__ void
kernel_push_nonatomic(int *d_left_weight, int *d_right_weight, int *d_down_weight, int *d_up_weight,
                      int *d_excess_flow, int *d_pull_left, int *d_pull_right, int *height_backup,
                      int *d_pull_down, int *d_pull_up, int *d_relabel_mask, int *d_graph_height,
                      int graph_size, int width, int height)
{
    int x1 = threadIdx.x ;
    int y1 = threadIdx.y ;
    int ix  = blockIdx.x * blockDim.x + threadIdx.x ;
    int iy  = blockIdx.y * blockDim.y + threadIdx.y ;
    int node_i =  iy * width + ix ;

        __shared__ int height_fn[180];

        int temp_mult = __umul24(y1+1 , 18) + x1 + 1 ;

        height_fn[temp_mult] = d_graph_height[node_i];

        (threadIdx.x == 15 && ix < width - 1) ? height_fn[temp_mult + 1] =  (d_graph_height[node_i + 1]) : 0;
        (threadIdx.x == 0  && ix > 0) ? height_fn[temp_mult - 1] = (d_graph_height[node_i - 1]) : 0;
        (threadIdx.y == 7  && iy < height - 1) ? height_fn[temp_mult + 18] = (d_graph_height[node_i + width]) : 0;
        (threadIdx.y == 0  && iy > 0) ? height_fn[temp_mult - 18] = (d_graph_height[node_i - width]) : 0;

    //    (ix >= width - 1) ? height_fn[temp_mult + 1] =  0 : 0;
    //    (ix <= 0) ? height_fn[temp_mult - 1] = N+2 : 0;
    //    printf("height_fn block(%d, %d)\n", blockIdx.x, blockIdx.y);
    //    printf("x1 = %d, y1 = %d\n", x1, y1);
    //    printf("%d\t%d\t%d\t%d\t%d\t", height_fn[temp_mult], height_fn[temp_mult-1], height_fn[temp_mult+1], height_fn[temp_mult-4], height_fn[temp_mult+4]);

        //int temp_mult2 = 340 + __umul24(y1,32) + x1 ;
        height_backup[0] = height_fn[temp_mult + 1];
        height_backup[1] = height_fn[temp_mult + 18];
        height_backup[2] = height_fn[temp_mult - 1];
        height_backup[3] = height_fn[temp_mult - 18];

        int flow_push = d_excess_flow[node_i] ;
        d_excess_flow[node_i] = 0;
        int min_flow_pushed = 0, temp_weight = 0;
        __syncthreads();
        //if(node_i%width != 0 && (node_i+1)%width != 0){
        if(ix != 0 && ix != width - 1){
        // push left
        //if(ix-1 >= 0){
            if(flow_push > 0){
                min_flow_pushed = flow_push ;
                temp_weight = d_left_weight[node_i] ;
                if(temp_weight > 0){
                    if(height_fn[temp_mult] == 1 + height_fn[temp_mult-1]){
                    (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
                    temp_weight = temp_weight - min_flow_pushed ;
                    d_left_weight[node_i] = temp_weight ;

                    d_right_weight[node_i - 1] += min_flow_pushed;
                    atomicAdd(&d_excess_flow[node_i - 1], min_flow_pushed);
                    //d_pull_left[node_i - 1] = min_flow_pushed ;
                    flow_push = flow_push - min_flow_pushed ;
                    printf("pushing node %d to the left \n", node_i);
                    }
                    else{
                        d_relabel_mask[node_i] = 1;
                        printf("in push left, node %d need relabel \n", node_i);
                    }
                }

            }
        __threadfence();
        // push to the right


        //if(ix+1 < width){
            if(flow_push > 0){
                min_flow_pushed = flow_push ;
                temp_weight = d_right_weight[node_i] ;
                if(temp_weight > 0){
                    if(height_fn[temp_mult] == 1 + height_fn[temp_mult+1]){
                    (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
                    temp_weight -= min_flow_pushed ;
                    d_right_weight[node_i] = temp_weight ;

                    d_left_weight[node_i + 1] += min_flow_pushed;
                    atomicAdd(&d_excess_flow[node_i + 1], min_flow_pushed);
                    //d_pull_right[node_i + 1] = min_flow_pushed ;
                    flow_push -= min_flow_pushed ;
                    //flow = min(d_right_weight[node_i], d_excess_flow[node_i]);
                    //d_excess_flow[node_i] -= flow;
                    //atomicSub(&d_excess_flow[node_i], flow);
                    //atomicAdd(&d_excess_flow[iy*width + ix+1], flow);
                    //atomicAdd(&d_excess_flow_backup[iy*width + ix+1], flow);
                    //d_excess_flow[iy*width + ix+1] += flow;
                    //d_right_flow[node_i] += flow;
                    //d_left_flow[iy*width + ix+1] -= flow;
                    //d_right_weight[node_i] -= flow;
                    //d_left_weight[iy*width + ix+1] += flow;
                    //printf("pushing node %d to the right \n", node_i);
                    //d_relabel_mask[node_i] = 0;
                    }
                    else{
                        d_relabel_mask[node_i] = 1;
                        printf("in push right, node %d need relabel \n", node_i);
                    }
                }

            }

        // push to the down

        if(iy+1 < height){
            if(flow_push > 0){
                min_flow_pushed = flow_push ;
                temp_weight = d_down_weight[node_i] ;
                if(temp_weight > 0){
                    if(height_fn[temp_mult] == 1 + height_fn[temp_mult+18]){
                        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
                        temp_weight -= min_flow_pushed ;
                        d_down_weight[node_i] = temp_weight ;

                        d_up_weight[node_i + width] += min_flow_pushed;
                        atomicAdd(&d_excess_flow[node_i + width], min_flow_pushed);
                        //d_pull_down[node_i + width] = min_flow_pushed ;
                        flow_push -= min_flow_pushed ;
                        //flow = min(d_down_weight[node_i], d_excess_flow[node_i]);
                        //d_excess_flow[node_i] -= flow;
                        //atomicSub(&d_excess_flow[node_i], flow);
                        //atomicAdd(&d_excess_flow[(iy+1)*width + ix], flow);
                        //atomicAdd(&d_excess_flow_backup[(iy+1)*width + ix], flow);
                        //d_excess_flow[(iy+1)*width + ix] += flow;
                        //d_down_flow[node_i] += flow;
                        //d_up_flow[(iy+1)*width + ix] -= flow;

                        //d_down_weight[node_i] -= flow;
                        //d_up_weight[(iy+1)*width + ix] += flow;
                        //printf("pushing node %d to the down\n", node_i);
                        //d_relabel_mask[node_i] = 0;
                    }
                    else{
                        d_relabel_mask[node_i] = 1;
                        printf("in push down, node %d need relabel \n", node_i);
                    }
                }
            }
        }
        //__syncthreads();


        //__syncthreads();

        //push up

        if(iy-1 >= 0){
            if(flow_push > 0){
                min_flow_pushed = flow_push ;
                temp_weight = d_up_weight[node_i] ;
                if(temp_weight > 0){
                    if(height_fn[temp_mult] == 1 + height_fn[temp_mult-18]){
                        (temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0 ;
                        temp_weight -= min_flow_pushed ;
                        d_up_weight[node_i] = temp_weight ;

                        d_down_weight[node_i - width] += min_flow_pushed;
                        atomicAdd(&d_excess_flow[node_i - width], min_flow_pushed);
                        //d_pull_up[node_i - width] = min_flow_pushed ;
                        flow_push -= min_flow_pushed ;
                        //flow = min(d_up_weight[node_i], d_excess_flow[node_i]);
                        //d_excess_flow[node_i] -= flow;
                        //atomicSub(&d_excess_flow[node_i], flow);
                        //atomicAdd(&d_excess_flow[(iy-1)*width + ix], flow);
                        //atomicAdd(&d_excess_flow_backup[(iy-1)*width + ix], flow);
                        //d_excess_flow[(iy-1)*width + ix] += flow;
                        //d_up_flow[node_i] += flow;
                        //d_down_flow[(iy-1)*width + ix] -= flow;
                        //d_up_weight[node_i] -= flow;
                        //d_down_weight[(iy-1)*width + ix] += flow;
                        //printf("pushing node %d to the up \n", node_i);
                        //d_relabel_mask[node_i] = 0;
                    }
                    else{
                        d_relabel_mask[node_i] = 1;
                        //printf("in push up, node %d need relabel \n", node_i);
                    }
                }
            }
        }
        }

        d_excess_flow[node_i] += flow_push;
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
