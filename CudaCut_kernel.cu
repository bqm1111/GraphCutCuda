#include "cudacut.h"

// push kernel
inline __global__ void
push_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
            int *d_excess_flow, int *d_graph_height, int *d_relabel_mask, int *d_height_backup, int *d_excess_flow_backup,
            int width, int height, int N){

    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = x1 + blockIdx.x*blockDim.x;
    int iy = y1 + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    //if(ix >= width) return;


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
                atomicAdd(&d_excess_flow_backup[node_i - 1], flow);
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
                atomicAdd(&d_excess_flow_backup[node_i + 1], flow);
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
                    atomicAdd(&d_excess_flow_backup[node_i + width], flow);
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
                    atomicAdd(&d_excess_flow_backup[node_i - width], flow);
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
            atomicAdd(d_finished_count,1);
            //printf("check finish condition, node (%d %d) has excessflow > 0\n", ix, iy);
        }
    }

    __syncthreads();
}

// relabel kernel
inline __global__ void
relabel_kernel(int *d_right_weight, int *d_left_weight, int *d_up_weight, int *d_down_weight,
               int *d_graph_height, int *d_relabel_mask, int *d_height_backup,
               int *d_excess_flow, int *d_excess_flow_backup,
               int width, int height, int N){

    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int node_i = iy*width + ix;
    int mh = 1<<26;
    int *height_backup = d_height_backup + 4*node_i;
    if(d_excess_flow_backup[node_i] > 0){
        d_excess_flow[node_i] += d_excess_flow_backup[node_i];
        d_excess_flow_backup[node_i] = 0;
    }

    if(ix > 0 && ix < width - 1){
    if(height_backup[0] == N && height_backup[2] == N
            && (height_backup[1] == N || height_backup[1] == -1)
            && (height_backup[3] == N || height_backup[3] == -1)){
        d_excess_flow[node_i] = 0;
        return;
    }


//    __syncthreads();
        if(d_relabel_mask[node_i] != 0){
            if(d_right_weight[node_i] > 0 && height_backup[0] != -1){
                if(height_backup[0] < mh){
                    mh = height_backup[0];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }
            if(d_left_weight[node_i] > 0 && height_backup[2] != -1){
                if(height_backup[2] < mh){
                    mh = height_backup[2];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }
            if(d_down_weight[node_i] > 0 && height_backup[1] != -1){
                if(height_backup[1] < mh){
                    mh = height_backup[1];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }
            if(d_up_weight[node_i] > 0 && height_backup[3] != -1){
                if(height_backup[3] < mh){
                    mh = height_backup[3];
                    d_graph_height[node_i] = mh + 1;
                    //tmp = mh + 1;
                }
            }

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

        if(iy >= 1 && d_down_weight[node_i - width] > 0){
            if(d_visited[node_i - width] == -1){
                d_visited[node_i - width] = d_visited[node_i] + 1;
                d_frontier[node_i - width] = true;
                //printf("down-d_visited node[%d] %d\n",node_i, d_visited[node_i]);
            }

        }

        if(iy < height-1 && d_up_weight[node_i + width] > 0){
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
