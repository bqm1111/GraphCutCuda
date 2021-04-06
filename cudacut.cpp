#include "cudacut.h"
#include "queue"
CudaCut::CudaCut()
{

}

void CudaCut::globalRelabelCpu(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight, bool *visited, int *h_graph_height){
    queue<int> que;
    int node1 = 79;
    int s;
    for(int i = 0; i < height; i++){
        que.push(node1 + i*width);
        //visited[node1 + i*width] = true;
    }

    while(!que.empty()){
        s = que.front();

        que.pop();
        int x,y;
        x = s%width;
        y = s / width;
        if(x > 1 && h_right_weight[s-1] > 0 && visited[s-1] == false){

            visited[s-1] = true;
            que.push(s-1);
        }
        if(y < height-1 && x >=1 && h_up_weight[s+width] > 0 && visited[s+width] == false){

            visited[s+width] = true;
            que.push(s + width);
        }
        if(y >= 1 && x >= 1 && h_down_weight[s-width] > 0 && visited[s-width] == false){
            visited[s-width] = true;
            que.push(s - width);
        }
        if(x < width-1 && h_left_weight[s+1] > 0 && visited[s+1] == false){
            visited[s+1] = true;
            que.push(s + 1);
        }
    }
    for(int i = 0; i < graph_size; i++){
        if(visited[i] == false){
            //std::cout << "visited false " << i << std::endl;
            h_graph_height[i] = graph_size;
        }
    }
}

