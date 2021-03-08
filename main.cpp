#include<iostream>
#include "graph.h"
#include "cudacut.h"


int main()
{
    int V = 16;
    Graph g(V);

//    // Creating above shown flow network
//    g.addEdge(0, 1, 3);
//    g.addEdge(0, 4, 3);
//    g.addEdge(1, 2, 3);
//    g.addEdge(1,0,1);
//    g.addEdge(1,5,3);
//    g.addEdge(2, 1, 1);
//    g.addEdge(2, 3, 3);
//    g.addEdge(2, 6, 3);
//    g.addEdge(4, 3, 7);
//    g.addEdge(4, 5, 4);

//    // Initialize source and sink
    int s = 0, t = 15;
    int x = 20;
    int idx;
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            idx = i*4 + j;
            if(idx == 0){
                g.addEdge(0,1,x);
                g.addEdge(0,4,3);
            }
            else{
                if(j+1 < 4)
                    g.addEdge(idx, i*4 + j+1, 3);
                if(i+1 < 4)
                    g.addEdge(idx, (i+1)*4 + j, 3);
                if(j-1 >= 0)
                    g.addEdge(idx, i*4 + j-1, 1);
                if(i-1 >= 0)
                    g.addEdge(idx, (i-1)*4 + j, 1);

            }
        }
    }


    //cout << "Maximum flow is " << g.getMaxFlow(s, t) << endl;
    cv::Mat img = cv::imread("/home/nvidia/imgs/images_filter/964_60.619567_1608895267105.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread("/home/nvidia/imgs/images_filter/972_75.791195_1608895267265.png", cv::IMREAD_GRAYSCALE);


    cv::Mat D (img, cv::Rect(35, 60, 640, 480) );
    cv::Mat E (img1, cv::Rect(35, 60, 640, 480) );
    cv::Mat m1 = D.clone();
    cv::Mat m2 = E.clone();
    CudaCut graphcut(480,80, m1, m2);
    graphcut.cudaCutsInit();
    //graphcut.cudaCutsSetupGraph();
    auto start = getMoment;
    graphcut.cudaCutsAtomicOptimize();
    auto end = getMoment;
    std::cout << "Optimize Time = "<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << std::endl;
    graphcut.cudaCutsFreeMem();
    cout << "Maximum flow CPU is " << g.getMaxFlow(s, t) << endl;
    return 0;
}
