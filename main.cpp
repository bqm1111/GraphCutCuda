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
    cv::Mat img0 = cv::imread("/home/nvidia/imgs/images_filter/152_0.591195_1608895250645.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img15 = cv::imread("/home/nvidia/imgs/images_filter/160_15.637790_1608895250805.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img30 = cv::imread("/home/nvidia/imgs/images_filter/168_30.608816_1608895250965.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img45 = cv::imread("/home/nvidia/imgs/images_filter/176_45.898608_1608895251125.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img60 = cv::imread("/home/nvidia/imgs/images_filter/185_60.669030_1608895251305.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img75 = cv::imread("/home/nvidia/imgs/images_filter/193_75.818675_1608895251465.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img90 = cv::imread("/home/nvidia/imgs/images_filter/201_90.642684_1608895251625.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img105 = cv::imread("/home/nvidia/imgs/images_filter/209_105.698897_1608895251785.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img120 = cv::imread("/home/nvidia/imgs/images_filter/217_120.694654_1608895251945.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img135 = cv::imread("/home/nvidia/imgs/images_filter/226_135.792087_1608895252125.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img150 = cv::imread("/home/nvidia/imgs/images_filter/234_150.613348_1608895252285.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img165 = cv::imread("/home/nvidia/imgs/images_filter/242_165.871538_1608895252445.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img180 = cv::imread("/home/nvidia/imgs/images_filter/250_180.747758_1608895252605.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img195 = cv::imread("/home/nvidia/imgs/images_filter/258_195.770995_1608895252765.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img210 = cv::imread("/home/nvidia/imgs/images_filter/267_210.670573_1608895252945.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img225 = cv::imread("/home/nvidia/imgs/images_filter/275_225.785868_1608895253105.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img240 = cv::imread("/home/nvidia/imgs/images_filter/283_240.732162_1608895253265.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img255 = cv::imread("/home/nvidia/imgs/images_filter/291_255.760895_1608895253425.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img270 = cv::imread("/home/nvidia/imgs/images_filter/300_270.667343_1608895253605.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img285 = cv::imread("/home/nvidia/imgs/images_filter/308_285.865078_1608895253765.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img300 = cv::imread("/home/nvidia/imgs/images_filter/316_300.768778_1608895253925.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img315 = cv::imread("/home/nvidia/imgs/images_filter/324_315.906057_1608895254085.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img330 = cv::imread("/home/nvidia/imgs/images_filter/333_330.675105_1608895254265.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img345 = cv::imread("/home/nvidia/imgs/images_filter/341_345.984133_1608895254425.png", cv::IMREAD_GRAYSCALE);

//    x = 80;
//    int y = (80-x)/2;
    cv::Mat D (img330, cv::Rect(35, 60, 640, 480) );
    cv::Mat E (img345, cv::Rect(35, 60, 640, 480) );
    cv::Mat m1 = D.clone();
    cv::Mat m2 = E.clone();
    cv::Mat img_resize1, img_resize2;
    cv::resize(D, img_resize1, cv::Size(D.cols * 0.5,D.rows * 0.5), 0, 0, CV_INTER_LINEAR);
    cv::resize(E, img_resize2, cv::Size(E.cols * 0.5,E.rows * 0.5), 0, 0, CV_INTER_LINEAR);
//    cv::imshow("img_resize", img_resize);
//    cv::imshow("m1", m1);
//    cv::imshow("m2", m2);
//    cv::waitKey();
    CudaCut graphcut(m1.rows,OVERLAP_WIDTH, m1, m2);
    cout << " stop ...." << endl;
    cv::Mat result(m1.rows, m1.cols*2-OVERLAP_WIDTH, CV_8UC1);
    cout << result.rows << " " << result.cols << endl;
    cv::Mat result1(m1.rows, m1.cols*2-OVERLAP_WIDTH, CV_8UC1);
    //D.copyTo(result1);
    m1.copyTo(result(cv::Rect(0,0,m1.cols, m1.rows)));
    m2.copyTo(result(cv::Rect(m1.cols-OVERLAP_WIDTH,0,m1.cols, m1.rows)));
    //cv::imshow("img330", D);
    //cv::imshow("img345", E);

    graphcut.cudaCutsInit();
    graphcut.cudaCutsSetupGraph();
    auto start = getMoment;
    graphcut.cudaCutsAtomic(result, result1);
    auto end = getMoment;
    std::cout << "Optimize Time = "<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << std::endl;
    graphcut.cudaCutsFreeMem();
    cout << "Maximum flow CPU is " << g.getMaxFlow(s, t) << endl;
    cv::imshow("result", result);
    cv::imshow("result1", result1);
    cv::waitKey(0);
    return 0;
}
