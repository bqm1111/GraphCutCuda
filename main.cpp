#include<iostream>
#include "graph.h"
#include "cudacut.h"
#include <math.h>
using namespace cv;

int main(int argc, char **argv)
{
//    cv::Mat img0 = cv::imread("/home/nvidia/imgs/images_filter/152_0.591195_1608895250645.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img15 = cv::imread("/home/nvidia/imgs/images_filter/160_15.637790_1608895250805.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat A1 (img0, cv::Rect(35, 60, 640, 480) );
//    cv::Mat B1 (img15, cv::Rect(35, 60, 640, 480) );
//    int overlap = 4;
//    int V = HEIGHT*overlap + 2;
//    Graph g(V);


//    int xoffset = WIDTH - overlap;

//    for(int y = 0; y < HEIGHT; y++)
//    {
//        for(int x = 0; x < overlap; x++)
//        {
//            int idx = y*overlap + x + 1;
//            uchar a0 = A1.at<uchar>(y, xoffset + x);
//            uchar b0 = B1.at<uchar>(y, x);
//            uchar cap0 = abs(a0 - b0);

////            uchar a1 = img1.at<uchar>(y, xoffset + x + 1);
////            uchar b1 = img2.at<uchar>(y, x + 1);
////            uchar cap1 = abs(a1 - b1);
////            g.addEdge(idx, idx+1, (int)(cap0+cap1));

//            if(x+1 < overlap) {
//                uchar a1 = A1.at<uchar>(y, xoffset + x + 1);
//                uchar b1 = B1.at<uchar>(y, x + 1);
//                uchar cap1 = abs(a1 - b1);

//                g.addEdge(idx, idx + 1, (int)(cap0 + cap1));
//                //g.addEdge(idx+1, idx, (int)(cap0 + cap1));
//            }

//            // Add bottom edge
//            if(y+1 < HEIGHT) {
////                Vec3b a2 = A.at<Vec3b>(y+1, xoffset + x);
////                Vec3b b2 = B.at<Vec3b>(y+1, x);
//                uchar a2 = A1.at<uchar>(y+1, xoffset + x);
//                uchar b2 = B1.at<uchar>(y+1, x);

//                uchar cap2 = abs(a2 - b2);
//                g.addEdge(idx, idx + overlap, (int)(cap0 + cap2));
//                //g.addEdge(idx + OVERLAP_WIDTH, idx, (int)(cap0 + cap2));
//            }
//            //            horizontal[y * graph.width + x] = 0;
//        }
//    }
//    for(int i = 1; i <= V-overlap-1; i += overlap){
//        cout << i << endl;
//        g.addEdge(0, i, 1<<20);
//        //g.addEdge(i, 0, 1<<21);
//    }

//    for(int i = overlap; i <= V-2; i+=overlap){
//        cout << i << endl;
//        g.addEdge(i, V-1, 1<<20);
//    }

////    int V = 14;
////        Graph g(V);
////        int idx;

////        // Creating above shown flow network
//////        for(int i = 0; i < 3; i++){
//////            for(int j = 0; j < 4; j++){
//////                idx = i*4 + j + 1;
//////            }
//////        }
////        g.addEdge(1, 2, 6);
////        g.addEdge(2, 3, 3);
////        g.addEdge(3, 4, 4);
////        g.addEdge(5, 6, 7);
////        g.addEdge(6, 7, 4);
////        g.addEdge(7, 8, 5);
////        g.addEdge(9, 10, 8);
////        g.addEdge(10, 11, 2);
////        g.addEdge(11, 12, 3);

////        g.addEdge(1, 5, 4);
////        g.addEdge(5, 9, 3);
////        g.addEdge(2, 6, 4);
////        g.addEdge(6, 10, 5);
////        g.addEdge(3, 7, 3);
////        g.addEdge(7, 11, 3);
////        g.addEdge(4, 8, 4);
////        g.addEdge(8, 12, 4);
////        g.addEdge(0, 1, 1<<20);
////        g.addEdge(0, 5, 1<<20);
////        g.addEdge(0, 9, 1<<20);
////        g.addEdge(4, 13, 1<<20);
////        g.addEdge(8, 13, 1<<20);
////        g.addEdge(12, 13, 1<<20);

////    // Initialize source and sink
//        int s = 0, t = 38401;
//    //int s = 0, t = 38401;
//    auto start1 = getMoment;
//    cout << "Maximum flow CPU is " << g.getMaxFlow(s, t) << endl;
//    auto end1 = getMoment;
//    cout << "CPU time " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1000 << std::endl;



    //cout << "Maximum flow is " << g.getMaxFlow(s, t) << endl;

//    cv::Mat img30 = cv::imread("/home/nvidia/imgs/images_filter/168_30.608816_1608895250965.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img45 = cv::imread("/home/nvidia/imgs/images_filter/176_45.898608_1608895251125.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img60 = cv::imread("/home/nvidia/imgs/images_filter/185_60.669030_1608895251305.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img75 = cv::imread("/home/nvidia/imgs/images_filter/193_75.818675_1608895251465.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img90 = cv::imread("/home/nvidia/imgs/images_filter/201_90.642684_1608895251625.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img105 = cv::imread("/home/nvidia/imgs/images_filter/209_105.698897_1608895251785.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img120 = cv::imread("/home/nvidia/imgs/images_filter/217_120.694654_1608895251945.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img135 = cv::imread("/home/nvidia/imgs/images_filter/226_135.792087_1608895252125.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img150 = cv::imread("/home/nvidia/imgs/images_filter/234_150.613348_1608895252285.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img165 = cv::imread("/home/nvidia/imgs/images_filter/242_165.871538_1608895252445.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img180 = cv::imread("/home/nvidia/imgs/images_filter/250_180.747758_1608895252605.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img195 = cv::imread("/home/nvidia/imgs/images_filter/258_195.770995_1608895252765.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img210 = cv::imread("/home/nvidia/imgs/images_filter/267_210.670573_1608895252945.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img225 = cv::imread("/home/nvidia/imgs/images_filter/275_225.785868_1608895253105.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img240 = cv::imread("/home/nvidia/imgs/images_filter/283_240.732162_1608895253265.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img255 = cv::imread("/home/nvidia/imgs/images_filter/291_255.760895_1608895253425.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img270 = cv::imread("/home/nvidia/imgs/images_filter/300_270.667343_1608895253605.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img285 = cv::imread("/home/nvidia/imgs/images_filter/308_285.865078_1608895253765.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img300 = cv::imread("/home/nvidia/imgs/images_filter/316_300.768778_1608895253925.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img315 = cv::imread("/home/nvidia/imgs/images_filter/324_315.906057_1608895254085.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat black_white_origin = cv::imread("/home/vietph/workspace/GraphCutCuda/build/image_test/backward_st_black_test.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat black_white_prune = cv::imread("/home/vietph/workspace/GraphCutCuda/build/image_test/backward_st_black_average.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat subtractImage(black_white_origin.rows, black_white_origin.cols, CV_8UC1);
//    for(int i = 0; i < black_white_origin.rows; i++)
//    {
//        for(int j = 0; j < black_white_origin.cols; j++)
//        {
//            subtractImage.at<uchar>(i,j) = abs(black_white_origin.at<uchar>(i,j) - black_white_prune.at<uchar>(i,j));
//        }
//    }
//    cv::imshow("subtractImage", subtractImage);
//    cv::imwrite("image_test/subtractImage_average.png", subtractImage);
//    return 0;

//    x = 80;
//    int y = (80-x)/2;
    cv::Mat image_read1 = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
    //assert(A.data);

    cv::Mat image_read2 = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
//    cv::resize(A, A, cv::Size(640, 480));
//    cv::resize(B, B, cv::Size(640, 480));

    cv::Mat A (image_read1, cv::Rect(35, 60, 640, 480) );
    cv::Mat B (image_read2, cv::Rect(35, 60, 640, 480) );
    int blockDimy = atoi(argv[1]);
    int number_loops = atoi(argv[2]);
//    cv::Mat D (img330, cv::Rect(35, 60, 640, 480) );
//    cv::Mat E (img345, cv::Rect(35, 60, 640, 480) );
    cv::Mat m1 = A.clone();
    cv::Mat m2 = B.clone();
//    cv::Mat img_resize1, img_resize2;
//    cv::resize(D, img_resize1, cv::Size(D.cols * 0.5,D.rows * 0.5), 0, 0, CV_INTER_LINEAR);
//    cv::resize(E, img_resize2, cv::Size(E.cols * 0.5,E.rows * 0.5), 0, 0, CV_INTER_LINEAR);
//    cv::imshow("img_resize", img_resize);
//    cv::imshow("m1", m1);
//    cv::imshow("m2", m2);
//    cv::waitKey();
    CudaCut graphcut(m1.cols,m1.rows, OVERLAP_WIDTH);
    cv::Mat result(m1.rows, m1.cols*2-OVERLAP_WIDTH, CV_8UC1);
    cout << result.rows << " " << result.cols << endl;
    cv::Mat result1(m1.rows, m1.cols*2-OVERLAP_WIDTH, CV_8UC1);
    //D.copyTo(result1);
    m1.copyTo(result(cv::Rect(0,0,m1.cols, m1.rows)));
    m2.copyTo(result(cv::Rect(m1.cols-OVERLAP_WIDTH,0,m1.cols, m1.rows)));
    //cv::imshow("result_no_stitching", result);
    //cv::waitKey();
    //cv::imshow("img345", E);
    int backwardCycle = atoi(argv[6]);
    int relabelCycle = atoi(argv[7]);
    int averageDistance = atoi(argv[8]);
    int stopPoint = atoi(argv[9]);
    graphcut.cudaCutsInit();
    graphcut.cudaWarmUp();
    float sum1 = 0, sum2 = 0, sum3 = 0;
    auto start = getMoment;
    for(int i = 0; i < atoi(argv[5]); i++){
        auto start1 = getMoment;
        graphcut.cudaCutsSetupGraph(m1,m2);
        auto end1 = getMoment;
        sum1 += TimeCpu(end1, start1) / 1000.0;
        auto start2 = getMoment;
        graphcut.cudaCutsAtomic(blockDimy, number_loops, backwardCycle,
                                relabelCycle, averageDistance, stopPoint);
        auto end2 = getMoment;
        sum2 += TimeCpu(end2, start2) / 1000.0;
        std::cout << "Kernel Time = "<< std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 1000.0 << std::endl;
        auto start3 = getMoment;
        graphcut.getStitchingImage(result, result1, Colors::Gray);
        auto end3 = getMoment;
        sum3 += TimeCpu(end3, start3) / 1000.0;
        std::cout << "getStitchingImage Time = "<< std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() / 1000.0 << std::endl;
    }
    auto end = getMoment;
    cout << "\n";
    cout << "Construct Graph Time = " << sum1/atoi(argv[5]) << "\n";
    cout << "Kernel Time = " << sum2/atoi(argv[5]) << "\n";
    cout << "Stitching Time = " << sum3/atoi(argv[5]) << "\n";
    std::cout << "Total Time = "<< TimeCpu(end, start) / 1000.0/atoi(argv[5]) << std::endl;
    graphcut.cudaCutsFreeMem();
//    cv::cvtColor(result, result, CV_GRAY2BGR);
//    cv::cvtColor(result1, result1, CV_GRAY2BGR);
    cv::imshow("result", result);
//    cv::imwrite("image_result/backward_forward.png", result);
//    cv::imwrite("image_result/backward_forward_cutline.png", result1);
    cv::waitKey(0);
    return 0;
}
