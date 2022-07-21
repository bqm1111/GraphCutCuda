#include "cudacut.h"
#include "queue"
CudaCut::CudaCut()
{

}

void CudaCut::graphCorrectionImage(unsigned char* data, int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight)
{
    for(int x = 0; x < OVERLAP_WIDTH; x++)
    {
        for(int y = 0; y < HEIGHT; y++)
        {
            data[y*OVERLAP_WIDTH + x] = 255;
        }
    }
    for(int x = 0; x < OVERLAP_WIDTH; x++)
    {
        for(int y = 0; y < HEIGHT; y++)
        {
            if(h_left_weight[y*OVERLAP_WIDTH + x] == 0 && x >= 1)
            {
                data[y*OVERLAP_WIDTH + x-1] = 0;
            }
            if(h_up_weight[y*OVERLAP_WIDTH + x] == 0 && y >=1)
            {
                data[(y-1)*OVERLAP_WIDTH + x] = 0;
            }
        }
    }
}

void CudaCut::globalRelabelCpu(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight,
                               bool *visited, int *h_graph_height, int *h_bfs_counter, int *d_excess_flow){
    queue<int> que;
    int node1 = width - 1;
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
            h_bfs_counter[s-1] = h_bfs_counter[s] + 1;
            h_graph_height[s-1] = h_bfs_counter[s-1];
        }
        if(y < height-1 && x >=1 && h_up_weight[s+width] > 0 && visited[s+width] == false){

            visited[s+width] = true;
            que.push(s + width);
            h_bfs_counter[s+width] = h_bfs_counter[s] + 1;
            h_graph_height[s+width] = h_bfs_counter[s+width];
        }
        if(y >= 1 && x >= 1 && h_down_weight[s-width] > 0 && visited[s-width] == false){
            visited[s-width] = true;
            que.push(s - width);
            h_bfs_counter[s-width] = h_bfs_counter[s] + 1;
            h_graph_height[s-width] = h_bfs_counter[s-width];
        }
        if(x < width-1 && h_left_weight[s+1] > 0 && visited[s+1] == false){
            visited[s+1] = true;
            que.push(s + 1);
            h_bfs_counter[s+1] = h_bfs_counter[s] + 1;
            h_graph_height[s+1] = h_bfs_counter[s+1];
        }
    }
    for(int i = 0; i < graph_size; i++){
        if(visited[i] == false){
            h_graph_height[i] = graph_size;
            d_excess_flow[i] = 0;
        }
    }
}

void CudaCut::forwardBfsCpu(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight,
                               bool *visited, int *h_graph_height, int *h_bfs_counter, int *d_excess_flow){
    queue<int> que;
    int node1 = 40;
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
        if(x > 15 && h_right_weight[s] > 0 && visited[s+1] == false){

            visited[s+1] = true;
            que.push(s+1);
        }
        if(y > 0 && h_up_weight[s] > 0 && visited[s-width] == false && x > 15){

            visited[s-width] = true;
            que.push(s - width);
        }
        if(y < height-1 && h_down_weight[s] > 0 && visited[s+width] == false && x > 15){
            visited[s+width] = true;
            que.push(s + width);
        }
        if(x < width-1 && h_left_weight[s] > 0 && visited[s-1] == false && x > 15){
            visited[s-1] = true;
            que.push(s - 1);
        }
    }
    for(int i = 0; i < graph_size; i++){
        if(visited[i] == false && d_excess_flow[i] > 0){
//            printf("%d ", i);
//            d_excess_flow[i] = 0;
        }
    }
}

int CudaCut::BfsCpuBackward(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight, bool *visited){
    queue<int> que;
    int node1 = width - 1;
    int s;
    int count = 0;
    for(int i = 0; i < height; i++){
        que.push(node1 + i*width);
        count++;
    }

    while(!que.empty()){
        s = que.front();

        que.pop();
        int x,y;
        x = s%width;
        y = s / width;
        if(x > 1 && h_right_weight[s-1] > 0 && visited[s-1] == true){
            visited[s-1] = false;
            que.push(s-1);
            count++;
        }
        if(y < height-1 && x >=1 && h_up_weight[s+width] > 0 && visited[s+width] == true){
            visited[s+width] = false;
            que.push(s + width);
            count++;
        }
        if(y >= 1 && x >= 1 && h_down_weight[s-width] > 0 && visited[s-width] == true){
            visited[s-width] = false;
            que.push(s - width);
            count++;
        }
        if(x < width-1 && h_left_weight[s+1] > 0 && visited[s+1] == true){
            visited[s+1] = false;
            que.push(s + 1);
            count++;
        }
    }
    return width*height-count;
    //return 0;
}
int CudaCut::BfsCpuForward(int *h_right_weight, int *h_left_weight, int *h_down_weight, int *h_up_weight, bool *visited, int *d_excess_flow){
    queue<int> que;
    int node1 = 1;
    int s;
    int count = 0;
    for(int i = 0; i < height; i++){
        que.push(node1 + i*width);
        count += 2;
    }

    while(!que.empty()){
        s = que.front();

        que.pop();
        int x,y;
        x = s%width;
        y = s / width;
        if(x < width-1 && h_right_weight[s] > 0 && visited[s+1] == false){
            visited[s+1] = true;
            que.push(s+1);
            count++;
        }
        if(y < height-1 && h_down_weight[s] > 0 && visited[s+width] == false){
            visited[s+width] = true;
            que.push(s+width);
            count++;
        }
        if(y > 0 && h_up_weight[s] > 0 && visited[s-width] == false){
            visited[s-width] = true;
            que.push(s-width);
            count++;
        }
        if(x > 1 && h_left_weight[s] > 0 && visited[s-1] == false){
            visited[s-1] = true;
            que.push(s-1);
            count++;
        }

    }
    for(int i = 0; i < graph_size; i++){
        if(visited[i] == false){
            d_excess_flow[i] = 0;
        }
    }
    return count;
}

void CudaCut::getStitchingImage(cv::Mat& result, cv::Mat& result1, Colors color){
    for(int i = 0; i < graph_size; i++){
        if((i+1)%width != 0)
            h_visited_backward[i] = true;
        else
            h_visited_backward[i] = false;
    }
    //auto start1 = getMoment;

    auto start = getMoment;
    int count1 = BfsCpuBackward(d_right_weight, d_left_weight, d_down_weight, d_up_weight, h_visited_backward);
    auto end = getMoment;
    std::cout << "Backward Time = "<< TimeCpu(end, start) / 1000.0 << "\n";

//    for(int i = 0; i < graph_size; i++){
//        if((i-1)%width != 0 && i%width != 0)
//            h_visited_forward[i] = false;
//        else
//            h_visited_forward[i] = true;
//    }
//    auto start1 = getMoment;
//    int count2 = BfsCpuForward(d_right_weight, d_left_weight, d_down_weight, d_up_weight,h_visited_forward);
//    auto end1 = getMoment;
//    vec3.push_back(int(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() ));
    //cout << "count2 = " << count2 << endl;
//    cout << "Forward Time " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1000 << std::endl;
//    if(count1 > graph_size/2){
//        //cout << "count1 > graph_size/2" << "\n";
//        count1 = graph_size-count1;
//    }
//    if(count2 > graph_size/2){
//        //cout << "count2 > graph_size/2" << "\n";
//        count2 = graph_size-count2;
//    }
//    if(count1 > count2){
//        //cout << "count1 = " << count1 << "\n";
//      selectPix(result, result1,h_visited_backward);
//    }
//    else{
        //cout << "count2 = " << count2 << "\n";
        selectPix(result, result1, h_visited_backward, color);
    //}

//    int count = 0;
//    //vector<int> left_point_x, left_point_y, right_point_x, right_point_y;
//    cv::Mat tmp(result, cv::Rect(img1.cols-OVERLAP_WIDTH,0, OVERLAP_WIDTH, img1.rows));

////    cv::Mat tmpA(img1, cv::Rect(img1.cols-OVERLAP_WIDTH,0, OVERLAP_WIDTH, img1.rows));
////    cv::Mat tmpB(img2, cv::Rect(0,0, OVERLAP_WIDTH, img1.rows));
////    for(int i = 0; i < height; i++){
////        tmpA.at<uchar>(i,0) = 255;
////        tmpA.at<uchar>(i,1) = 255;
////        tmpA.at<uchar>(i,2) = 255;
////        tmpA.at<uchar>(i,77) = 255;
////        tmpA.at<uchar>(i,78) = 255;
////        tmpA.at<uchar>(i,79) = 255;

////        tmpB.at<uchar>(i,0) = 255;
////        tmpB.at<uchar>(i,1) = 255;
////        tmpB.at<uchar>(i,2) = 255;
////        tmpB.at<uchar>(i,77) = 255;
////        tmpB.at<uchar>(i,78) = 255;
////        tmpB.at<uchar>(i,79) = 255;
////    }
////    cv::imshow("img1", img1);
////    cv::imshow("img2", img2);
////    cv::waitKey();

//    for(int i = 0; i < height; i++){
//        for(int j = 0; j < width; j++){
//            if(h_visited_backward[i*width+j] == true){
////                if(i < 60){
////                left_point_x.push_back(j);
////                left_point_y.push_back(i*(-10));
////                }
//                tmp.at<uchar>(i,j) = h_m1[i*width+j];
//                //tmpB.at<uchar>(i,j) = 0;
//                count++;
//            }
//            else{
////                if(i < 60){
////                right_point_x.push_back(j);
////                right_point_y.push_back(i*(-10));
////                }
//                tmp.at<uchar>(i,j) = h_m2[i*width+j];
//                //tmpA.at<uchar>(i,j) = 0;
//            }

//        }
//    }
//    cout << "count = " << count << endl;
//    result.copyTo(result1);
//    //tmp.copyTo(tmpA);
//    //tmp.copyTo(tmpB);
//    cv::Mat tmp1(result1, cv::Rect(img1.cols-OVERLAP_WIDTH,0, OVERLAP_WIDTH, img1.rows));
//    for(int i = 0; i < height; i++){
//        for(int j = 0; j < width; j++){
//            if(j > 0 && j< width-1 && h_visited_backward[i*width+j] != h_visited_backward[i*width+j+1]){
//                tmp1.at<uchar>(i,j) = 255;
//                tmp1.at<uchar>(i,j+1) = 255;
//                tmp1.at<uchar>(i,j-1) = 255;
////                tmpA.at<uchar>(i,j) = 255;
////                tmpA.at<uchar>(i,j+1) = 255;
////                tmpA.at<uchar>(i,j-1) = 255;
////                tmpB.at<uchar>(i,j) = 255;
////                tmpB.at<uchar>(i,j+1) = 255;
////                tmpB.at<uchar>(i,j-1) = 255;

//            }
//            if(i > 0 && i< height-1 && h_visited_backward[i*width+j] != h_visited_backward[(i+1)*width+j]){
//                tmp1.at<uchar>(i,j) = 255;
//                tmp1.at<uchar>(i+1,j) = 255;
//                tmp1.at<uchar>(i-1,j) = 255;
////                tmpA.at<uchar>(i,j) = 255;
////                tmpA.at<uchar>(i+1,j) = 255;
////                tmpA.at<uchar>(i-1,j) = 255;
////                tmpB.at<uchar>(i,j) = 255;
////                tmpB.at<uchar>(i+1,j) = 255;
////                tmpB.at<uchar>(i-1,j) = 255;
//            }

//        }
//    }
//    for(int i = 0; i < height; i++){
//        tmp1.at<uchar>(i,0) = 255;
//        tmp1.at<uchar>(i,1) = 255;
//        tmp1.at<uchar>(i,2) = 255;

//        tmp1.at<uchar>(i,77) = 255;
//        tmp1.at<uchar>(i,78) = 255;
//        tmp1.at<uchar>(i,79) = 255;
//    }
//    for(int i = 0; i < height; i++){
//        tmpA.at<uchar>(i,0) = 255;
//        tmpA.at<uchar>(i,1) = 255;
//        tmpA.at<uchar>(i,78) = 255;
//        tmpA.at<uchar>(i,79) = 255;
//        tmpB.at<uchar>(i,0) = 255;
//        tmpB.at<uchar>(i,1) = 255;
//        tmpB.at<uchar>(i,78) = 255;
//        tmpB.at<uchar>(i,79) = 255;
//    }
//    for(int i = 0; i < OVERLAP_WIDTH; i++){
//        tmpA.at<uchar>(0,i) = 255;
//        tmpA.at<uchar>(1,i) = 255;
//        tmpA.at<uchar>(479,i) = 255;
//        tmpA.at<uchar>(478,i) = 255;
//        tmpB.at<uchar>(0,i) = 255;
//        tmpB.at<uchar>(1,i) = 255;
//        tmpB.at<uchar>(479,i) = 255;
//        tmpB.at<uchar>(478,i) = 255;
//    }
//    std::cout << "count " << count << endl;
//    int *left_point_x_arr, *left_point_y_arr, *right_point_x_arr, *right_point_y_arr;
//    left_point_x_arr = left_point_x.data();
//    left_point_y_arr = left_point_y.data();
//    right_point_x_arr = right_point_x.data();
//    right_point_y_arr = right_point_y.data();

//    writeToFile("../variable/left_point_x_arr.txt", left_point_x_arr, left_point_x.size(), 1);
//    writeToFile("../variable/left_point_y_arr.txt", left_point_y_arr, left_point_y.size(), 1);
//    writeToFile("../variable/right_point_x_arr.txt", right_point_x_arr, right_point_x.size(), 1);
//    writeToFile("../variable/right_point_y_arr.txt", right_point_y_arr, right_point_y.size(), 1);

    //gpuErrChk(cudaFree(d_finish_bfs));
//    cv::imshow("tmp1", result1);
//    cv::imshow("img1", img1);
//    cv::imshow("img2", img2);
//    cv::waitKey();

}

void CudaCut::selectPix(cv::Mat& result, cv::Mat& result1, bool *visited, Colors color){
    cv::Mat tmp(result, cv::Rect(image_width-width,0, width, height));
    cv::Mat black;
    if(color == Colors::Gray)
    {
        black = cv::Mat(tmp.rows, tmp.cols, CV_8UC1);
    }
    else if(color == Colors::RGB)
    {
        black = cv::Mat(tmp.rows, tmp.cols, CV_8UC3);
    }

    // correct pixels
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(visited[i*width+j] == true){
                tmp.at<uchar>(i,j) = h_m1[i*width+j];
                if(color == Colors::Gray)
                    black.at<uchar>(i,j) = 0;
                else if(color == Colors::RGB)
                    black.at<Vec3b>(i, j) = cv::Vec3b(237,83,17);//17, 83, 237
            }
            else{
                tmp.at<uchar>(i,j) = h_m2[i*width+j];
                if(color == Colors::Gray)
                    black.at<uchar>(i,j) = 255;
                else if(color == Colors::RGB)
                    black.at<Vec3b>(i, j) = cv::Vec3b(70, 67, 240); //240, 67, 70
            }

        }
    }
    cv::imwrite("image_test/backward_st_black.png", black);
//    if(color == Colors::Gray)
//    {
//        result.copyTo(result1);
//        cv::Mat tmp1(result1, cv::Rect(image_width-width,0, width, height));
//        for(int i = 0; i < height; i++){
//            for(int j = 0; j < width; j++){
////                if(j == 0 || j == width-1)
////                {
////                    tmp1.at<uchar>(i,j) = 255;
////                }
//                if(j > 0 && j< width-1 && visited[i*width+j] != visited[i*width+j+1]){
//                    tmp1.at<uchar>(i,j) = 255;
//                    tmp1.at<uchar>(i,j+1) = 255;
//                    tmp1.at<uchar>(i,j-1) = 255;

//                }
//                if(i > 0 && i< height-1 && visited[i*width+j] != visited[(i+1)*width+j]){
//                    tmp1.at<uchar>(i,j) = 255;
//                    tmp1.at<uchar>(i+1,j) = 255;
//                    tmp1.at<uchar>(i-1,j) = 255;
//                }

//            }
//        }
//    }
//    else if(color == Colors::RGB)
//    {
//        result.copyTo(result1);
//        cv::Mat tmp1(result1, cv::Rect(image_width-width,0, width, height));
//        cv::Mat colorTmp;
//        cv::cvtColor(tmp1, colorTmp, cv::COLOR_GRAY2BGR);
//        for(int i = 0; i < height; i++){
//            for(int j = 0; j < width; j++){
//                if(j > 0 && j< width-1 && visited[i*width+j] != visited[i*width+j+1]){
//                    colorTmp.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 255);
//                    colorTmp.at<cv::Vec3b>(i,j + 1) = cv::Vec3b(0, 0, 255);
//                    colorTmp.at<cv::Vec3b>(i,j - 1) = cv::Vec3b(0, 0, 255);

//                }
//                if(i > 0 && i< height-1 && visited[i*width+j] != visited[(i+1)*width+j]){
//                    colorTmp.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 255);
//                    colorTmp.at<cv::Vec3b>(i + 1,j) = cv::Vec3b(0, 0, 255);
//                    colorTmp.at<cv::Vec3b>(i - 1,j) = cv::Vec3b(0, 0, 255);
//                }

//            }
//        }
//        cv::cvtColor(result1, result1, cv::COLOR_GRAY2BGR);
//        colorTmp.copyTo(result1(cv::Rect(image_width-width,0, width, height)));
//        cv::imshow("rgbImage", result1);
//        cv::waitKey();
//    }

}
