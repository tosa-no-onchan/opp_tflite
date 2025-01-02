/*
/home/nishi/Documents/VisualStudio-CPP/tflite/opp_tflite/dummy.cpp

1) build
$ make -fMakefile-dummy

2) run 
$ export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-2.16.2-lite-flex:$LD_LIBRARY_PATH
$ ./dummy
*/
#include <iostream>
//#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opp_tflite/opp_tflite.hpp"

void one_shot(opp_tflite::Opp_Tflite &opp_tfl,std::string image,int x_interval,int x_max){
    // テスト画像を読み込む。
    cv::Mat img;
    img = cv::imread(image,0);      // read gray jpeg
    // error handling
    if(img.data == NULL) {
        std::cerr << "Error: not found such file. " << image<< std::endl;
        exit(1);
    }
    cv::Mat img3=img.clone();
    int width = img.cols;

    cv::imshow("img", img);
    cv::waitKey(100);

    std::vector<u_int8_t> pred_y;
    opp_tfl.predict(&img, pred_y);

    // 結果を、プロットする。
    int x=x_interval;
    if(pred_y.size() > 0){
        for(int k=0;k<x_max;k++){
            //std::cout << soft_max_idx[k] <<std::endl;
            int idx= pred_y[k];
            if (x < width && idx < 49){
                std::cout << std::to_string(idx) << ",";
                int y = 12 + idx*2;
                cv::circle(img3, cv::Point(x,y), 2, cv::Scalar(128),-1);
            }
            x +=x_interval;
        }
        std::cout << std::endl;
    }
    else{
        std::cout << "predict nothing !!" << std::endl;
    }
    cv::imshow("img3", img3);
    //cv::waitKey(100);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    std::vector<std::string> images ={
        "/home/nishi/colcon_ws/src/turtlebot3_navi_my/ml_data/image/1.jpg",
        "/home/nishi/colcon_ws/src/turtlebot3_navi_my/ml_data/image/2.jpg",
        "/home/nishi/colcon_ws/src/turtlebot3_navi_my/ml_data/image/3.jpg",
        "/home/nishi/colcon_ws/src/turtlebot3_navi_my/ml_data/image/4.jpg",
    };

    opp_tflite::Settings s;
    int x_interval;
    int x_max;

    #define USE_LSTM
    #if defined(USE_LSTM)
        s.model_path = "/home/nishi/Documents/VisualStudio-TF/opp_with_lstm/a.model.tflite";
        x_interval=2;
        x_max=300;
    #else
        s.model_path ="/home/nishi/Documents/VisualStudio-TF/opp_with_transformer_mltu/a.model.tflite";
        x_interval=4;
        x_max=150;
    #endif

    opp_tflite::Opp_Tflite opp_tfl;
    opp_tfl.init(&s);
    for(std::string image : images){
        one_shot(opp_tfl, image,x_interval,x_max);
    }
    std::cout << " prog end" << std::endl;
    return 0;

}
