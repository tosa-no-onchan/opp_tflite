# opp_tflite  2.18.0  
Obstacl Path Planner TensorFlow 2.18.0 Lite C++ library  
opp with Lstm or Transformer tflite を、ROS2 C++ から predict できるようにしたライブラリー。  
  
詳しくは、各関連ページを参照してください。  
1. 最初に、TensorFlow 2.16.2 Lite C++ library と libtensorflowlite_flex.so を作る。  
[TensorFlow 2.16.2 Lite C++ library build.](https://www.netosa.com/blog/2024/12/tensorflow-2162-lite-c-library-build.html)

  ~/local/tensorflow/deploy-lite-nishi.sh  

3.  Ros2 C++ から使えるように、 Opp tflite library を作る。  
[Opp TensorFlow 2.16.2 Lite C++ library build.](https://www.netosa.com/blog/2024/12/opp-tensorflow-2162-lite-c-library-build.html)

1) libtf-lite_tools.a libtf-lite_core.a のビルド。    
$ make -fMakefile-Archive   
$ make -fMakefile-Archive install  
$ make -fMakefile-Archive clean  

2) ライブラリ libopp_tflite.a のビルド。  
$ make -fMakefile-Archive-opp_tflite  
$ make -fMakefile-Archive-opp_tflite install  
$ make -fMakefile-Archive-opp_tflite clear  

3) dummy プログラムのビルドと実行  
$ make -fMakefile-dummy  
$ export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-lite-flex:$LD_LIBRARY_PATH  
$ ./dummy  

##### 参照  
[ROS2 自作 Turtlebot3 による 草刈りロボット開発。#9 LSTM で経路計画をする。](https://www.netosa.com/blog/2024/11/ros2-turtlebot3-9-lstm.html)  
