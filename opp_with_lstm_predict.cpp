/*
 opp_with_lstm_predict.cpp

    /home/nishi/Documents/VisualStudio-CPP/tf2_cpp/sample3/opp_with_lstm_predict.cpp

    refference
    https://stackoverflow.com/questions/61552420/tensorflow-2-0-c-load-pre-trained-model
    https://www.tensorflow.org/guide/saved_model?hl=ja#c_%E3%81%AB%E3%82%88%E3%82%8B_savedmodel_%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF
    https://gist.github.com/OneRaynyDay/c79346890dda095aecc6e9249a9ff3e1

    https://stackoverflow.com/questions/77299905/tensorflow-2-serve-prediction-issue-in-cpp

    image classify
    https://github.com/jhjin/tensorflow-cpp


    How to determine a shape of input and output nodes in graph. (Tensorflow C++)
    https://datascience.stackexchange.com/questions/60127/how-to-determine-a-shape-of-input-and-output-nodes-in-graph-tensorflow-c

    reshape
    https://docs1.w3cub.com/tensorflow~cpp/class/tensorflow/ops/reshape/


# export CPATH=/home/nishi/usr/local/include/tensorflow-2.16.2:$CPATH
# export LIBRARY_PATH=/home/nishi/usr/local/share/libtensorflow-cpu-2.16.2/lib:$LIBRARY_PATH
$ export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-lite-flex:$LD_LIBRARY_PATH


1) lib path
export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-lite:$LD_LIBRARY_PATH

2) include path
 /home/nishi/usr/local/include/tensorflow-2.18.0

1. build
   Makefile を作ったので、Makefile-Archive-opp_tflite を使います。
$ make -fMakefile-Archive-opp_tflite
2. run
$ export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-lite-flex:$LD_LIBRARY_PATH
$ ./opp_with_lstm_predict
*/

// tensorflow/cc/example/example.cc
#include <string>
#include <iostream>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"


#include "absl/memory/memory.h"
//#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
//#include "tensorflow/lite/examples/label_image/get_top_n.h"
//#include "tensorflow/lite/examples/label_image/log.h"
//#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#include "label_image.h"

#include <vector>

#include <iostream>
#include <opencv2/opencv.hpp>


using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;

class DelegateProviders {
 public:
  DelegateProviders() : delegate_list_util_(&params_) {
    delegate_list_util_.AddAllDelegateParams();
    delegate_list_util_.AppendCmdlineFlags(flags_);

    // Remove the "help" flag to avoid printing "--help=false"
    params_.RemoveParam("help");
    delegate_list_util_.RemoveCmdlineFlag(flags_, "help");
  }

  // Initialize delegate-related parameters from parsing command line arguments,
  // and remove the matching arguments from (*argc, argv). Returns true if all
  // recognized arg values are parsed correctly.
  bool InitFromCmdlineArgs(int* argc, const char** argv) {
    // Note if '--help' is in argv, the Flags::Parse return false,
    // see the return expression in Flags::Parse.
    return tflite::Flags::Parse(argc, argv, flags_);
  }

  // According to passed-in settings `s`, this function sets corresponding
  // parameters that are defined by various delegate execution providers. See
  // lite/tools/delegates/README.md for the full list of parameters defined.
  void MergeSettingsIntoParams(const tflite::label_image::Settings& s) {
    // Parse settings related to GPU delegate.
    // Note that GPU delegate does support OpenCL. 'gl_backend' was introduced
    // when the GPU delegate only supports OpenGL. Therefore, we consider
    // setting 'gl_backend' to true means using the GPU delegate.
    if (s.gl_backend) {
      if (!params_.HasParam("use_gpu")) {
        //LOG(WARN)
        std::cout << "GPU delegate execution provider isn't linked or GPU "
                     "delegate isn't supported on the platform!"<< std::endl;
      } 
      else {
        params_.Set<bool>("use_gpu", true);
        // The parameter "gpu_inference_for_sustained_speed" isn't available for
        // iOS devices.
        if (params_.HasParam("gpu_inference_for_sustained_speed")) {
          params_.Set<bool>("gpu_inference_for_sustained_speed", true);
        }
        params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
      }
    }

    // Parse settings related to NNAPI delegate.
    if (s.accel) {
      if (!params_.HasParam("use_nnapi")) {
        //LOG(WARN)
        std::cout << "NNAPI delegate execution provider isn't linked or NNAPI "
                     "delegate isn't supported on the platform!"<< std::endl;
      } 
      else {
        params_.Set<bool>("use_nnapi", true);
        params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
      }
    }

    // Parse settings related to Hexagon delegate.
    if (s.hexagon_delegate) {
      if (!params_.HasParam("use_hexagon")) {
        //LOG(WARN)
        std::cout << "Hexagon delegate execution provider isn't linked or " <<
                     "Hexagon delegate isn't supported on the platform!"<< std::endl;
      } 
      else {
        params_.Set<bool>("use_hexagon", true);
        params_.Set<bool>("hexagon_profiling", s.profiling);
      }
    }

    // Parse settings related to XNNPACK delegate.
    if (s.xnnpack_delegate) {
      if (!params_.HasParam("use_xnnpack")) {
        //LOG(WARN)
        std::cout << "XNNPACK delegate execution provider isn't linked or " <<
                     "XNNPACK delegate isn't supported on the platform!"<< std::endl;
      } 
      else {
        params_.Set<bool>("use_xnnpack", true);
        params_.Set<int32_t>("num_threads", s.number_of_threads);
      }
    }
  }

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates()
      const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

  std::string GetHelpMessage(const std::string& cmdline) const {
    return tflite::Flags::Usage(cmdline, flags_);
  }

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tflite::tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;

  // Contains valid flags
  std::vector<tflite::Flag> flags_;
};


void print_info(const cv::Mat& mat)
{
	//using namespace std;
	// 要素の型とチャンネル数の組み合わせ。
	// 紙面の都合により、サンプルで使用する値のみ記述
	std::cout << "type: " << (
		mat.type() == CV_8UC3 ? "CV_8UC3" :
		mat.type() == CV_16SC1 ? "CV_16SC1" :
		mat.type() == CV_64FC2 ? "CV_64FC2" :
		"other"
		) << std::endl;
	std::cout << "type: " << mat.type() << std::endl;
	// 要素の型
	std::cout << "depth: " << (
		mat.depth() == CV_8U ? "CV_8U" :
		mat.depth() == CV_16S ? "CV_16S" :
		mat.depth() == CV_64F ? "CV_64F" :
		"other"
		) << std::endl;
	// チャンネル数
	std::cout << "channels: " << mat.channels() << std::endl;
	// バイト列が連続しているか
	std::cout << "continuous: " <<
		(mat.isContinuous() ? "true" : "false")<< std::endl;
}

/*
* SoftMaxIdx()
* (1,500,53) の3次元配列の中で、axis=1 の行の中の一番大きい col idx を求める。
*  ただし、 各行の一番最後の col は、除外する。
*/
void SoftMaxIdx(float* &outputs,std::vector<int> &soft_max_idx,int ax1_max=300,int ax2_max=53){
  //auto items_x = it->shaped<float, 2>({ax1_max, ax2_max});
  //std::cout << " items_x:" << items_x <<std::endl;
  //std::cout << " items_x(0,0):" << items_x(0,0)<<std::endl;
  for(int i300=0; i300 < ax1_max; i300++){
      int idx=ax2_max-1;
      float val_f=0.0;
      // 1番最後は、対象から外す。
      for(int i53=0;i53 < ax2_max-1 ; i53++){
          if(outputs[i300*ax2_max+i53] > val_f){
              idx=i53;
              val_f=outputs[i300*ax2_max+i53];
          }
      }
      // 一番大きい idx を求める。
      soft_max_idx.push_back(idx);
  }
}


int main(int argc, char* argv[]) {
  //using namespace tensorflow;
  //using namespace tensorflow::ops;

  tflite::label_image::Settings s;
  tflite::label_image::Settings *settings = &s;

  DelegateProviders delegate_providers;

  //#define USE_TRANSFORMER
  #if defined(USE_TRANSFORMER)
    std::string model_dir="/home/nishi/Documents/VisualStudio-TF/opp_with_transformer_mltu";
  #else
    std::string model_dir="/home/nishi/Documents/VisualStudio-TF/opp_with_lstm";
  #endif

  std::string model_file=model_dir+"/a.model.tflite";

  //std::string image = "/home/nishi/colcon_ws/src/turtlebot3_navi_my/ml_data/image/1.jpg";
  std::string image = "/home/nishi/colcon_ws/src/turtlebot3_navi_my/ml_data/image/2.jpg";

  delegate_providers.MergeSettingsIntoParams(s);

  // Load model.
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  if (!model) {
      std::cerr << "FlatBufferModel::BuildFromFile(\"" << model_file.c_str() << "\") failed." << std::endl;
      return -1;
  }

  settings->model = model.get();
  std::cout << "Loaded model " << settings->model_name<< std::endl;
  model->error_reporter();
  std::cout << "resolved reporter"<< std::endl;

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  if (!interpreter) {
      std::cout << ">>> Failed to construct interpreter"<< std::endl;
      return -1;
  }

  if (settings->number_of_threads != -1) {
    interpreter->SetNumThreads(settings->number_of_threads);
  }

  auto profiler = std::make_unique<tflite::profiling::Profiler>(
      settings->max_profiling_buffer_entries);
  interpreter->SetProfiler(profiler.get());

  std::cout << ">>> delegates"<< std::endl;
  auto delegates = delegate_providers.CreateAllDelegates();

  for (auto& delegate : delegates) {
    const auto delegate_name = delegate.provider->GetName();
    if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
      //LOG(ERROR) 
      std::cout << "Failed to apply " << delegate_name << " delegate."<< std::endl;
      exit(-1);
    } 
    else {
      //LOG(INFO)
      std::cout << "Applied " << delegate_name << " delegate."<< std::endl;
    }
  }

  std::cout << ">>> allocate interpreter tensors"<< std::endl;
  if (interpreter->AllocateTensors() != kTfLiteOk){
      std::cerr << ">>> Cannot allocate interpreter tensors" << std::endl;
      return 1;
  }

  int input = interpreter->inputs()[0];
  std::cout << "input: " << input << std::endl;

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  std::cout << "number of inputs: " << inputs.size() << std::endl;
  std::cout << "number of outputs: " << outputs.size()<< std::endl;

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  std::cout << "wanted_height: " << wanted_height << std::endl;
  std::cout << "wanted_width: " << wanted_width << std::endl;
  std::cout << "wanted_channels: " << wanted_channels << std::endl;

  auto input_type=interpreter->tensor(input)->type;

  // 転置した値か?
  int32_t input_width = 122;
  int32_t input_height = 600;

  //std::string input_layer = "x:0";
  //std::string output_layer = "Identity:0";

  std::cout << " passed:#2" << std::endl;
  // テスト画像を読み込む。
  cv::Mat img;
  img = cv::imread(image,0);      // read gray jpeg
  // error handling
  if(img.data == NULL) {
      std::cerr << "Error: not found such file. " << image<< std::endl;
      exit(1);
  }
  cv::Mat img3=img.clone();

  // img : (h:122,w:?)
  int width = img.cols;
  int height = img.rows;
  int depth = img.channels();
  std::cout << " width:" << width << " height:"<< height << " depth:"<< depth <<std::endl;

  // 画像の拡大、縮小をしない方法で、 サイズ(h:122,w:600) にする。
  // 余白: 0
  // https://catalina1344.hatenablog.jp/entry/2014/04/05/103947
  cv::Mat img2(cv::Size(600,122), CV_8UC1, cv::Scalar(0));    // cv::Size(w,h) なので、注意!!
  cv::Mat dar16_9_roi(img2, cv::Rect(0, 0, img.cols,img.rows));
  img.copyTo(dar16_9_roi);

  // 転置する。(h:122,w:600) -> (h:600,w:122)  
  // model には、画像の上から下へ、122 px 毎に読ませる。   600 line
  // 注) 上下が逆になった気がする。  -> 後で、補正する。
  cv::transpose(img2,img2);

  //print_info(img2);
  cv::imshow("img", img);
  cv::waitKey(100);

  //cv::imshow("img2", img2);
  //cv::waitKey(1000);
  //cv::waitKey(0);

  // uint8 -> float32 に変換
  cv::Mat img2_f;
  img2.convertTo(img2_f, CV_32F);

  // ノーマライズする。
  img2_f = img2_f/255.0;

  switch (input_type) {
      case kTfLiteFloat32:
          std::cout << "input_type: kTfLiteFloat32 "<< std::endl;
          //resize<float>(interpreter->typed_tensor<float>(input), in.data(),
          //                image_height, image_width, image_channels, wanted_height,
          //                wanted_width, wanted_channels, settings);
      break;
      case kTfLiteInt8:
          std::cout << "input_type: kTfLiteInt8 "<< std::endl;
          //resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
          //                image_height, image_width, image_channels, wanted_height,
          //                wanted_width, wanted_channels, settings);
          exit(-1);
          break;
      case kTfLiteUInt8:
          std::cout << "input_type: kTfLiteUInt8 "<< std::endl;
          //resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
          //                image_height, image_width, image_channels, wanted_height,
          //                wanted_width, wanted_channels, settings);
          exit(-1);
          break;
      default:
          std::cout << "cannot handle input type "
                      << interpreter->tensor(input)->type << " yet"<< std::endl;
          exit(-1);
  }

  auto ptr = interpreter->typed_input_tensor<float>(0);
  if(ptr == 0){
      std::cerr << "error interpreter->typed_input_tensor <float>(0" << std::endl;
      exit(-1);
  }

  //std::cout << "ptr:" << ptr << std::endl;
  //exit(0);

  cv::Mat tensor_image(600, 122, CV_32FC1, ptr);      // Tensor (batch:1,low:600,col:122) が入力形式
  img2_f.convertTo(tensor_image, CV_32FC1);

  // Run inference.
  if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Cannot invoke interpreter" << std::endl;
      exit(-1);
  }
  std::cout << "invoke end!!" << std::endl;

  // Get Output
  int output = interpreter->outputs()[0];
  //std::cout << "output:"<< output << std::endl;

  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;

  #if defined(USE_TRANSFORMER)
    auto output_size = output_dims->data[output_dims->size - 1];
    std::cout << "output_size:"<< output_size << std::endl;
  #else
    auto output_x_size = output_dims->data[output_dims->size - 2];
    auto output_y_size = output_dims->data[output_dims->size - 1];
    std::cout << "output_x_size:"<< output_x_size << std::endl;
    std::cout << "output_y_size:"<< output_y_size << std::endl;
  #endif

  // check out put dimension
  for (int i=0;i<output_dims->size;i++ ){
     std::cout <<"output_dims->data[" << i<<"]:" << output_dims->data[i] << std::endl;
  }
  // lstm [1,300,53]   float
  // transformer [1,150]  int_32

  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      std::cout << "output type: kTfLiteFloat32"<< std::endl;
      //get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
      //                 settings->number_of_results, threshold, &top_results,
      //                 settings->input_type);
      break;
    case kTfLiteInt32:
       std::cout << "output type: kTfLiteInt32"<< std::endl;
      break;
    case kTfLiteInt8:
      std::cout << "output type: kTfLiteInt8"<< std::endl;
      //get_top_n<int8_t>(interpreter->typed_output_tensor<int8_t>(0),
      //                  output_size, settings->number_of_results, threshold,
      //                  &top_results, settings->input_type);
      break;
    case kTfLiteUInt8:
      std::cout << "output type: kTfLiteUInt8"<< std::endl;
      //get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
      //                   output_size, settings->number_of_results, threshold,
      //                   &top_results, settings->input_type);
      exit(-1);
      break;
    default:
      //LOG(ERROR)
      std::cout << "cannot handle output type "
                 << interpreter->tensor(output)->type << " yet"<< std::endl;
      exit(-1);
  }

  #if defined(USE_TRANSFORMER)
    // Transformer model
    std::string prediction = "";
    auto out_ptr = interpreter->typed_output_tensor<int32_t>(0);

    std::cout << "out_ptr:"<< out_ptr << std::endl;
    if (out_ptr !=0){
      int x=4;
      for(int i=1;i < output_size;i++){
        int idx = out_ptr[i];
        if(idx < 49){
          std::cout << std::to_string(idx) << ",";
          int y = 12 + idx*2;
          cv::circle(img3, cv::Point(x,y), 2, cv::Scalar(128),-1);
        }
        x+=4;
      }
      std::cout << std::endl;
    }

  #else
    // LSTM model
    std::string prediction = "";
    auto out_ptr = interpreter->typed_output_tensor<float>(0);
    std::cout << "out_ptr:"<< out_ptr << std::endl;

    if(out_ptr !=0){
      std::vector<int> soft_max_idx;
      SoftMaxIdx(out_ptr,soft_max_idx);

      // 結果を、プロットする。
      int x=2;
      for(int k=0;k<300;k++){
          //std::cout << soft_max_idx[k] <<std::endl;
          int idx= soft_max_idx[k];
          if (x < width && idx < 49){
            std::cout << std::to_string(idx) << ",";
            int y = 12 + idx*2;
            cv::circle(img3, cv::Point(x,y), 2, cv::Scalar(128),-1);
          }
          x +=2;
      }
      std::cout << std::endl;
    }

  #endif

  cv::imshow("img3", img3);
  cv::waitKey(0);

  std::cout << " prog end2" << std::endl;
  return 0;
}

