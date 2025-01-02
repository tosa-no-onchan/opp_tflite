/*
 opp_tflite.cpp
*/

#include "opp_tflite/opp_tflite.hpp"

namespace opp_tflite{

/*
* SoftMaxIdx()
* (1,500,53) の3次元配列の中で、axis=1 の行の中の一番大きい col idx を求める。
*  ただし、 各行の一番最後の col は、除外する。
*/
template<typename T>
void SoftMaxIdx(float* &outputs,T &soft_max_idx,int ax1_max=300,int ax2_max=53){
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

/*
* void Opp_Tflite::init(Settings *settings)
*/
void Opp_Tflite::init(Settings *settings){
  std::cout << "Opp_Tflite::init()" << std::endl;
  is_act_=false;
  settings_= settings;

  //model_dir_ = settings_->model_path;

  //model_file_=model_dir_+"/a.model.tflite";
  //model_file_=model_dir_+ settings_->model_name;
  model_file_ = settings_->model_path;

  std::cout << "model_file_:"<< model_file_<< std::endl;
  delegate_providers_.MergeSettingsIntoParams(*settings_);

  model_ = tflite::FlatBufferModel::BuildFromFile(model_file_.c_str());
  if (!model_) {
      std::cerr << "FlatBufferModel::BuildFromFile(\"" << model_file_.c_str() << "\") failed." << std::endl;
      //return rc_f;
      return;
  }

  settings_->model = model_.get();
  std::cout << "Loaded model " << settings_->model_path << std::endl;
  model_->error_reporter();
  std::cout << "resolved reporter"<< std::endl;

  is_act_=true;

  std::cout << "Opp_Tflite::init(): end" << std::endl;
}

/*
* void Opp_Tflite::predict(cv::Mat *img, std::shared_ptr<std::vector<u_int8_t>> &pred_y)
* 注) 下記、model_、interpreter_ が、std::unique_ptr なので、複数のルーチンにまたがって使えない。
*  std::shared_ptr だと、OK みたい
*  同一のルーチン内で使わないと、メモリーが開放されてしまうみたい。
*  std::unique_ptr<tflite::FlatBufferModel> model_;  <-- こちらは、 std::shared_ptr OK みたい。
*  std::unique_ptr<tflite::Interpreter> interpreter_;
*
*/
bool Opp_Tflite::predict(cv::Mat *img, std::vector<u_int8_t> &pred_y){
    std::cout << "Opp_Tflite::predict()" << std::endl;

    bool rc_f=false;
    std::unique_ptr<tflite::Interpreter> interpreter;       // この変数は、std::unique_ptr でしか定義できないから、
                                                            //  このルーチン内のみで使う。

    #define TEST_OP1
    #if defined(TEST_OP0)
        auto resolver2 = tflite::CreateOpResolver();   // sub
        tflite::InterpreterBuilder(*model_, *resolver2)(&interpreter);

    // Transformer のときは、良いかも!!
    #elif defined(TEST_OP1)
        //auto resolver2 = tflite::MutableOpResolver();  // class
        tflite::MutableOpResolver resolver2;
        //resolver2.AddBuiltin(BuiltinOperator_ADD, tflite::CreateOpResolver());
        resolver2.AddAll(*(tflite::CreateOpResolver()));

        tflite::InterpreterBuilder(*model_, resolver2)(&interpreter);
    //#elif defined(TEST_OP2)
    //    static tflite::MicroMutableOpResolver<1> resolver2; // no of times I have added any ops through ops resolver
    //    resolver2.AddExpand_dims();         /*  The operation reported missing in error */
    //    tflite::InterpreterBuilder(*model_, resolver2)(&interpreter);
    #else
        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter);
    #endif

    if (!interpreter) {
        std::cout << ">>> Failed to construct interpreter"<< std::endl;
        return rc_f;
    }
    if (settings_->number_of_threads != -1) {
        interpreter->SetNumThreads(settings_->number_of_threads);
    }

    auto profiler = std::make_unique<tflite::profiling::Profiler>(
        settings_->max_profiling_buffer_entries);
    interpreter->SetProfiler(profiler.get());

    std::cout << ">>> delegates"<< std::endl;
    auto delegates = delegate_providers_.CreateAllDelegates();

    for (auto& delegate : delegates) {
        const auto delegate_name = delegate.provider->GetName();
        if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
            //LOG(ERROR) 
            std::cout << "Failed to apply " << delegate_name << " delegate."<< std::endl;
            return rc_f;
        } 
        else {
            //LOG(INFO)
            std::cout << "Applied " << delegate_name << " delegate."<< std::endl;
        }
    }

    std::cout << ">>> allocate interpreter tensors"<< std::endl;
    if (interpreter->AllocateTensors() != kTfLiteOk){
        std::cerr << ">>> Cannot allocate interpreter tensors" << std::endl;
        return rc_f;
    }

    input_ = interpreter->inputs()[0];
    std::cout << "input_: " << input_ << std::endl;

    inputs_ = interpreter->inputs();
    outputs_ = interpreter->outputs();

    std::cout << "number of inputs: " << inputs_.size() << std::endl;
    std::cout << "number of outputs: " << outputs_.size()<< std::endl;

    // get input dimension from the input tensor metadata
    // assuming one input only
    TfLiteIntArray* dims = interpreter->tensor(input_)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    //int wanted_channels = dims->data[3];

    std::cout << "wanted_height: " << wanted_height << std::endl;
    std::cout << "wanted_width: " << wanted_width << std::endl;
    //std::cout << "wanted_channels: " << wanted_channels << std::endl;

    input_type_=interpreter->tensor(input_)->type;


    cv::Mat img3=img->clone();
    int width = img->cols;
    int height = img->rows;
    int depth = img->channels();
    std::cout << " width:" << width << " height:"<< height << " depth:"<< depth <<std::endl;

    // 画像の拡大、縮小をしない方法で、 サイズ(h:122,w:600) にする。
    // 余白: 0
    // https://catalina1344.hatenablog.jp/entry/2014/04/05/103947
    cv::Mat img2(cv::Size(600,122), CV_8UC1, cv::Scalar(0));    // cv::Size(w,h) なので、注意!!
    cv::Mat dar16_9_roi(img2, cv::Rect(0, 0, img->cols,img->rows));
    img->copyTo(dar16_9_roi);

    // 転置する。(h:122,w:600) -> (h:600,w:122)  
    // model には、画像の上から下へ、122 px 毎に読ませる。   600 line
    // 注) 上下が逆になった気がする。  -> 後で、補正する。
    cv::transpose(img2,img2);

    //print_info(img2);
    //cv::imshow("img", *img);
    //cv::waitKey(100);

    //cv::imshow("img2", img2);
    //cv::waitKey(1000);
    //cv::waitKey(0);

    // uint8 -> float32 に変換
    cv::Mat img2_f;
    img2.convertTo(img2_f, CV_32F);

    // ノーマライズする。
    img2_f = img2_f/255.0;

    switch (input_type_) {
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
            return rc_f;
            break;
        case kTfLiteUInt8:
            std::cout << "input_type: kTfLiteUInt8 "<< std::endl;
            //resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
            //                image_height, image_width, image_channels, wanted_height,
            //                wanted_width, wanted_channels, settings);
            return rc_f;
            break;
        default:
            std::cout << "cannot handle input type "
                        << interpreter->tensor(input_)->type << " yet"<< std::endl;
            return rc_f;
    }

    auto ptr = interpreter->typed_input_tensor<float>(0);
    if(ptr == 0){
        std::cerr << "error interpreter->typed_input_tensor <float>(0" << std::endl;
        return rc_f;
    }

    cv::Mat tensor_image(600, 122, CV_32FC1, ptr);      // Tensor (batch:1,low:600,col:122) が入力形式
    img2_f.convertTo(tensor_image, CV_32FC1);

    std::cout << "exec invoke" << std::endl;

    // Run inference.
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Cannot invoke interpreter" << std::endl;
        return rc_f;
    }
    std::cout << "invoke end!!" << std::endl;

    // Get Output
    int output = interpreter->outputs()[0];
    //std::cout << "output:"<< output << std::endl;

    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;

    auto output_size = output_dims->data[output_dims->size - 1];
    std::cout << "output_dims->size:"<< output_dims->size << std::endl;
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
        //exit(-1);
        return rc_f;
        break;
    default:
        //LOG(ERROR)
        std::cout << "cannot handle output type "
                    << interpreter->tensor(output)->type << " yet"<< std::endl;
        //exit(-1);
        return rc_f;
    }

    switch(output_dims->size){
    // Transformer model
    case 2:
    {
        auto out_ptr = interpreter->typed_output_tensor<int32_t>(0);
        if(out_ptr !=0){
            auto output_size = output_dims->data[output_dims->size - 1];
            for(int i=0;i < output_size;i++){
                pred_y.push_back((u_int8_t)(out_ptr[i]));
            }
        }
    }
    break;

    // LSTM model
    case 3:
    {
        auto out_ptr = interpreter->typed_output_tensor<float>(0);
        std::cout << "out_ptr:"<< out_ptr << std::endl;
        if(out_ptr !=0){
            //std::vector<u_int8_t> soft_max_idx;
            //SoftMaxIdx<std::vector<u_int8_t>>(out_ptr,soft_max_idx);
            // pred_y
            SoftMaxIdx<std::vector<u_int8_t>>(out_ptr,pred_y);
        }
    }

    break;

    }

    return true;


}

}