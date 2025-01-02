/*
* Machine Learning Planner
*  tflite/opp_tflite/include/opp_ml_tflite/opp_tflite.hpp
*
*/

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
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

// add 2024.12.18
#include "tensorflow/lite/core/create_op_resolver.h"
#include "tensorflow/lite/mutable_op_resolver.h"

//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"    // nothing

#include <vector>

#include <iostream>
#include <opencv2/opencv.hpp>

#pragma once

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;

namespace opp_tflite{

struct Settings {
  bool verbose = false;
  bool accel = false;
  //TfLiteType input_type = kTfLiteFloat32;
  bool profiling = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  bool hexagon_delegate = false;
  bool xnnpack_delegate = true;
  //int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  std::string model_path = "/home/nishi/Documents/VisualStudio-TF/opp_with_lstm/a.model.tflite";
  //std::string model_path ="/home/nishi/Documents/VisualStudio-TF/opp_with_transformer_mltu/a.model.tflite";
  //std::string model_name = "/a.model.tflite";
  tflite::FlatBufferModel* model;
  //string input_bmp_name = "./grace_hopper.bmp";
  //string labels_file_name = "./labels.txt";
  //int number_of_threads = 4;
  int number_of_threads = 1;      // 1 で良いみたい。 by nishi 2024.12.18
  int number_of_results = 5;
  int max_profiling_buffer_entries = 1024;
  int number_of_warmup_runs = 2;
};

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
  void MergeSettingsIntoParams(const opp_tflite::Settings& s) {
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
        std::cout << "NNAPI delegate execution provider isn't linked or NNAPI " <<
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

class Opp_Tflite{
public:
  bool is_act_;
  int input_;

  std::vector<int> inputs_;
  std::vector<int> outputs_;

  int input_type_;
  //std::unique_ptr<tflite::FlatBufferModel> model_;
  std::shared_ptr<tflite::FlatBufferModel> model_;
  tflite::ops::builtin::BuiltinOpResolver resolver_;


  Opp_Tflite(){}
  void init(Settings *settings);
  bool predict(cv::Mat *img, std::vector<u_int8_t> &pred_y);
  //bool predict(cv::Mat *img);

private:
  Settings *settings_;
  DelegateProviders delegate_providers_;
  std::string model_dir_="/home/nishi/Documents/VisualStudio-TF/opp_with_lstm";
  std::string model_file_="/a.model.tflite";

};

}