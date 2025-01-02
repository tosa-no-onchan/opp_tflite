/*
* tflite/opp_tflite/predict-test.cpp
*
* https://github.com/tensorflow/tensorflow/issues/26501
*
1) lib path
export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-lite:$LD_LIBRARY_PATH

2) include path
 /home/nishi/usr/local/include/tensorflow-lite

1. build
$ make
 
2. run
$ export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-lite:$LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=/home/nishi/usr/local/lib/tensorflow-2.16.2-lite-flex:$LD_LIBRARY_PATH
$ ./predict-test

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
//#include <opencv2/opencv.hpp>
#include <cstdint>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

//#include "tensorflow/lite/delegates/flex/delegate.h"

int main(int argc, char *argv[])
{
    std::string model_dir="/home/nishi/Documents/VisualStudio-TF/opp_with_lstm";
    std::string model_file=model_dir+"/a.model.tflite";
    // Load model.
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model) {
        std::cerr << "FlatBufferModel::BuildFromFile(\"" << model_file.c_str() << "\") failed." << std::endl;
        return -1;
    }

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    std::cout << "allocate interpreter tensors"<< std::endl;

    if (!interpreter) {
        std::cout << "Failed to construct interpreter"<< std::endl;
        return -1;
    }

    #define VERBOSE
    #if defined(VERBOSE)
        std::cout << "tensors size: " << interpreter->tensors_size();
        std::cout << " nodes size: " << interpreter->nodes_size();
        std::cout << " inputs: " << interpreter->inputs().size();
        std::cout << " input(0) name: " << interpreter->GetInputName(0) << std::endl;;
        
        #if defined(VERBOSE2)
            int t_size = interpreter->tensors_size();
            for (int i = 0; i < t_size; i++) {
            if (interpreter->tensor(i)->name)
                std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                        << interpreter->tensor(i)->bytes << ", "
                        << interpreter->tensor(i)->type << ", "
                        << interpreter->tensor(i)->params.scale << ", "
                        << interpreter->tensor(i)->params.zero_point << std::endl;;
            }
        #endif

    #endif

    //auto delegate = tflite::FlexDelegate::Create();
    //tflite::FlexDelegate flexDelegate = new tflite::FlexDelegate();

    std::cout << ">>> allocate interpreter tensors"<< std::endl;
    if (interpreter->AllocateTensors() != kTfLiteOk){
        std::cerr << ">>> Cannot allocate interpreter tensors" << std::endl;
        return 1;
    }

    std::cout << " prog end" << std::endl;

    return 0;
}