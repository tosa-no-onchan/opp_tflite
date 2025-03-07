#!/bin/bash -eu
# -*- coding: utf-8 -*-

# Referece:
# [copy_tensorflow_headers.sh](https://gist.github.com/saitodev/3cde48806a32272962899693700d9669)

# $ sudo ./deploy-lite-flex-nishi.sh

#VER=2.16.2
VER=2.18.0

#DIST_DIR=/home/nishi/usr/local/share/libtensorflow-cpu-${VER}
DIST_DIR=/home/nishi/usr/local

#HEADER_DIR=${DIST_DIR}/include/tensorflow-${VER}-lite/tensorflow
HEADER_DIR=${DIST_DIR}/include/tensorflow-${VER}-lite-flex
LIB_DIR=${DIST_DIR}/lib/tensorflow-${VER}-lite-flex

ROOT_DIR=./tensorflow-${VER}
SRC_DIR=./tensorflow-${VER}/tensorflow

BAZEL_BIN_DIR=./tensorflow-${VER}/bazel-bin

if [ ! -e $HEADER_DIR ];
then
    mkdir -p $HEADER_DIR
fi

if [ ! -e $HEADER_DIR/tensorflow ];
then
    mkdir -p ${HEADER_DIR}/tensorflow
fi

if [ ! -e $LIB_DIR ];
then
    mkdir -p $LIB_DIR
fi

cp -af ${BAZEL_BIN_DIR}/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so ${LIB_DIR}

#find ${ROOT_DIR}/build-nishi/ -name "lib*.a" | xargs -i cp -af {} ${LIB_DIR}
#cp -af ${ROOT_DIR}/build-nishi/libtensorflow-lite.a ${LIB_DIR}
#cp -af ${ROOT_DIR}/build-nishi/_deps/flatbuffers-build/libflatbuffers.a ${LIB_DIR}

#pushd $SRC_DIR
#find lite -follow -type f -name "*.h" -exec cp --parents {} ${HEADER_DIR}/tensorflow \;
#popd

#pushd ${ROOT_DIR}/build-nishi/
#pushd abseil-cpp
#find absl -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
#find ci -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
#popd

#pushd flatbuffers/include
#find flatbuffers -follow -type f -name "*.h" -exec cp --parents {} $HEADER_DIR \;
#find flatbuffers -follow -type f -name "*.inc" -exec cp --parents {} $HEADER_DIR \;
#popd
#popd
