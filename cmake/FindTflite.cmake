# opp_tflite/cmake/FindTflite.cmake
# for Tensorflow 2.18.0
#  here are lib paths
# /home/nishi/usr/local/lib/tensorflow-lite /home/nishi/usr/local/lib/tensorflow-lite-flex
# https://qiita.com/shohirose/items/d9bda00a39a113965c5c
# /usr/share/cmake-3.22/Modules/FindLAPACK.cmake

set(Tflite_INCLUDE_DIR "/home/nishi/usr/local/include/tensorflow-lite")
set(Tflite_LIBRARY 
  "-L/home/nishi/usr/local/lib/tensorflow-lite -L/home/nishi/usr/local/lib/tensorflow-lite-flex -ltensorflow-lite -lprofiling_info_proto -lprotobuf -labsl_flags -labsl_flags_internal -labsl_flags_marshalling -labsl_flags_reflection -labsl_flags_config -labsl_flags_program_name -labsl_flags_private_handle_accessor -labsl_flags_commandlineflag -labsl_flags_commandlineflag_internal -labsl_raw_hash_set -labsl_hashtablez_sampler -labsl_hash -labsl_city -labsl_low_level_hash -labsl_status -labsl_cord -labsl_bad_optional_access -labsl_cordz_info -labsl_cord_internal -labsl_cordz_functions -labsl_exponential_biased -labsl_cordz_handle -labsl_crc_cord_state -labsl_crc32c -labsl_crc_internal -labsl_crc_cpu_detect -labsl_str_format_internal -labsl_strerror -labsl_synchronization -labsl_stacktrace -labsl_symbolize -labsl_debugging_internal -labsl_demangle_internal -labsl_graphcycles_internal -labsl_kernel_timeout_internal -labsl_malloc_internal -labsl_time -labsl_strings -labsl_string_view -labsl_strings_internal -labsl_throw_delegate -labsl_base -labsl_spinlock_wait -lrt -labsl_int128 -labsl_civil_time -labsl_time_zone -labsl_bad_variant_access -labsl_raw_logging_internal -labsl_log_severity -lfarmhash -lfft2d_fftsg2d -lfft2d_fftsg -lm -ldl -lxnnpack-delegate -lflatbuffers -leight_bit_int_gemm -lruy_context_get_ctx -lruy_context -lruy_frontend -lruy_kernel_arm -lruy_kernel_avx -lruy_kernel_avx2_fma -lruy_kernel_avx512 -lruy_apply_multiplier -lruy_pack_arm -lruy_pack_avx -lruy_pack_avx2_fma -lruy_pack_avx512 -lruy_prepare_packed_matrices -lruy_trmul -lruy_ctx -lruy_allocator -lruy_prepacked_cache -lruy_system_aligned_alloc -lruy_have_built_path_for_avx -lruy_have_built_path_for_avx2_fma -lruy_have_built_path_for_avx512 -lruy_thread_pool -lruy_blocking_counter -lruy_wait -lruy_denormal -lruy_block_map -lruy_tune -lruy_cpuinfo -lruy_profiler_instrumentation -pthread -lXNNPACK -lmicrokernels-prod -lpthreadpool -lm -lcpuinfo -lprotobuf -ltf-lite_tools -ltf-lite_core"
  "-ltensorflowlite_flex"
)
#set(Tflite_LIBRARY_2_16_2 
#  "-L/home/nishi/usr/local/lib/tensorflow-lite -L/home/nishi/usr/local/lib/tensorflow-lite-flex -ltensorflow-lite -labsl_flags -labsl_flags_internal -labsl_flags_marshalling -labsl_flags_reflection -labsl_flags_config -labsl_flags_program_name -labsl_flags_private_handle_accessor -labsl_flags_commandlineflag -labsl_flags_commandlineflag_internal -labsl_raw_hash_set -labsl_hashtablez_sampler -labsl_hash -labsl_city -labsl_low_level_hash -labsl_status -labsl_cord -labsl_bad_optional_access -labsl_cordz_info -labsl_cord_internal -labsl_cordz_functions -labsl_exponential_biased -labsl_cordz_handle -labsl_crc_cord_state -labsl_crc32c -labsl_crc_internal -labsl_crc_cpu_detect -labsl_str_format_internal -labsl_strerror -labsl_synchronization -labsl_stacktrace -labsl_symbolize -labsl_debugging_internal -labsl_demangle_internal -labsl_graphcycles_internal -labsl_kernel_timeout_internal -labsl_malloc_internal -labsl_time -labsl_strings -labsl_string_view -labsl_strings_internal -labsl_throw_delegate -labsl_base -labsl_spinlock_wait -lrt -labsl_int128 -labsl_civil_time -labsl_time_zone -labsl_bad_variant_access -labsl_raw_logging_internal -labsl_log_severity -lfarmhash -lfft2d_fftsg2d -lfft2d_fftsg -lm -lflatbuffers -leight_bit_int_gemm -lruy_context_get_ctx -lruy_context -lruy_frontend -lruy_kernel_arm -lruy_kernel_avx -lruy_kernel_avx2_fma -lruy_kernel_avx512 -lruy_apply_multiplier -lruy_pack_arm -lruy_pack_avx -lruy_pack_avx2_fma -lruy_pack_avx512 -lruy_prepare_packed_matrices -lruy_trmul -lruy_ctx -lruy_allocator -lruy_prepacked_cache -lruy_system_aligned_alloc -lruy_have_built_path_for_avx -lruy_have_built_path_for_avx2_fma -lruy_have_built_path_for_avx512 -lruy_thread_pool -lruy_blocking_counter -lruy_wait -lruy_denormal -lruy_block_map -lruy_tune -lruy_cpuinfo -lruy_profiler_instrumentation -pthread -ldl -lXNNPACK -lpthreadpool -lm -lcpuinfo -ltf-lite_tools -ltf-lite_core"
#  "-ltensorflowlite_flex"
#)

#message(STATUS "Tflite_LIBRARY=${Tflite_LIBRARY}")

mark_as_advanced(
  Tflite_INCLUDE_DIR
  Tflite_LIBRARY     # ヘッダーのみのライブラリの場合は不要
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tflite
  REQUIRED_VARS
    Tflite_INCLUDE_DIR
    Tflite_LIBRARY      # ヘッダーのみのライブラリの場合は不要
  )

if(Tflite_FOUND AND NOT TARGET Tflite::Tflite)
  add_library(Tflite::Tflite UNKNOWN IMPORTED)
  set_target_properties(Tflite::Tflite PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"  # ヘッダーのみのライブラリの場合は不要
    IMPORTED_LOCATION "${Tflite_LIBRARY}"      # ヘッダーのみのライブラリの場合は不要
    INTERFACE_INCLUDE_DIRECTORIES "${Tflite_INCLUDE_DIR}"
    )
endif()
