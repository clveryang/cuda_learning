find_package(CUDAToolkit)

add_executable(my_reduce_v7_num_shuffle_down my_reduce_v7_num_shuffle_down.cu)
target_link_libraries(my_reduce_v7_num_shuffle_down PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v7_num_shuffle_down PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


# add_executable(my_reduce_v0_global_memory my_reduce_v0_global_memory.cu)
# target_link_libraries(my_reduce_v0_global_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(my_reduce_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()

