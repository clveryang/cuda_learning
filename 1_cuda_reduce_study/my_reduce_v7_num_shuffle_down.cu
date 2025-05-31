#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstdio>

#define thread_per_block 256
#define warp_size 32

template<unsigned int NUM_PRO_BLOCK>
__global__ void reduce_0(float * device_input, float * device_output){

    float * block_pointer = device_input + blockIdx.x * NUM_PRO_BLOCK;
    
    int thread_id = threadIdx.x;
    float sum = 0.f;
    
    for (int i = 0; i < NUM_PRO_BLOCK / thread_per_block; i++){
        sum += block_pointer[thread_id + i * thread_per_block];
    }
    
/*
    1 2 3 4 5 6 7 8 
    move 4 
    5 6 7 8 5 6 7 8
    move 2
    3 4 5 6 7 8 7 8
    move 1
    2 3 4 5 6 7 8 8
*/
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__  float warpLevelSum[warp_size];
    const int laneId = threadIdx.x % warp_size;
    const int warpId = threadIdx.x / warp_size;

    if (laneId == 0){
        warpLevelSum [warpId] = sum;
    }
    __syncthreads();
    
    if (warpId == 0){
        sum = (laneId < blockDim.x / 32) ? warpLevelSum[laneId] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

    }
    if(thread_id == 0)
        device_output[blockIdx.x] = sum;
}



bool check(float *gpu_result, float * cpu_result, size_t N){

    for(int i = 0; i < N; i++){

        if(abs(gpu_result[i] - cpu_result[i]) > 0.005)
            return false;

    }
    return true;
}

int main(){

    // 生成随机数
    const int num_float = 48 * 1024 * 1024;

    float * host_input = (float*)malloc(num_float * sizeof(float));

    for(int i = 0; i < num_float; i++)
    {
        host_input[i] = 2.0 * (float)drand48() -1.0;
        //printf("host_input[%d]:%f \n", i, host_input[i]);
    }

    //固定block的数量 让一个block中的线程做更多的work
    const int block = 1024;
    const int num_pro_block = num_float / block;


    // 分配device空间
    float * device_input;
    cudaMalloc((void**)&device_input, num_float*sizeof(float));
    cudaMemcpy(device_input, host_input, num_float * sizeof(float), cudaMemcpyHostToDevice);

    float * device_output;
    cudaMalloc((void**)&device_output, (block) * sizeof(float));

    // gpu计算reduce

    dim3 grids(block, 1), blocks(thread_per_block, 1);

    reduce_0<num_pro_block><<< grids, blocks >>>(device_input, device_output);

    float * gpu_output = (float*)malloc(block * sizeof(float));
    cudaMemcpy(gpu_output, device_output, block * sizeof(float), cudaMemcpyDeviceToHost);
    
    // cpu计算数组结果
    float* host_output = (float*)malloc(block * sizeof(float));

    for(int i = 0; i < block; i++){
        host_output[i] = 0; 
        for (int j = 0; j < num_pro_block; j++){
            host_output[i] += host_input[i * num_pro_block + j];
        }
    }
 
    //cpu 加法
    if (check(gpu_output, host_output, block)){
        printf("answer is right \n");
    }else{
        printf("answer is wrong \n");
    }


    printf("down \n");
    return 0;
}
