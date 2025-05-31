#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstdio>

#define thread_per_block 256

__global__ void reduce_0(float * device_input, float * device_output){

    __shared__ float data[thread_per_block];
    // float * block_pointer = device_input + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    data[thread_id] = device_input[thread_id + blockIdx.x * blockDim.x];
    __syncthreads();

    for (int i = 1; i < thread_per_block; i*=2)
    {
        
        if(thread_id < (thread_per_block / (2*i))){
            
            int index = 2 * thread_id * i;
            data[index] += data[index + i];

        }
        __syncthreads();
    }

    if (thread_id == 0)
        device_output[blockIdx.x] = data[0];


    // if( thread_id = 0 or thread_id = 2 or thread_id = 4 or thread_id = 6)
    //     block_pointer[thread_id] += block_pointer[thread_id + 1];
    // if (thread_id = 0 or thread_id = 4)
    //     block_pointer[thread_id] += block_pointer[thread_id + 2];
    // if (thread_id = 0)
    //     block_pointer[thread_id] += block_pointer[thread_id + 4];

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
    size_t num_float = 48 * 1024 * 1024;

    float * host_input = (float*)malloc(num_float * sizeof(float));

    for(int i = 0; i < num_float; i++)
    {
        host_input[i] =2.0 * (float)drand48() -1.0;
        //printf("host_input[%d]:%f \n", i, host_input[i]);
    }
    int block = num_float / thread_per_block;

    // 分配device空间
    float * device_input;
    cudaMalloc((void**)&device_input, num_float*sizeof(float));
    cudaMemcpy(device_input, host_input, num_float * sizeof(float), cudaMemcpyHostToDevice);

    float * device_output;
    cudaMalloc((void**)&device_output, (block) * sizeof(float));

    // gpu计算reduce

    dim3 grids(block, 1), blocks(thread_per_block, 1);
    printf("check1 \n");
    reduce_0<<< grids, blocks >>>(device_input, device_output);

    float * gpu_output = (float*)malloc(block * sizeof(float));
    cudaMemcpy(gpu_output, device_output, block * sizeof(float), cudaMemcpyDeviceToHost);
    
    // cpu计算数组结果

    float* host_output = (float*)malloc(block * sizeof(float));
    printf("check2 \n");

    for(int i = 0; i < block; i++){
        host_output[i] = 0; 
        for (int j = 0; j < thread_per_block; j++){
            host_output[i] += host_input[i * thread_per_block + j];
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
