#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstdio>

#define thread_per_block 256

template<unsigned int NUM_PRO_BLOCK>
__global__ void reduce_0(float * device_input, float * device_output){

    __shared__ float data[thread_per_block];
    float * block_pointer = device_input + blockIdx.x * NUM_PRO_BLOCK;
    
    int thread_id = threadIdx.x;
    
    data[thread_id] = 0;
    for (int i = 0; i < NUM_PRO_BLOCK / thread_per_block; i++){
        data[thread_id] += block_pointer[thread_id + i * thread_per_block];
    }
    
    __syncthreads();

// #pragma unroll
//     for (int i = blockDim.x / 2; i > 32; i/=2)
//     {
        
//         if(thread_id < i){
            
//             //int index = 2 * thread_id * i;
//             data[thread_id] += data[thread_id + i];

//         }
//         __syncthreads();
//     }

    if(thread_id < 128)
    {
        data[thread_id] += data[thread_id + 128];
        __syncthreads();
    }
    
    if(thread_id < 64)
    {
        data[thread_id] += data[thread_id + 64];
        __syncthreads();
    }    

    if (thread_id < 32){
        data[thread_id] += data[thread_id + 32];
        data[thread_id] += data[thread_id + 16];
        data[thread_id] += data[thread_id + 8];
        data[thread_id] += data[thread_id + 4];
        data[thread_id] += data[thread_id + 2];
        data[thread_id] += data[thread_id + 1];
    }

    if (thread_id == 0)
        device_output[blockIdx.x] = data[0];

    /*
    减少blockIdx.x的一半

    */

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
