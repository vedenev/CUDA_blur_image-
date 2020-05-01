//tutorial: https://medium.com/@akshathvarugeese/cuda-c-functions-in-python-through-dll-and-ctypes-for-windows-os-c29f56361089
//how to make cross-platform code for shared library https://stackoverflow.com/questions/2164827/explicitly-exporting-shared-library-functions-in-linux


#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT extern "C" __declspec(dllexport)
    #define IMPORT extern "C" __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#include <iostream>

#include <cuda.h>

#include <cmath>

#include <helper_cuda.h>

# define M_PI           3.14159265358979323846  /* pi */

const int N_CANNELS = 3;



const unsigned int core_gpu_max_size = 64 * 64;
__constant__ unsigned short core_gpu[core_gpu_max_size];

__global__ void blurKernel(unsigned char* image_gpu, int n_channels, int image_size_x, int image_strade, int kernel_size, unsigned char* image_out, int image_out_size_x, int image_out_size_y, int image_out_strade)
{
	
	const unsigned int index_output_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (index_output_y < image_out_size_y)
	{
		const unsigned int index_output_x = blockIdx.x * blockDim.x + threadIdx.x;
		if (index_output_x < image_out_size_x)
		{
			const unsigned int index_output_channel = blockIdx.z * blockDim.z + threadIdx.z;
			int index_input_offset = image_strade * index_output_channel + index_output_y * image_size_x + index_output_x;
			unsigned short sum = 0;
			int core_index = 0;
			for (int y = 0; y < kernel_size; y++)
			{
				int index_input_offset_2 = index_input_offset + y * image_size_x;
				for (int x = 0; x < kernel_size; x++)
				{
					int index_input = index_input_offset_2 + x;
					sum += ((unsigned short)image_gpu[index_input]) * core_gpu[core_index];
					core_index++;
				}
			}
			int index_output = image_out_strade * index_output_channel + index_output_y * image_out_size_x + index_output_x;
			sum += 128; // for rounding
			image_out[index_output] = (unsigned char)(sum / 256);
		}
	}
}

int integer_division_ceiling(int x, int y)
{
	// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
	return x / y + (x % y != 0);
}

EXPORT void blur(unsigned char* image, int image_size_x, int image_size_y, int kernel_size, float sigma, unsigned char* image_out) {

	// numpy indexing [channel_index, y_index, x_index]

	// create core:
	float mean = (kernel_size - 1.0) / 2.0;
	float variance = sigma * sigma;
	float variance2 = 2 * variance;
	float variance2pi = variance2 * M_PI;
	int kernel_size2 = kernel_size * kernel_size;
	float* core_float_cpu = (float*)malloc(sizeof(float) * kernel_size2);
	int index = 0;
	float core_float_cpu_sum = 0.0;
	for (int y = 0; y < kernel_size; y++)
	{
		float dy = (float)y - mean;
		float dy2 = dy * dy;
		for (int x = 0; x < kernel_size; x++)
		{
			float dx = (float)x - mean;
			float dx2 = dx * dx;

			float argument_tmp = (dx2 + dy2) / variance2;
			
			core_float_cpu[index] = exp(-argument_tmp) / (variance2pi);
			core_float_cpu_sum += core_float_cpu[index];
			index++;
		}
	}

	for (index = 0; index < kernel_size2; index++)
	{
		core_float_cpu[index] = core_float_cpu[index] / core_float_cpu_sum;
	}

	unsigned short* core_cpu;
	core_cpu = (unsigned short*)malloc(sizeof(unsigned short) * kernel_size2);


	for (index = 0; index < kernel_size2; index++)
	{
		core_cpu[index] = (unsigned short)(roundf(256.0 * core_float_cpu[index]));
	}


	checkCudaErrors(cudaMemcpyToSymbol(core_gpu, core_cpu, sizeof(unsigned short) * kernel_size2));

	int image_size = image_size_x * image_size_y * N_CANNELS;
	unsigned char* image_gpu;
	if (cudaMalloc((void**)&image_gpu, sizeof(unsigned char) * image_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU image_gpu\n";
	}
	cudaMemcpy(image_gpu, image, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
	
	int image_out_size_x = image_size_x - kernel_size + 1;
	int image_out_size_y = image_size_y - kernel_size + 1;
	int image_out_size = image_out_size_x * image_out_size_y * N_CANNELS;
	unsigned char* image_out_gpu;
	if (cudaMalloc((void**)&image_out_gpu, sizeof(unsigned char) * image_out_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU image_out_gpu\n";
	}

	int image_out_thread_size_x = 16;
	int image_out_thread_size_y = 16;
	int image_out_thread_size_z = 1;
	int image_out_grid_size_x = integer_division_ceiling(image_out_size_x, image_out_thread_size_x);
	int image_out_grid_size_y = integer_division_ceiling(image_out_size_y, image_out_thread_size_y);
	dim3 grid_image_out(image_out_grid_size_x, image_out_grid_size_y, image_out_thread_size_z);
	dim3 threadBlock_image_out(image_out_thread_size_x, image_out_thread_size_y, N_CANNELS);
	int image_strade = image_size_x * image_size_y;
	int image_out_strade = image_out_size_x * image_out_size_y;
	blurKernel<<<grid_image_out, threadBlock_image_out>>>(image_gpu, N_CANNELS, image_size_x, image_strade,  kernel_size, image_out_gpu, image_out_size_x, image_out_size_y, image_out_strade);
	cudaMemcpy(image_out, image_out_gpu, sizeof(char) * image_out_size, cudaMemcpyDeviceToHost);
	
}