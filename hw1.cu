/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_HEIGHT 256
#define IMG_WIDTH 256
//#define N_IMAGES 10000
#define N_IMAGES 500

#define NUM_THREADS 256


typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int arr_min(int arr[], int arr_size) {//return the min
	int tid = threadIdx.x;
	int half_length = arr_size / 2;

	while (half_length >= 1) {
		for (int i = tid; i < half_length; i += blockDim.x) {
			if(arr[tid + i] < arr[i]) arr[i] = arr[tid + i];
		}
		__syncthreads();
		half_length /= 2;
	}
    return arr[0]; //TODO
}

// this function implements the Kiggle-Stone algorithm
__device__ void prefix_sum(int arr[], int arr_size, int histogram[]) {

	int tbsize = blockDim.x;
	int tid = threadIdx.x;
	int inc;
 
 	for (int stride = 1; stride < tbsize; stride *= 2) {
	
		if (tid >= arr_size)
			continue;

 		if (tid >= stride) {
 		inc = arr[tid - stride];
 		}
 		__syncthreads();

 		if (tid >= stride) {
			arr[tid] += inc;
 		}

 		__syncthreads();
 	}

    return;
}

__device__ void mapCalc(int map[], int min, int cdf[]) {
	int id = threadIdx.x;
	int map_value = ((double)(cdf[id] - min)/(IMG_WIDTH * IMG_HEIGHT - min)) * 255;
    map[id] = map_value;
}

__global__ void process_image_kernel(uchar *in, uchar *out) {

    __shared__ int l_histogram[256];
    __shared__ int l_cdf[256];
    __shared__ int map[256];
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int tbsize = blockDim.x;

	// zero histogram
	l_histogram[tid] = 0;

	__syncthreads();

	for(int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += tbsize)
		atomicAdd(&l_histogram[in[(IMG_WIDTH * IMG_HEIGHT)*bid + i]], 1);

	__syncthreads();

	// prepare the cdf array in advance
	l_cdf[tid] = l_histogram[tid];

	__syncthreads();

	prefix_sum(l_cdf, 256, l_histogram);

	__syncthreads();

    int min = arr_min(l_histogram, 256);

	__syncthreads();

	mapCalc(map, min, l_cdf);

	__syncthreads();
   

    for(int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += tbsize) {
		out[(IMG_WIDTH * IMG_HEIGHT)*bid + i] =
			map[in[(IMG_WIDTH * IMG_HEIGHT)*bid + i]];
    }

	__syncthreads();

    return ; //TODO
}

int main() {
///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    /* instead of loading real images, we'll load the arrays with random data */
    srand(0);
    for (long long int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        images_in[i] = rand() % 256;
    }

    double t_start, t_finish;

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
		process_image(img_in, img_out);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;
///////////////////////////////////////////////////////////////////////////////////////////////////////////
	uchar *image_in;
	uchar *image_out;

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n"); //Do not change

    //TODO: allocate GPU memory for a single input image and a single output image
    CUDA_CHECK( cudaMalloc((void **)&image_in, IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc((void **)&image_out, IMG_HEIGHT * IMG_WIDTH) );

    t_start = get_time_msec(); //Do not change

    //TODO: in a for loop:
    for (int i=0; i < N_IMAGES; i++) {
		// Copying src image from the input images
		CUDA_CHECK(cudaMemcpy(image_in, &images_in[i * IMG_WIDTH*IMG_HEIGHT], IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault));

		process_image_kernel <<< 1, NUM_THREADS >>> (image_in, image_out);  

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(&images_out_gpu_serial[i * IMG_HEIGHT*IMG_WIDTH], image_out, IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault));
    }

	cudaFree(image_in);
	cudaFree(image_out);

    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not change

    // GPU bulk
    printf("\n=== GPU Bulk ===\n"); //Do not change
    //TODO: allocate GPU memory for a all input images and all output images
	CUDA_CHECK( cudaMalloc((void **)&image_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc((void **)&image_out, N_IMAGES * IMG_HEIGHT * IMG_WIDTH) );

    //TODO: copy all input images from images_in to the GPU memory you allocated
    t_start = get_time_msec(); //Do not change
	CUDA_CHECK(cudaMemcpy(image_in, images_in, N_IMAGES*IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault));

    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
	process_image_kernel <<< N_IMAGES, NUM_THREADS >>> (image_in, image_out);
	CUDA_CHECK(cudaDeviceSynchronize());

    //TODO: copy output images from GPU memory to images_out_gpu_bulk
	CUDA_CHECK(cudaMemcpy(images_out_gpu_bulk, image_out, N_IMAGES*IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault));

    t_finish = get_time_msec(); //Do not change

	cudaFree(image_in);
	cudaFree(image_out);

    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not chhange

    return 0;
}
