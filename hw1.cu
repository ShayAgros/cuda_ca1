/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_HEIGHT 256
#define IMG_WIDTH 256
//#define N_IMAGES 10000
#define N_IMAGES 1

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

__device__ void mapCalc(int map[], int min, int cdf[]){
	int id = threadIdx.x;
	int map_value = (cdf[id] - min) / (IMG_WIDTH * IMG_HEIGHT - min) * 255;
    map[id] = map_value;
}

__global__ void process_image_kernel(uchar *in, uchar *out, int temp_histogram[], int temp_cdf[]) {

    __shared__ int l_histogram[256];
    __shared__ int l_cdf[256];
	int tid = threadIdx.x;
	int tbsize = blockDim.x;

	// zero histogram
	l_histogram[tid] = 0;

	for(int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += tbsize)
		atomicAdd(&l_histogram[in[i]], 1);

	__syncthreads();

	// for debug purposes TODO: delete
	temp_histogram[tid] = l_histogram[tid];

	// prepare the cdf array in advance
	l_cdf[tid] = l_histogram[tid];

	__syncthreads();

//	prefix_sum <<< 254/2, 2 >>> (cdf, 256, histogram);

	prefix_sum(l_cdf, 256, l_histogram);

	temp_cdf[tid] = l_cdf[tid];

	__syncthreads();

    int min = arr_min(l_histogram, 256);
	
	temp_cdf[tid] = l_histogram[tid];
	
	__syncthreads();

    int map[256] = { 0 };
	mapCalc(map, min, l_cdf);
	
	__syncthreads();
    

    for(int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += tbsize){
		out[i] = map[in[i]];
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

	int *temp_histogram;
	int *temp_cdf;
	int cpu_histogram[256] = { 0 };
	int cpu_cdf[256] = { 0 };
	int total_sum = 0;

    // TODO: debug, remove later
    CUDA_CHECK( cudaMalloc((void **)&temp_histogram, 256 * sizeof(*temp_histogram)) );
    CUDA_CHECK( cudaMalloc((void **)&temp_cdf, 256 * sizeof(*temp_cdf)) );

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n"); //Do not change

    //TODO: allocate GPU memory for a single input image and a single output image
    CUDA_CHECK( cudaMalloc((void **)&image_in, IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc((void **)&image_out, IMG_HEIGHT * IMG_WIDTH) );

    t_start = get_time_msec(); //Do not change

    //TODO: in a for loop:
    for (int i=0; i < N_IMAGES; i++) {
		// Copying src image from the input images
		cudaMemcpy(image_in, &images_in[i * IMG_WIDTH*IMG_HEIGHT], IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault);
		
		// TODO: debug, remove later
		cudaMemset(temp_histogram, 0, 256 * sizeof(*temp_histogram));
		cudaMemset(temp_cdf, 0, 256 * sizeof(*temp_cdf));

		process_image_kernel <<< 1, NUM_THREADS >>> (image_in, image_out, temp_histogram, temp_cdf);   

		cudaDeviceSynchronize();

		// TODO: debug, remove later
		cudaMemcpy(cpu_histogram, temp_histogram, 256 * sizeof(*temp_histogram), cudaMemcpyDefault);
		cudaMemcpy(cpu_cdf, temp_cdf, 256 * sizeof(*temp_cdf), cudaMemcpyDefault);

		// Debug prints TODO: remove later
		printf ("\n\nHistogram array is as followed:\n");
		for (int i=0; i< 4; i++) {
			for (int j = 0; j < 64; j++) {
				printf("h[%d] = %d  ",i*64 + j , cpu_histogram[i*64 + j]);
			}

		printf("\n\n");
		}

		for(int i = 0; i < 256; i++)
			total_sum += cpu_histogram[i];

		printf("Total sum is: %d\n", total_sum);

		printf ("\n\nCDF array is as followed:\n");
		for (int i=0; i< 4; i++) {
			for (int j = 0; j < 64; j++) {
				printf("h[%d] = %d  ",i*64 + j , cpu_cdf[i*64 + j]);
			}

		printf("\n\n");
		}
		cudaMemcpy(&images_out_gpu_serial[i*IMG_HEIGHT*IMG_WIDTH], image_out, IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault);

    }
	
	cudaFree(image_in);
	cudaFree(image_out);
	//////////////////////////TODO: remove later
	cudaFree(temp_histogram);
	cudaFree(temp_cdf);
	
	
    //   1. copy the relevant image from images_in to the GPU memory you allocated CHECKED
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not change

    // GPU bulk
    printf("\n=== GPU Bulk ===\n"); //Do not change
    //TODO: allocate GPU memory for a all input images and all output images
    t_start = get_time_msec(); //Do not change
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out_gpu_bulk
	CUDA_CHECK( cudaMalloc((void **)&image_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc((void **)&image_out, N_IMAGES * IMG_HEIGHT * IMG_WIDTH) );
	cudaMemcpy(image_in, images_in, N_IMAGES*IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault);
	process_image_kernel <<< N_IMAGES, NUM_THREADS >>> (image_in, image_out, temp_histogram, temp_cdf);   
	cudaMemcpy(images_out_gpu_bulk, image_out, N_IMAGES*IMG_WIDTH*IMG_HEIGHT, cudaMemcpyDefault);	
	cudaFree(image_in);
	cudaFree(image_out);
	
    t_finish = get_time_msec(); //Do not change
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk); // Do not change
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not chhange

    return 0;
}
