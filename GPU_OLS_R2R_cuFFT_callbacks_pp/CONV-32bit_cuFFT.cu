//********************************************************************************************
//* This is GPU implementation of a Overlap-and-save method for calculating convolution. 
//* Copyright (C) 2019  Adámek Karel
//* 
//* Authors: Karel Adamek ( ORCID:0000-0003-2797-0595; https://github.com/KAdamek ), Wesley Armour ( ORCID:0000-0003-1756-3064 ), Sofia Dimoudi 
//********************************************************************************************

#include <iostream>
#include <fstream>
#include <iomanip> 

#include <cufft.h>
#include <cufftXt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"

#include "params.h"

#define WARP 32

int device=DEVICEID;

typedef cufftComplex* FP; 

struct ConvolveMetadata{
	int nConvolutions;
	int nFilters;
	FP filtersPtr;
};

//--------------------------- Callbacks -----------------------------
__device__ void CB_RemoveAliasedSamples(void *dataOut, size_t external_pos, cufftReal element, void *callerInfo, void *sharedPtr) {
    int *length = ((int*)callerInfo);
	int pos_within_fft = external_pos % CONV_SIZE;
	int fft = (int) (external_pos/CONV_SIZE);
	int useful_part_size = CONV_SIZE - length[0] + 1;
	useful_part_size = 2*((useful_part_size)>>1);
	int template_offset = length[0]/2;
	
	int pos = fft*useful_part_size + pos_within_fft - template_offset;
	if(pos_within_fft>=template_offset && pos_within_fft<(useful_part_size + template_offset)){
		((cufftReal*)dataOut)[pos] = 0.5*element;
	}
}


__device__ void CB_ConvolveFilters(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {
	ConvolveMetadata *l_cnvmd = ((ConvolveMetadata*) callerInfo);
	int conv_size = ((CONV_SIZE>>1)+1);
	int nConvolutions = l_cnvmd->nConvolutions;
	int nFilters = l_cnvmd->nFilters;
	cufftComplex *filterPtr;
	filterPtr = l_cnvmd->filtersPtr;
	int pos_within_fft = offset % conv_size;
	int fft = (int) (offset/conv_size);
	
	if(pos_within_fft<conv_size){
		for(int f=0; f<nFilters; f++){
			int filter_pos = f*conv_size + pos_within_fft;
			int store_pos  = f*nConvolutions*conv_size + fft*conv_size + pos_within_fft;
			
			cufftComplex temp;
			temp.x = filterPtr[filter_pos].x*element.x - filterPtr[filter_pos].y*element.y;
			temp.y = filterPtr[filter_pos].x*element.y + filterPtr[filter_pos].y*element.x;

			((cufftComplex*)dataOut)[store_pos] = temp;
		}
	}
}


__device__ cufftCallbackStoreR d_backwardFFTCallbackPtr = CB_RemoveAliasedSamples;
__device__ cufftCallbackStoreC d_forwardFFTCallbackPtr  = CB_ConvolveFilters;
//--------------------------- Callbacks -----------------------------


__global__ void prepare_signal(float* d_extended_data, float* d_input_signal, int signal_length, int convolution_size, int useful_part_size, int offset, int multiple) {
	int extpos = blockIdx.x*convolution_size;
	int sigpos = blockIdx.x*useful_part_size;
	
	if(blockIdx.x==0){ //leading part
		for(int i=0; i<multiple; i++){
			int pos = sigpos + i*1024 + threadIdx.x - offset;
			if( pos<0 ){
				d_extended_data[extpos + i*1024 + threadIdx.x] = 0;
			}
			else {
				d_extended_data[extpos + i*1024 + threadIdx.x] = d_input_signal[pos];
			}
		}
	}
	else if(blockIdx.x==(gridDim.x-1)){ //trailing part
		for(int i=0; i<multiple; i++){
			int pos = sigpos + i*1024 + threadIdx.x - offset;
			if (pos<signal_length){
				d_extended_data[extpos + i*1024 + threadIdx.x] = d_input_signal[pos];
			}
			else {
				d_extended_data[extpos + i*1024 + threadIdx.x] = 0;
			}
		}	
	}
	else { //middle parts
		for(int i=0; i<multiple; i++){
			d_extended_data[extpos + i*1024 + threadIdx.x] = d_input_signal[sigpos - offset + i*1024 + threadIdx.x];
		}
	}
}

__global__ void post_process(float *d_output, float* d_input, int nTimesamples, float h){
	float left, right;
	int pos_x = blockIdx.x*blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y*nTimesamples;
	if(blockIdx.x == 0) {
		if(threadIdx.x > 0) {
			left = d_input[pos_y + pos_x - 1];
		}
		else {
			left = d_input[pos_y + pos_x];
		}
		right = d_input[pos_y + pos_x + 1];
	}
	else {
		left = d_input[pos_y + pos_x - 1];
		if( (pos_x + 1) < nTimesamples ) {
			right = d_input[pos_y + pos_x + 1];
		}
		else {
			right = d_input[pos_y + nTimesamples - 1];
		}
	}
	
	float result;
	result = (left - right)/(2.0f*h);
	
	if(pos_x<nTimesamples) d_output[pos_y + pos_x] = result;
}


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

void Calculate_Convolution_Parameters(int *offset, int *useful_part_size, int *nConvolutions, int signal_length, int filter_length, int convolution_size){ 
	*offset           = (int) (filter_length/2); 
	*useful_part_size = convolution_size - filter_length + 1;
	*nConvolutions    = (int) ((signal_length + (*useful_part_size) - 1)/(*useful_part_size));
}


int CONV_cuFFT_benchmark(
		float *d_input, 
		float *d_extended_input, 
		float2 *d_extended_input_freqdom,
		float2 *d_output_plane_1, 
		float *d_filters_timedom, 
		float2 *d_filters_freqdom, 
		float *d_output_plane_2, 
		int signal_length, int filter_length, int nFilters, float h, double *CONV_time){
	GpuTimer timer, individual_timer;
	
	//---------> Convolution parameters
	int offset, useful_part_size, nConvolutions;
	Calculate_Convolution_Parameters(&offset, &useful_part_size, &nConvolutions, signal_length, filter_length, CONV_SIZE);
	
	
	//---------> Grid and Blocks for segment preparation
	int OaS_thread_size, OaS_multiplicator;
	if(CONV_SIZE>1024) {
		OaS_thread_size=1024;
		OaS_multiplicator=CONV_SIZE/1024;
	}
	else  {
		OaS_multiplicator=1;
		OaS_thread_size=CONV_SIZE;
	}
	dim3 OaS_gridSize(nConvolutions, 1, 1);
	dim3 OaS_blockSize(OaS_thread_size, 1, 1);
	//---------<
	
	//---------> cuFFT plans
	cufftHandle plan_forwardFFT, plan_inverseFFT, plan_forwardFFT_filters;
	cufftResult_t cuFFTResult;
	if ( cufftPlan1d(&plan_forwardFFT_filters, CONV_SIZE, CUFFT_R2C, nFilters) != CUFFT_SUCCESS) printf("CUFFT error.\n");
	
	// backward plan
	cufftCreate(&plan_inverseFFT);
    int signalSize = CONV_SIZE;
	size_t workSize;
	!!cufftMakePlanMany(plan_inverseFFT, 1, &signalSize, 0,0,0,0,0,0, CUFFT_C2R, nConvolutions*nFilters, &workSize);
	cufftCallbackStoreR h_backwardFFTCallbackPtr;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_backwardFFTCallbackPtr, d_backwardFFTCallbackPtr, sizeof(h_backwardFFTCallbackPtr)));
	int *d_length;
	int h_length = filter_length;
	checkCudaErrors(cudaMalloc((void **) &d_length,  sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_length, &h_length, sizeof(int), cudaMemcpyHostToDevice));
	cuFFTResult = cufftXtSetCallback(plan_inverseFFT, (void **)&h_backwardFFTCallbackPtr, CUFFT_CB_ST_REAL, (void **) &d_length);
	if( cuFFTResult != CUFFT_SUCCESS) {
		printf("CUDA API Error while creating cuFFT plan for output plane! Error: %d\n", cuFFTResult);
		return(1);
	}
	
	// forward FFT plan
	ConvolveMetadata *d_conv_metadata;
	cudaMalloc(&d_conv_metadata, sizeof(ConvolveMetadata));
	cudaMemcpy(&d_conv_metadata->nConvolutions, &nConvolutions, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_conv_metadata->nFilters, &nFilters, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_conv_metadata->filtersPtr, &d_filters_freqdom, sizeof(FP), cudaMemcpyHostToDevice);
	
	cufftCreate(&plan_forwardFFT);
	cufftMakePlanMany(plan_forwardFFT, 1, &signalSize, 0,0,0,0,0,0, CUFFT_R2C, nConvolutions, &workSize);
	cufftCallbackStoreC h_forwardFFTCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&h_forwardFFTCallbackPtr, d_forwardFFTCallbackPtr, sizeof(h_forwardFFTCallbackPtr)));
	cuFFTResult = cufftXtSetCallback(plan_forwardFFT, (void **)&h_forwardFFTCallbackPtr, CUFFT_CB_ST_COMPLEX, (void **) &d_conv_metadata);
	if( cuFFTResult != CUFFT_SUCCESS) {
		printf("CUDA API Error while creating cuFFT plan for output plane! Error: %d\n", cuFFTResult);
		return(1);
	}
	
	//----------------------------------------------> Measurements start
	timer.Start();
	
	//---------> DFT of filters
	individual_timer.Start();
	cuFFTResult = cufftExecR2C(plan_forwardFFT_filters, (cufftReal *) d_filters_timedom, (cufftComplex *)d_filters_freqdom);
	if(cuFFTResult!=CUFFT_SUCCESS) return(1);
	individual_timer.Stop();
	if(DEBUG) printf("Filter preparation: %0.3f\n", individual_timer.Elapsed());
	
	//---------> Segment preparation
	individual_timer.Start();
	prepare_signal<<<OaS_gridSize,OaS_blockSize>>>(d_extended_input, d_input, signal_length, CONV_SIZE, useful_part_size, offset, OaS_multiplicator);
	cuFFTResult = cufftExecR2C(plan_forwardFFT, (cufftReal *) d_extended_input, (cufftComplex *) d_output_plane_1);
	if(cuFFTResult!=CUFFT_SUCCESS) return(1);
	individual_timer.Stop();
	if(DEBUG) printf("Segment preparation: %0.3f\n", individual_timer.Elapsed());
	
	//---------> Inverse FFT of resulting plane
	individual_timer.Start();
	cuFFTResult = cufftExecC2R(plan_inverseFFT, (cufftComplex *) d_output_plane_1, (cufftReal *) d_output_plane_2);
	if(cuFFTResult!=CUFFT_SUCCESS) return(1);
	individual_timer.Stop();
	if(DEBUG) printf("Inverse cuFFT of resulting plane: %0.3f\n", individual_timer.Elapsed());
	
	
	//---------> Non-local post processing
	individual_timer.Start();
	int pp_nThreads = 256;
	int pp_nBlocks_x = (int) ((nConvolutions*useful_part_size + pp_nThreads - 1)/pp_nThreads);
	dim3 pp_grid(pp_nBlocks_x, nFilters, 1);
	dim3 pp_block(pp_nThreads, 1, 1);
	post_process<<<pp_grid,pp_block>>>((float*) d_output_plane_1, (float*) d_output_plane_2, nConvolutions*useful_part_size, h);
	individual_timer.Stop();
	if(DEBUG) printf("Non-local post-processing took: %0.3f\n", individual_timer.Elapsed());
	
	
	timer.Stop();
	*CONV_time += timer.Elapsed();
	//----------------------------------------------< Measurements end
	
	cufftDestroy(plan_forwardFFT_filters);
	cufftDestroy(plan_forwardFFT);
	cufftDestroy(plan_inverseFFT);
	return(0);
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

int GPU_CONV(float *h_input, float *h_output, float *h_filters_timedom, int signal_length, int filter_length, int nFilters, int nRuns, float h, double *execution_time){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount){
		checkCudaErrors(cudaSetDevice(device));
	}
	else {
		printf("ERROR! Selected device is not available\n");
		return(1);
	}
	cudaMemGetInfo(&free_mem,&total_mem);
	
	//---------> Measurements
	double transfer_in, transfer_out, CONV_cuFFT_time, total_CONV_cuFFT_time;
	transfer_in=0.0; transfer_out=0.0; CONV_cuFFT_time=0.0; total_CONV_cuFFT_time=0;
	GpuTimer timer;
	
	//---------> Convolution parameters
	int offset, useful_part_size, nConvolutions;
	Calculate_Convolution_Parameters(&offset, &useful_part_size, &nConvolutions, signal_length, filter_length, CONV_SIZE);
	if(DEBUG) printf("offset=%d; useful_part_size=%d; nConvolutions=%d;\n", offset, useful_part_size, nConvolutions);
	
	//---------> Defining variables and their sizes
	float *d_input_signal;
	float *d_input_signal_extended;
	float2 *d_input_signal_extended_freqdom;
	float2 *d_output_plane;
	float *d_output_plane_reduced;
	float *d_filters_timedom;
	float2 *d_filters_freqdom;
	size_t input_size                  = useful_part_size*nConvolutions + CONV_SIZE;
	size_t input_size_extended         = CONV_SIZE*nConvolutions;
	size_t input_size_extended_freqdom = ((CONV_SIZE>>1)+1)*nConvolutions;
	size_t output_size                 = ((CONV_SIZE>>1)+1)*nConvolutions*nFilters;
	size_t output_size_reduced         = (CONV_SIZE+1)*nConvolutions*nFilters;
	size_t filters_size                = ((CONV_SIZE>>1)+1)*nFilters;
	
	//---------> Checking memory
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required=(( 2*((float) input_size) + 3*((float) output_size) + ((float) filters_size))*sizeof(float2))/(1024.0*1024.0);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("DEBUG:\n");
	if(DEBUG) printf("    Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(DEBUG) printf("    d_input_signal:          %0.3f MB\n", ((float) input_size*sizeof(float))/(1024.0*1024.0) ); 
	if(DEBUG) printf("    d_input_signal_extended: %0.3f MB\n", ((float) input_size*sizeof(float))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_templates:             %0.3f MB\n", ((float) filters_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_output_plane:          %0.3f MB\n", ((float) output_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_output_plane_reduced:  %0.3f MB\n", ((float) output_size*sizeof(float))/(1024.0*1024.0) );
	if(memory_required>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(-3);}
	
	//------------------------------------------------------------------------------
	//---------> Memory allocation
	if (VERBOSE) printf("Device memory allocation...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input_signal,  sizeof(float)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output_plane, sizeof(float2)*output_size));
	checkCudaErrors(cudaMalloc((void **) &d_output_plane_reduced, sizeof(float)*output_size_reduced));
	checkCudaErrors(cudaMalloc((void **) &d_input_signal_extended, sizeof(float)*input_size_extended));
	checkCudaErrors(cudaMalloc((void **) &d_input_signal_extended_freqdom, sizeof(float2)*input_size_extended_freqdom));
	checkCudaErrors(cudaMalloc((void **) &d_filters_timedom, sizeof(float)*CONV_SIZE*nFilters));
	checkCudaErrors(cudaMalloc((void **) &d_filters_freqdom, sizeof(float2)*((CONV_SIZE>>1)+1)*nFilters));
	timer.Stop();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
	//------------------------------------------------------------------------------

	
	//------------------------------------------------------------------------------
	//---------> CONV calculation
	
	//-----> Copy chunk of input data to a device
	if (VERBOSE) printf("Transferring data into device memory...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMemcpy(d_input_signal, h_input, (input_size)*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filters_timedom, h_filters_timedom, sizeof(float)*CONV_SIZE*nFilters, cudaMemcpyHostToDevice));
	timer.Stop();
	transfer_in+=timer.Elapsed();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
    
	
	if (VERBOSE) printf("Calculating convolution via cuFFT...: \t\t");
	total_CONV_cuFFT_time = 0;
	int Conv_error = 0;
	for(int r=0; r<nRuns; r++){
		checkCudaErrors(cudaMemcpy(d_input_signal, h_input, (input_size)*sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_filters_timedom, h_filters_timedom, filters_size*sizeof(float), cudaMemcpyHostToDevice));
		Conv_error = CONV_cuFFT_benchmark(d_input_signal, 
							d_input_signal_extended, 
							d_input_signal_extended_freqdom, 
							d_output_plane, 
							d_filters_timedom, 
							d_filters_freqdom, 
							d_output_plane_reduced, 
							signal_length, filter_length, nFilters, h, &total_CONV_cuFFT_time);
		if(Conv_error!=0) {
			total_CONV_cuFFT_time = -1000.0;
			break;
		}
	}
	CONV_cuFFT_time = total_CONV_cuFFT_time/nRuns;
	*execution_time = CONV_cuFFT_time;
	if (VERBOSE) printf("done in %g ms.\n\n", CONV_cuFFT_time);
	
	
	//-----> Copy chunk of output data to host
	if (VERBOSE) printf("Transferring data to host...: \t\t");
	timer.Start();
	#ifdef POST_PROCESS
	checkCudaErrors(cudaMemcpy( h_output, d_output_plane, useful_part_size*nConvolutions*nFilters*sizeof(float), cudaMemcpyDeviceToHost));
	#else
	checkCudaErrors(cudaMemcpy( h_output, d_output_plane_reduced, useful_part_size*nConvolutions*nFilters*sizeof(float), cudaMemcpyDeviceToHost));	
	#endif
	timer.Stop();
	transfer_out+=timer.Elapsed();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
	//------------------------------------------------------------------------------
	

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_output_plane));
	checkCudaErrors(cudaFree(d_output_plane_reduced));
	checkCudaErrors(cudaFree(d_input_signal));
	checkCudaErrors(cudaFree(d_input_signal_extended));
	checkCudaErrors(cudaFree(d_input_signal_extended_freqdom));
	checkCudaErrors(cudaFree(d_filters_timedom));	
	checkCudaErrors(cudaFree(d_filters_freqdom));	
	
	if(Conv_error!=0) return(1);
	else return(0);
}
