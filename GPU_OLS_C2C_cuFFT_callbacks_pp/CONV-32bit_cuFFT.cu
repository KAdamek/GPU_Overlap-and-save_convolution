//********************************************************************************************
//* This is GPU implementation of a overlap-and-save method for calculating convolution using cuFFT with callbacks. 
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
__device__ void CB_RemoveAliasedSamples(void *dataOut, size_t external_pos, cufftComplex element, void *callerInfo, void *sharedPtr) {
    int *length = ((int*)callerInfo);
	int pos_within_fft = external_pos % CONV_SIZE;
	int fft = (int) (external_pos/CONV_SIZE);
	int useful_part = CONV_SIZE - length[0] + 1;
	int template_offset = length[0]/2;
	
	int pos = fft*useful_part + pos_within_fft - template_offset;
	if(pos_within_fft>=template_offset && pos_within_fft<(useful_part+template_offset)){
		((cufftComplex*)dataOut)[pos] = element;
	}
	
}

__device__ void CB_ConvolveFilters(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {
	ConvolveMetadata *l_cnvmd = ((ConvolveMetadata*) callerInfo);
	int nConvolutions = l_cnvmd->nConvolutions;
	int nFilters = l_cnvmd->nFilters;
	cufftComplex *filterPtr;
	filterPtr = l_cnvmd->filtersPtr;
	int pos_within_fft = offset % CONV_SIZE;
	int fft = (int) (offset/CONV_SIZE);
	
	for(int f=0; f<nFilters; f++){
		int filter_pos = f*CONV_SIZE + pos_within_fft;
		int store_pos  = f*nConvolutions*CONV_SIZE + fft*CONV_SIZE + pos_within_fft;
		
		cufftComplex temp;
		temp.x = filterPtr[filter_pos].x*element.x - filterPtr[filter_pos].y*element.y;
		temp.y = filterPtr[filter_pos].x*element.y + filterPtr[filter_pos].y*element.x;

		((cufftComplex*)dataOut)[store_pos] = temp;
	}
}

__device__ cufftCallbackStoreC d_backwardFFTCallbackPtr = CB_RemoveAliasedSamples;
__device__ cufftCallbackStoreC d_forwardFFTCallbackPtr = CB_ConvolveFilters;
//--------------------------- Callbacks -----------------------------

__global__ void prepare_signal(float2* d_extended_data, float2* d_input_signal, int signal_length, int convolution_size, int useful_part_size, int offset, int multiple) {
	int extpos = blockIdx.x*convolution_size;
	int sigpos = blockIdx.x*useful_part_size;
	
	if(blockIdx.x==0){ //leading part
		for(int i=0; i<multiple; i++){
			int pos = sigpos + i*1024 + threadIdx.x - offset;
			if( pos<0 ){
				d_extended_data[extpos + i*1024 + threadIdx.x].x = 0;
				d_extended_data[extpos + i*1024 + threadIdx.x].y = 0;
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
				d_extended_data[extpos + i*1024 + threadIdx.x].x = 0;
				d_extended_data[extpos + i*1024 + threadIdx.x].y = 0;
			}
		}	
	}
	else { //middle parts
		for(int i=0; i<multiple; i++){
			d_extended_data[extpos + i*1024 + threadIdx.x] = d_input_signal[sigpos - offset + i*1024 + threadIdx.x];
		}
	}
}


__global__ void post_process(float2 *d_output, float2* d_input, int nTimesamples, float h){
	float2 left, right;
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
	
	float2 f2temp;
	f2temp.x = (left.x - right.x)/(2.0f*h);
	f2temp.y = (left.y - right.y)/(2.0f*h);
	
	if(pos_x<nTimesamples) d_output[pos_y + pos_x] = f2temp;
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

void Calculate_Convolution_Parameters(int *offset, int *useful_part_size, int *nConvolutions, int signal_length, int filter_length, int convolution_size){ 
	*offset           = (int) (filter_length/2); // we assume that filter is centred around zero
	*useful_part_size = convolution_size - filter_length + 1;
	*nConvolutions    = (int) ((signal_length + (*useful_part_size) - 1)/(*useful_part_size));
}


int CONV_cuFFT_benchmark(float2 *d_input, float2 *d_extended_input, float2 *d_output, float2 *d_template, float2 *d_reduced_plane, int signal_length, int filter_length, int nFilters, float h, double *CONV_time){
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
	if (cufftPlan1d(&plan_forwardFFT_filters, CONV_SIZE, CUFFT_C2C, nFilters) != CUFFT_SUCCESS) {
		printf("CUDA API Error while creating cuFFT plan for filters!\n");
		return(1);
	}

	// Inverse FFT plan with aliased samples removal callback
	cufftCreate(&plan_inverseFFT);
    int signalSize = CONV_SIZE;
	size_t workSize;
    cufftMakePlanMany(plan_inverseFFT, 1, &signalSize, 0,0,0,0,0,0, CUFFT_C2C, nConvolutions*nFilters, &workSize);
	cufftCallbackStoreC h_backwardFFTCallbackPtr;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_backwardFFTCallbackPtr, d_backwardFFTCallbackPtr, sizeof(h_backwardFFTCallbackPtr)));
	
	int *d_length; // Making a copy of the filter length on the device for callback
	int h_length = filter_length;
	checkCudaErrors(cudaMalloc((void **) &d_length,  sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_length, &h_length, sizeof(int), cudaMemcpyHostToDevice));
	
	cuFFTResult = cufftXtSetCallback(plan_inverseFFT, (void **)&h_backwardFFTCallbackPtr, CUFFT_CB_ST_COMPLEX, (void **) &d_length);
	if( cuFFTResult != CUFFT_SUCCESS) {
		printf("CUDA API Error while creating cuFFT plan for output plane! Error: %d\n", cuFFTResult);
		return(1);
	}
	
	// forward FFT plan with frequency domain convolution callback
	ConvolveMetadata *d_conv_metadata;
	cudaMalloc(&d_conv_metadata, sizeof(ConvolveMetadata));
	cudaMemcpy(&d_conv_metadata->nConvolutions, &nConvolutions, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_conv_metadata->nFilters, &nFilters, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_conv_metadata->filtersPtr, &d_template, sizeof(FP), cudaMemcpyHostToDevice);
	
	cufftCreate(&plan_forwardFFT);
	cufftMakePlanMany(plan_forwardFFT, 1, &signalSize, 0,0,0,0,0,0, CUFFT_C2C, nConvolutions, &workSize);
	cufftCallbackStoreC h_forwardFFTCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&h_forwardFFTCallbackPtr, d_forwardFFTCallbackPtr, sizeof(h_forwardFFTCallbackPtr)));
	cuFFTResult = cufftXtSetCallback(plan_forwardFFT, (void **)&h_forwardFFTCallbackPtr, CUFFT_CB_ST_COMPLEX, (void **) &d_conv_metadata);
	if( cuFFTResult != CUFFT_SUCCESS) {
		printf("CUDA API Error while creating cuFFT plan for output plane! Error: %d\n", cuFFTResult);
		return(1);
	}	
	//---------<
	
	//----------------------------------------------> Measurements start
	timer.Start();
	
	//---------> Forward FFT of filters
	individual_timer.Start();
	cuFFTResult = cufftExecC2C(plan_forwardFFT_filters, (cufftComplex *)d_template, (cufftComplex *)d_template, CUFFT_FORWARD);
	if(cuFFTResult!=CUFFT_SUCCESS) {printf("cuFFT Error while performing cuFFT plan for filters!\n"); return(1);}
	individual_timer.Stop();
	if(DEBUG) printf("Filter preparation: %0.3f\n", individual_timer.Elapsed());
	
	//---------> Segment preparation
	individual_timer.Start();
	prepare_signal<<<OaS_gridSize,OaS_blockSize>>>(d_extended_input, d_input, signal_length, CONV_SIZE, useful_part_size, offset, OaS_multiplicator);
	cuFFTResult = cufftExecC2C(plan_forwardFFT, (cufftComplex *)d_extended_input, (cufftComplex *)d_output, CUFFT_FORWARD);
	if(cuFFTResult!=CUFFT_SUCCESS) {printf("cuFFT Error while performing cuFFT plan for input signal!\n"); return(1);}
	individual_timer.Stop();
	if(DEBUG) printf("Segment preparation: %0.3f\n", individual_timer.Elapsed());
	
	//---------> Inverse FFT of resulting plane
	individual_timer.Start();
	cuFFTResult = cufftExecC2C(plan_inverseFFT, (cufftComplex *)d_output, (cufftComplex *)d_reduced_plane, CUFFT_INVERSE);
	if(cuFFTResult!=CUFFT_SUCCESS) {printf("cuFFT Error while performing cuFFT plan for output plane!\n"); return(1);}
	individual_timer.Stop();
	if(DEBUG) printf("Inverse cuFFT of resulting plane: %0.3f\n", individual_timer.Elapsed());
	
	#ifdef POST_PROCESS
	//---------> Non-local post processing
	individual_timer.Start();
	int pp_nThreads = 256;
	int pp_nBlocks_x = (int) ((nConvolutions*useful_part_size + pp_nThreads - 1)/pp_nThreads);
	dim3 pp_grid(pp_nBlocks_x, nFilters, 1);
	dim3 pp_block(pp_nThreads, 1, 1);
	post_process<<<pp_grid,pp_block>>>(d_output, d_reduced_plane, nConvolutions*useful_part_size, h);
	individual_timer.Stop();
	if(DEBUG) printf("Non-local post-processing took: %0.3f\n", individual_timer.Elapsed());
	#endif
	
	timer.Stop();
	*CONV_time += timer.Elapsed();
	//----------------------------------------------< Measurements end
	
	cufftDestroy(plan_forwardFFT_filters);
	cufftDestroy(plan_forwardFFT);
	cufftDestroy(plan_inverseFFT);
	cudaFree(d_length);
	cudaFree(d_conv_metadata);
	return(0);
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

int GPU_CONV(float2 *h_input, float2 *h_output, float2 *h_filters, int signal_length, int filter_length, int nFilters, float h, int nRuns, double *execution_time){
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
	
	//---------> Defining variables and their sizes
	float2 *d_input_signal;
	float2 *d_input_signal_extended;
	float2 *d_output_plane;
	float2 *d_output_plane_reduced;
	float2 *d_filters;
	size_t input_size          = useful_part_size*nConvolutions + CONV_SIZE;
	size_t input_size_extended = CONV_SIZE*nConvolutions;
	size_t output_size         = CONV_SIZE*nConvolutions*nFilters;
	size_t output_size_reduced = useful_part_size*nConvolutions*nFilters;
	size_t template_size       = CONV_SIZE*nFilters;
	
	//---------> Checking memory
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required=(( 2*((float) input_size) + 3*((float) output_size) + ((float) template_size))*sizeof(float2))/(1024.0*1024.0);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("DEBUG:\n");
	if(DEBUG) printf("    Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(DEBUG) printf("    d_input_signal:          %0.3f MB\n", ((float) input_size*sizeof(float2))/(1024.0*1024.0) ); 
	if(DEBUG) printf("    d_input_signal_extended: %0.3f MB\n", ((float) input_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_filters:               %0.3f MB\n", ((float) template_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_output_plane:          %0.3f MB\n", ((float) output_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_output_plane_reduced:  %0.3f MB\n", ((float) output_size*sizeof(float2))/(1024.0*1024.0) );
	if(memory_required>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(-3);}
	
	//------------------------------------------------------------------------------
	//---------> Memory allocation
	if (VERBOSE) printf("Device memory allocation...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input_signal,  sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output_plane, sizeof(float2)*output_size));
	checkCudaErrors(cudaMalloc((void **) &d_output_plane_reduced, sizeof(float2)*output_size_reduced));
	checkCudaErrors(cudaMalloc((void **) &d_input_signal_extended, sizeof(float2)*input_size_extended));
	checkCudaErrors(cudaMalloc((void **) &d_filters, sizeof(float2)*template_size));
	checkCudaErrors(cudaMemset((void*) &d_input_signal[(nConvolutions-1)*useful_part_size], 0, CONV_SIZE*sizeof(float2)));
	timer.Stop();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
	//------------------------------------------------------------------------------

	
	//------------------------------------------------------------------------------
	//---------> CONV calculation
	
	//-----> Copy chunk of input data to a device
	if (VERBOSE) printf("Transferring data into device memory...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMemcpy(d_input_signal, h_input, (input_size)*sizeof(float2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filters, h_filters, template_size*sizeof(float2), cudaMemcpyHostToDevice));
	timer.Stop();
	transfer_in+=timer.Elapsed();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
    
	
	if (VERBOSE) printf("Calculating convolution via cuFFT...: \t\t");
	total_CONV_cuFFT_time = 0;
	int Conv_error = 0;
	for(int r=0; r<nRuns; r++){
		checkCudaErrors(cudaMemcpy(d_input_signal, h_input, (input_size-offset)*sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_filters, h_filters, template_size*sizeof(float2), cudaMemcpyHostToDevice));
		Conv_error = CONV_cuFFT_benchmark(d_input_signal, d_input_signal_extended, d_output_plane, d_filters, d_output_plane_reduced, signal_length, filter_length, nFilters, h, &total_CONV_cuFFT_time);	
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
	checkCudaErrors(cudaMemcpy( h_output, d_output_plane, output_size_reduced*sizeof(float2), cudaMemcpyDeviceToHost));
	#else
	checkCudaErrors(cudaMemcpy( h_output, d_output_plane_reduced, output_size_reduced*sizeof(float2), cudaMemcpyDeviceToHost));
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
	checkCudaErrors(cudaFree(d_filters));	
	
	return(0);
}
