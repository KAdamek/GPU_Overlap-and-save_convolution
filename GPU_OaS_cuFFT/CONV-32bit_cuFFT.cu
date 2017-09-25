//********************************************************************************************
//* This is GPU implementation of a Overlap-and-save method for calculating convolution. 
//* Copyright (C) 2017  Adámek Karel
//* 
//* Authors: Karel Adamek ( ORCID:0000-0003-2797-0595; https://github.com/KAdamek ), Wesley Armour ( ORCID:0000-0003-1756-3064 ), Sofia Dimoudi 
//********************************************************************************************


#include <iostream>
#include <fstream>
#include <iomanip> 

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"

#include "params.h"

#define WARP 32

int device=0;

__global__ void prepare_signal(float2* d_extended_data, float2* d_input_signal, int convolution_size, int useful_part_size, int offset, int multiple) {
	int extpos = blockIdx.x*convolution_size;
	int sigpos = blockIdx.x*useful_part_size;
	
	if(blockIdx.x==0){ //leading part
		for(int i=0; i<multiple; i++){
			if( (threadIdx.x+i*1024)<offset ){
				d_extended_data[extpos + i*1024 + threadIdx.x].x = 0;
				d_extended_data[extpos + i*1024 + threadIdx.x].y = 0;
			}
			else {
				d_extended_data[extpos + i*1024 + threadIdx.x] = d_input_signal[sigpos + i*1024 + threadIdx.x - offset];
			}
		}
	}
	else { //middle parts
		for(int i=0; i<multiple; i++){
			d_extended_data[extpos + i*1024 + threadIdx.x] = d_input_signal[sigpos - offset + i*1024 + threadIdx.x];
		}
	}
	
	// Checked!
}

__global__ void reduce_plane(float2* d_extended_plane, float2* d_reduced_plane, int useful_part_size, int offset, int nConvolutions, int multiplicator){
	for (int i = 0; i < multiplicator; i++){
		if ( (threadIdx.x+i*1024) < useful_part_size){
			d_reduced_plane[(blockIdx.y*useful_part_size*nConvolutions) + i*1024  + blockIdx.x*useful_part_size + threadIdx.x] = d_extended_plane[(blockIdx.y*CONV_SIZE*nConvolutions) + i*1024  + blockIdx.x*CONV_SIZE + offset + threadIdx.x];
		}
	}
}

__global__ void reduce_plane_power(float2* d_extended_plane, float* d_reduced_plane, int useful_part_size, int offset, int nConvolutions, int multiplicator){
	float2 f2temp;
	
	for (int i = 0; i < multiplicator; i++){
		if ( (threadIdx.x+i*1024) < useful_part_size){
			f2temp = d_extended_plane[(blockIdx.y*CONV_SIZE*nConvolutions) + i*1024  + blockIdx.x*CONV_SIZE + offset + threadIdx.x];
			d_reduced_plane[(blockIdx.y*useful_part_size*nConvolutions) + i*1024  + blockIdx.x*useful_part_size + threadIdx.x] = f2temp.x*f2temp.x + f2temp.y*f2temp.y;
		}
	}
}

// Even newer one
__global__ void CONV_GPU(float2 const* __restrict__ d_input, float2* d_output, float2 const* __restrict__ d_templates, int nConvolutions, int nTemplates) {
	float2 r_input;
	float2 r_templates;
	float2 ftemp;
	
	r_input = d_input[CONV_SIZE*blockIdx.x + blockDim.x*blockIdx.z + threadIdx.x];
	
	for(int t=0; t<nTemplates; t++){
		r_templates = __ldg(&d_templates[CONV_SIZE*nTemplates*blockIdx.y + t*CONV_SIZE + blockDim.x*blockIdx.z + threadIdx.x]);
		
		ftemp.x=r_templates.x*r_input.x - r_templates.y*r_input.y;
		ftemp.y=r_templates.x*r_input.y + r_templates.y*r_input.x;
		
		d_output[CONV_SIZE*blockIdx.x + blockDim.x*blockIdx.z + nConvolutions*CONV_SIZE*nTemplates*blockIdx.y + t*CONV_SIZE*nConvolutions + threadIdx.x] = ftemp;
	}
}


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

void CONV_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}


void Extend_and_fft_input_signal_manually(float2 *d_input_extended, float2 *d_input, int useful_part_size, int offset, int nConvolutions, double *time){
	GpuTimer timer;
	int OaS_thread_size, OaS_multiplicator;
	cufftHandle plan_input;
	
	if(CONV_SIZE>1024) {
		OaS_thread_size=1024;
		OaS_multiplicator=CONV_SIZE/1024;
	}
	else  {
		OaS_multiplicator=0;
		OaS_thread_size=CONV_SIZE;
	}
	dim3 OaS_gridSize(nConvolutions, 1, 1);
	dim3 OaS_blockSize(OaS_thread_size, 1, 1);
	
	// cuFFT plan
	if ( cufftPlan1d(&plan_input, CONV_SIZE, CUFFT_C2C, nConvolutions) != CUFFT_SUCCESS) printf("CUFFT error");
	
	timer.Start();
	
	// ---> Overlap and save
	prepare_signal<<<OaS_gridSize,OaS_blockSize>>>(d_input_extended, d_input, CONV_SIZE, useful_part_size, offset, OaS_multiplicator);
	
	// ---> Forward cuFFT of prepared signal
	cufftExecC2C(plan_input, (cufftComplex *)d_input_extended, (cufftComplex *)d_input_extended, CUFFT_FORWARD);
	
	timer.Stop();
	*time = timer.Elapsed();
	
	cufftDestroy(plan_input);
}


void Extend_and_fft_input_signal_allcufft(float2 *d_input_extended, float2 *d_input, int useful_part_size, int offset, int nConvolutions, double *time){
	GpuTimer timer;
	cufftHandle plan_input;
	
	// cuFFT plan
	int rank=1;
	int N=CONV_SIZE;
	int istride=1, idist=useful_part_size;
	int ostride=1, odist=CONV_SIZE;
	if ( cufftPlanMany(&plan_input, rank, &N, &N, istride, idist, &N, ostride, odist, CUFFT_C2C, nConvolutions) != CUFFT_SUCCESS) {
		printf("CUFFT error");
	}
	
	timer.Start();
	
	// ---> Forward cuFFT of prepared signal
	cufftExecC2C(plan_input, (cufftComplex *)d_input, (cufftComplex *)d_input_extended, CUFFT_FORWARD);
	
	timer.Stop();
	*time = timer.Elapsed();
	
	cufftDestroy(plan_input);
}


int CONV_cuFFT_benchmark(float2 *d_input, float2 *d_extended_input, float2 *d_output, float2 *d_template, float2 *d_reduced_plane, int useful_part_size, int offset, int nConvolutions, int nTemplates, int nThreads, double *CONV_time){
	GpuTimer timer;
	
	// ----------------------------------------------->
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x = nConvolutions;
	int nCUDAblocks_y = 1;
	int nCUDAblocks_z = CONV_SIZE/nThreads;
	if(CONV_SIZE%nThreads!=0) {
		printf("Error: CONV_SIZE is not divisible by nThreads\n");
		return(1);
	}
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, nCUDAblocks_z);
	dim3 blockSize(nThreads, 1, 1);
	
	if(DEBUG){
		printf("\n");
		printf("nConvolutions:%d; nTemplates:%d; CONV_SIZE:%d; nThreads:%d;\n", nConvolutions, nTemplates, CONV_SIZE, nThreads);
		printf("Convolution gridsize:  %d; %d; %d;\n", gridSize.x, gridSize.y, gridSize.z);
		printf("Convolution blocksize: %d; %d; %d;\n", blockSize.x, blockSize.y, blockSize.z);
	}
	
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
	
	int RP_thread_size, RP_multiplicator;
	if(CONV_SIZE>1024) {
		RP_thread_size=1024;
		RP_multiplicator=CONV_SIZE/1024;
	}
	else {
		RP_multiplicator=1;
		RP_thread_size=CONV_SIZE;
	}
	dim3 RP_gridSize(nConvolutions, nTemplates, 1);
	dim3 RP_blockSize(RP_thread_size, 1, 1);
	// ----------------------------------------------->
	
	cufftHandle plan_input, plan_plane, plan_templates;
	cufftResult error;

	error = cufftPlan1d(&plan_plane, CONV_SIZE, CUFFT_C2C, nConvolutions*nTemplates);
	if (CUFFT_SUCCESS != error){
		printf("CUFFT error: %d", error);
	}
	
	// ----------------------------------------------->
	//---------> DFT of templates
	if (cufftPlan1d(&plan_templates, CONV_SIZE, CUFFT_C2C, nTemplates) != CUFFT_SUCCESS) printf("CUFFT error: %d", error);
	cufftExecC2C(plan_templates, (cufftComplex *)d_template, (cufftComplex *)d_template, CUFFT_FORWARD);
	cufftDestroy(plan_templates);
	// ----------------------------------------------->
	
	if ( cufftPlan1d(&plan_input, CONV_SIZE, CUFFT_C2C, nConvolutions) != CUFFT_SUCCESS) printf("CUFFT error: %d", error);
	
	// ----------------------------------------------->
	// --------> Measured part (Convolution)
	timer.Start();
	// ---> Overlap and save
	prepare_signal<<<OaS_gridSize,OaS_blockSize>>>(d_extended_input, d_input, CONV_SIZE, useful_part_size, offset, OaS_multiplicator);
	timer.Stop();
	*CONV_time += timer.Elapsed();
	if(DEBUG) printf("Overlap_and_save: %0.3f\n", timer.Elapsed());
	
	
	timer.Start();
	// ---> Forward cuFFT of prepared signal
	cufftExecC2C(plan_input, (cufftComplex *)d_extended_input, (cufftComplex *)d_extended_input, CUFFT_FORWARD);
	timer.Stop();
	*CONV_time += timer.Elapsed();
	if(DEBUG) printf("cuFFT of the signal: %0.3f\n", timer.Elapsed());


	timer.Start();	
	// ---> Convolution
	CONV_init();
	CONV_GPU<<<gridSize,blockSize>>>(d_extended_input, d_output, d_template, nConvolutions, nTemplates);
	timer.Stop();
	*CONV_time += timer.Elapsed();
	if(DEBUG) printf("Convolution: %0.3f\n", timer.Elapsed());
	

	timer.Start();
	// ---> Inverse cuFFT of resulting plane
	cufftExecC2C(plan_plane, (cufftComplex *)d_output, (cufftComplex *)d_output, CUFFT_INVERSE);
	timer.Stop();
	*CONV_time += timer.Elapsed();
	if(DEBUG) printf("IcuFFT: %0.3f\n", timer.Elapsed());
	
	timer.Start();
	// ---> Selecting correct parts of the resulting signal (Sofia)
	reduce_plane<<<RP_gridSize,RP_blockSize>>>(d_output, d_reduced_plane, useful_part_size, offset, nConvolutions, RP_multiplicator);
	timer.Stop();
	*CONV_time += timer.Elapsed();
	if(DEBUG) printf("reduction: %0.3f\n", timer.Elapsed());
	
	cufftDestroy(plan_input);
	cufftDestroy(plan_plane);
	return(0);
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

int GPU_CONV(float2 *h_input, float2 *h_output, float2 *h_templates, int useful_part_size, int offset, int nConvolutions, int nTemplates, int nRuns, double *execution_time){
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
	
	//---------> Defining variables and their sizes
	float2 *d_input_signal;
	float2 *d_input_signal_extended;
	float2 *d_output_plane;
	float2 *d_output_plane_reduced;
	float2 *d_templates;
	size_t input_size          = useful_part_size*nConvolutions + CONV_SIZE - useful_part_size;
	size_t input_size_extended = CONV_SIZE*nConvolutions;
	size_t output_size         = CONV_SIZE*nConvolutions*nTemplates;
	size_t output_size_reduced = useful_part_size*nConvolutions*nTemplates;
	size_t template_size       = CONV_SIZE*nTemplates;
	
	//---------> Checking memory
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required=(( 2*((float) input_size) + 3*((float) output_size) + ((float) template_size))*sizeof(float2))/(1024.0*1024.0);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("DEBUG:\n");
	if(DEBUG) printf("    Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(DEBUG) printf("    d_input_signal:          %0.3f MB\n", ((float) input_size*sizeof(float2))/(1024.0*1024.0) ); 
	if(DEBUG) printf("    d_input_signal_extended: %0.3f MB\n", ((float) input_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_templates:             %0.3f MB\n", ((float) template_size*sizeof(float2))/(1024.0*1024.0) );
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
	checkCudaErrors(cudaMalloc((void **) &d_templates, sizeof(float2)*template_size));
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
	checkCudaErrors(cudaMemcpy(d_templates, h_templates, template_size*sizeof(float2), cudaMemcpyHostToDevice));
	timer.Stop();
	transfer_in+=timer.Elapsed();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
    
	
	if (VERBOSE) printf("Calculating convolution via cuFFT...: \t\t");
	total_CONV_cuFFT_time = 0;
	int nThreads = 512;
	for(int r=0; r<nRuns; r++){
		checkCudaErrors(cudaMemcpy(d_input_signal, h_input, (input_size-offset)*sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_templates, h_templates, template_size*sizeof(float2), cudaMemcpyHostToDevice));
		CONV_cuFFT_benchmark(d_input_signal, d_input_signal_extended, d_output_plane, d_templates, d_output_plane_reduced, useful_part_size, offset, nConvolutions, nTemplates, nThreads, &total_CONV_cuFFT_time);		
	}
	CONV_cuFFT_time = total_CONV_cuFFT_time/nRuns;
	*execution_time = CONV_cuFFT_time;
	if (VERBOSE) printf("done with %d threads in %g ms.\n\n", nThreads, CONV_cuFFT_time);
	
	
	//-----> Copy chunk of output data to host
	if (VERBOSE) printf("Transferring data to host...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMemcpy( h_output, d_output_plane_reduced, output_size_reduced*sizeof(float2), cudaMemcpyDeviceToHost));
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
	checkCudaErrors(cudaFree(d_templates));	
	
	return(0);
}
