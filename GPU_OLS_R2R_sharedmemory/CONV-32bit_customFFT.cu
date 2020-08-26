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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"
#include "params.h"

#define WARP 32

class FFT_ConstParams {
public:
	static const int fft_exp = -1;
	static const int fft_length = -1;
	static const int fft_half = -1;
	static const int warp = 32;
};

class FFT_256 : public FFT_ConstParams {
	public:
	static const int fft_exp = 8;
	static const int fft_quarter = 64;
	static const int fft_half = 128;
	static const int fft_threequarters = 192;
	static const int fft_length = 256;
};

class FFT_512 : public FFT_ConstParams {
	public:
	static const int fft_exp = 9;
	static const int fft_quarter = 128;
	static const int fft_half = 256;
	static const int fft_threequarters = 384;
	static const int fft_length = 512;
};

class FFT_1024 : public FFT_ConstParams {
	public:
	static const int fft_exp = 10;
	static const int fft_quarter = 256;
	static const int fft_half = 512;
	static const int fft_threequarters = 768;
	static const int fft_length = 1024;
};

class FFT_2048 : public FFT_ConstParams {
	public:
	static const int fft_exp = 11;
	static const int fft_quarter = 512;
	static const int fft_half = 1024;
	static const int fft_threequarters = 1536;
	static const int fft_length = 2048;
};

class FFT_4096 : public FFT_ConstParams {
	public:
	static const int fft_exp = 12;
	static const int fft_quarter = 1024;
	static const int fft_half = 2048;
	static const int fft_threequarters = 3072;
	static const int fft_length = 4096;
};

class FFT_ConstDirection {
public:
	static const int fft_direction = -1;
};

class FFT_forward : public FFT_ConstDirection {
public:
	static const int fft_direction = 0;
};

class FFT_inverse : public FFT_ConstDirection {
public:
	static const int fft_direction = 1;
};



__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	sincosf(-6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ float2 Get_W_value_inverse(int N, int m){
	float2 ctemp;
	sincosf(6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ float shfl(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_sync(0xffffffff, (*value), par));
	#else
		return(__shfl((*value), par));
	#endif
}


template<class const_params, class const_direction>
__device__ void do_FFT_Stockham_C2C(float2 *s_input){ // in-place
	float2 SA_DFT_value_even, SA_DFT_value_odd;
	float2 SB_DFT_value_even, SB_DFT_value_odd; 
	float2 SA_ftemp2, SA_ftemp;
	float2 SB_ftemp2, SB_ftemp;
	float2 W;
	
	int r, j, k, PoT, PoTm1;

	//-----> FFT
	//--> 
	
	PoT=1;
	PoTm1=0;
	//------------------------------------------------------------
	// First iteration
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x;
		
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + SA_ftemp2.x;
		SA_DFT_value_even.y = SA_ftemp.y + SA_ftemp2.y;
		SA_DFT_value_odd.x  = SA_ftemp.x - SA_ftemp2.x;
		SA_DFT_value_odd.y  = SA_ftemp.y - SA_ftemp2.y;
		
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + SB_ftemp2.x;
		SB_DFT_value_even.y = SB_ftemp.y + SB_ftemp2.y;
		SB_DFT_value_odd.x  = SB_ftemp.x - SB_ftemp2.x;
		SB_DFT_value_odd.y  = SB_ftemp.y - SB_ftemp2.y;
		
		__syncthreads();
		s_input[j*PoT]         = SA_DFT_value_even;
		s_input[j*PoT + PoTm1] = SA_DFT_value_odd;
		s_input[j*PoT + const_params::fft_half]         = SB_DFT_value_even;
		s_input[j*PoT + PoTm1 + const_params::fft_half] = SB_DFT_value_odd;
		__syncthreads();
	// First iteration
	//------------------------------------------------------------
	
	for(r=2;r<6;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		if(const_direction::fft_direction==0) {
			W = Get_W_value(PoT,k);
		}
		else {
			W = Get_W_value_inverse(PoT,k);
		}
		
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
		SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
		SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
		SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
		
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + W.x*SB_ftemp2.x - W.y*SB_ftemp2.y;
		SB_DFT_value_even.y = SB_ftemp.y + W.x*SB_ftemp2.y + W.y*SB_ftemp2.x;
		SB_DFT_value_odd.x  = SB_ftemp.x - W.x*SB_ftemp2.x + W.y*SB_ftemp2.y;
		SB_DFT_value_odd.y  = SB_ftemp.y - W.x*SB_ftemp2.y - W.y*SB_ftemp2.x;
		
		__syncthreads();
		s_input[j*PoT + k]         = SA_DFT_value_even;
		s_input[j*PoT + k + PoTm1] = SA_DFT_value_odd;
		s_input[j*PoT + k + const_params::fft_half]         = SB_DFT_value_even;
		s_input[j*PoT + k + PoTm1 + const_params::fft_half] = SB_DFT_value_odd;
		__syncthreads();
	}
	
	
	for(r=6;r<=const_params::fft_exp-1;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		if(const_direction::fft_direction==0) {
			W = Get_W_value(PoT,k);
		}
		else {
			W = Get_W_value_inverse(PoT,k);
		}
		
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
		SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
		SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
		SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
		
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + W.x*SB_ftemp2.x - W.y*SB_ftemp2.y;
		SB_DFT_value_even.y = SB_ftemp.y + W.x*SB_ftemp2.y + W.y*SB_ftemp2.x;
		SB_DFT_value_odd.x  = SB_ftemp.x - W.x*SB_ftemp2.x + W.y*SB_ftemp2.y;
		SB_DFT_value_odd.y  = SB_ftemp.y - W.x*SB_ftemp2.y - W.y*SB_ftemp2.x;
		
		__syncthreads();
		s_input[j*PoT + k]         = SA_DFT_value_even;
		s_input[j*PoT + k + PoTm1] = SA_DFT_value_odd;
		s_input[j*PoT + k + const_params::fft_half]         = SB_DFT_value_even;
		s_input[j*PoT + k + PoTm1 + const_params::fft_half] = SB_DFT_value_odd;
		__syncthreads();
	}
	// Last iteration
	{
		j = 0;
		k = threadIdx.x;
		
		float2 WA;
		if(const_direction::fft_direction==0) {
			WA = Get_W_value(const_params::fft_length, threadIdx.x);
		}
		else {
			WA = Get_W_value_inverse(const_params::fft_length, threadIdx.x);
		}
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + WA.x*SA_ftemp2.x - WA.y*SA_ftemp2.y;
		SA_DFT_value_even.y = SA_ftemp.y + WA.x*SA_ftemp2.y + WA.y*SA_ftemp2.x;
		SA_DFT_value_odd.x  = SA_ftemp.x - WA.x*SA_ftemp2.x + WA.y*SA_ftemp2.y;
		SA_DFT_value_odd.y  = SA_ftemp.y - WA.x*SA_ftemp2.y - WA.y*SA_ftemp2.x;
		
		float2 WB;
		if(const_direction::fft_direction==0) {
			WB = Get_W_value(const_params::fft_length, threadIdx.x + const_params::fft_quarter);
		}
		else {
			WB = Get_W_value_inverse(const_params::fft_length, threadIdx.x + const_params::fft_quarter);
		}
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + WB.x*SB_ftemp2.x - WB.y*SB_ftemp2.y;
		SB_DFT_value_even.y = SB_ftemp.y + WB.x*SB_ftemp2.y + WB.y*SB_ftemp2.x;
		SB_DFT_value_odd.x  = SB_ftemp.x - WB.x*SB_ftemp2.x + WB.y*SB_ftemp2.y;
		SB_DFT_value_odd.y  = SB_ftemp.y - WB.x*SB_ftemp2.y - WB.y*SB_ftemp2.x;
		
		__syncthreads();
		s_input[threadIdx.x]                          = SA_DFT_value_even;
		s_input[threadIdx.x + const_params::fft_half] = SA_DFT_value_odd;
		s_input[threadIdx.x + const_params::fft_quarter]       = SB_DFT_value_even;
		s_input[threadIdx.x + const_params::fft_threequarters] = SB_DFT_value_odd;
		__syncthreads();
	}
	//-------> END
	
	__syncthreads();
}


template<class const_params, class const_direction>
__device__ void do_FFT_Stockham_R2C_C2R(float2 *s_input){
	float2 one_half;
	if(const_direction::fft_direction==0) {
		one_half.x = 0.5f;
		one_half.y = -0.5f;
		do_FFT_Stockham_C2C<const_params,FFT_forward>(s_input);
	}
	else {
		one_half.x = -0.5f;
		one_half.y = 0.5f;
		if(threadIdx.x==0) {
			float2 L, F;
			L = s_input[0];
			F.x = 0.5f*(L.x + L.y);
			F.y = 0.5f*(L.x - L.y); 
			s_input[0] = F;			
		}
	}

	float2 SA_A, SA_B, SA_W, SA_H1, SA_H2, SA_F1, SA_F2;
	SA_A = s_input[threadIdx.x + 1];
	SA_B = s_input[const_params::fft_length - threadIdx.x - 1];
	SA_H1.x =       0.5f*(SA_A.x + SA_B.x);
	SA_H1.y =       0.5f*(SA_A.y - SA_B.y);
	SA_H2.x = one_half.x*(SA_A.y + SA_B.y);
	SA_H2.y = one_half.y*(SA_A.x - SA_B.x);
	if(const_direction::fft_direction==0) {
		SA_W = Get_W_value(const_params::fft_length*2, threadIdx.x + 1);
	}
	else {
		SA_W = Get_W_value_inverse(const_params::fft_length*2, threadIdx.x + 1);
	}
	SA_F1.x =  SA_H1.x + SA_W.x*SA_H2.x - SA_W.y*SA_H2.y;
	SA_F1.y =  SA_H1.y + SA_W.x*SA_H2.y + SA_W.y*SA_H2.x;
	SA_F2.x =  SA_H1.x - SA_W.x*SA_H2.x + SA_W.y*SA_H2.y;
	SA_F2.y = -SA_H1.y + SA_W.x*SA_H2.y + SA_W.y*SA_H2.x;
	s_input[threadIdx.x + 1] = SA_F1;
	s_input[const_params::fft_length - threadIdx.x - 1] = SA_F2;
	

	float2 SB_A, SB_B, SB_W, SB_H1, SB_H2, SB_F1, SB_F2;
	SB_A = s_input[threadIdx.x + 1 + const_params::fft_quarter];
	SB_B = s_input[const_params::fft_length - threadIdx.x - 1 - const_params::fft_quarter];
	SB_H1.x =       0.5f*(SB_A.x + SB_B.x);
	SB_H1.y =       0.5f*(SB_A.y - SB_B.y);
	SB_H2.x = one_half.x*(SB_A.y + SB_B.y);
	SB_H2.y = one_half.y*(SB_A.x - SB_B.x);
	if(const_direction::fft_direction==0) {
		SB_W = Get_W_value(const_params::fft_length*2, threadIdx.x + 1 + const_params::fft_quarter);
	}
	else {
		SB_W = Get_W_value_inverse(const_params::fft_length*2, threadIdx.x + 1 + const_params::fft_quarter);
	}
	SB_F1.x =  SB_H1.x + SB_W.x*SB_H2.x - SB_W.y*SB_H2.y;
	SB_F1.y =  SB_H1.y + SB_W.x*SB_H2.y + SB_W.y*SB_H2.x;
	SB_F2.x =  SB_H1.x - SB_W.x*SB_H2.x + SB_W.y*SB_H2.y;
	SB_F2.y = -SB_H1.y + SB_W.x*SB_H2.y + SB_W.y*SB_H2.x;
	s_input[threadIdx.x + 1 + const_params::fft_quarter] = SB_F1;
	s_input[const_params::fft_length - threadIdx.x - 1 - const_params::fft_quarter] = SB_F2;

	__syncthreads();
	
	if(const_direction::fft_direction==0) {
		if(threadIdx.x==0) {
			float2 L, F;
			L = s_input[0];
			F.x = L.x + L.y;
			F.y = L.x - L.y;
			s_input[0] = F;
		}
	}
	else {
		do_FFT_Stockham_C2C<const_params,FFT_inverse>(s_input);
	}
}


template<class const_params, class const_direction>
__global__ void FFT_GPU_R2C_C2R_external(float2 *d_input, float2* d_output) {
	__shared__ float2 s_input[const_params::fft_length + 1];
	s_input[threadIdx.x]                                   = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_quarter]       = d_input[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_half]          = d_input[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_threequarters] = d_input[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length];
	__syncthreads();
	
	do_FFT_Stockham_R2C_C2R<const_params, const_direction>(s_input);

	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                   = s_input[threadIdx.x];
	d_output[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length]       = s_input[threadIdx.x + const_params::fft_quarter];
	d_output[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length]          = s_input[threadIdx.x + const_params::fft_half];
	d_output[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length] = s_input[threadIdx.x + const_params::fft_threequarters];
}


template<class const_params>
__device__ __inline__ void prepare_signal_4elem(float* s_signal, float const* __restrict__ d_input_signal, int signal_length, int useful_part_size, int offset) {
	int pos = blockIdx.x*useful_part_size;
	
	s_signal[threadIdx.x]                                   = 0;
	s_signal[threadIdx.x + const_params::fft_quarter]       = 0;
	s_signal[threadIdx.x + const_params::fft_half]          = 0;
	s_signal[threadIdx.x + const_params::fft_threequarters] = 0;
	
	pos = blockIdx.x*useful_part_size + threadIdx.x - offset;
	if( pos>=0 && pos<signal_length ) 
		s_signal[threadIdx.x] = d_input_signal[pos];
	
	if( (pos + const_params::fft_quarter)>=0 && (pos + const_params::fft_quarter)<signal_length ) 
		s_signal[threadIdx.x + const_params::fft_quarter] = d_input_signal[pos + const_params::fft_quarter];
		
	if( (pos + const_params::fft_half)>=0 && (pos + const_params::fft_half)<signal_length ) 	
		s_signal[threadIdx.x + const_params::fft_half] = d_input_signal[pos + const_params::fft_half];
	
	if( (pos + const_params::fft_threequarters)>=0 && (pos + const_params::fft_threequarters)<signal_length ) 
		s_signal[threadIdx.x + const_params::fft_threequarters] = d_input_signal[pos + const_params::fft_threequarters];
	
	
	s_signal[threadIdx.x + const_params::fft_length]                                   = 0;
	s_signal[threadIdx.x + const_params::fft_length + const_params::fft_quarter]       = 0;
	s_signal[threadIdx.x + const_params::fft_length + const_params::fft_half]          = 0;
	s_signal[threadIdx.x + const_params::fft_length + const_params::fft_threequarters] = 0;
	
	pos = blockIdx.x*useful_part_size + threadIdx.x + const_params::fft_length - offset;
	if( pos>=0 && pos<signal_length ) 
		s_signal[threadIdx.x + const_params::fft_length] = d_input_signal[pos];
	
	if( (pos + const_params::fft_quarter)>=0 && (pos + const_params::fft_quarter)<signal_length ) 
		s_signal[threadIdx.x + const_params::fft_length + const_params::fft_quarter] = d_input_signal[pos + const_params::fft_quarter];
		
	if( (pos + const_params::fft_half)>=0 && (pos + const_params::fft_half)<signal_length ) 	
		s_signal[threadIdx.x + const_params::fft_length + const_params::fft_half] = d_input_signal[pos + const_params::fft_half];
	
	if( (pos + const_params::fft_threequarters)>=0 && (pos + const_params::fft_threequarters)<signal_length ) 
		s_signal[threadIdx.x + const_params::fft_length + const_params::fft_threequarters] = d_input_signal[pos + const_params::fft_threequarters];
}

template<class const_params>
__inline__ __device__ void GPU_conv_assign_signal(float2 *r_signal, float2* s_input){
	//*r_signal0 = s_input_1[threadIdx.x];
	//*r_signal1 = s_input_1[threadIdx.x + const_params::fft_quarter];
	//*r_signal2 = s_input_1[threadIdx.x + const_params::fft_half];
	//*r_signal3 = s_input_1[threadIdx.x + const_params::fft_threequarters];
	r_signal[0] = s_input[threadIdx.x];
	r_signal[1] = s_input[threadIdx.x + const_params::fft_quarter];
	r_signal[2] = s_input[threadIdx.x + const_params::fft_half];
	r_signal[3] = s_input[threadIdx.x + const_params::fft_threequarters];
}

template<class const_params>
__inline__ __device__  void GPU_conv_complex_multiplication(int filter_id, float2 *r_signal, float2 const* __restrict__ d_filters, float2 *s_input) {
	float2 r_filter_1[4];
	int pos;
	
	pos = filter_id*const_params::fft_length + threadIdx.x;
	r_filter_1[0]=__ldg(&d_filters[pos]);
	r_filter_1[1]=__ldg(&d_filters[pos + const_params::fft_quarter]);
	r_filter_1[2]=__ldg(&d_filters[pos + const_params::fft_half]);
	r_filter_1[3]=__ldg(&d_filters[pos + const_params::fft_threequarters]);
	
	// Convolution (complex multiplication)
	s_input[threadIdx.x].x                                   = r_filter_1[0].x*r_signal[0].x - r_filter_1[0].y*r_signal[0].y;
	s_input[threadIdx.x].y                                   = r_filter_1[0].x*r_signal[0].y + r_filter_1[0].y*r_signal[0].x;
	s_input[threadIdx.x + const_params::fft_quarter].x       = r_filter_1[1].x*r_signal[1].x - r_filter_1[1].y*r_signal[1].y;
	s_input[threadIdx.x + const_params::fft_quarter].y       = r_filter_1[1].x*r_signal[1].y + r_filter_1[1].y*r_signal[1].x;
	s_input[threadIdx.x + const_params::fft_half].x          = r_filter_1[2].x*r_signal[2].x - r_filter_1[2].y*r_signal[2].y;
	s_input[threadIdx.x + const_params::fft_half].y          = r_filter_1[2].x*r_signal[2].y + r_filter_1[2].y*r_signal[2].x;
	s_input[threadIdx.x + const_params::fft_threequarters].x = r_filter_1[3].x*r_signal[3].x - r_filter_1[3].y*r_signal[3].y;
	s_input[threadIdx.x + const_params::fft_threequarters].y = r_filter_1[3].x*r_signal[3].y + r_filter_1[3].y*r_signal[3].x;
	
	if(threadIdx.x==0){
		s_input[threadIdx.x].x = r_filter_1[0].x*r_signal[0].x;
		s_input[threadIdx.x].y = r_filter_1[0].y*r_signal[0].y;
	}
}

template<class const_params>
__global__ void k_GPU_conv_OLS_R2R_via_customFFT(
			float const* __restrict__ d_input_signal, 
			float *d_output_plane, 
			float2 const* __restrict__ d_filters_freqdom, 
			int signal_length, 
			int useful_part_size, 
			int offset, 
			int nConvolutions, 
			int nFilters) {
	__shared__ float s_input_1[2*const_params::fft_length];
	float2 r_signal[4];
	int pos, t;
	
	// Loading signal segment
	prepare_signal_4elem<const_params>(s_input_1, d_input_signal, signal_length, useful_part_size, offset);
	offset = (2*const_params::fft_length - useful_part_size + 1)>>1;
	
	__syncthreads();
	
	// Forward FFT on input signal
	do_FFT_Stockham_R2C_C2R<const_params,FFT_forward>((float2 *) s_input_1);
	
	// Storing FFTed signal for reuse
	GPU_conv_assign_signal<const_params>(r_signal, (float2 *) s_input_1);
	
	for(t=0; t<nFilters; t++){
		GPU_conv_complex_multiplication<const_params>(t, r_signal, d_filters_freqdom, (float2 *) s_input_1);
		
		__syncthreads();
		
		//----------> Inverse FFT
		do_FFT_Stockham_R2C_C2R<const_params,FFT_inverse>((float2 *) s_input_1);
		//----------<
		
		
		pos = t*useful_part_size*nConvolutions + blockIdx.x*useful_part_size + threadIdx.x;
		//if(blockIdx.x==0 && (t==0)) printf("th:%d; pos=%d; offset=%d; useful=%d; output:%f\n", threadIdx.x, pos, offset, useful_part_size, s_input_1[threadIdx.x]);
		if( threadIdx.x>=offset && threadIdx.x<(useful_part_size+offset) ) {
			d_output_plane[pos - offset] = s_input_1[threadIdx.x]/((float) const_params::fft_length);
		}
		if( (threadIdx.x + const_params::fft_quarter)>=offset && (threadIdx.x + const_params::fft_quarter)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_quarter - offset] = s_input_1[threadIdx.x + const_params::fft_quarter]/((float) const_params::fft_length);
		}
		if( (threadIdx.x + const_params::fft_half)>=offset && (threadIdx.x + const_params::fft_half)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_half - offset] = s_input_1[threadIdx.x + const_params::fft_half]/((float) const_params::fft_length);
		}
		if( (threadIdx.x + const_params::fft_threequarters)>=offset && (threadIdx.x + const_params::fft_threequarters)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_threequarters - offset] = s_input_1[threadIdx.x + const_params::fft_threequarters]/((float) const_params::fft_length);
		}

		pos = t*useful_part_size*nConvolutions + blockIdx.x*useful_part_size + threadIdx.x + const_params::fft_length;
		//if(blockIdx.x==0 && (t==0)) printf("th:%d; pos=%d; offset=%d; useful=%d; output:%f\n", threadIdx.x, pos, offset, useful_part_size, s_input_1[threadIdx.x]);
		int tmp = threadIdx.x +  + const_params::fft_length;
		if( tmp>=offset && tmp<(useful_part_size+offset) ) {
			d_output_plane[pos - offset] = s_input_1[tmp]/((float) const_params::fft_length);
		}
		if( (tmp + const_params::fft_quarter)>=offset && (tmp + const_params::fft_quarter)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_quarter - offset] = s_input_1[tmp + const_params::fft_quarter]/((float) const_params::fft_length);
		}
		if( (tmp + const_params::fft_half)>=offset && (tmp + const_params::fft_half)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_half - offset] = s_input_1[tmp + const_params::fft_half]/((float) const_params::fft_length);
		}
		if( (tmp + const_params::fft_threequarters)>=offset && (tmp + const_params::fft_threequarters)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_threequarters - offset] = s_input_1[tmp + const_params::fft_threequarters]/((float) const_params::fft_length);
		}
		
		__syncthreads();
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


void forwardCustomFFT(float2 *d_output, float *d_input, int FFT_size, int nFFTs, int inverse){
	dim3 gridSize(nFFTs, 1, 1);
	dim3 blockSize((FFT_size>>1)/4, 1, 1);
	
	switch(FFT_size) {
		case 512:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_256,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			else FFT_GPU_R2C_C2R_external<FFT_256,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			break;
		
		case 1024:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_512,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			else FFT_GPU_R2C_C2R_external<FFT_512,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			break;

		case 2048:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_1024,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			else FFT_GPU_R2C_C2R_external<FFT_1024,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			break;
			
		case 4096:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_2048,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			else FFT_GPU_R2C_C2R_external<FFT_2048,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, d_output);
			break;
		
		default : 
			break;
	}
}


void conv_OLS_R2R_customFFT(float *d_input_signal, float *d_output_plane, float2 *d_filters, int signal_length, int convolution_length, int useful_part_size, int offset, int nConvolutions, int nFilters){
	dim3 gridSize(nConvolutions, 1, 1);
	dim3 blockSize((convolution_length>>1)/4, 1, 1);
	//printf("gridSize:[%d;%d;%d]; blockSize:[%d;%d;%d];\n",gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
	
	switch(convolution_length) {
		case 512:
			k_GPU_conv_OLS_R2R_via_customFFT<FFT_256><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;
		
		case 1024:
			k_GPU_conv_OLS_R2R_via_customFFT<FFT_512><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;

		case 2048:
			k_GPU_conv_OLS_R2R_via_customFFT<FFT_1024><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;
			
		case 4096:
			k_GPU_conv_OLS_R2R_via_customFFT<FFT_2048><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;
			
		case 8192:
			k_GPU_conv_OLS_R2R_via_customFFT<FFT_4096><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;
		
		default : 
			break;
	}
}


void convolution_via_customFFT_benchmark(float *d_input_signal, float *d_output_plane, float *d_filters_timedom, int signal_length, int convolution_length, int useful_part_size, int offset, int nConvolutions, int nFilters, double *CONV_time){
	GpuTimer timer;
	
	// ----------------------------------------------->
	// --------> Measured part (Convolution)
	timer.Start();
	
	float2 *d_filters_freqdom;
	int filters_size_bytes = (convolution_length>>1)*nFilters*sizeof(float2);
	if( cudaSuccess!=cudaMalloc((void **) &d_filters_freqdom, filters_size_bytes) ) printf("CUDA API error while allocating filter memory for convolution\n");
	forwardCustomFFT(d_filters_freqdom, d_filters_timedom, convolution_length, nFilters, 0);
	
	checkCudaErrors(cudaGetLastError());
	
	CONV_init();
	conv_OLS_R2R_customFFT(d_input_signal, d_output_plane, d_filters_freqdom, signal_length, convolution_length, useful_part_size, offset, nConvolutions, nFilters);
	
	checkCudaErrors(cudaGetLastError());
	
	checkCudaErrors(cudaFree(d_filters_freqdom));
	
	timer.Stop();
	*CONV_time += timer.Elapsed();
	// --------> Measured part (Convolution)
	// ----------------------------------------------->
}


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

int GPU_convolution_OLS_customFFT(float *h_input_signal, float *h_output_plane, float *h_filters_timedom, int signal_length, int convolution_length, int filter_length, int past_filter_samples, int nFilters, int nRuns, int device, double *execution_time){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem, total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(device<devCount){
		checkCudaErrors(cudaSetDevice(device));
	}
	else {
		printf("ERROR! Selected device is not available\n");
		return(1);
	}
	cudaMemGetInfo(&free_mem,&total_mem);
	
	//---------> Time measurements
	double transfer_in, transfer_out, CONV_kFFT_time, total_CONV_kFFT_time;
	transfer_in=0.0; transfer_out=0.0; CONV_kFFT_time=0.0; total_CONV_kFFT_time=0;
	GpuTimer timer;
	
	//----> Calculating variables for overlap-and-save
	int offset           = past_filter_samples;
	int useful_part_size = convolution_length - filter_length + 1;
	//useful_part_size = 2*(useful_part_size>>1);
	int nConvolutions    = (signal_length + useful_part_size - 1)/useful_part_size;
	
	//int itemp = (int) ((convolution_length - filter_length)>>1);
	//int useful_part_size = itemp*2;
	//int offset = (convolution_length - useful_part_size)/2;
	//int nConvolutions    = (signal_length + useful_part_size - 1)/useful_part_size;
	
	
	if(DEBUG) printf("signal_length: %d; filter_length: %d; segment_size: %d;\n", signal_length, filter_length, convolution_length);
	if(DEBUG) printf("offset: %d; nConvolutions: %d; useful_part_size: %d;\n", offset, nConvolutions, useful_part_size);
	
	//---------> Defining variables and their sizes
	float  *d_output_plane;
	float  *d_input_signal;
	float  *d_filters_timedom;
	size_t input_size   = signal_length;
	size_t output_size  = nConvolutions*useful_part_size*nFilters;
	size_t filters_size = convolution_length*nFilters;
	
	//---------> Checking memory
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required=(( ((float) input_size) + ((float) output_size) + ((float) filters_size))*sizeof(float))/(1024.0*1024.0);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("DEBUG:\n");
	if(DEBUG) printf("    Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(DEBUG) printf("    d_input_signal:  %0.3f MB\n", ((float) input_size*sizeof(float))/(1024.0*1024.0) ); 
	if(DEBUG) printf("    d_filters:       %0.3f MB\n", ((float) filters_size*sizeof(float))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_output_plane:  %0.3f MB\n", ((float) output_size*sizeof(float))/(1024.0*1024.0) );
	if(memory_required>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(-3);}
	
	//---------> Memory allocation
	if (VERBOSE) printf("Device memory allocation...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input_signal, sizeof(float)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output_plane, sizeof(float)*output_size));
	checkCudaErrors(cudaMalloc((void **) &d_filters_timedom, sizeof(float)*filters_size));
	timer.Stop();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());	
	//------------------------------------------------------------------------------
	//---------> CONV calculation
	
		//-----> Copy chunk of input data to a device
		if (VERBOSE) printf("Transferring data into device memory...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy(d_input_signal, h_input_signal, input_size*sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_filters_timedom, h_filters_timedom, filters_size*sizeof(float), cudaMemcpyHostToDevice));
		timer.Stop();
		transfer_in+=timer.Elapsed();
		if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
		
		if (DEBUG) printf("Calculating convolution via kFFT...: \t\t");
		total_CONV_kFFT_time = 0;
		for(int f=0; f<nRuns; f++){
			checkCudaErrors(cudaMemcpy(d_input_signal, h_input_signal, input_size*sizeof(float), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_filters_timedom, h_filters_timedom, filters_size*sizeof(float), cudaMemcpyHostToDevice));
			convolution_via_customFFT_benchmark(d_input_signal, d_output_plane, d_filters_timedom, signal_length, convolution_length, useful_part_size, offset, nConvolutions, nFilters, &total_CONV_kFFT_time);
			checkCudaErrors(cudaGetLastError());
		}
		CONV_kFFT_time=total_CONV_kFFT_time/nRuns;
		if (DEBUG) printf("done in %g ms.\n", CONV_kFFT_time);
		*execution_time=CONV_kFFT_time;
		
		//-----> Copy chunk of output data to host
		if (DEBUG) printf("Transferring data to host...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy( h_output_plane, d_output_plane, output_size*sizeof(float), cudaMemcpyDeviceToHost));
		timer.Stop();
		transfer_out+=timer.Elapsed();
		if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());
	
	
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input_signal));
	checkCudaErrors(cudaFree(d_output_plane));
	checkCudaErrors(cudaFree(d_filters_timedom));
	
	return(0);
}
