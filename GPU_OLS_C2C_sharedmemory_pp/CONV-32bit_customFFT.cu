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
//#define TESTING

int device=DEVICEID;

class FFT_Params {
public:
	static const int fft_exp = -1;
	static const int fft_length = -1;
	static const int warp = 32;
};

class FFT_256 : public FFT_Params {
	public:
	static const int fft_exp = 8;
	static const int fft_length = 256;
	static const int fft_length_quarter = 64;
	static const int fft_length_half = 128;
	static const int fft_length_three_quarters = 192;
};

class FFT_512 : public FFT_Params {
	public:
	static const int fft_exp = 9;
	static const int fft_length = 512;
	static const int fft_length_quarter = 128;
	static const int fft_length_half = 256;
	static const int fft_length_three_quarters = 384;
};

class FFT_1024 : public FFT_Params {
	public:
	static const int fft_exp = 10;
	static const int fft_length = 1024;
	static const int fft_length_quarter = 256;
	static const int fft_length_half = 512;
	static const int fft_length_three_quarters = 768;
};

class FFT_2048 : public FFT_Params {
	public:
	static const int fft_exp = 11;
	static const int fft_length = 2048;
	static const int fft_length_quarter = 512;
	static const int fft_length_half = 1024;
	static const int fft_length_three_quarters = 1536;
};

class FFT_4096 : public FFT_Params {
	public:
	static const int fft_exp = 12;
	static const int fft_length = 4096;
	static const int fft_length_quarter = 1024;
	static const int fft_length_half = 2048;
	static const int fft_length_three_quarters = 3072;
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
	sincosf ( -6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ float2 Get_W_value_inverse(int N, int m){
	float2 ctemp;
	sincosf ( 6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}


__device__ __inline__ float shfl_xor(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_xor_sync(0xffffffff, (*value), par));
	#else
		return(__shfl_xor((*value), par));
	#endif
}


template<class const_params>
__inline__ __device__ void CT_DIT_FFT_4way(float2 *s_input){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (const_params::warp - 1);
	warp_id = threadIdx.x/const_params::warp;

	#ifdef TESTING
	int A_load_id, B_load_id, i, A_n, B_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + const_params::fft_length_quarter;
	A_n=threadIdx.x;
	B_n=threadIdx.x + const_params::fft_length_quarter;
	for(i=1; i<const_params::fft_exp; i++) {
		A_n >>= 1;
		B_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
    }
    A_load_id &= const_params::fft_length-1;
	B_load_id &= const_params::fft_length-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[A_load_id];
	B_DFT_value=s_input[A_load_id + 1];
	C_DFT_value=s_input[B_load_id];
	D_DFT_value=s_input[B_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]         = A_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_half]   = B_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_quarter]   = C_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = D_DFT_value;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp];
	B_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp + const_params::warp];
	C_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp + 2*const_params::warp];
	D_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp + 3*const_params::warp];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
	A_DFT_value.y=parity*A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
	B_DFT_value.x=parity*B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
	B_DFT_value.y=parity*B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
	C_DFT_value.x=parity*C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
	C_DFT_value.y=parity*C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
	D_DFT_value.x=parity*D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
	D_DFT_value.y=parity*D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp = m_param>>q;
		parity=((itemp<<1)-1);
		W = Get_W_value_inverse(PoTp1, itemp*m_param);
		
		Aftemp.x = W.x*A_DFT_value.x - W.y*A_DFT_value.y;
		Aftemp.y = W.x*A_DFT_value.y + W.y*A_DFT_value.x;
		Bftemp.x = W.x*B_DFT_value.x - W.y*B_DFT_value.y;
		Bftemp.y = W.x*B_DFT_value.y + W.y*B_DFT_value.x;
		Cftemp.x = W.x*C_DFT_value.x - W.y*C_DFT_value.y;
		Cftemp.y = W.x*C_DFT_value.y + W.y*C_DFT_value.x;
		Dftemp.x = W.x*D_DFT_value.x - W.y*D_DFT_value.y;
		Dftemp.y = W.x*D_DFT_value.y + W.y*D_DFT_value.x;
		
		A_DFT_value.x = Aftemp.x + parity*shfl_xor(&Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*shfl_xor(&Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*shfl_xor(&Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*shfl_xor(&Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*shfl_xor(&Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*shfl_xor(&Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*shfl_xor(&Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*shfl_xor(&Dftemp.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*const_params::warp;
	s_input[itemp]                        = A_DFT_value;
	s_input[itemp + const_params::warp]   = B_DFT_value;
	s_input[itemp + 2*const_params::warp] = C_DFT_value;
	s_input[itemp + 3*const_params::warp] = D_DFT_value;
	
	for(q=5;q<(const_params::fft_exp-1);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value_inverse(PoTp1,m_param);

		A_read_index=j*(PoTp1<<1) + m_param;
		B_read_index=j*(PoTp1<<1) + m_param + PoT;
		C_read_index=j*(PoTp1<<1) + m_param + PoTp1;
		D_read_index=j*(PoTp1<<1) + m_param + 3*PoT;
		
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
		A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;		
		B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
		B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;
		
		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x=Cftemp.x + W.x*Dftemp.x - W.y*Dftemp.y;
		C_DFT_value.y=Cftemp.y + W.x*Dftemp.y + W.y*Dftemp.x;		
		D_DFT_value.x=Cftemp.x - W.x*Dftemp.x + W.y*Dftemp.y;
		D_DFT_value.y=Cftemp.y - W.x*Dftemp.y - W.y*Dftemp.x;
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	//last iteration
	__syncthreads();
	m_param = threadIdx.x;
	
	W=Get_W_value_inverse(PoTp1,m_param);
    
	A_read_index = m_param;
	B_read_index = m_param + PoT;
	C_read_index = m_param + (PoT>>1);
	D_read_index = m_param + 3*(PoT>>1);
	
	Aftemp = s_input[A_read_index];
	Bftemp = s_input[B_read_index];
	A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
	A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;		
	B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
	B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;
	
	Cftemp = s_input[C_read_index];
	Dftemp = s_input[D_read_index];
	C_DFT_value.x=Cftemp.x - W.y*Dftemp.x - W.x*Dftemp.y;
	C_DFT_value.y=Cftemp.y - W.y*Dftemp.y + W.x*Dftemp.x;		
	D_DFT_value.x=Cftemp.x + W.y*Dftemp.x + W.x*Dftemp.y;
	D_DFT_value.y=Cftemp.y + W.y*Dftemp.y - W.x*Dftemp.x;
	
	s_input[A_read_index]=A_DFT_value;
	s_input[B_read_index]=B_DFT_value;
	s_input[C_read_index]=C_DFT_value;
	s_input[D_read_index]=D_DFT_value;

	__syncthreads();	
}


template<class const_params>
__inline__ __device__ void CT_DIF_FFT_4way(float2 *s_input){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int local_id, warp_id;
	int j, m_param, parity;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTm1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;
	
	
	//-----> FFT
	//-->
	PoTm1 = const_params::fft_length_half;
	PoT   = const_params::fft_length;
	
	//Highest iteration
	m_param = threadIdx.x;
	j=0;
	A_read_index = m_param;
	B_read_index = m_param + PoTm1;
	C_read_index = m_param + (PoTm1>>1);
	D_read_index = m_param + 3*(PoTm1>>1);
	
	W=Get_W_value(PoT, m_param);
	
	Aftemp = s_input[A_read_index];
	Bftemp = s_input[B_read_index];
	Cftemp = s_input[C_read_index];
	Dftemp = s_input[D_read_index];
	
	A_DFT_value.x = Aftemp.x + Bftemp.x;
	A_DFT_value.y = Aftemp.y + Bftemp.y;
	B_DFT_value.x = W.x*(Aftemp.x - Bftemp.x) - W.y*(Aftemp.y - Bftemp.y);
	B_DFT_value.y = W.x*(Aftemp.y - Bftemp.y) + W.y*(Aftemp.x - Bftemp.x);
	
	C_DFT_value.x = Cftemp.x + Dftemp.x;
	C_DFT_value.y = Cftemp.y + Dftemp.y;
	D_DFT_value.x = W.y*(Cftemp.x - Dftemp.x) + W.x*(Cftemp.y - Dftemp.y);
	D_DFT_value.y = W.y*(Cftemp.y - Dftemp.y) - W.x*(Cftemp.x - Dftemp.x);
	
	s_input[A_read_index]=A_DFT_value;
	s_input[B_read_index]=B_DFT_value;
	s_input[C_read_index]=C_DFT_value;
	s_input[D_read_index]=D_DFT_value;
	
	PoT=PoT>>1;
	PoTm1=PoTm1>>1;
	
	for(q=(const_params::fft_exp-2);q>4;q--){
		__syncthreads();
		m_param = threadIdx.x & (PoTm1 - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoT, m_param);

		A_read_index=j*(PoT<<1) + m_param;
		B_read_index=j*(PoT<<1) + m_param + PoTm1;
		C_read_index=j*(PoT<<1) + m_param + PoT;
		D_read_index=j*(PoT<<1) + m_param + 3*PoTm1;
		
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		
		A_DFT_value.x = Aftemp.x + Bftemp.x;
		A_DFT_value.y = Aftemp.y + Bftemp.y;
		C_DFT_value.x = Cftemp.x + Dftemp.x;
		C_DFT_value.y = Cftemp.y + Dftemp.y;
		
		B_DFT_value.x = W.x*(Aftemp.x - Bftemp.x) - W.y*(Aftemp.y - Bftemp.y);
		B_DFT_value.y = W.x*(Aftemp.y - Bftemp.y) + W.y*(Aftemp.x - Bftemp.x);
		D_DFT_value.x = W.x*(Cftemp.x - Dftemp.x) - W.y*(Cftemp.y - Dftemp.y);
		D_DFT_value.y = W.x*(Cftemp.y - Dftemp.y) + W.y*(Cftemp.x - Dftemp.x);
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}

	__syncthreads();
	j = local_id + (warp_id<<2)*WARP;
	A_DFT_value = s_input[j];
	B_DFT_value = s_input[j + WARP];
	C_DFT_value = s_input[j + 2*WARP];
	D_DFT_value = s_input[j + 3*WARP];
	
	for(q=4;q>=0;q--){
		m_param = (local_id & (PoT - 1));
		j = m_param>>q;
		parity=(1-j*2);
		W = Get_W_value(PoT, j*(m_param-PoTm1));
		
		Aftemp.x = parity*A_DFT_value.x + shfl_xor(&A_DFT_value.x, PoTm1);
		Aftemp.y = parity*A_DFT_value.y + shfl_xor(&A_DFT_value.y, PoTm1);
		Bftemp.x = parity*B_DFT_value.x + shfl_xor(&B_DFT_value.x, PoTm1);
		Bftemp.y = parity*B_DFT_value.y + shfl_xor(&B_DFT_value.y, PoTm1);
		Cftemp.x = parity*C_DFT_value.x + shfl_xor(&C_DFT_value.x, PoTm1);
		Cftemp.y = parity*C_DFT_value.y + shfl_xor(&C_DFT_value.y, PoTm1);
		Dftemp.x = parity*D_DFT_value.x + shfl_xor(&D_DFT_value.x, PoTm1);
		Dftemp.y = parity*D_DFT_value.y + shfl_xor(&D_DFT_value.y, PoTm1);
		
		A_DFT_value.x = W.x*Aftemp.x - W.y*Aftemp.y; 
		A_DFT_value.y = W.x*Aftemp.y + W.y*Aftemp.x;
		B_DFT_value.x = W.x*Bftemp.x - W.y*Bftemp.y; 
		B_DFT_value.y = W.x*Bftemp.y + W.y*Bftemp.x;
		C_DFT_value.x = W.x*Cftemp.x - W.y*Cftemp.y; 
		C_DFT_value.y = W.x*Cftemp.y + W.y*Cftemp.x;
		D_DFT_value.x = W.x*Dftemp.x - W.y*Dftemp.y; 
		D_DFT_value.y = W.x*Dftemp.y + W.y*Dftemp.x;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}
	
	j = local_id + (warp_id<<2)*WARP;
	s_input[j]          = A_DFT_value;
	s_input[j + WARP]   = B_DFT_value;
	s_input[j + 2*WARP] = C_DFT_value;
	s_input[j + 3*WARP] = D_DFT_value;
	
	__syncthreads();
	
	#ifdef TESTING
	__syncthreads();
	int A_load_id, B_load_id, i, A_n, B_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + const_params::fft_length_quarter;
	A_n=threadIdx.x;
	B_n=threadIdx.x + const_params::fft_length_quarter;
	for(i=1; i<const_params::fft_exp; i++) {
		A_n >>= 1;
		B_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
    }
    A_load_id &= const_params::fft_length-1;
	B_load_id &= const_params::fft_length-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[A_load_id];
	B_DFT_value=s_input[A_load_id + 1];
	C_DFT_value=s_input[B_load_id];
	D_DFT_value=s_input[B_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]                                           = A_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_half]           = B_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_quarter]        = C_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = D_DFT_value;
	__syncthreads();
	#endif
}



template<class const_params>
__inline__ __device__ void FFT_CT_DIT_4elem_2vertical_no_reorder(float2 *s_input1, float2 *s_input2){
	float2 A_DFT_value1, B_DFT_value1, C_DFT_value1, D_DFT_value1;
	float2 A_DFT_value2, B_DFT_value2, C_DFT_value2, D_DFT_value2;
	float2 W;
	float2 Aftemp1, Bftemp1, Cftemp1, Dftemp1;
	float2 Aftemp2, Bftemp2, Cftemp2, Dftemp2;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP];
	B_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP + 3*WARP];
	A_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP];
	B_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP + 3*WARP];
	
	__syncthreads();
	
	A_DFT_value1.x=parity*A_DFT_value1.x + shfl_xor(&A_DFT_value1.x,1);
	A_DFT_value1.y=parity*A_DFT_value1.y + shfl_xor(&A_DFT_value1.y,1);
	B_DFT_value1.x=parity*B_DFT_value1.x + shfl_xor(&B_DFT_value1.x,1);
	B_DFT_value1.y=parity*B_DFT_value1.y + shfl_xor(&B_DFT_value1.y,1);
	C_DFT_value1.x=parity*C_DFT_value1.x + shfl_xor(&C_DFT_value1.x,1);
	C_DFT_value1.y=parity*C_DFT_value1.y + shfl_xor(&C_DFT_value1.y,1);
	D_DFT_value1.x=parity*D_DFT_value1.x + shfl_xor(&D_DFT_value1.x,1);
	D_DFT_value1.y=parity*D_DFT_value1.y + shfl_xor(&D_DFT_value1.y,1);
	
	A_DFT_value2.x=parity*A_DFT_value2.x + shfl_xor(&A_DFT_value2.x,1);
	A_DFT_value2.y=parity*A_DFT_value2.y + shfl_xor(&A_DFT_value2.y,1);
	B_DFT_value2.x=parity*B_DFT_value2.x + shfl_xor(&B_DFT_value2.x,1);
	B_DFT_value2.y=parity*B_DFT_value2.y + shfl_xor(&B_DFT_value2.y,1);
	C_DFT_value2.x=parity*C_DFT_value2.x + shfl_xor(&C_DFT_value2.x,1);
	C_DFT_value2.y=parity*C_DFT_value2.y + shfl_xor(&C_DFT_value2.y,1);
	D_DFT_value2.x=parity*D_DFT_value2.x + shfl_xor(&D_DFT_value2.x,1);
	D_DFT_value2.y=parity*D_DFT_value2.y + shfl_xor(&D_DFT_value2.y,1);
	
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp = m_param>>q;
		parity=((itemp<<1)-1);
		W = Get_W_value_inverse(PoTp1, itemp*m_param);
		
		Aftemp1.x = W.x*A_DFT_value1.x - W.y*A_DFT_value1.y;
		Aftemp1.y = W.x*A_DFT_value1.y + W.y*A_DFT_value1.x;
		Bftemp1.x = W.x*B_DFT_value1.x - W.y*B_DFT_value1.y;
		Bftemp1.y = W.x*B_DFT_value1.y + W.y*B_DFT_value1.x;
		Cftemp1.x = W.x*C_DFT_value1.x - W.y*C_DFT_value1.y;
		Cftemp1.y = W.x*C_DFT_value1.y + W.y*C_DFT_value1.x;
		Dftemp1.x = W.x*D_DFT_value1.x - W.y*D_DFT_value1.y;
		Dftemp1.y = W.x*D_DFT_value1.y + W.y*D_DFT_value1.x;
		
		Aftemp2.x = W.x*A_DFT_value2.x - W.y*A_DFT_value2.y;
		Aftemp2.y = W.x*A_DFT_value2.y + W.y*A_DFT_value2.x;
		Bftemp2.x = W.x*B_DFT_value2.x - W.y*B_DFT_value2.y;
		Bftemp2.y = W.x*B_DFT_value2.y + W.y*B_DFT_value2.x;
		Cftemp2.x = W.x*C_DFT_value2.x - W.y*C_DFT_value2.y;
		Cftemp2.y = W.x*C_DFT_value2.y + W.y*C_DFT_value2.x;
		Dftemp2.x = W.x*D_DFT_value2.x - W.y*D_DFT_value2.y;
		Dftemp2.y = W.x*D_DFT_value2.y + W.y*D_DFT_value2.x;
		
		A_DFT_value1.x = Aftemp1.x + parity*shfl_xor(&Aftemp1.x,PoT);
		A_DFT_value1.y = Aftemp1.y + parity*shfl_xor(&Aftemp1.y,PoT);
		B_DFT_value1.x = Bftemp1.x + parity*shfl_xor(&Bftemp1.x,PoT);
		B_DFT_value1.y = Bftemp1.y + parity*shfl_xor(&Bftemp1.y,PoT);
		C_DFT_value1.x = Cftemp1.x + parity*shfl_xor(&Cftemp1.x,PoT);
		C_DFT_value1.y = Cftemp1.y + parity*shfl_xor(&Cftemp1.y,PoT);
		D_DFT_value1.x = Dftemp1.x + parity*shfl_xor(&Dftemp1.x,PoT);
		D_DFT_value1.y = Dftemp1.y + parity*shfl_xor(&Dftemp1.y,PoT);	
		
		A_DFT_value2.x = Aftemp2.x + parity*shfl_xor(&Aftemp2.x,PoT);
		A_DFT_value2.y = Aftemp2.y + parity*shfl_xor(&Aftemp2.y,PoT);
		B_DFT_value2.x = Bftemp2.x + parity*shfl_xor(&Bftemp2.x,PoT);
		B_DFT_value2.y = Bftemp2.y + parity*shfl_xor(&Bftemp2.y,PoT);
		C_DFT_value2.x = Cftemp2.x + parity*shfl_xor(&Cftemp2.x,PoT);
		C_DFT_value2.y = Cftemp2.y + parity*shfl_xor(&Cftemp2.y,PoT);
		D_DFT_value2.x = Dftemp2.x + parity*shfl_xor(&Dftemp2.x,PoT);
		D_DFT_value2.y = Dftemp2.y + parity*shfl_xor(&Dftemp2.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*WARP;
	s_input1[itemp]          = A_DFT_value1;
	s_input1[itemp + WARP]   = B_DFT_value1;
	s_input1[itemp + 2*WARP] = C_DFT_value1;
	s_input1[itemp + 3*WARP] = D_DFT_value1;
	
	s_input2[itemp]          = A_DFT_value2;
	s_input2[itemp + WARP]   = B_DFT_value2;
	s_input2[itemp + 2*WARP] = C_DFT_value2;
	s_input2[itemp + 3*WARP] = D_DFT_value2;
	
	for(q=5;q<(const_params::fft_exp-1);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value_inverse(PoTp1,m_param);

		A_read_index=j*(PoTp1<<1) + m_param;
		B_read_index=j*(PoTp1<<1) + m_param + PoT;
		C_read_index=j*(PoTp1<<1) + m_param + PoTp1;
		D_read_index=j*(PoTp1<<1) + m_param + 3*PoT;
		
		Aftemp1 = s_input1[A_read_index];
		Bftemp1 = s_input1[B_read_index];
		A_DFT_value1.x=Aftemp1.x + W.x*Bftemp1.x - W.y*Bftemp1.y;
		A_DFT_value1.y=Aftemp1.y + W.x*Bftemp1.y + W.y*Bftemp1.x;		
		B_DFT_value1.x=Aftemp1.x - W.x*Bftemp1.x + W.y*Bftemp1.y;
		B_DFT_value1.y=Aftemp1.y - W.x*Bftemp1.y - W.y*Bftemp1.x;
		
		Aftemp2 = s_input2[A_read_index];
		Bftemp2 = s_input2[B_read_index];
		A_DFT_value2.x=Aftemp2.x + W.x*Bftemp2.x - W.y*Bftemp2.y;
		A_DFT_value2.y=Aftemp2.y + W.x*Bftemp2.y + W.y*Bftemp2.x;		
		B_DFT_value2.x=Aftemp2.x - W.x*Bftemp2.x + W.y*Bftemp2.y;
		B_DFT_value2.y=Aftemp2.y - W.x*Bftemp2.y - W.y*Bftemp2.x;
		
		Cftemp1 = s_input1[C_read_index];
		Dftemp1 = s_input1[D_read_index];
		C_DFT_value1.x=Cftemp1.x + W.x*Dftemp1.x - W.y*Dftemp1.y;
		C_DFT_value1.y=Cftemp1.y + W.x*Dftemp1.y + W.y*Dftemp1.x;		
		D_DFT_value1.x=Cftemp1.x - W.x*Dftemp1.x + W.y*Dftemp1.y;
		D_DFT_value1.y=Cftemp1.y - W.x*Dftemp1.y - W.y*Dftemp1.x;

		Cftemp2 = s_input2[C_read_index];
		Dftemp2 = s_input2[D_read_index];
		C_DFT_value2.x=Cftemp2.x + W.x*Dftemp2.x - W.y*Dftemp2.y;
		C_DFT_value2.y=Cftemp2.y + W.x*Dftemp2.y + W.y*Dftemp2.x;		
		D_DFT_value2.x=Cftemp2.x - W.x*Dftemp2.x + W.y*Dftemp2.y;
		D_DFT_value2.y=Cftemp2.y - W.x*Dftemp2.y - W.y*Dftemp2.x;
		
		s_input1[A_read_index]=A_DFT_value1;
		s_input1[B_read_index]=B_DFT_value1;
		s_input1[C_read_index]=C_DFT_value1;
		s_input1[D_read_index]=D_DFT_value1;
		
		s_input2[A_read_index]=A_DFT_value2;
		s_input2[B_read_index]=B_DFT_value2;
		s_input2[C_read_index]=C_DFT_value2;
		s_input2[D_read_index]=D_DFT_value2;
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	//last iteration
	__syncthreads();
	m_param = threadIdx.x;
	
	W=Get_W_value_inverse(PoTp1,m_param);
    
	A_read_index = m_param;
	B_read_index = m_param + PoT;
	C_read_index = m_param + (PoT>>1);
	D_read_index = m_param + 3*(PoT>>1);
	
	Aftemp1 = s_input1[A_read_index];
	Bftemp1 = s_input1[B_read_index];
	A_DFT_value1.x=Aftemp1.x + W.x*Bftemp1.x - W.y*Bftemp1.y;
	A_DFT_value1.y=Aftemp1.y + W.x*Bftemp1.y + W.y*Bftemp1.x;		
	B_DFT_value1.x=Aftemp1.x - W.x*Bftemp1.x + W.y*Bftemp1.y;
	B_DFT_value1.y=Aftemp1.y - W.x*Bftemp1.y - W.y*Bftemp1.x;

	Aftemp2 = s_input2[A_read_index];
	Bftemp2 = s_input2[B_read_index];
	A_DFT_value2.x=Aftemp2.x + W.x*Bftemp2.x - W.y*Bftemp2.y;
	A_DFT_value2.y=Aftemp2.y + W.x*Bftemp2.y + W.y*Bftemp2.x;		
	B_DFT_value2.x=Aftemp2.x - W.x*Bftemp2.x + W.y*Bftemp2.y;
	B_DFT_value2.y=Aftemp2.y - W.x*Bftemp2.y - W.y*Bftemp2.x;	
	
	Cftemp1 = s_input1[C_read_index];
	Dftemp1 = s_input1[D_read_index];
	C_DFT_value1.x=Cftemp1.x - W.y*Dftemp1.x - W.x*Dftemp1.y;
	C_DFT_value1.y=Cftemp1.y - W.y*Dftemp1.y + W.x*Dftemp1.x;		
	D_DFT_value1.x=Cftemp1.x + W.y*Dftemp1.x + W.x*Dftemp1.y;
	D_DFT_value1.y=Cftemp1.y + W.y*Dftemp1.y - W.x*Dftemp1.x;
	
	Cftemp2 = s_input2[C_read_index];
	Dftemp2 = s_input2[D_read_index];
	C_DFT_value2.x=Cftemp2.x - W.y*Dftemp2.x - W.x*Dftemp2.y;
	C_DFT_value2.y=Cftemp2.y - W.y*Dftemp2.y + W.x*Dftemp2.x;		
	D_DFT_value2.x=Cftemp2.x + W.y*Dftemp2.x + W.x*Dftemp2.y;
	D_DFT_value2.y=Cftemp2.y + W.y*Dftemp2.y - W.x*Dftemp2.x;
	
	s_input1[A_read_index]=A_DFT_value1;
	s_input1[B_read_index]=B_DFT_value1;
	s_input1[C_read_index]=C_DFT_value1;
	s_input1[D_read_index]=D_DFT_value1;	
	
	s_input2[A_read_index]=A_DFT_value2;
	s_input2[B_read_index]=B_DFT_value2;
	s_input2[C_read_index]=C_DFT_value2;
	s_input2[D_read_index]=D_DFT_value2;

	__syncthreads();	
}


template<class const_params>
__global__ void k_customFFT_GPU_forward(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]                                            = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_length_quarter]        = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter];
	s_input[threadIdx.x + const_params::fft_length_half]           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half];
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters];
	
	__syncthreads();
	CT_DIF_FFT_4way<const_params>(s_input);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                            = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = s_input[threadIdx.x + const_params::fft_length_quarter];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = s_input[threadIdx.x + const_params::fft_length_half];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = s_input[threadIdx.x + const_params::fft_length_three_quarters];
}


template<class const_params>
__device__ __inline__ void prepare_signal_4elem(float2* s_signal, float2 const* __restrict__ d_input_signal, int signal_length, int useful_part_size, int offset) {
	int pos = blockIdx.x*useful_part_size;
	
	pos = blockIdx.x*useful_part_size + threadIdx.x - offset;
	s_signal[threadIdx.x].x                                           = 0;
	s_signal[threadIdx.x].y                                           = 0;
	s_signal[threadIdx.x + const_params::fft_length_quarter].x        = 0;
	s_signal[threadIdx.x + const_params::fft_length_quarter].y        = 0;
	s_signal[threadIdx.x + const_params::fft_length_half].x           = 0;
	s_signal[threadIdx.x + const_params::fft_length_half].y           = 0;
	s_signal[threadIdx.x + const_params::fft_length_three_quarters].x = 0;
	s_signal[threadIdx.x + const_params::fft_length_three_quarters].y = 0;
	
	if( pos>=0 && pos<signal_length ) 
		s_signal[threadIdx.x] = d_input_signal[pos];
	
	if( (pos + const_params::fft_length_quarter)>=0 && (pos + const_params::fft_length_quarter)<signal_length ) 
		s_signal[threadIdx.x + const_params::fft_length_quarter] = d_input_signal[pos + const_params::fft_length_quarter];
		
	if( (pos + const_params::fft_length_half)>=0 && (pos + const_params::fft_length_half)<signal_length ) 	
		s_signal[threadIdx.x + const_params::fft_length_half] = d_input_signal[pos + const_params::fft_length_half];
	
	if( (pos + const_params::fft_length_three_quarters)>=0 && (pos + const_params::fft_length_three_quarters)<signal_length ) 
		s_signal[threadIdx.x + const_params::fft_length_three_quarters] = d_input_signal[pos + const_params::fft_length_three_quarters];
}


__inline__ __device__ float2 post_process(float2 *s_input, int sm_pos, int offset, float h){
	float2 left, right, result;
	if(blockIdx.x==0){
		if((sm_pos - offset)==0){
			left  = s_input[sm_pos];
			right = s_input[sm_pos+1];
		}
		else {
			left  = s_input[sm_pos-1];
			right = s_input[sm_pos+1];
		}
	}
	else {
		left  = s_input[sm_pos-1];
		right = s_input[sm_pos+1];
	}
	
	result.x = (left.x - right.x)/(2.0f*h);
	result.y = (left.y - right.y)/(2.0f*h);
	return(result);
}

template<class const_params>
__global__ void k_GPU_conv_OLS_via_customFFT(
			float2 const* __restrict__ d_input_signal, 
			float2 *d_output_plane, 
			float2 const* __restrict__ d_filters, 
			int signal_length, 
			int useful_part_size, 
			float h, 
			int offset, 
			int nConvolutions, 
			int nFilters) {
	extern __shared__ float2 s_input_1[];
	float2 r_filter_1[4];
	float2 signal[4];
	int pos, t;
	
	// Loading signal segment
	prepare_signal_4elem<const_params>(s_input_1, d_input_signal, signal_length, useful_part_size, offset);
	offset = ((const_params::fft_length - useful_part_size + 1)>>1);
	// Forward FFT on input signal
	CT_DIF_FFT_4way<const_params>(s_input_1);
	
	// Storing FFTed signal for reuse
	signal[0]=s_input_1[threadIdx.x];
	signal[1]=s_input_1[threadIdx.x + const_params::fft_length_quarter];
	signal[2]=s_input_1[threadIdx.x + const_params::fft_length_half];
	signal[3]=s_input_1[threadIdx.x + const_params::fft_length_three_quarters];
	
	for(t=0; t<nFilters; t++){
		// Loading filters
		pos = t*const_params::fft_length + threadIdx.x;
		r_filter_1[0]=__ldg(&d_filters[pos]);
		r_filter_1[1]=__ldg(&d_filters[pos + const_params::fft_length_quarter]);
		r_filter_1[2]=__ldg(&d_filters[pos + const_params::fft_length_half]);
		r_filter_1[3]=__ldg(&d_filters[pos + const_params::fft_length_three_quarters]);

		// Convolution (complex multiplication)
		s_input_1[threadIdx.x].x                                           = (r_filter_1[0].x*signal[0].x - r_filter_1[0].y*signal[0].y)/((float) const_params::fft_length);
		s_input_1[threadIdx.x].y                                           = (r_filter_1[0].x*signal[0].y + r_filter_1[0].y*signal[0].x)/((float) const_params::fft_length);
		s_input_1[threadIdx.x + const_params::fft_length_quarter].x        = (r_filter_1[1].x*signal[1].x - r_filter_1[1].y*signal[1].y)/((float) const_params::fft_length);
		s_input_1[threadIdx.x + const_params::fft_length_quarter].y        = (r_filter_1[1].x*signal[1].y + r_filter_1[1].y*signal[1].x)/((float) const_params::fft_length);
		s_input_1[threadIdx.x + const_params::fft_length_half].x           = (r_filter_1[2].x*signal[2].x - r_filter_1[2].y*signal[2].y)/((float) const_params::fft_length);
		s_input_1[threadIdx.x + const_params::fft_length_half].y           = (r_filter_1[2].x*signal[2].y + r_filter_1[2].y*signal[2].x)/((float) const_params::fft_length);
		s_input_1[threadIdx.x + const_params::fft_length_three_quarters].x = (r_filter_1[3].x*signal[3].x - r_filter_1[3].y*signal[3].y)/((float) const_params::fft_length);
		s_input_1[threadIdx.x + const_params::fft_length_three_quarters].y = (r_filter_1[3].x*signal[3].y + r_filter_1[3].y*signal[3].x)/((float) const_params::fft_length);
		
		__syncthreads();
		
		//----------> Inverse FFT
		CT_DIT_FFT_4way<const_params>(s_input_1);
		//----------<
		
		//----------> Post-processing
		
		//----------<
		// Writing out the clean part of the segment
		pos = t*useful_part_size*nConvolutions + blockIdx.x*useful_part_size + threadIdx.x;
		if( threadIdx.x>=offset && threadIdx.x<(useful_part_size+offset) ) {
			#ifdef POST_PROCESS
			d_output_plane[pos - offset] = post_process(s_input_1, threadIdx.x, offset, h);
			#else
			d_output_plane[pos - offset] = s_input_1[threadIdx.x];
			#endif
		}
		if( (threadIdx.x + const_params::fft_length_quarter)>=offset && (threadIdx.x + const_params::fft_length_quarter)<(useful_part_size+offset) ) {
			#ifdef POST_PROCESS
			d_output_plane[pos + const_params::fft_length_quarter - offset] = post_process(s_input_1, threadIdx.x + const_params::fft_length_quarter, offset, h);
			#else
			d_output_plane[pos + const_params::fft_length_quarter - offset] = s_input_1[threadIdx.x + const_params::fft_length_quarter];
			#endif
		}
		if( (threadIdx.x + const_params::fft_length_half)>=offset && (threadIdx.x + const_params::fft_length_half)<(useful_part_size+offset) ) {
			#ifdef POST_PROCESS
			d_output_plane[pos + const_params::fft_length_half - offset] = post_process(s_input_1, threadIdx.x + const_params::fft_length_half, offset, h);
			#else
			d_output_plane[pos + const_params::fft_length_half - offset] = s_input_1[threadIdx.x + const_params::fft_length_half];
			#endif
		}
		if( (threadIdx.x + const_params::fft_length_three_quarters)>=offset && (threadIdx.x + const_params::fft_length_three_quarters)<(useful_part_size+offset) ) {
			#ifdef POST_PROCESS
			d_output_plane[pos + const_params::fft_length_three_quarters - offset] = post_process(s_input_1, threadIdx.x + const_params::fft_length_three_quarters, offset, h);
			#else
			d_output_plane[pos + const_params::fft_length_three_quarters - offset] = s_input_1[threadIdx.x + const_params::fft_length_three_quarters];
			#endif
		}
		
		__syncthreads();
	}
}


template<class const_params>
__global__ void k_GPU_conv_OLS_via_customFFT_2filters(
			float2 const* __restrict__ d_input_signal, 
			float2 *d_output_plane, 
			float2 const* __restrict__ d_filters,
			int signal_length,			
			int useful_part_size, 
			int offset, 
			int nConvolutions, 
			int nFilters) {
	__shared__ float2 s_input_1[const_params::fft_length];
	__shared__ float2 s_input_2[const_params::fft_length];
	float2 r_filter_1[4];
	float2 r_filter_2[4];
	float2 signal[4];
	int pos, t;
	
	// Loading data
	prepare_signal_4elem<const_params>(s_input_1, d_input_signal, signal_length, useful_part_size, offset);
	
	// Forward FFT on input signal
	CT_DIF_FFT_4way<const_params>(s_input_1);
	
	// Storing FFTed signal for reuse
	signal[0]=s_input_1[threadIdx.x];
	signal[1]=s_input_1[threadIdx.x + const_params::fft_length_quarter];
	signal[2]=s_input_1[threadIdx.x + const_params::fft_length_half];
	signal[3]=s_input_1[threadIdx.x + const_params::fft_length_three_quarters];
	
	for(t=0; t<(nFilters>>1); t++){
		// Loading filters
		pos = 2*t*const_params::fft_length + threadIdx.x;
		
		r_filter_1[0]=__ldg(&d_filters[pos]);
		r_filter_1[1]=__ldg(&d_filters[pos + const_params::fft_length_quarter]);
		r_filter_1[2]=__ldg(&d_filters[pos + const_params::fft_length_half]);
		r_filter_1[3]=__ldg(&d_filters[pos + const_params::fft_length_three_quarters]);
		
		r_filter_2[0]=__ldg(&d_filters[pos + const_params::fft_length]);
		r_filter_2[1]=__ldg(&d_filters[pos + const_params::fft_length + const_params::fft_length_quarter]);
		r_filter_2[2]=__ldg(&d_filters[pos + const_params::fft_length + const_params::fft_length_half]);
		r_filter_2[3]=__ldg(&d_filters[pos + const_params::fft_length + const_params::fft_length_three_quarters]);

		// Convolution (complex multiplication)
		s_input_1[threadIdx.x].x                                            = r_filter_1[0].x*signal[0].x - r_filter_1[0].y*signal[0].y;
		s_input_1[threadIdx.x].y                                            = r_filter_1[0].x*signal[0].y + r_filter_1[0].y*signal[0].x;
		s_input_1[threadIdx.x + const_params::fft_length_quarter].x        = r_filter_1[1].x*signal[1].x - r_filter_1[1].y*signal[1].y;
		s_input_1[threadIdx.x + const_params::fft_length_quarter].y        = r_filter_1[1].x*signal[1].y + r_filter_1[1].y*signal[1].x;
		s_input_1[threadIdx.x + const_params::fft_length_half].x           = r_filter_1[2].x*signal[2].x - r_filter_1[2].y*signal[2].y;
		s_input_1[threadIdx.x + const_params::fft_length_half].y           = r_filter_1[2].x*signal[2].y + r_filter_1[2].y*signal[2].x;
		s_input_1[threadIdx.x + const_params::fft_length_three_quarters].x = r_filter_1[3].x*signal[3].x - r_filter_1[3].y*signal[3].y;
		s_input_1[threadIdx.x + const_params::fft_length_three_quarters].y = r_filter_1[3].x*signal[3].y + r_filter_1[3].y*signal[3].x;
		
		s_input_2[threadIdx.x].x                                            = r_filter_2[0].x*signal[0].x - r_filter_2[0].y*signal[0].y;
		s_input_2[threadIdx.x].y                                            = r_filter_2[0].x*signal[0].y + r_filter_2[0].y*signal[0].x;
		s_input_2[threadIdx.x + const_params::fft_length_quarter].x        = r_filter_2[1].x*signal[1].x - r_filter_2[1].y*signal[1].y;
		s_input_2[threadIdx.x + const_params::fft_length_quarter].y        = r_filter_2[1].x*signal[1].y + r_filter_2[1].y*signal[1].x;
		s_input_2[threadIdx.x + const_params::fft_length_half].x           = r_filter_2[2].x*signal[2].x - r_filter_2[2].y*signal[2].y;
		s_input_2[threadIdx.x + const_params::fft_length_half].y           = r_filter_2[2].x*signal[2].y + r_filter_2[2].y*signal[2].x;
		s_input_2[threadIdx.x + const_params::fft_length_three_quarters].x = r_filter_2[3].x*signal[3].x - r_filter_2[3].y*signal[3].y;
		s_input_2[threadIdx.x + const_params::fft_length_three_quarters].y = r_filter_2[3].x*signal[3].y + r_filter_2[3].y*signal[3].x;
		
		__syncthreads();
		
		//----------> Inverse FFT
		FFT_CT_DIT_4elem_2vertical_no_reorder<const_params>(s_input_1, s_input_2);
		//----------<
		
		// Writing out the clean part of the segment
		// First convolution
		pos = 2*t*useful_part_size*nConvolutions + blockIdx.x*useful_part_size + threadIdx.x;
		if( threadIdx.x>=offset && threadIdx.x<(useful_part_size+offset) ) {
			d_output_plane[pos - offset] = s_input_1[threadIdx.x];
		}
		if( (threadIdx.x + const_params::fft_length_quarter)>=offset && (threadIdx.x + const_params::fft_length_quarter)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_length_quarter - offset] = s_input_1[threadIdx.x + const_params::fft_length_quarter];
		}
		if( (threadIdx.x + const_params::fft_length_half)>=offset && (threadIdx.x + const_params::fft_length_half)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_length_half - offset] = s_input_1[threadIdx.x + const_params::fft_length_half];
		}
		if( (threadIdx.x + const_params::fft_length_three_quarters)>=offset && (threadIdx.x + const_params::fft_length_three_quarters)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_length_three_quarters - offset] = s_input_1[threadIdx.x + const_params::fft_length_three_quarters];
		}
		// Second convolution
		pos = pos + useful_part_size*nConvolutions;
		if( threadIdx.x>=offset && threadIdx.x<(useful_part_size+offset) ) {
			d_output_plane[pos - offset] = s_input_2[threadIdx.x];
		}
		if( (threadIdx.x + const_params::fft_length_quarter)>=offset && (threadIdx.x + const_params::fft_length_quarter)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_length_quarter - offset] = s_input_2[threadIdx.x + const_params::fft_length_quarter];
		}
		if( (threadIdx.x + const_params::fft_length_half)>=offset && (threadIdx.x + const_params::fft_length_half)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_length_half - offset] = s_input_2[threadIdx.x + const_params::fft_length_half];
		}
		if( (threadIdx.x + const_params::fft_length_three_quarters)>=offset && (threadIdx.x + const_params::fft_length_three_quarters)<(useful_part_size+offset) ) {
			d_output_plane[pos + const_params::fft_length_three_quarters - offset] = s_input_2[threadIdx.x + const_params::fft_length_three_quarters];
		}
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


void forwardCustomFFT(float2 *d_filters, int FFT_size, int nFilters){
	dim3 gridSize(nFilters, 1, 1);
	dim3 blockSize(FFT_size/4, 1, 1);
	
	switch(FFT_size) {
		case 256:
			k_customFFT_GPU_forward<FFT_256><<<gridSize, blockSize, FFT_size*8>>>(d_filters, d_filters);
			break;
			
		case 512:
			k_customFFT_GPU_forward<FFT_512><<<gridSize, blockSize, FFT_size*8>>>(d_filters, d_filters);
			break;
		
		case 1024:
			k_customFFT_GPU_forward<FFT_1024><<<gridSize, blockSize, FFT_size*8>>>(d_filters, d_filters);
			break;

		case 2048:
			k_customFFT_GPU_forward<FFT_2048><<<gridSize, blockSize, FFT_size*8>>>(d_filters, d_filters);
			break;
			
		case 4096:
			k_customFFT_GPU_forward<FFT_4096><<<gridSize, blockSize, FFT_size*8>>>(d_filters, d_filters);
			break;
		
		default : 
			break;
	}
}


void conv_OLS_customFFT(float2 *d_input_signal, float2 *d_output_plane, float2 *d_filters, int signal_length, int convolution_length, int useful_part_size, float h, int offset, int nConvolutions, int nFilters){
	dim3 gridSize(nConvolutions, 1, 1);
	dim3 blockSize(convolution_length/4, 1, 1);
	
	switch(convolution_length) {
		case 256:
			k_GPU_conv_OLS_via_customFFT<FFT_256><<<gridSize, blockSize, convolution_length*8>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, h, offset, nConvolutions, nFilters);
			break;
			
		case 512:
			k_GPU_conv_OLS_via_customFFT<FFT_512><<<gridSize, blockSize, convolution_length*8>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, h, offset, nConvolutions, nFilters);
			break;
		
		case 1024:
			k_GPU_conv_OLS_via_customFFT<FFT_1024><<<gridSize, blockSize, convolution_length*8>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, h, offset, nConvolutions, nFilters);
			break;

		case 2048:
			k_GPU_conv_OLS_via_customFFT<FFT_2048><<<gridSize, blockSize, convolution_length*8>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, h, offset, nConvolutions, nFilters);
			break;
			
		case 4096:
			k_GPU_conv_OLS_via_customFFT<FFT_4096><<<gridSize, blockSize, convolution_length*8>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, h, offset, nConvolutions, nFilters);
			break;
		
		default : 
			break;
	}
}


void conv_OLS_customFFT_2filters(float2 *d_input_signal, float2 *d_output_plane, float2 *d_filters, int signal_length, int convolution_length, int useful_part_size, int offset, int nConvolutions, int nFilters){
	dim3 gridSize(nConvolutions, 1, 1);
	dim3 blockSize(convolution_length/4, 1, 1);
	/*
	switch(convolution_length) {
		case 256:
			k_GPU_conv_OLS_via_customFFT_2filters<FFT_256><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;
			
		case 512:
			k_GPU_conv_OLS_via_customFFT_2filters<FFT_512><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;
		
		case 1024:
			k_GPU_conv_OLS_via_customFFT_2filters<FFT_1024><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;

		case 2048:
			k_GPU_conv_OLS_via_customFFT_2filters<FFT_2048><<<gridSize, blockSize>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters);
			break;
		
		default : 
			break;
	}
	*/
}


void convolution_via_customFFT_benchmark(float2 *d_input_signal, float2 *d_output_plane, float2 *d_filters, int signal_length, int convolution_length, int useful_part_size, float h, int offset, int nConvolutions, int nFilters, double *CONV_time, int kernel_type){
	GpuTimer timer;
	
	// --------> Preparing filters for convolution
	forwardCustomFFT(d_filters, convolution_length, nFilters);
	
	// ----------------------------------------------->
	// --------> Measured part (Convolution)
	timer.Start();
	
	CONV_init();
	if(kernel_type==1){
		conv_OLS_customFFT(d_input_signal, d_output_plane, d_filters, signal_length, convolution_length, useful_part_size, h, offset, nConvolutions, nFilters);
	}
	
	if(kernel_type==2){
		conv_OLS_customFFT_2filters(d_input_signal, d_output_plane, d_filters, signal_length, convolution_length, useful_part_size, offset, nConvolutions, nFilters);
	}
	
	timer.Stop();
	*CONV_time += timer.Elapsed();
	// --------> Measured part (Convolution)
	// ----------------------------------------------->
}


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

int GPU_convolution_OLS_customFFT(float2 *h_input_signal, float2 *h_output_plane, float2 *h_filters, int signal_length, int convolution_length, int filter_length, int past_filter_samples, int nFilters, int nRuns, float h, int offset_modifier, int kernel_type, double *execution_time){
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
	int offset           = past_filter_samples + offset_modifier/2;
	int useful_part_size = convolution_length - (filter_length + offset_modifier) + 1;
	int nConvolutions    = (signal_length + useful_part_size - 1)/useful_part_size;
	
	if(DEBUG) printf("signal_length: %d; filter_length: %d; segment_size: %d;\n", signal_length, filter_length, convolution_length);
	if(DEBUG) printf("offset: %d; nConvolutions: %d; useful_part_size: %d;\n", offset, nConvolutions, useful_part_size);
	
	//---------> Defining variables and their sizes
	float2 *d_output_plane;
	float2 *d_input_signal;
	float2 *d_filters;
	size_t input_size    = signal_length;
	size_t output_size   = nConvolutions*useful_part_size*nFilters;
	size_t template_size = convolution_length*nFilters;
	
	//---------> Checking memory
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required=(( ((float) input_size) + ((float) output_size) + ((float) template_size))*sizeof(float2))/(1024.0*1024.0);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("DEBUG:\n");
	if(DEBUG) printf("    Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(DEBUG) printf("    d_input_signal:          %0.3f MB\n", ((float) input_size*sizeof(float2))/(1024.0*1024.0) ); 
	if(DEBUG) printf("    d_filters:             %0.3f MB\n", ((float) template_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_output_plane:  %0.3f MB\n", ((float) output_size*sizeof(float2))/(1024.0*1024.0) );
	if(memory_required>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(-3);}
	
	//---------> Memory allocation
	if (VERBOSE) printf("Device memory allocation...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input_signal, sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output_plane, sizeof(float2)*output_size));
	checkCudaErrors(cudaMalloc((void **) &d_filters, sizeof(float2)*template_size));
	timer.Stop();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());	
	//------------------------------------------------------------------------------
	//---------> CONV calculation
	
		//-----> Copy chunk of input data to a device
		if (VERBOSE) printf("Transferring data into device memory...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy(d_input_signal, h_input_signal, input_size*sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_filters, h_filters, template_size*sizeof(float2), cudaMemcpyHostToDevice));
		timer.Stop();
		transfer_in+=timer.Elapsed();
		if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
		
		if (DEBUG) printf("Calculating convolution via kFFT...: \t\t");
		total_CONV_kFFT_time = 0;
		for(int f=0; f<nRuns; f++){
			checkCudaErrors(cudaMemcpy(d_input_signal, h_input_signal, input_size*sizeof(float2), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_filters, h_filters, template_size*sizeof(float2), cudaMemcpyHostToDevice));
			convolution_via_customFFT_benchmark(d_input_signal, d_output_plane, d_filters, signal_length, convolution_length, useful_part_size, h, offset, nConvolutions, nFilters, &total_CONV_kFFT_time, kernel_type);
			checkCudaErrors(cudaGetLastError());
		}
		CONV_kFFT_time=total_CONV_kFFT_time/nRuns;
		if (DEBUG) printf("done in %g ms.\n", CONV_kFFT_time);
		*execution_time=CONV_kFFT_time;
		
		//-----> Copy chunk of output data to host
		if (DEBUG) printf("Transferring data to host...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy( h_output_plane, d_output_plane, output_size*sizeof(float2), cudaMemcpyDeviceToHost));
		timer.Stop();
		transfer_out+=timer.Elapsed();
		if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());
	
	
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input_signal));
	checkCudaErrors(cudaFree(d_output_plane));
	checkCudaErrors(cudaFree(d_filters));
	
	return(0);
}
