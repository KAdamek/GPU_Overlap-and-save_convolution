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


__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	ctemp.x=cosf( -2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	ctemp.y=sinf( -2.0f*3.141592654f*fdividef( (float) m, (float) N) );	
	return(ctemp);
}

__device__ __inline__ float2 Get_W_value_inverse(int N, int m){
	float2 ctemp;
	ctemp.x=cosf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	ctemp.y=sinf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	return(ctemp);
}


__inline__ __device__ void do_FFT_CT_DIF_2elem_no_reorder(float2 *s_input){
	float2 A_DFT_value, B_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp;

	int local_id, warp_id;
	int j, m_param, parity;
	int A_read_index, B_read_index;
	int PoT, PoTm1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;
	
	
	//-----> FFT
	//-->
	PoTm1 = (CONV_SIZE>>1);
	PoT   = CONV_SIZE;

	for(q=(FFT_EXP-1);q>4;q--){
		__syncthreads();
		m_param = threadIdx.x & (PoTm1 - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoT, m_param);

		A_read_index=j*PoT + m_param;
		B_read_index=j*PoT + m_param + PoTm1;
		
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		
		A_DFT_value.x = Aftemp.x + Bftemp.x;
		A_DFT_value.y = Aftemp.y + Bftemp.y;
		
		B_DFT_value.x = W.x*(Aftemp.x - Bftemp.x) - W.y*(Aftemp.y - Bftemp.y);
		B_DFT_value.y = W.x*(Aftemp.y - Bftemp.y) + W.y*(Aftemp.x - Bftemp.x);
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}

	__syncthreads();
	A_DFT_value=s_input[local_id + warp_id*2*WARP];
	B_DFT_value=s_input[local_id + warp_id*2*WARP + WARP];
	
	for(q=4;q>=0;q--){
		m_param = (local_id & (PoT - 1));
		j = m_param>>q;
		parity=(1-j*2);
		W = Get_W_value(PoT, j*(m_param-PoTm1));
		
		Aftemp.x = parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x, PoTm1);
		Aftemp.y = parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y, PoTm1);
		Bftemp.x = parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x, PoTm1);
		Bftemp.y = parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y, PoTm1);
		
		A_DFT_value.x = W.x*Aftemp.x - W.y*Aftemp.y; 
		A_DFT_value.y = W.x*Aftemp.y + W.y*Aftemp.x;
		B_DFT_value.x = W.x*Bftemp.x - W.y*Bftemp.y; 
		B_DFT_value.y = W.x*Bftemp.y + W.y*Bftemp.x;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}
	
	s_input[local_id + warp_id*2*WARP] = A_DFT_value;
	s_input[local_id + warp_id*2*WARP + WARP] = B_DFT_value;
	
	__syncthreads();
}

__inline__ __device__ void do_FFT_mk11_4elem_no_reorder(float2 *s_input){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

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
	A_DFT_value=s_input[local_id + (warp_id<<2)*WARP];
	B_DFT_value=s_input[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 3*WARP];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x,1);
	A_DFT_value.y=parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y,1);
	B_DFT_value.x=parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x,1);
	B_DFT_value.y=parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y,1);
	C_DFT_value.x=parity*C_DFT_value.x + __shfl_xor(C_DFT_value.x,1);
	C_DFT_value.y=parity*C_DFT_value.y + __shfl_xor(C_DFT_value.y,1);
	D_DFT_value.x=parity*D_DFT_value.x + __shfl_xor(D_DFT_value.x,1);
	D_DFT_value.y=parity*D_DFT_value.y + __shfl_xor(D_DFT_value.y,1);
	
	
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
		
		A_DFT_value.x = Aftemp.x + parity*__shfl_xor(Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*__shfl_xor(Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*__shfl_xor(Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*__shfl_xor(Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*__shfl_xor(Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*__shfl_xor(Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*__shfl_xor(Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*__shfl_xor(Dftemp.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*WARP;
	s_input[itemp]          = A_DFT_value;
	s_input[itemp + WARP]   = B_DFT_value;
	s_input[itemp + 2*WARP] = C_DFT_value;
	s_input[itemp + 3*WARP] = D_DFT_value;
	
	for(q=5;q<(FFT_EXP-1);q++){
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

__inline__ __device__ void do_FFT_CT_DIF_4elem_no_reorder(float2 *s_input){
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
	PoTm1 = (CONV_SIZE>>1);
	PoT   = CONV_SIZE;
	
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
	
	for(q=(FFT_EXP-2);q>4;q--){
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
		
		Aftemp.x = parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x, PoTm1);
		Aftemp.y = parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y, PoTm1);
		Bftemp.x = parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x, PoTm1);
		Bftemp.y = parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y, PoTm1);
		Cftemp.x = parity*C_DFT_value.x + __shfl_xor(C_DFT_value.x, PoTm1);
		Cftemp.y = parity*C_DFT_value.y + __shfl_xor(C_DFT_value.y, PoTm1);
		Dftemp.x = parity*D_DFT_value.x + __shfl_xor(D_DFT_value.x, PoTm1);
		Dftemp.y = parity*D_DFT_value.y + __shfl_xor(D_DFT_value.y, PoTm1);
		
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
}

__inline__ __device__ void do_FFT_mk11_4elem_2vertical_no_reorder(float2 *s_input1, float2 *s_input2){
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
	
	A_DFT_value1.x=parity*A_DFT_value1.x + __shfl_xor(A_DFT_value1.x,1);
	A_DFT_value1.y=parity*A_DFT_value1.y + __shfl_xor(A_DFT_value1.y,1);
	B_DFT_value1.x=parity*B_DFT_value1.x + __shfl_xor(B_DFT_value1.x,1);
	B_DFT_value1.y=parity*B_DFT_value1.y + __shfl_xor(B_DFT_value1.y,1);
	C_DFT_value1.x=parity*C_DFT_value1.x + __shfl_xor(C_DFT_value1.x,1);
	C_DFT_value1.y=parity*C_DFT_value1.y + __shfl_xor(C_DFT_value1.y,1);
	D_DFT_value1.x=parity*D_DFT_value1.x + __shfl_xor(D_DFT_value1.x,1);
	D_DFT_value1.y=parity*D_DFT_value1.y + __shfl_xor(D_DFT_value1.y,1);
	
	A_DFT_value2.x=parity*A_DFT_value2.x + __shfl_xor(A_DFT_value2.x,1);
	A_DFT_value2.y=parity*A_DFT_value2.y + __shfl_xor(A_DFT_value2.y,1);
	B_DFT_value2.x=parity*B_DFT_value2.x + __shfl_xor(B_DFT_value2.x,1);
	B_DFT_value2.y=parity*B_DFT_value2.y + __shfl_xor(B_DFT_value2.y,1);
	C_DFT_value2.x=parity*C_DFT_value2.x + __shfl_xor(C_DFT_value2.x,1);
	C_DFT_value2.y=parity*C_DFT_value2.y + __shfl_xor(C_DFT_value2.y,1);
	D_DFT_value2.x=parity*D_DFT_value2.x + __shfl_xor(D_DFT_value2.x,1);
	D_DFT_value2.y=parity*D_DFT_value2.y + __shfl_xor(D_DFT_value2.y,1);
	
	
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
		
		A_DFT_value1.x = Aftemp1.x + parity*__shfl_xor(Aftemp1.x,PoT);
		A_DFT_value1.y = Aftemp1.y + parity*__shfl_xor(Aftemp1.y,PoT);
		B_DFT_value1.x = Bftemp1.x + parity*__shfl_xor(Bftemp1.x,PoT);
		B_DFT_value1.y = Bftemp1.y + parity*__shfl_xor(Bftemp1.y,PoT);
		C_DFT_value1.x = Cftemp1.x + parity*__shfl_xor(Cftemp1.x,PoT);
		C_DFT_value1.y = Cftemp1.y + parity*__shfl_xor(Cftemp1.y,PoT);
		D_DFT_value1.x = Dftemp1.x + parity*__shfl_xor(Dftemp1.x,PoT);
		D_DFT_value1.y = Dftemp1.y + parity*__shfl_xor(Dftemp1.y,PoT);	
		
		A_DFT_value2.x = Aftemp2.x + parity*__shfl_xor(Aftemp2.x,PoT);
		A_DFT_value2.y = Aftemp2.y + parity*__shfl_xor(Aftemp2.y,PoT);
		B_DFT_value2.x = Bftemp2.x + parity*__shfl_xor(Bftemp2.x,PoT);
		B_DFT_value2.y = Bftemp2.y + parity*__shfl_xor(Bftemp2.y,PoT);
		C_DFT_value2.x = Cftemp2.x + parity*__shfl_xor(Cftemp2.x,PoT);
		C_DFT_value2.y = Cftemp2.y + parity*__shfl_xor(Cftemp2.y,PoT);
		D_DFT_value2.x = Dftemp2.x + parity*__shfl_xor(Dftemp2.x,PoT);
		D_DFT_value2.y = Dftemp2.y + parity*__shfl_xor(Dftemp2.y,PoT);	
		
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
	
	for(q=5;q<(FFT_EXP-1);q++){
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



__global__ void FFT_External_2elem_no_reorder(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*CONV_SIZE];
	s_input[threadIdx.x + (CONV_SIZE>>1)]=d_input[threadIdx.x + (CONV_SIZE>>1) + blockIdx.x*CONV_SIZE];
	
	__syncthreads();
	do_FFT_CT_DIF_2elem_no_reorder(s_input);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*CONV_SIZE]=s_input[threadIdx.x];
	d_output[threadIdx.x + (CONV_SIZE>>1) + blockIdx.x*CONV_SIZE]=s_input[threadIdx.x + (CONV_SIZE>>1)];
}

__device__ __inline__ void prepare_signal_4elem(float2* s_signal, float2 const* __restrict__ d_input_signal, int useful_part_size, int offset) {
	int pos = blockIdx.x*useful_part_size;
	
	if(blockIdx.x==0){ //leading part
		pos = blockIdx.x*useful_part_size + threadIdx.x - offset;
		if( threadIdx.x<offset ){
			s_signal[threadIdx.x].x = 0;
			s_signal[threadIdx.x].y = 0;
		}
		else {
			s_signal[threadIdx.x] = __ldg(&d_input_signal[pos]);
		}
		
		if( (threadIdx.x + (CONV_HALF>>1))<offset ){
			s_signal[threadIdx.x + (CONV_HALF>>1)].x = 0;
			s_signal[threadIdx.x + (CONV_HALF>>1)].y = 0;
		}
		else {
			s_signal[threadIdx.x + (CONV_HALF>>1)] = __ldg(&d_input_signal[pos + (CONV_HALF>>1)]);
		}
		
		// I do not need to test if (threadIdx.x + CONV_HALF)<offset because if it is then unspoiled lenght = 0
		s_signal[threadIdx.x + CONV_HALF] = __ldg(&d_input_signal[pos + CONV_HALF]);
		s_signal[threadIdx.x + 3*(CONV_HALF>>1)] = __ldg(&d_input_signal[pos + 3*(CONV_HALF>>1)]);
	}
	else { //middle parts
		pos = blockIdx.x*useful_part_size - offset + threadIdx.x;
		s_signal[threadIdx.x]             = __ldg(&d_input_signal[pos]);
		s_signal[threadIdx.x + (CONV_HALF>>1)] = __ldg(&d_input_signal[pos + (CONV_HALF>>1)]);
		s_signal[threadIdx.x + CONV_HALF] = __ldg(&d_input_signal[pos + CONV_HALF]);
		s_signal[threadIdx.x + 3*(CONV_HALF>>1)] = __ldg(&d_input_signal[pos + 3*(CONV_HALF>>1)]);
	}	
}



__global__ void GPU_CONV_kFFT_mk11_4elem_dynamic(float2 const* __restrict__ d_input_signal, float2 *d_output_plane_reduced, float2 const* __restrict__ d_templates, int useful_part_size, int offset, int nConvolutions, int nTemplates) {
	__shared__ float2 s_input_1[CONV_SIZE];
	// Convolution
	float2 r_templates_1[4];
	float2 signal[4];
	int pos, t;
	// Loading data
	prepare_signal_4elem(s_input_1, d_input_signal, useful_part_size, offset);

	do_FFT_CT_DIF_4elem_no_reorder(s_input_1);
	
	signal[0]=s_input_1[threadIdx.x];
	signal[1]=s_input_1[threadIdx.x + (CONV_HALF>>1)];
	signal[2]=s_input_1[threadIdx.x + CONV_HALF];
	signal[3]=s_input_1[threadIdx.x + 3*(CONV_HALF>>1)];
	
	for(t=0; t<nTemplates; t++){
		// Loading templates
		pos = t*CONV_SIZE + threadIdx.x;
		r_templates_1[0]=__ldg(&d_templates[pos]);
		r_templates_1[1]=__ldg(&d_templates[pos + (CONV_HALF>>1)]);
		r_templates_1[2]=__ldg(&d_templates[pos + CONV_HALF]);
		r_templates_1[3]=__ldg(&d_templates[pos + 3*(CONV_HALF>>1)]);

		// Convolution
		s_input_1[threadIdx.x].x                    = r_templates_1[0].x*signal[0].x - r_templates_1[0].y*signal[0].y;
		s_input_1[threadIdx.x].y                    = r_templates_1[0].x*signal[0].y + r_templates_1[0].y*signal[0].x;
		s_input_1[threadIdx.x + (CONV_HALF>>1)].x   = r_templates_1[1].x*signal[1].x - r_templates_1[1].y*signal[1].y;
		s_input_1[threadIdx.x + (CONV_HALF>>1)].y   = r_templates_1[1].x*signal[1].y + r_templates_1[1].y*signal[1].x;
		s_input_1[threadIdx.x + CONV_HALF].x        = r_templates_1[2].x*signal[2].x - r_templates_1[2].y*signal[2].y;
		s_input_1[threadIdx.x + CONV_HALF].y        = r_templates_1[2].x*signal[2].y + r_templates_1[2].y*signal[2].x;
		s_input_1[threadIdx.x + 3*(CONV_HALF>>1)].x = r_templates_1[3].x*signal[3].x - r_templates_1[3].y*signal[3].y;
		s_input_1[threadIdx.x + 3*(CONV_HALF>>1)].y = r_templates_1[3].x*signal[3].y + r_templates_1[3].y*signal[3].x;
		
		__syncthreads();
		
		//----------> IFFT
		do_FFT_mk11_4elem_no_reorder(s_input_1);
		//----------<
		
		// Saving data
		pos = t*useful_part_size*nConvolutions + blockIdx.x*useful_part_size + threadIdx.x;
		if( threadIdx.x>=offset && threadIdx.x<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos - offset] = s_input_1[threadIdx.x];
		}
		if( (threadIdx.x+(CONV_HALF>>1))>=offset && (threadIdx.x+(CONV_HALF>>1))<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + (CONV_HALF>>1) - offset] = s_input_1[threadIdx.x + (CONV_HALF>>1)];
		}
		if( (threadIdx.x+CONV_HALF)>=offset && (threadIdx.x+CONV_HALF)<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + CONV_HALF - offset] = s_input_1[threadIdx.x + CONV_HALF];
		}
		if( (threadIdx.x+3*(CONV_HALF>>1))>=offset && (threadIdx.x+3*(CONV_HALF>>1))<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + 3*(CONV_HALF>>1) - offset] = s_input_1[threadIdx.x + 3*(CONV_HALF>>1)];
		}
	}
}



__global__ void GPU_CONV_kFFT_mk11_4elem_2v_dynamic(float2 const* __restrict__ d_input_signal, float2 *d_output_plane_reduced, float2 const* __restrict__ d_templates, int useful_part_size, int offset, int nConvolutions, int nTemplates) {
	__shared__ float2 s_input_1[CONV_SIZE];
	__shared__ float2 s_input_2[CONV_SIZE];
	// Convolution
	float2 r_templates_1[4];
	float2 r_templates_2[4];
	float2 signal[4];
	int pos, t;
	// Loading data
	prepare_signal_4elem(s_input_1, d_input_signal, useful_part_size, offset);

	do_FFT_CT_DIF_4elem_no_reorder(s_input_1);
	
	signal[0]=s_input_1[threadIdx.x];
	signal[1]=s_input_1[threadIdx.x + (CONV_HALF>>1)];
	signal[2]=s_input_1[threadIdx.x + CONV_HALF];
	signal[3]=s_input_1[threadIdx.x + 3*(CONV_HALF>>1)];
	
	for(t=0; t<(nTemplates>>1); t++){
		// Loading templates
		pos = 2*t*CONV_SIZE + threadIdx.x;
		
		r_templates_1[0]=__ldg(&d_templates[pos]);
		r_templates_1[1]=__ldg(&d_templates[pos + (CONV_HALF>>1)]);
		r_templates_1[2]=__ldg(&d_templates[pos + CONV_HALF]);
		r_templates_1[3]=__ldg(&d_templates[pos + 3*(CONV_HALF>>1)]);
		
		r_templates_2[0]=__ldg(&d_templates[pos + CONV_SIZE]);
		r_templates_2[1]=__ldg(&d_templates[pos + CONV_SIZE + (CONV_HALF>>1)]);
		r_templates_2[2]=__ldg(&d_templates[pos + CONV_SIZE + CONV_HALF]);
		r_templates_2[3]=__ldg(&d_templates[pos + CONV_SIZE + 3*(CONV_HALF>>1)]);

		// Convolution
		s_input_1[threadIdx.x].x                    = r_templates_1[0].x*signal[0].x - r_templates_1[0].y*signal[0].y;
		s_input_1[threadIdx.x].y                    = r_templates_1[0].x*signal[0].y + r_templates_1[0].y*signal[0].x;
		s_input_1[threadIdx.x + (CONV_HALF>>1)].x   = r_templates_1[1].x*signal[1].x - r_templates_1[1].y*signal[1].y;
		s_input_1[threadIdx.x + (CONV_HALF>>1)].y   = r_templates_1[1].x*signal[1].y + r_templates_1[1].y*signal[1].x;
		s_input_1[threadIdx.x + CONV_HALF].x        = r_templates_1[2].x*signal[2].x - r_templates_1[2].y*signal[2].y;
		s_input_1[threadIdx.x + CONV_HALF].y        = r_templates_1[2].x*signal[2].y + r_templates_1[2].y*signal[2].x;
		s_input_1[threadIdx.x + 3*(CONV_HALF>>1)].x = r_templates_1[3].x*signal[3].x - r_templates_1[3].y*signal[3].y;
		s_input_1[threadIdx.x + 3*(CONV_HALF>>1)].y = r_templates_1[3].x*signal[3].y + r_templates_1[3].y*signal[3].x;
		
		s_input_2[threadIdx.x].x                    = r_templates_2[0].x*signal[0].x - r_templates_2[0].y*signal[0].y;
		s_input_2[threadIdx.x].y                    = r_templates_2[0].x*signal[0].y + r_templates_2[0].y*signal[0].x;
		s_input_2[threadIdx.x + (CONV_HALF>>1)].x   = r_templates_2[1].x*signal[1].x - r_templates_2[1].y*signal[1].y;
		s_input_2[threadIdx.x + (CONV_HALF>>1)].y   = r_templates_2[1].x*signal[1].y + r_templates_2[1].y*signal[1].x;
		s_input_2[threadIdx.x + CONV_HALF].x        = r_templates_2[2].x*signal[2].x - r_templates_2[2].y*signal[2].y;
		s_input_2[threadIdx.x + CONV_HALF].y        = r_templates_2[2].x*signal[2].y + r_templates_2[2].y*signal[2].x;
		s_input_2[threadIdx.x + 3*(CONV_HALF>>1)].x = r_templates_2[3].x*signal[3].x - r_templates_2[3].y*signal[3].y;
		s_input_2[threadIdx.x + 3*(CONV_HALF>>1)].y = r_templates_2[3].x*signal[3].y + r_templates_2[3].y*signal[3].x;
		
		__syncthreads();
		
		//----------> IFFT
		do_FFT_mk11_4elem_2vertical_no_reorder(s_input_1, s_input_2);
		//----------<
		
		// Saving data
		pos = 2*t*useful_part_size*nConvolutions + blockIdx.x*useful_part_size + threadIdx.x;
		if( threadIdx.x>=offset && threadIdx.x<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos - offset] = s_input_1[threadIdx.x];
		}
		if( (threadIdx.x+(CONV_HALF>>1))>=offset && (threadIdx.x+(CONV_HALF>>1))<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + (CONV_HALF>>1) - offset] = s_input_1[threadIdx.x + (CONV_HALF>>1)];
		}
		if( (threadIdx.x+CONV_HALF)>=offset && (threadIdx.x+CONV_HALF)<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + CONV_HALF - offset] = s_input_1[threadIdx.x + CONV_HALF];
		}
		if( (threadIdx.x+3*(CONV_HALF>>1))>=offset && (threadIdx.x+3*(CONV_HALF>>1))<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + 3*(CONV_HALF>>1) - offset] = s_input_1[threadIdx.x + 3*(CONV_HALF>>1)];
		}
		pos = pos + useful_part_size*nConvolutions;
		if( threadIdx.x>=offset && threadIdx.x<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos - offset] = s_input_2[threadIdx.x];
		}
		if( (threadIdx.x+(CONV_HALF>>1))>=offset && (threadIdx.x+(CONV_HALF>>1))<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + (CONV_HALF>>1) - offset] = s_input_2[threadIdx.x + (CONV_HALF>>1)];
		}
		if( (threadIdx.x+CONV_HALF)>=offset && (threadIdx.x+CONV_HALF)<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + CONV_HALF - offset] = s_input_2[threadIdx.x + CONV_HALF];
		}
		if( (threadIdx.x+3*(CONV_HALF>>1))>=offset && (threadIdx.x+3*(CONV_HALF>>1))<(useful_part_size+offset) ) {
			d_output_plane_reduced[pos + 3*(CONV_HALF>>1) - offset] = s_input_2[threadIdx.x + 3*(CONV_HALF>>1)];
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

void CONV_kFFT_benchmark(float2 *d_input_signal, float2 *d_output_plane_reduced, float2 *d_template, int useful_part_size, int offset, int nConvolutions, int nTemplates, double *CONV_time, int kernel_type, int dynamic){
	GpuTimer timer;
	
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nConvolutions;
	int nCUDAblocks_y=1;
	nCUDAblocks_y = 1;
	
	dim3 template_gridSize(nTemplates, 1, 1);
	dim3 template_blockSize(CONV_HALF, 1, 1);
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
	dim3 blockSize((CONV_HALF>>1), 1, 1);
	
	// --------> Preparing templates for convolution
	FFT_External_2elem_no_reorder<<<template_gridSize, template_blockSize, CONV_SIZE*8>>>( d_template, d_template);
	
	// ----------------------------------------------->
	// --------> Measured part (Convolution)
	timer.Start();
	
	CONV_init();
	if(kernel_type==1){
		GPU_CONV_kFFT_mk11_4elem_dynamic<<<gridSize,blockSize>>>(d_input_signal, d_output_plane_reduced, d_template, useful_part_size, offset, nConvolutions, nTemplates);
	}
	
	if(kernel_type==2){
		GPU_CONV_kFFT_mk11_4elem_2v_dynamic<<<gridSize,blockSize>>>(d_input_signal, d_output_plane_reduced, d_template, useful_part_size, offset, nConvolutions, nTemplates);
	}
	
	timer.Stop();
	*CONV_time += timer.Elapsed();
	// --------> Measured part (Convolution)
	// ----------------------------------------------->
}


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

int GPU_CONV(float2 *h_input_signal, float2 *h_output_plane_reduced, float2 *h_templates, int useful_part_size, int offset, int nConvolutions, int nTemplates, int nRuns, int kernel_type, double *execution_time){
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
	
	//---------> Time measurements
	double transfer_in, transfer_out, CONV_kFFT_time, total_CONV_kFFT_time;
	transfer_in=0.0; transfer_out=0.0; CONV_kFFT_time=0.0; total_CONV_kFFT_time=0;
	GpuTimer timer;
	
	//---------> Defining variables and their sizes
	float2 *d_output_plane_reduced;
	float2 *d_input_signal;
	float2 *d_templates;
	size_t input_size          = nConvolutions*useful_part_size + offset*2;
	size_t output_size_reduced = nConvolutions*useful_part_size*nTemplates;
	size_t template_size = CONV_SIZE*nTemplates;
	
	//---------> Checking memory
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required=(( ((float) input_size) + ((float) output_size_reduced) + ((float) template_size))*sizeof(float2))/(1024.0*1024.0);
	if(DEBUG) printf("\n");
	if(DEBUG) printf("DEBUG:\n");
	if(DEBUG) printf("    Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(DEBUG) printf("    d_input_signal:          %0.3f MB\n", ((float) input_size*sizeof(float2))/(1024.0*1024.0) ); 
	if(DEBUG) printf("    d_templates:             %0.3f MB\n", ((float) template_size*sizeof(float2))/(1024.0*1024.0) );
	if(DEBUG) printf("    d_output_plane_reduced:  %0.3f MB\n", ((float) output_size_reduced*sizeof(float2))/(1024.0*1024.0) );
	if(memory_required>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(-3);}
	
	//---------> Memory allocation
	if (VERBOSE) printf("Device memory allocation...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input_signal,  sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output_plane_reduced, sizeof(float2)*output_size_reduced));
	checkCudaErrors(cudaMalloc((void **) &d_templates, sizeof(float2)*template_size));
	checkCudaErrors(cudaMemset((void*) &d_input_signal[(nConvolutions-1)*useful_part_size], 0, CONV_SIZE*sizeof(float2)));
	timer.Stop();
	if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());	
	//------------------------------------------------------------------------------
	//---------> CONV calculation
	
		//-----> Copy chunk of input data to a device
		if (VERBOSE) printf("Transferring data into device memory...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy(d_input_signal, h_input_signal, input_size*sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_templates, h_templates, template_size*sizeof(float2), cudaMemcpyHostToDevice));
		timer.Stop();
		transfer_in+=timer.Elapsed();
		if (VERBOSE) printf("done in %g ms.\n", timer.Elapsed());
		
		if (DEBUG) printf("Calculating convolution via kFFT...: \t\t");
		total_CONV_kFFT_time = 0;
		for(int f=0; f<nRuns; f++){
			checkCudaErrors(cudaMemcpy(d_input_signal, h_input_signal, input_size*sizeof(float2), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_templates, h_templates, template_size*sizeof(float2), cudaMemcpyHostToDevice));
			CONV_kFFT_benchmark(d_input_signal, d_output_plane_reduced, d_templates, useful_part_size, offset, nConvolutions, nTemplates, &total_CONV_kFFT_time, kernel_type, 1);
		}
		CONV_kFFT_time=total_CONV_kFFT_time/nRuns;
		if (DEBUG) printf("done in %g ms.\n", CONV_kFFT_time);
		*execution_time=CONV_kFFT_time;
		
		//-----> Copy chunk of output data to host
		if (DEBUG) printf("Transferring data to host...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy( h_output_plane_reduced, d_output_plane_reduced, output_size_reduced*sizeof(float2), cudaMemcpyDeviceToHost));
		timer.Stop();
		transfer_out+=timer.Elapsed();
		if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());
	
	
	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input_signal));
	checkCudaErrors(cudaFree(d_output_plane_reduced));
	checkCudaErrors(cudaFree(d_templates));
	
	return(0);
}
