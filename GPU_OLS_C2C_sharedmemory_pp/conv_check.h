#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

double max_error = 1.0e-4;

void CPU_time_domain(float2 *h_input, float2 *h_CPU_output_timedomain, float2 *h_filters, int signal_length, int filter_length, int past_filter_samples, int nFilters){
	for(int f=0; f<nFilters; f++){
		printf(".");  fflush(stdout);
		for(int s=0; s<signal_length; s++){
			float2 ac;
			ac.x = 0; ac.y = 0;
			for(int i=0; i<filter_length; i++){
				int filter_pos = filter_length - 1 - i;
				float2 fv, sv;
				fv = h_filters[f*filter_length + filter_pos];
				int signal_pos = (s + i - past_filter_samples);
				if(signal_pos>=0 && signal_pos<signal_length) sv = h_input[signal_pos];
				else {sv.x = 0; sv.y = 0;}
				ac.x = ac.x + sv.x*fv.x - sv.y*fv.y;
				ac.y = ac.y + sv.x*fv.y + sv.y*fv.x;
			}
			h_CPU_output_timedomain[f*(signal_length + filter_length - 1) + s] = ac;
		}
	}
	printf("\n");
}

void CPU_postprocess(float2 *h_CPU_postprocessed, float2 *h_CPU_output_reduced, int nTimesamples, int nFilters, float h){
	float2 left, right, result;
	
	for(int f=0; f<nFilters; f++){
		for(int s=0; s<nTimesamples-1; s++){
			int pos = f*nTimesamples + s;
			if( s==0 ) {
				left = h_CPU_output_reduced[pos];
			}
			else {
				left = h_CPU_output_reduced[pos-1];
			}
			
			if( s>=(nTimesamples-1) ) {
				right = h_CPU_output_reduced[f*nTimesamples + nTimesamples - 1];
			}
			else {
				right = h_CPU_output_reduced[pos+1];
			}
			
			result.x = (left.x - right.x)/(2.0*h);
			result.y = (left.y - right.y)/(2.0*h);
			h_CPU_postprocessed[pos] = result;
		}
	}
}


float get_error(float2 A_f2, float2 B_f2){
	float error, div_error=10000, per_error=10000, order=0;
	int power;
	float A = max(A_f2.x, A_f2.y);
	float B = max(B_f2.x, B_f2.y);
	if(A<0) A = -A;
	if(B<0) B = -B;
	
	if (A>B) {
		div_error = A-B;
		if(B>10){
			power = (int) log10(B);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	else {
		div_error = B-A;
		if(A>10){
			power = (int) log10(A);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	
	if(div_error<per_error) error = div_error;
	else error = per_error;
	return(error);
}

int Compare_data(float2 *CPU_result, float2 *GPU_result, float CPU_scale, float GPU_scale, int CPU_offset, int GPU_offset, int CPU_dim_x, int GPU_dim_x, int dim_y, int nSamples, int useful_part_size, double *total_error, double *mean_error){
	double total_error_l = 0, mean_error_l = 0;
	size_t nErrors = 0;
	int cislo = 0;
	float error;
	
	for(int y=0; y<dim_y; y++){
		for(int x=0; x<nSamples; x++){
			int CPU_pos = y*CPU_dim_x + x + CPU_offset;
			int GPU_pos = y*GPU_dim_x + x + GPU_offset;
			if((x + CPU_offset)<CPU_dim_x && (x + GPU_offset)<GPU_dim_x){
				float2 CPU, GPU;
				CPU.x = CPU_result[CPU_pos].x/CPU_scale; CPU.y = CPU_result[CPU_pos].y/CPU_scale;
				GPU.x = GPU_result[GPU_pos].x/GPU_scale; GPU.y = GPU_result[GPU_pos].y/GPU_scale;
				
				
				error = get_error(CPU, GPU);
				total_error_l = total_error_l + error;
				if( error > max_error ){
					nErrors++;
					if(cislo<40){
						printf("Error [%f] CPU [%f;%f] GPU [%f;%f] x=%d; y=%d segment=%d; s.x=%d\n", error, CPU.x, CPU.y, GPU.x, GPU.y, x, y, (int) ((x + GPU_offset)/useful_part_size), (x + GPU_offset)%useful_part_size);
						cislo++;
					}
				}
			}
		}
	}
	mean_error_l = total_error_l/(((double) nSamples)*((double) dim_y));
	(*total_error) = total_error_l;
	(*mean_error) = mean_error_l;
	return(nErrors);
}


void Full_CONV_check(float2 *GPU_result, float2 *h_input, float2 *h_filters, int signal_length, int filter_length, int past_filter_samples, int useful_part_size, int offset, int conv_length, int nConvolutions, int nFilters, float h, double *cumulative_error, double *mean_error){
	size_t output_size_timedomain = (signal_length + filter_length - 1)*nFilters;
	float2 *h_CPU_output_timedomain;
	float2 *h_CPU_postprocessed;
	h_CPU_output_timedomain = (float2 *)malloc(output_size_timedomain*sizeof(float2));
	h_CPU_postprocessed     = (float2 *)malloc(output_size_timedomain*sizeof(float2));
	memset(h_CPU_output_timedomain, 0.0, output_size_timedomain*sizeof(float2));
	memset(h_CPU_postprocessed, 0.0, output_size_timedomain*sizeof(float2));
	
	printf("\n--> Time-domain convolution:");
	CPU_time_domain(h_input, h_CPU_output_timedomain, h_filters, signal_length, filter_length, past_filter_samples, nFilters);
	
	printf("\n--> Post-processing:\n");
	CPU_postprocess(h_CPU_postprocessed, h_CPU_output_timedomain, (signal_length + filter_length - 1), nFilters, h);
		
	float GPU_scale, CPU_scale;
	int CPU_offset, GPU_offset, CPU_dim_x, GPU_dim_x, nSamples;

	#ifdef POST_PROCESS
	
	printf("\n--> Comparison to CPU time-domain with post-processing:\n");
	GPU_scale = 1.0;
	CPU_scale = 1.0;
	GPU_offset = 0;
	CPU_offset = 0;
	GPU_dim_x = nConvolutions*useful_part_size;
	CPU_dim_x = (signal_length + filter_length - 1);
	nSamples = signal_length - offset;	
	Compare_data(h_CPU_postprocessed, GPU_result, CPU_scale, GPU_scale, CPU_offset, GPU_offset, CPU_dim_x, GPU_dim_x, nFilters, nSamples, useful_part_size, cumulative_error, mean_error);
	//printf("----> Total error: %e; Mean error: %e\n", (double) *cumulative_error, (double) *mean_error);
	if((*mean_error)<1.0e-4) printf("PASSED\n");
	else printf("FAILED\n");	
	#else
	
	printf("\n--> Comparison to CPU time-domain:\n");
	GPU_scale = conv_length;
	GPU_scale = 1.0;
	CPU_scale = 1.0;
	GPU_offset = 0;
	CPU_offset = 0;
	GPU_dim_x = nConvolutions*useful_part_size;
	CPU_dim_x = (signal_length + filter_length - 1);
	nSamples = signal_length-offset;
	Compare_data(h_CPU_output_timedomain, GPU_result, CPU_scale, GPU_scale, CPU_offset, GPU_offset, CPU_dim_x, GPU_dim_x, nFilters, nSamples, useful_part_size, cumulative_error, mean_error);
	//printf("----> Total error: %e; Mean error: %e\n", (double) *cumulative_error, (double) *mean_error);
	if((*mean_error)<1.0e-4) printf("PASSED\n");
	else printf("FAILED\n");
	
	#endif
	
	free(h_CPU_output_timedomain);
	free(h_CPU_postprocessed);
}
