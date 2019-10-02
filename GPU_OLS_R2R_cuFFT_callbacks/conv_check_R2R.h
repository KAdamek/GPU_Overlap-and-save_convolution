#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

double max_error = 1.0e-4;

void CPU_time_domain(float *h_input, float *h_CPU_output_timedomain, float *h_filters, int signal_length, int filter_length, int nFilters){
	for(int f=0; f<nFilters; f++){
		printf(".");  fflush(stdout);
		for(int s=0; s<signal_length; s++){
			float ac;
			ac = 0;
			for(int i=0; i<filter_length; i++){
				int filter_pos = filter_length - 1 - i;
				float fv, sv;
				fv = h_filters[f*filter_length + filter_pos];
				int signal_pos = (s + i - (filter_length>>1));
				if(signal_pos>=0 && signal_pos<signal_length) sv = h_input[signal_pos];
				else {sv = 0;}
				ac = ac + sv*fv;
			}
			h_CPU_output_timedomain[f*(signal_length + filter_length - 1) + s] = ac;
		}
	}
	printf("\n");
}

float get_error(float A, float B){
	float error, div_error=10000, per_error=10000, order=0;
	int power;
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

int Compare_data(float *CPU_result, float *GPU_result, int dim_x, int dim_y, int signal_length, int useful_part_size, double *total_error, double *mean_error){
	double total_error_l = 0, mean_error_l = 0;
	size_t nErrors = 0;
	int cislo = 0;
	float error;
	
	for(int y=0; y<dim_y; y++){
		for(int x=0; x<signal_length; x++){
			int pos = y*dim_x + x;
			error = get_error(CPU_result[pos], GPU_result[pos]);
			total_error_l = total_error_l + error;
			if( error > max_error ){
				nErrors++;
				if(cislo<40){
					printf("Error [%f] CPU [%f] GPU [%f] x=%d; y=%d segment=%d\n", error, CPU_result[pos], GPU_result[pos], x, y, (int) (x/useful_part_size));
					cislo++;
				}
			}
		}
	}
	mean_error_l = total_error_l/(((double) dim_x)*((double) dim_y));
	(*total_error) = total_error_l;
	(*mean_error) = mean_error_l;
	return(nErrors);
}

int Compare_data(float *CPU_result, float *GPU_result, float CPU_scale, float GPU_scale, int CPU_offset, int GPU_offset, int CPU_dim_x, int GPU_dim_x, int dim_y, int nSamples, int useful_part_size, double *total_error, double *mean_error){
	double total_error_l = 0, mean_error_l = 0;
	size_t nErrors = 0;
	int cislo = 0;
	float error;
	
	for(int y=0; y<dim_y; y++){
		for(int x=0; x<nSamples; x++){
			int CPU_pos = y*CPU_dim_x + x + CPU_offset;
			int GPU_pos = y*GPU_dim_x + x + GPU_offset;
			float CPU, GPU;
			CPU = CPU_result[CPU_pos]/CPU_scale;
			GPU = GPU_result[GPU_pos]/GPU_scale;
			
			
			error = get_error(CPU, GPU);
			total_error_l = total_error_l + error;
			if( error > max_error ){
				nErrors++;
				if(cislo<40){
					printf("Error [%f] CPU [%f] GPU [%f] x=%d; y=%d segment=%d; s.x=%d\n", error, CPU, GPU, x, y, (int) (x/useful_part_size), x%useful_part_size);
					cislo++;
				}
			}
		}
	}
	mean_error_l = total_error_l/(((double) nSamples)*((double) dim_y));
	(*total_error) = total_error_l;
	(*mean_error) = mean_error_l;
	return(nErrors);
}


void Full_CONV_check(float *GPU_result, float *h_input_real, float *h_filters, int signal_length, int filter_length, int useful_part_size, int offset, int conv_length, int nConvolutions, int nFilters, double *cumulative_error, double *mean_error){
	float GPU_scale, CPU_scale;
	int CPU_offset, GPU_offset, CPU_dim_x, GPU_dim_x, nSamples;
	
	//----------------------- CPU time-domain
	size_t output_size_timedomain = (signal_length + filter_length - 1)*nFilters;
	float *h_CPU_output_timedomain;
	h_CPU_output_timedomain = (float *)malloc(output_size_timedomain*sizeof(float));
	memset(h_CPU_output_timedomain, 0.0, output_size_timedomain*sizeof(float));
	
	printf("\n--> Time-domain convolution:");
	CPU_time_domain(h_input_real, h_CPU_output_timedomain, h_filters, signal_length, filter_length, nFilters);
	
	printf("\n--> Comparison to CPU time-domain:\n");
	GPU_scale = conv_length/2;
	CPU_scale = 1.0;
	GPU_offset = 0;
	CPU_offset = 0;
	GPU_dim_x = nConvolutions*useful_part_size;
	CPU_dim_x = (signal_length + filter_length - 1);
	nSamples = signal_length - offset;	
	Compare_data(h_CPU_output_timedomain, GPU_result, CPU_scale, GPU_scale, CPU_offset, GPU_offset, CPU_dim_x, GPU_dim_x, nFilters, nSamples, useful_part_size, cumulative_error, mean_error);
	//printf("----> Total error: %e; Mean error: %e\n", (double) *cumulative_error, (double) *mean_error);
	if((*mean_error)<1.0e-4) printf("PASSED\n");
	else printf("FAILED\n");
	
	
	free(h_CPU_output_timedomain);
	//-------------------------------------------------<	
}
