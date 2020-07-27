//********************************************************************************************
//* This is GPU implementation of a Overlap-and-save method for calculating convolution. 
//* Copyright (C) 2019  Adámek Karel
//* 
//* Authors: Karel Adamek ( ORCID:0000-0003-2797-0595; https://github.com/KAdamek ), Wesley Armour ( ORCID:0000-0003-1756-3064 ), Sofia Dimoudi 
//********************************************************************************************

#include "debug.h"
#include "params.h"
#include "results.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "conv_check_R2R.h"

void Generate_signal(float *h_input, int signal_length){
	for(int f=0; f<signal_length; f++){
		h_input[f] = rand() / (float)RAND_MAX;
	}
}

void Generate_random_filter(float *h_filters, int nSamples, int nFilters){
	for(int t=0; t<nFilters; t++){
		for(int f=0; f<nSamples; f++){
			h_filters[t*nSamples + f] = rand() / (float)RAND_MAX;
		}
	}
}

void Pad_templates(float *h_filters_time, float *h_filters_padded, int filter_length, int corrected_filter_length, int convolution_size, int nFilters){
	float *tmp_filter;
	tmp_filter = new float[corrected_filter_length];
	for(int f=0; f<nFilters*convolution_size; f++){
		h_filters_padded[f] = 0;
	}
	
	for(int t=0; t<nFilters; t++){
		// copy to temporary filter
		if(filter_length!=corrected_filter_length) {
			tmp_filter[0] = 0;
			for(int f=0; f<filter_length; f++){
				tmp_filter[f + 1] = h_filters_time[t*filter_length + f];
			}
		}
		else {
			for(int f=0; f<filter_length; f++){
				tmp_filter[f] = h_filters_time[t*filter_length + f];
			}
		}
		
		for(int f=0; f<corrected_filter_length; f++){
			// padding for centered filter
			if(f>=corrected_filter_length/2) {
				h_filters_padded[t*convolution_size + f - corrected_filter_length/2] = tmp_filter[f];
			}
			else if(f<corrected_filter_length/2) {
				h_filters_padded[t*convolution_size + f + convolution_size - corrected_filter_length/2] = tmp_filter[f];
			}
		}
	}
}


int Write_output(float *h_output, int signal_length, int nFilters, char *output_signal_file){
	int error=0;
	ofstream FILEOUT;
	FILEOUT.open(output_signal_file);
	if (!FILEOUT.fail()){
		if (VERBOSE) printf("Writing output\n");
		for(int f=0; f<nFilters; f++){
			if (VERBOSE) printf("[");
			for(int Ts=0;Ts<signal_length;Ts++){
				if(Ts%100000==0) {
					if (VERBOSE) {
						printf(".");
						fflush(stdout);
					}
				}
				FILEOUT << f << " " << Ts << " " << h_output[f*signal_length+Ts] << endl;
			}
			FILEOUT << endl;
			if (VERBOSE) printf("] filter=%d\n",f);
		}
	}
	else {
		cout << "Write to a file failed!" << endl;
		error++;
	}
	FILEOUT.close();
	return(error);
}


long int File_size_row_signal(ifstream &FILEIN){
	std::size_t count=0;
	FILEIN.seekg(0,ios::beg);
	for(std::string line; std::getline(FILEIN, line); ++count){}
	return((long int)count);
}


int Load_signal(char *filename, int *nSamples, float **data){
	float real, imaginary;
	int file_size, cislo, error;
	error=0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		error=0;
		file_size=File_size_row_signal(FILEIN);
		(*nSamples) = file_size;
		printf("nSamples:%d;\n", (*nSamples) );

		if(file_size>0){
			*data = (float*)malloc(file_size*sizeof(float));
			memset( (*data), 0.0, file_size*sizeof(float));
			if(*data==NULL){
				printf("\nAllocation error!\n");
				error++;
			}
		
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
			
			cislo=0;
			while (!FILEIN.eof()){
				FILEIN >> real >> imaginary;
				(*data)[cislo] = sqrt(real*real + imaginary*imaginary);
				cislo++;
			}
		}
		else {
			printf("\nFile is void of any content!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}


int Load_filters(char *filename, int *nFilters, int *filter_length, float **data){
	float real, imaginary;
	int file_size, cislo, error, filter_size;
	error=0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		error=0;
		file_size = File_size_row_signal(FILEIN);
		(*filter_length) = file_size/(*nFilters);
		filter_size = (*nFilters)*(*filter_length);
		printf("filter_length:%d; file_size:%d; filter_size:%d;\n", (*filter_length), file_size, filter_size);

		if(file_size>0){
			*data = (float*)malloc( filter_size*sizeof(float));
			memset( (*data), 0.0, filter_size*sizeof(float));
			
			if(*data==NULL){
				printf("\nAllocation error!\n");
				error++;
			}
		
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);

			cislo=0;
			while (!FILEIN.eof()){
				FILEIN >> real >> imaginary;
				(*data)[cislo] = real;
				cislo++;
			}
		}
		else {
			printf("\nFile is void of any content!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}


int GPU_convolution_OLS_customFFT(float *h_input_signal, float *h_output_plane, float *h_filters, int signal_length, int convolution_length, int filter_length, int past_filter_samples, int nFilters, int nRuns, int device, double *execution_time);


int main(int argc, char* argv[]) {
	int signal_length;
	int filter_length;
	int past_filter_samples;
	int convolution_length;
	int nFilters;
	int nRuns;
	char input_type='0';
	char input_filter_file[255];
	char input_signal_file[255];
	char output_signal_file[255];
	
	char * pEnd;
	if (argc>2) {
		if (strlen(argv[1])!=1) {printf("Specify input: \n'r' - random input generated by the code\n 'f' - file input provided by user\n"); exit(2);}
		input_type=*argv[1];
	}
	if (input_type == 'f' && argc==8) {
		if (strlen(argv[2])>255) {printf("Filename of input signal file is too long\n"); exit(2);}
		sprintf(input_signal_file,"%s",argv[2]);
		if (strlen(argv[3])>255) {printf("Filename of input filter file is too long\n"); exit(2);}
		sprintf(input_filter_file,"%s",argv[3]);
		if (strlen(argv[4])>255) {printf("Filename of output signal file is too long\n"); exit(2);}
		sprintf(output_signal_file,"%s",argv[4]);
		
		convolution_length = strtol(argv[5],&pEnd,10);
		nFilters = strtol(argv[6],&pEnd,10);
		past_filter_samples = strtol(argv[7],&pEnd,10);
		nRuns = 1;
	}
	else if (input_type == 'r' && argc==8) {
		signal_length  = strtol(argv[2],&pEnd,10);
		filter_length = strtol(argv[3],&pEnd,10);
		past_filter_samples = strtol(argv[4],&pEnd,10);
		convolution_length = strtol(argv[5],&pEnd,10);
		nFilters      = strtol(argv[6],&pEnd,10);
		
		nRuns = strtol(argv[7],&pEnd,10);
	}
	else {
		printf("Parameters error!\n");
		printf(" 1) Input type: 'r' or 'f' \n");
		printf("----------------------------------\n");
		printf("Parameters if input type is 'f' - file input provided by user\n");
		printf(" 2) Input signal file\n");
		printf(" 3) Input filter file\n");
		printf(" 4) Output signal file\n");
		printf(" 5) Convolution length in samples\n");
		printf(" 6) number of filters\n");
		printf(" 7) number of past samples in the filter.\n");
		printf("    for past filter (causal) it is (filter_length - 1)\n");
		printf("    for odd centered filter it is floor(filter_length/2)\n");
		printf("    for future filter it is 0\n");
		printf(" Example: CONV.exe f signal.dat filter.dat output.dat 2048 32 192\n");
		printf("----------------------------------\n");
		printf("Parameters if input type is 'r' - random input generated by the code\n");
		printf(" 2) Signal length in number of time samples\n");
		printf(" 3) Filter length in samples\n");
		printf(" 4) number of past samples in the filter.\n");
		printf("    for past filter (causal) it is (filter_length - 1)\n");
		printf("    for odd centered filter it is floor(filter_length/2)\n");
		printf("    for future filter it is 0\n");
		printf(" 5) Convolution length in samples\n");
		printf(" 6) Number of filters\n");
		printf(" 7) number of GPU kernel runs\n");
		printf(" Example: CONV.exe r 2097152 193 192 2048 32 10\n");
        return 1;
	}
	
	if (DEBUG) {
		printf("Parameters:\n");
		printf("Input signal and filters are ");
		if (input_type == 'r') {
			printf("randomly generated.\n");
			printf("Signal length:      %d samples\n", signal_length);
			printf("Filter length:      %d samples\n", filter_length);
			printf("# of past samples:  %d samples\n", past_filter_samples);
			printf("Convolution length: %d samples\n", convolution_length);
			printf("Number of filters:  %d\n", nFilters);
			printf("nRuns:              %d\n", nRuns);
		}
		if (input_type == 'f') {
			printf("read from file.\n");
			printf("Input signal:       %s\n", input_signal_file);
			printf("Input filter:       %s\n", input_filter_file);
			printf("Output signal:      %s\n", output_signal_file);
			printf("Convolution length: %d samples\n", convolution_length);
			printf("nFilters:           %d\n", nFilters);
			printf("# of past samples:  %d samples\n", past_filter_samples);
			printf("nRuns:              %d\n", nRuns);
			printf("-----------------\n");
		}
	}

	float *h_input;
	float *h_output;
	float *h_filters;		    // filters in time-domain
	float *h_filters_padded;	// filters in time-domain padded with zeroes
	
	if (input_type == 'f') {
		int error=0;
		error += Load_signal(input_signal_file, &signal_length, &h_input);
		error += Load_filters(input_filter_file, &nFilters, &filter_length, &h_filters);
		if( error>0 ){exit(1);}
		else if (VERBOSE) printf("File loaded\n");
	}

	//----------------> Results
	double execution_time = 0;
	Performance_results CONV_cuFFT;
	CONV_cuFFT.Assign(signal_length, filter_length, nFilters, nRuns, 0, convolution_length, nFilters, "CONV_R2R_kFFT.dat", "one");
	
	int corrected_filter_length;
	if( filter_length%2==0 ) corrected_filter_length = filter_length + 1;
	else corrected_filter_length = filter_length;
	int useful_part_size = convolution_length - corrected_filter_length + 1;
	useful_part_size = 2*(useful_part_size>>1);
	int nConvolutions    = (signal_length + useful_part_size - 1)/useful_part_size;
	if( useful_part_size<=1) {printf("Filter length is too long. Increase FFT length.\n");exit(1);}

	
	if (input_type == 'r') {
		h_input          = (float *)malloc(signal_length*sizeof(float));
		h_filters	     = (float *)malloc(filter_length*nFilters*sizeof(float));
		srand(time(NULL));
		Generate_signal(h_input, signal_length);
		Generate_random_filter(h_filters, filter_length, nFilters);
		if (VERBOSE) printf("Signal and filters generated\n");
	}
	
	size_t filter_size_padded = nFilters*convolution_length;
	h_filters_padded = (float*)malloc(filter_size_padded*sizeof(float));
	Pad_templates(h_filters, h_filters_padded, filter_length, corrected_filter_length,  convolution_length, nFilters);
	
	size_t output_size = nFilters*useful_part_size*nConvolutions;
	h_output = (float*)malloc(output_size*sizeof(float));
	
	if (VERBOSE) printf("Convolution - kFFT\n");

	//----------------> GPU kernel
	GPU_convolution_OLS_customFFT(h_input, h_output, h_filters_padded, signal_length, convolution_length, corrected_filter_length, past_filter_samples, nFilters, nRuns, DEVICEID, &execution_time);
	CONV_cuFFT.GPU_time = execution_time;
	if(VERBOSE) printf("     Execution time:\033[32m%0.3f\033[0mms\n", CONV_cuFFT.GPU_time);
	if(VERBOSE) {cout << "     All parameters: "; CONV_cuFFT.Print();}
	if(WRITE) CONV_cuFFT.Save();
	//----------------> GPU kernel
	
	if(CHECK){
		double total_error, mean_error;
		printf("Checking results...\n");
		Full_CONV_check(h_output, h_input, h_filters, signal_length, filter_length, past_filter_samples, useful_part_size, (filter_length>>1), convolution_length, nConvolutions, nFilters, &total_error, &mean_error);
		//printf("Total error: %e; Mean error: %e\n", total_error, mean_error);
	}
	
	if (input_type == 'f') {
		Write_output(h_output, useful_part_size*nConvolutions, nFilters, output_signal_file);
	}
	
	free(h_input);
	free(h_output);
	free(h_filters_padded);
	free(h_filters);

	cudaDeviceReset();

	if (VERBOSE) printf("Finished!\n");

	return (0);
}
