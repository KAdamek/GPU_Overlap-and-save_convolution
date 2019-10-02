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


#include "conv_check.h"


void Generate_random_signal(float2 *h_input, int signal_length){
	for(int f=0; f<signal_length; f++){
		h_input[f].y=rand() / (float)RAND_MAX;
		h_input[f].x=rand() / (float)RAND_MAX;
		//h_input[f].y=1;
		//h_input[f].x=2;
	}
}

void Generate_signal(float2 *h_input, int nTimesamples){
	for(int f=0; f<nTimesamples; f++){
		h_input[f].y=rand() / (float)RAND_MAX;
		h_input[f].x=rand() / (float)RAND_MAX;
	}
	
	if(nTimesamples>15000){
		for(int f=15000; f<nTimesamples; f++){
			h_input[f].x = f%4096;
		}
		
		for(int f=0; f<192; f++){
			h_input[f + 5300].x = 10.0;
		}
		
		for(int f=0; f<128; f++){
			h_input[f + 8626].x = 10.0;
		}
		
		for(int f=0; f<36; f++){
			h_input[f + 9626].x = 10.0;
		}
		
		for(int f=0; f<83; f++){
			h_input[f + 10626].x = 10.0;
		}
		
		for(int f=0; f<138; f++){
			h_input[f + 11626].x = 10.0;
		}
	}
}

void Generate_random_filter(float2 *h_filters, int nSamples, int nFilters){
	for(int t=0; t<nFilters; t++){
		for(int f=0; f<nSamples; f++){
			h_filters[t*nSamples + f].y=rand() / (float)RAND_MAX;
			h_filters[t*nSamples + f].x=rand() / (float)RAND_MAX;
			//h_filters[t*nSamples + f].y=0.5;
			//h_filters[t*nSamples + f].x=0.3;
		}
	}
}

void Generate_boxcar_filter(float2 *h_filters, int nSamples, int nFilters){
	int boxcar_width;
	for(int t=0; t<nFilters; t++){
		boxcar_width = ((t+1)*8);
		for(int f=0; f<nSamples; f++){
			if( f>=(nSamples/2-boxcar_width/2) && f<( nSamples/2+boxcar_width/2) ){
				h_filters[t*nSamples + f].x=1;
				h_filters[t*nSamples + f].y=0;
			}
			else {
				h_filters[t*nSamples + f].x=0;
				h_filters[t*nSamples + f].y=0;
			}
		}
	}
}

void Pad_templates(float2 *h_filters_time, float2 *h_filters, int template_size, int convolution_size, int nFilters){
	for(int f=0; f<nFilters*convolution_size; f++){
		h_filters[f].x=0;
		h_filters[f].y=0;
	}
	
	for(int t=0; t<nFilters; t++){
		for(int f=0; f<template_size; f++){
			// padding for centered filter
			if(f>=template_size/2) {
				h_filters[t*convolution_size + f - template_size/2].x=h_filters_time[t*template_size + f].x;
				h_filters[t*convolution_size + f - template_size/2].y=h_filters_time[t*template_size + f].y;
			}
			else if(f<template_size/2) {
				h_filters[t*convolution_size + f + convolution_size - template_size/2].x = h_filters_time[t*template_size + f].x;
				h_filters[t*convolution_size + f + convolution_size - template_size/2].y = h_filters_time[t*template_size + f].y;
			}
		}
	}
}


int Write_output(float2 *h_output, int signal_length, int nFilters, char *output_signal_file){
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
				FILEOUT << f << " " << Ts << " " << h_output[f*signal_length+Ts].x << " " << h_output[f*signal_length+Ts].y << endl;
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


int Load_signal(char *filename, int *nSamples, float2 **data){
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
			*data = (float2*)malloc(file_size*sizeof(float2));
			memset( (*data), 0.0, file_size*sizeof(float2));
			if(*data==NULL){
				printf("\nAllocation error!\n");
				error++;
			}
		
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);
			
			cislo=0;
			while (!FILEIN.eof()){
				FILEIN >> real >> imaginary;
				(*data)[cislo].x = real;
				(*data)[cislo].y = imaginary;
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


int Load_filters(char *filename, int *nFilters, int *filter_length, float2 **data){
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
			*data = (float2*)malloc( filter_size*sizeof(float2));
			memset( (*data), 0.0, filter_size*sizeof(float2));
			
			if(*data==NULL){
				printf("\nAllocation error!\n");
				error++;
			}
		
			FILEIN.clear();
			FILEIN.seekg(0,ios::beg);

			cislo=0;
			while (!FILEIN.eof()){
				FILEIN >> real >> imaginary;
				(*data)[cislo].x = real;
				(*data)[cislo].y = imaginary;
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


int GPU_convolution_OLS_customFFT(float2 *h_input_signal, float2 *h_output_plane, float2 *h_filters, int signal_length, int convolution_length, int filter_length, int nFilters, int nRuns, int kernel_type, double *execution_time);


int main(int argc, char* argv[]) {
	int signal_length;
	int filter_length;
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
	if (input_type == 'f' && argc==7) {
		if (strlen(argv[2])>255) {printf("Filename of input signal file is too long\n"); exit(2);}
		sprintf(input_signal_file,"%s",argv[2]);
		if (strlen(argv[3])>255) {printf("Filename of input filter file is too long\n"); exit(2);}
		sprintf(input_filter_file,"%s",argv[3]);
		if (strlen(argv[4])>255) {printf("Filename of output signal file is too long\n"); exit(2);}
		sprintf(output_signal_file,"%s",argv[4]);
		
		convolution_length = strtol(argv[5],&pEnd,10);
		nFilters = strtol(argv[6],&pEnd,10);
		nRuns = 1;
	}
	else if (input_type == 'r' && argc==7) {
		signal_length  = strtol(argv[2],&pEnd,10);
		filter_length = strtol(argv[3],&pEnd,10);
		convolution_length = strtol(argv[4],&pEnd,10);
		nFilters      = strtol(argv[5],&pEnd,10);
		
		nRuns = strtol(argv[6],&pEnd,10);
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
		printf(" Example: CONV.exe f signal.dat filter.dat output.dat 2048 32\n");
		printf("----------------------------------\n");
		printf("Parameters if input type is 'r' - random input generated by the code\n");
		printf(" 2) Signal length in number of time samples\n");
		printf(" 3) Filter length in samples\n");
		printf(" 4) Convolution length in samples\n");
		printf(" 5) Number of filters\n");
		printf(" 6) number of GPU kernel runs\n");
		printf(" Example: CONV.exe r 2097152 193 2048 32 10\n");
        return 1;
	}
	
	if (DEBUG) {
		printf("Parameters:\n");
		printf("Input signal and filters are ");
		if (input_type == 'r') {
			printf("randomly generated.\n");
			printf("Signal length:      %d samples\n", signal_length);
			printf("Filter length:      %d samples\n", filter_length);
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
			printf("nRuns:              %d\n", nRuns);
			printf("-----------------\n");
		}
	}

	float2 *h_input;
	float2 *h_output;
	float2 *h_filters;		    // filters in time-domain
	float2 *h_filters_padded;	// filters in time-domain padded with zeroes
	
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
	CONV_cuFFT.Assign(signal_length, filter_length, nFilters, nRuns, 0, convolution_length, nFilters, "CONV_kFFT.dat", "one");
	
	// we assume that filter is centred around zero
	int useful_part_size = convolution_length - filter_length + 1;
	int nConvolutions    = signal_length/useful_part_size;
	if( (signal_length%useful_part_size)>0 ) nConvolutions++;
	if( useful_part_size<=1) {printf("Filter length is too long. Increase FFT length.\n");exit(1);}

	
	if (input_type == 'r') {
		h_input          = (float2 *)malloc(signal_length*sizeof(float2));
		h_filters	     = (float2 *)malloc(filter_length*nFilters*sizeof(float2));
		srand(time(NULL));
		Generate_random_signal(h_input, signal_length);
		Generate_random_filter(h_filters, filter_length, nFilters);
		if (VERBOSE) printf("Signal and filters generated\n");
	}
	
	size_t filter_size_padded = nFilters*convolution_length;
	h_filters_padded = (float2*)malloc(filter_size_padded*sizeof(float2));
	Pad_templates(h_filters, h_filters_padded, filter_length, convolution_length, nFilters);
	
	size_t output_size = nFilters*useful_part_size*nConvolutions;
	h_output = (float2*)malloc(output_size*sizeof(float2));
	
	if (VERBOSE) printf("Convolution - kFFT\n");

	//----------------> GPU kernel
	int kernel_type=1; //one filter per iteration
	GPU_convolution_OLS_customFFT(h_input, h_output, h_filters_padded, signal_length, convolution_length, filter_length, nFilters, nRuns, kernel_type, &execution_time);
	CONV_cuFFT.GPU_time = execution_time;
	if(VERBOSE) printf("     Execution time:\033[32m%0.3f\033[0mms\n", CONV_cuFFT.GPU_time);
	if(VERBOSE) {cout << "     All parameters: "; CONV_cuFFT.Print();}
	if(WRITE) CONV_cuFFT.Save();
	//----------------> GPU kernel

	if(CHECK){
		double total_error, mean_error;
		printf("Checking results...\n");
		Full_CONV_check(h_output, h_input, h_filters, signal_length, filter_length, useful_part_size, convolution_length, nConvolutions, nFilters, &total_error, &mean_error);
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