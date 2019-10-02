//********************************************************************************************
//* This is GPU implementation of a Overlap-and-save method for calculating convolution. 
//* Copyright (C) 2017  Adámek Karel
//* 
//* Authors: Karel Adamek ( ORCID:0000-0003-2797-0595; https://github.com/KAdamek ), Wesley Armour ( ORCID:0000-0003-1756-3064 ), Sofia Dimoudi 
//********************************************************************************************


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip> 

struct float2 {
	float x;
	float y;
};

void Generate_signal(float2 *h_input, int nTimesamples){
	for(int f=0; f<nTimesamples; f++){
		h_input[f].y=rand() / (float)RAND_MAX;
		h_input[f].x=rand() / (float)RAND_MAX;
	}
	
	for(int f=15000; f<nTimesamples; f++){
		h_input[f].x = (f%4096)/500.0;
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

void Generate_templates(float2 *h_templates, int nSamples, int nTemplates){
	int boxcar_width, itemp;
	itemp = (((float) nSamples)*0.8)/((float) nTemplates);
	if (itemp==0) itemp++;
	for(int t=0; t<nTemplates; t++){
		boxcar_width = ((t+1)*itemp);
		if(boxcar_width>nSamples) boxcar_width=nSamples;
		for(int f=0; f<nSamples; f++){
			if( f>=(nSamples/2-boxcar_width/2) && f<( nSamples/2+boxcar_width/2) ){
				h_templates[t*nSamples + f].x=1;
				h_templates[t*nSamples + f].y=0;
			}
			else {
				h_templates[t*nSamples + f].x=0;
				h_templates[t*nSamples + f].y=0;
			}
		}
	}
}



int GPU_CONV(float2 *h_input_signal, float2 *h_output_plane_reduced, float2 *h_templates, int useful_part_size, int offset, int template_length, int nConvolutions, int nTemplates, int nRuns, double *execution_time);
int GPU_CONV_debug(float2 *h_input_signal, float2 *h_GPU_input_signal_extended, float2 *h_GPU_input_signal_extended_FFT, float2 *h_output_plane, float2 *h_output_plane_IFFT, float2 *h_output_plane_reduced, float2 *h_templates, int useful_part_size, int offset, int template_length, int nConvolutions, int nTemplates, int nRuns, double *execution_time);


int main(int argc, char* argv[]) {
	int nTimesamples;
	int template_length;
	int nTemplates;
	char filter_file[100];
	char signal_file[100];

	char * pEnd;
	if (argc==6) {
		nTimesamples    = strtol(argv[1],&pEnd,10);
		template_length = strtol(argv[2],&pEnd,10);
		nTemplates      = strtol(argv[3],&pEnd,10);
		if (strlen(argv[4])>100) {printf("Filename of input signal file is too long\n"); exit(2);}
		sprintf(signal_file,"%s",argv[4]);
		if (strlen(argv[5])>100) {printf("Filename of input filter file is too long\n"); exit(2);}
		sprintf(filter_file,"%s",argv[5]);
	}
	else {
		printf("Argument error!\n");
		printf(" 1) Signal length in number of time samples (min 15000 samples)\n");
		printf(" 2) Filter length Example:129\n");
		printf(" 3) Number of filters\n");
		printf(" 4) Name of the file to export signal to\n");
		printf(" 5) Name of the file to export filters to\n");
        return 1;
	}
	
	if (nTimesamples<15000) {printf("Number of samples must be higher then 15000 samples\n"); exit(1);}
	
	size_t input_size            = nTimesamples;
	size_t template_size_time    = nTemplates*template_length;

	float2 *h_input_signal;
	float2 *h_templates;

	h_input_signal = (float2 *)malloc(input_size*sizeof(float2));
	h_templates    = (float2 *)malloc(template_size_time*sizeof(float2));

	memset(h_input_signal, 0.0, input_size*sizeof(float2));
	memset(h_templates, 0.0, template_size_time*sizeof(float2));

	Generate_signal(h_input_signal, nTimesamples);
	Generate_templates(h_templates, template_length, nTemplates);
	
	
	std::ofstream FILEOUT;
	FILEOUT.open(signal_file);
	for(int ts=0; ts<nTimesamples; ts++){
		FILEOUT << h_input_signal[ts].x << " " << h_input_signal[ts].y << std::endl;
	}
	FILEOUT.close();
	
	FILEOUT.open(filter_file);
	for(int f=0; f<nTemplates; f++){
		for(int ts=0; ts<template_length; ts++){
			FILEOUT << h_templates[f*template_length + ts].x << " " << h_templates[f*template_length + ts].y << std::endl;
		}
	}
	FILEOUT.close();
	
	free(h_input_signal);
	free(h_templates);

	return (0);
}
