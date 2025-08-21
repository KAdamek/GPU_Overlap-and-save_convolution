README v0.1 / 1 OCTOBER 2019
27 JULY 2020 - convolutions with shared memory FFT can now process both odd and even length filters with arbitrary number of past and present samples not only odd time centred filters as before
25 AUGUST 2020 - convolved signal is now multiplied by the normalization constant equal to the size of the FFT

### Publication
If you find these codes useful please cite:
https://dl.acm.org/doi/10.1145/3394116
  
# Convolution using shared memory overlap-and-save method on NVIDIA GPUs
Overlap-and-save method of calculation linear one-dimensional convolution on NVIDIA GPUs using shared memory. Out implementation of the overlap-and-save method uses shared memory implementation of the FFT algorithm to increase performance of one-dimensional complex-to-complex or real-to-real convolutions. The speed-up achieved depends on the filter length up to 2.5x faster for filter length 257 samples for complex-to-complex (C2C) and up to 4x for real-to-real (R2R) convolution.

Note: This implementation of OLS convolution uses modified version of GPU shared memory FFT which does not reorder elements of the output. For GPU shared memory FFT with reordering step which please go to [SMFFT repository](https://github.com/KAdamek/SMFFT)

## Introduction
Convolution is a standard tool in signal processing. It is a linear operation where a signal s is modified by a filter (response function) r. There are two fundamental ways how to calculate convolution. We can calculate convolution in time-domain using formulae for discrete convolution (more in any signal processing literature for example R. G. Lyons, Understanding digital signal processing 3rd ed, Prentice 180 Hall, 2011). We can also invoke convolution theorem and calculate convolution in frequency domain. To do this we need to perform discrete Fourier transformation first, then perform convolution in frequency-domain and finally use inverse discrete Fourier transformation to return to time-domain. The overlap-and-save (OLS) method (more in Press et.al. Numerical Recipes) of calculating convolution is designed for special case where we have very long input signal, but relatively short filter. The disadvantage of Fourier-domain convolution method in cases such as this is that the convolution theorem demands that both the signal and the filter in Fourier-domain must be of the same length. This can be performance prohibitive if we are dealing with very long signals and multiple short filters. The overlap-and-save method separates the input signal into smaller segments which are then independently processed, which makes this method ideal for parallel processing for example on GPUs. At the end the overlap-and-save method add all these segments together in such a way as to produce linear convolution.

We provide two implementations of overlap-and-save method, first is using vendor provided FFT library the NVIDIA cuFFT library (cuFFT-OSL) for calculating necessary FFTs, the second implementation is using our shared memory implementation of the FFT algorithm and performs overlap-and-save method in shared memory (SM-OLS) without accessing the device memory. The advantage of having a shared memory FFT algorithm is that we can perform all necessary steps of the overlap-and-save inside one CUDA kernel thus saving costly device memory transactions thus providing significant speedup over cuFFT implementation. 

This technique was used in GPU implementation of the Fourier domain acceleration search for time-domain radio astronomy and it is a part of AstroAccelerate (https://github.com/AstroAccelerateOrg/astro-accelerate).

Current version of the code aims to demonstrate performance gain of the shared memory implementation and for performance testing. 

The suffix "_pp"  means that the code is also performing custom post-processing step.

## Usage
In total there are eight different implementations of the overlap-and-save (OLS) method. There is OLS which uses NVIDIA cuFFT library (cuFFT-OLS) and shared memory implementation of the OLS method (SM-OLS) which uses shared memory implementation of the FFT algorithm. Both of these are for one-dimensional complex-to-complex or real-to-real convolutions. Each implementation has also version with non-local post-processing in for of numerical differentiation (distinguished by "_pp" in the directory name). 
There are two modes of operation, one for performance testing (first argument 'r') which does not require user to provide the input data and one for processing user data (first argument 'f'). The command line arguments are slightly different between cuFFT-OLS and SM-OLS.

### Convolutions using NVIDIA cuFFT library
The arguments expected by the cuFFT-OLS depends on chosen mode of operation. 
 1) Input type: 'r' or 'f'

Parameters if input type is 'f' - file input provided by user
 2) Input signal file
 3) Input filter file
 4) Output signal file
 5) number of filters
 Example: CONV.exe f signal.dat filter.dat output.dat 32

Parameters if input type is 'r' - random input generated by the code
 2) Signal length in number of time samples
 3) Filter length in samples
 4) Number of templates
 5) number of GPU kernel runs
 Example: CONV.exe r 2097152 193 32 10
 
The cuFFT-OLS implementation expects that CONV_SIZE would be #defined in 'params.h'. This constant determines size of the segment which will be processed, best performing value is in the most cases 8192. 

### Convolutions using shared memory FFT
The command line arguments are slightly different for SM-OLS. In addition to the same parameters from cuFFT-OLS it also requires the user to give the segment size (FFT size):
 1) Input type: 'r' or 'f'
----------------------------------
Parameters if input type is 'f' - file input provided by user
 2) Input signal file
 3) Input filter file
 4) Output signal file
 5) Convolution length in samples
 6) number of filters
 7) number of past samples in the filter.
    for past filter (causal) it is (filter_length - 1)
    for odd centred filter it is floor(filter_length/2)
    for future filter it is 0
 Example: CONV.exe f signal.dat filter.dat output.dat 2048 32 192
----------------------------------
Parameters if input type is 'r' - random input generated by the code
 2) Signal length in number of time samples
 3) Filter length in samples
 4) number of past samples in the filter.
    for past filter (causal) it is (filter_length - 1)
    for odd centred filter it is floor(filter_length/2)
    for future filter it is 0
 5) Convolution length in samples
 6) Number of filters
 7) number of GPU kernel runs
 Example: CONV.exe r 2097152 193 192 2048 32 10

		
In case of SM-FFT implementation there is no universal segment size. The maximum segment size is 4096. 

Beware when using very long signals, for example signal with 2^23 time samples with 100 filter will take 6.4GB of memory.

If enabled in debug.h the code will write out timing and parameters of given run for analysis of the results. The columns of the output file are:
1) number of time samples
2) filter length 
3) number of filters
4) average execution time of convolution kernel (no transfer time from/to host are included)
5) number of executions from which the execution time is calculated
6) limitation on number of registers if set
7) CONV_SIZE
8) number of filters processed per CUDA thread block
9) type of the kernel used



## Generating example files
The example files could be generated by using 'GPU_OaS_generate_files'. The code for generating example files expects following arguments:
1) Signal length in number of time samples (min 15000 samples)
2) Filter length Example:129 (odd because we have assumed centered filter)
3) Number of filters
4) Name of the file to export signal to
5) Name of the file to export filters to
For example:
Example_files.exe 50000 129 32 signal.dat filter.dat

Structure of input and output files
In the input file (signal or filter), each time complex sample is written on individual line where real part is in the first column and imaginary part is in second column. Filters are written one after another without any additional lines. Input for the real-to-real convolution is the same, with the difference that for the input signal is used the power sqrt(real^2+imaginary^) and for the filter the imaginary part is ignored.

The structure of the output file is divided into blocks by filters and they are separated by empty lines. The column in output file are as follows:
1) filter index 0<=i<(number of filters)
2) time sample
3) real part of convolved signal
4) imaginary part of convolved signal (only for complex-to-complex convolutions)

in gnuplot one can display powers of complex-to-complex convolved signal as: splot 'output.dat' using 1:2:($3*$3+$4*$4) palette 
for three-dimensional plot. 
Results for one particular filter can be displayed using
plot 'output.dat' using ($1==X?$2:1/0):($3*$3+$4*$4) w lines
where X is the number of the filter starting with zero.

Be advised that the code might produce very big files for long signals and large number of filters.

For real-to-real the output has only real component thus one can use "using ...:3" instead of "using ...:($3*$3+$4*$4)".




## Installation

### Requirements

NVIDIA GPU and CUDA Toolkit

### Installation
We have provided a make file which should take care of the compilation step. The make file assumes that environmental variable 'CUDA_HOME' is set and that it points to a folder containing installation of CUDA Toolkit. The compute capability also needs to be set in the make file. You can change the architecture by using a flag -arch=X where X is compute capability. For TitanV from Volta generation that is -arch=sm_70.

The code does not require any other dependencies.


### Configuration

The code behavoiur could be changed by editing debug.h which is located in each of the directories. The debug.h contains the following options:
VERBOSE enables more output verbose output to console
DEBUG displays debugging information
CHECK enables test which checks the output of the overlap-and-save convolution with timedomain convolution
WRITE enables writeing of the execution time and other parameters into a file

DEVICEID 0 with this you can set the id of the device which should be used for during code execution
POST_PROCESS a flag which enables non-local used in for the article. If commented out code will perform normal convolution without post-processing.

### Additional files
Together with the code we also provide scripts for benchmarking and R script for processing resulting data.

## Future work
	We would like to improve the code and create proper library which could be used directly without modification.

## Contributors
	Karel Adamek
	Sofia Dimoudi
	Wes Armour
	Mike Giles

## Contact
	You can contact me using my email karel.adamek@gmail.com

## License

This project is licensed under [insert license]. The license should be in a separate file called LICENSE, so don't explain it in detail within your documentation. Also, don't forget to specify licenses of third-party libraries and programs you use.

Sometimes including a Table of Contents (TOC) at the beginning of the documentation makes sense, especially when your README file is more than a few paragraphs. If you think that the README file has grown too large, put some of the more detailed parts, such as installation or configuration sections, into their own files.

