#!/bin/bash

for tempsize in 64 96 128 192 256 384 512;
do
	for templates in 2 4 8 16 32 64 96;
	do
		for reg in 0 24 32 40 48 56 64 72 128;
		do
			echo "#define CONV_SIZE 1024" > params.h
			echo "#define CONV_HALF 512" >> params.h
			echo "#define FFT_EXP 10" >> params.h
			
			rm CONV_1f.exe;
			rm CONV_2f.exe;
			rm *.o;
			make reglim=$reg > /dev/null 2>&1
			./CONV_1f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_1f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_1f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_1f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_1f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_1f.exe r 8388608 $tempsize $templates 20 $reg
			
			./CONV_2f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_2f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_2f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_2f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_2f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_2f.exe r 8388608 $tempsize $templates 20 $reg

			
			echo "#define CONV_SIZE 512" > params.h
			echo "#define CONV_HALF 256" >> params.h
			echo "#define FFT_EXP 9" >> params.h
			
			rm CONV_1f.exe;
			rm CONV_2f.exe;
			rm *.o;
			make reglim=$reg > /dev/null 2>&1
			./CONV_1f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_1f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_1f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_1f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_1f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_1f.exe r 8388608 $tempsize $templates 20 $reg
			
			./CONV_2f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_2f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_2f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_2f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_2f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_2f.exe r 8388608 $tempsize $templates 20 $reg

			
			echo "#define CONV_SIZE 2048" > params.h
			echo "#define CONV_HALF 1024" >> params.h
			echo "#define FFT_EXP 11" >> params.h
			
			rm CONV_1f.exe;
			rm CONV_2f.exe;
			rm *.o;
			make reglim=$reg > /dev/null 2>&1
			./CONV_1f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_1f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_1f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_1f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_1f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_1f.exe r 8388608 $tempsize $templates 20 $reg
			
			./CONV_2f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_2f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_2f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_2f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_2f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_2f.exe r 8388608 $tempsize $templates 20 $reg

			
			echo "#define CONV_SIZE 256" > params.h
			echo "#define CONV_HALF 128" >> params.h
			echo "#define FFT_EXP 8" >> params.h
			
			rm CONV_1f.exe;
			rm CONV_2f.exe;
			rm *.o;
			make reglim=$reg > /dev/null 2>&1
			./CONV_1f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_1f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_1f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_1f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_1f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_1f.exe r 8388608 $tempsize $templates 20 $reg
			
			./CONV_2f.exe r 262144 $tempsize $templates 20 $reg
			./CONV_2f.exe r 524288 $tempsize $templates 20 $reg
			./CONV_2f.exe r 1048576 $tempsize $templates 20 $reg
			./CONV_2f.exe r 2097152 $tempsize $templates 20 $reg
			./CONV_2f.exe r 4194304 $tempsize $templates 20 $reg
			./CONV_2f.exe r 8388608 $tempsize $templates 20 $reg
		done
	done
done
