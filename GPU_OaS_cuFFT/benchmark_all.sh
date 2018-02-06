#!/bin/bash

for tempsize in 64 96 128 192 256 384 512 768 1024;
do
	for convsize in 1024 2048 4096 8192 16384;
	do
		for templates in 2 4 8 11 16 32 51 64 96;
		do
			for reg in 0;
			do
				echo "#define CONV_SIZE $convsize" > params.h
				
				rm CONV.exe
				make
				./CONV.exe r 262144 $tempsize $templates 20 $reg
				./CONV.exe r 524288 $tempsize $templates 20 $reg
				./CONV.exe r 1048576 $tempsize $templates 20 $reg
				./CONV.exe r 2097152 $tempsize $templates 20 $reg
				./CONV.exe r 4194304 $tempsize $templates 20 $reg
				./CONV.exe r 8388608 $tempsize $templates 20 $reg
			done
		done
	done
done

