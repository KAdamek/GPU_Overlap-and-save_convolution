#!/bin/bash

for tempsize in 64 96 128 192 256 384 512;
do
	for convsize in 1024 2048 4096 8192 16384;
	do
		for templates in 2 4 8 11 16 32 51 64 96;
		do
			for reg in 0 24 32 40 48 56 64 72 128;
			do
				echo "#define CONV_SIZE $convsize" > params.h
				
				rm CONV.exe
				make
				./CONV.exe 262144 $tempsize $templates 20 $reg
				./CONV.exe 524288 $tempsize $templates 20 $reg
				./CONV.exe 1048576 $tempsize $templates 20 $reg
				./CONV.exe 2097152 $tempsize $templates 20 $reg
				./CONV.exe 4194304 $tempsize $templates 20 $reg
				./CONV.exe 8388608 $tempsize $templates 20 $reg
			done
		done
	done
done

