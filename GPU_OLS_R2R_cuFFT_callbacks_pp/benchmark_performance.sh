#!/bin/bash

for convsize in 1024 2048 4096 8192 16384;
do

echo "#define CONV_SIZE $convsize" > params.h		
rm CONV.exe
make
	for tempsize in 65 97 129 193 257 385 513 769 1025 2049 3073;
	do	
		for templates in 2 4 8 11 16 32 51 64 96;
		do
			./CONV.exe r 262144 $tempsize $templates 20
			./CONV.exe r 524288 $tempsize $templates 20
			./CONV.exe r 1048576 $tempsize $templates 20
			./CONV.exe r 2097152 $tempsize $templates 20
			./CONV.exe r 4194304 $tempsize $templates 20
			./CONV.exe r 8388608 $tempsize $templates 20
		done
	done
done

