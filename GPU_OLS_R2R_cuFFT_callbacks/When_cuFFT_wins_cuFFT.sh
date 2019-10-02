#!/bin/bash

for convsize in 1024 2048 4096 8192 16384;
do
	echo "#define CONV_SIZE $convsize" > params.h
	
	rm CONV.exe
	make
	for tempsize in {64..2048..32}
	do
		for templates in 32;
		do
			./CONV.exe r 2097152 $tempsize $templates 20 0
		done
	done
done
