#!/bin/bash


rm CONV_1f.exe;
rm CONV_2f.exe;
rm *.o;
make reglim=$reg > /dev/null 2>&1
for convlength in 256 512 1024 2048 4096
do
	for tempsize in 64 96 128 192 256 384 512 768 1024 2048 3072;
	do
		for templates in 2 4 8 16 32 64 96;
		do
			./CONV_1f.exe r 262144 $tempsize $convlength $templates 20
			./CONV_1f.exe r 524288 $tempsize $convlength $templates 20
			./CONV_1f.exe r 1048576 $tempsize $convlength $templates 20
			./CONV_1f.exe r 2097152 $tempsize $convlength $templates 20
			./CONV_1f.exe r 4194304 $tempsize $convlength $templates 20
			./CONV_1f.exe r 8388608 $tempsize $convlength $templates 20
			
			./CONV_2f.exe r 262144 $tempsize $convlength $templates 20
			./CONV_2f.exe r 524288 $tempsize $convlength $templates 20
			./CONV_2f.exe r 1048576 $tempsize $convlength $templates 20
			./CONV_2f.exe r 2097152 $tempsize $convlength $templates 20
			./CONV_2f.exe r 4194304 $tempsize $convlength $templates 20
			./CONV_2f.exe r 8388608 $tempsize $convlength $templates 20
		done
	done
done
