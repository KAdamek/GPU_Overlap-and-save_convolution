#!/bin/bash


rm CONV.exe;
rm *.o;
make reglim=$reg > /dev/null 2>&1
for convlength in 256 512 1024 2048 4096
do
	for tempsize in 65 97 129 193 257 385 513 769 1025 2049 3073;
	do
		for templates in 2 4 8 16 32 64 96;
		do
			./CONV.exe r 262144 $tempsize $convlength $templates 20
			./CONV.exe r 524288 $tempsize $convlength $templates 20
			./CONV.exe r 1048576 $tempsize $convlength $templates 20
			./CONV.exe r 2097152 $tempsize $convlength $templates 20
			./CONV.exe r 4194304 $tempsize $convlength $templates 20
			./CONV.exe r 8388608 $tempsize $convlength $templates 20
		done
	done
done
