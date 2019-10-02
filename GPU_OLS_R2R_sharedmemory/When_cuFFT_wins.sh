#!/bin/bash

rm CONV_1f.exe;
rm CONV_2f.exe;
rm *.o;
make reglim=0 > /dev/null 2>&1
for convlength in 256 512 1024 2048 4096;
do 
	for tempsize in {64..4096..32}
	do
		./CONV.exe r 2097152 $tempsize $convlength 32 20
	done
done
