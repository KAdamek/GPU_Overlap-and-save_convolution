#!/bin/bash

rm CONV_kFFT.dat;

./benchmark_performance.sh;
mv CONV_kFFT.dat OLS_SM_C2C_pp_perf.dat;

./When_cuFFT_wins.sh
mv CONV_kFFT.dat OLS_SM_C2C_pp_whencuFFTwins.dat;