#!/bin/bash

rm CONV_R2R_kFFT.dat;

./benchmark_performance.sh;
mv CONV_R2R_kFFT.dat OLS_SM_R2R_perf.dat;

./When_cuFFT_wins.sh
mv CONV_R2R_kFFT.dat OLS_SM_R2R_whencuFFTwins.dat;