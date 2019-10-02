#!/bin/bash

rm CONV_R2R_kFFT.dat;

./benchmark_performance.sh;
mv CONV_R2R_kFFT.dat OLS_SM_R2R_pp_perf.dat;

./When_cuFFT_wins.sh
mv CONV_R2R_kFFT.dat OLS_SM_R2R_pp_whencuFFTwins.dat;