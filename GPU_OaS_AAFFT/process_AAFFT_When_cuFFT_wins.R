extension=".txt";
inpath="";
outpath="";
input_filename = "CONV_kFFT.dat";
output_filename="When_cuFFT_wins_kFFT_results";

data1 = read.table(paste(inpath,input_filename,sep=""));
alldata<-rbind(data1);


newline <- ""

filter_length <- seq(from=64, to=4096, by=32);
convolution_length <- c(256, 512, 1024, 2048, 4096);
bestdata<-alldata[1,];bestdata<-bestdata[-1,];
best256data<-alldata[1,];best256data<-best256data[-1,];
best512data<-alldata[1,];best512data<-best512data[-1,];
best1024data<-alldata[1,];best1024data<-best1024data[-1,];
best2048data<-alldata[1,];best2048data<-best2048data[-1,];

#process data
for (convLength in convolution_length){
	dataFixedConv<-alldata[(alldata[[7]]==convLength),];
	for (filLen in filter_length){
		dataFiltered<-dataFixedConv[(dataFixedConv[[2]]==filLen),];	
		if (length(dataFiltered[,1]) > 0){
			best_line <- which.min(dataFiltered[[4]]);
			bestdata<-rbind(bestdata,dataFiltered[best_line,]);
		}
	}	
}

#export data
output_filename<-paste(output_filename,extension,sep="");
unlink(output_filename);
write.table(bestdata, file = output_filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
