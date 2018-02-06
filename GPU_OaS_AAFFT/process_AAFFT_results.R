inpath="";
outpath="";
file1 = "CONV_kFFT.dat";

data1 = read.table(paste(inpath,file1,sep=""));
alldata<-rbind(data1);


newline <- ""

template_length <- c(64,96,128,192,256,384,512,768,1024);
templates <- c(2,4,8,16,32,64,96);
signal_length <- c(262144,524288,1048576,2097152,4194304,8388608);
bestdata<-alldata[1,];bestdata<-bestdata[-1,];
best10data<-alldata[1,];best10data<-best10data[-1,];


for (nLenght in template_length){
  templengthdata<-alldata[(alldata[[2]]==nLenght),];
  for (nTemplates in templates){
    templatedata<-templengthdata[(templengthdata[[3]]==nTemplates),];
    for (slength in signal_length){
      slengthdata<-templatedata[(templatedata[[1]]==slength),];
      if (length(slengthdata[,1]) > 0){
        best_line <- which.min(slengthdata[[4]]);
        bestdata<-rbind(bestdata,slengthdata[best_line,]);
      }
    }
  }
}


#writing out the results
extension=".txt";
for (nLenght in template_length){
  templengthdata<-bestdata[(bestdata[[2]]==nLenght),];
  for (slength in signal_length){
    slengthdata<-templengthdata[(templengthdata[[1]]==slength),];
    filename="Best_kFFT";
    filename<-paste(filename,slength,nLenght,sep="_");
    filename<-paste(filename,extension,sep="");
    unlink(filename);
    write.table(slengthdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
  }
}


#export data grouped by template width
for (nLenght in template_length){
  templengthdata<-bestdata[(bestdata[[2]]==nLenght),];
  #set up container
  slengthdata<-templengthdata[(templengthdata[[1]]==signal_length[[1]]),];
  resultdata<-cbind(slengthdata[[3]]);
  #add columns with signal length and time for each signal length
  for (slength in signal_length){
    slengthdata<-templengthdata[(templengthdata[[1]]==slength),];
    resultdata<-cbind(resultdata,slengthdata[[1]],slengthdata[[4]]);
  }
  #export data
  filename="kFFT_results";
  filename<-paste(filename,nLenght,sep="_");
  filename<-paste(filename,extension,sep="");
  unlink(filename);
  write.table(resultdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
}





#export data grouped by template width with speedup
cuFFTfilename="Best_cuFFT";
for (nLenght in template_length){
  templengthdata<-bestdata[(bestdata[[2]]==nLenght),];
  #set up container
  slengthdata<-templengthdata[(templengthdata[[1]]==signal_length[[1]]),];
  resultdata<-cbind(slengthdata[[3]]);
  #add columns with signal length and time for each signal length
  for (slength in signal_length){
    slengthdata<-templengthdata[(templengthdata[[1]]==slength),];
    #Loading cuFFTresults
    tempcuFFTfilename<-paste(cuFFTfilename,slength,nLenght,sep="_");
    tempcuFFTfilename<-paste(tempcuFFTfilename,extension,sep="");
    cuFFTdata = read.table(paste(inpath,tempcuFFTfilename,sep=""));
    #Creating dataframe with results
    resultdata<-cbind(resultdata,slengthdata[[1]],cuFFTdata[[4]],slengthdata[[4]],cuFFTdata[[4]]/slengthdata[[4]]);
  }
  #export data
  filename="Results_speedup";
  filename<-paste(filename,nLenght,sep="_");
  filename<-paste(filename,extension,sep="");
  unlink(filename);
  write.table(resultdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
}
