output_extension=".txt";
inpath="";
outpath="";
AAFFT_file = "Conv_CUDA10.0_TitanV_OLS_R2R_customFFT_perf.dat";
cuFFT_file = "Conv_CUDA10.0_TitanV_OLS_R2R_cuFFT_callbacks_mk2_perf.dat";

newline <- ""

#Flags
export_best_performance = 0;
export_data_grouped_by_template_width = 0;
export_nTemplates_time_speedup_gr_template_width_one_file = 1;
export_signal_length_time_speedup_gr_nTemplates_one_file = 1;

#Read data and find dimensions of it
AAFFTdata = read.table(paste(inpath, AAFFT_file, sep=""));
cuFFTdata = read.table(paste(inpath, cuFFT_file, sep=""));

alldata <- rbind(AAFFTdata, cuFFTdata);
template_lengths    = unique(alldata[[2]], incomparables = FALSE);
number_of_templates = unique(alldata[[3]], incomparables = FALSE);
signal_lengths      = unique(alldata[[1]], incomparables = FALSE);
rm(alldata);


#-------------------------------------------------
#Process AAFFT performance data
bestAAFFTperformance <- AAFFTdata[1,]; bestAAFFTperformance <- bestAAFFTperformance[-1,];

#Find best performing configuration for each case (template width, convolution size, ...)
for (nLenght in template_lengths){
  templengthdata <- AAFFTdata[(AAFFTdata[[2]]==nLenght),];
  for (nTemplates in number_of_templates){
    templatedata <- templengthdata[(templengthdata[[3]]==nTemplates),];
    for (slength in signal_lengths){
      slengthdata<-templatedata[(templatedata[[1]]==slength),];
      if (length(slengthdata[,1]) > 0){
        best_line <- which.min(slengthdata[[4]]);
        bestAAFFTperformance<-rbind(bestAAFFTperformance,slengthdata[best_line,]);
      }
    }
  }
}
#rm(AAFFTdata);
#--------------------------------------------------<

#-------------------------------------------------
#Process cuFFT performance data
bestcuFFTperformance <- cuFFTdata[1,]; bestcuFFTperformance <- bestcuFFTperformance[-1,];

#Find best performing configuration for each case (template width, convolution size, ...)
for (nLenght in template_lengths){
  templengthdata <- cuFFTdata[(cuFFTdata[[2]]==nLenght),];
  for (nTemplates in number_of_templates){
    templatedata <- templengthdata[(templengthdata[[3]]==nTemplates),];
    for (slength in signal_lengths){
      slengthdata<-templatedata[(templatedata[[1]]==slength),];
      if (length(slengthdata[,1]) > 0){
        best_line <- which.min(slengthdata[[4]]);
        bestcuFFTperformance<-rbind(bestcuFFTperformance,slengthdata[best_line,]);
      }
    }
  }
}
#rm(cuFFTdata);
#--------------------------------------------------<


#writing best performance grouped by filter length
if(export_best_performance==1){
  for (nLenght in template_lengths){
    templengthdata<-bestAAFFTperformance[(bestAAFFTperformance[[2]]==nLenght),];
    for (slength in signal_lengths){
      slengthdata<-templengthdata[(templengthdata[[1]]==slength),];
      filename="Best_kFFT";
      filename<-paste(filename, slength, nLenght, sep="_");
      filename<-paste(filename, output_extension, sep="");
      unlink(filename);
      write.table(slengthdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
    }
  }
  
  #writing out the results
  extension=".txt";
  for (nLenght in template_lengths){
    templengthdata<-bestcuFFTperformance[(bestcuFFTperformance[[2]]==nLenght),];
    for (slength in signal_lengths){
      slengthdata<-templengthdata[(templengthdata[[1]]==slength),];
      filename="Best_cuFFT";
      filename<-paste(filename, slength, nLenght, sep="_");
      filename<-paste(filename, output_extension, sep="");
      unlink(filename);
      write.table(slengthdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
    }
  }
}


#export data grouped by template width
if(export_data_grouped_by_template_width==1){
  for (nLenght in template_lengths){
    AAFFTdata_fixedTemplate <- bestAAFFTperformance[(bestAAFFTperformance[[2]]==nLenght),];
    #set up container
    AAFFTdata_temp<-AAFFTdata_fixedTemplate[(AAFFTdata_fixedTemplate[[1]]==signal_lengths[[1]]),];
    resultdata<-cbind(AAFFTdata_temp[[3]]);
    #add columns with signal length and time for each signal length
    for (slength in signal_lengths){
      AAFFTdata_temp<-templengthdata[(templengthdata[[1]]==slength),];
	  if (length(AAFFTdata_temp[,1]) > 0){
        resultdata<-cbind(resultdata, AAFFTdata_temp[[1]], AAFFTdata_temp[[4]]);
      }
    }
    #export data
    filename="AAFFT_results";
    filename<-paste(filename, nLenght, sep="_");
    filename<-paste(filename, output_extension, sep="");
    unlink(filename);
    write.table(resultdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
  }

  for (nLenght in template_lengths){
    cuFFTdata_fixedTemplate <- bestcuFFTperformance[(bestcuFFTperformance[[2]]==nLenght),];
    #set up container
    cuFFTdata_temp<-cuFFTdata_fixedTemplate[(cuFFTdata_fixedTemplate[[1]]==signal_lengths[[1]]),];
    resultdata<-cbind(cuFFTdata_temp[[3]]);
    #add columns with signal length and time for each signal length
    for (slength in signal_lengths){
      cuFFTdata_temp<-templengthdata[(templengthdata[[1]]==slength),];
	  if (length(cuFFTdata_temp[,1]) > 0){
        resultdata<-cbind(resultdata, cuFFTdata_temp[[1]], cuFFTdata_temp[[4]]);
      }
    }
    #export data
    filename="cuFFT_results";
    filename<-paste(filename, nLenght, sep="_");
    filename<-paste(filename, output_extension, sep="");
    unlink(filename);
    write.table(resultdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
  }
}


#export data time and speedup vs nTemplates grouped by template width, both AAFFT and cuFFT in one file
if(export_nTemplates_time_speedup_gr_template_width_one_file==1){
  for (nLenght in template_lengths){
    AAFFTdata_fixedTemplate <- bestAAFFTperformance[(bestAAFFTperformance[[2]]==nLenght),];
    cuFFTdata_fixedTemplate <- bestcuFFTperformance[(bestcuFFTperformance[[2]]==nLenght),];
    
    #set up container
    AAFFTdata_temp <- AAFFTdata_fixedTemplate[(AAFFTdata_fixedTemplate[[1]]==signal_lengths[[1]]),];
    resultdata<-cbind(AAFFTdata_temp[[3]]);
    #add columns with signal length and time for each signal length
    for (slength in signal_lengths){
      AAFFTdata_temp <- AAFFTdata_fixedTemplate[(AAFFTdata_fixedTemplate[[1]]==slength),];
      cuFFTdata_temp <- cuFFTdata_fixedTemplate[(cuFFTdata_fixedTemplate[[1]]==slength),];
      #Creating dataframe with results
      resultdata<-cbind(resultdata, AAFFTdata_temp[[1]], cuFFTdata_temp[[4]], AAFFTdata_temp[[4]], cuFFTdata_temp[[4]]/AAFFTdata_temp[[4]]);
    }
    #export data
    filename="Results_TitanV_cuFFT_callbacks_R2R_speedup";
    filename<-paste(filename,nLenght,sep="_");
    filename<-paste(filename,output_extension,sep="");
    unlink(filename);
    write.table(resultdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
  }
}
rm(resultdata)

#export data time and speedup vs signal length grouped by number of templates, both AAFFT and cuFFT in one file
if(export_signal_length_time_speedup_gr_nTemplates_one_file==1){
	for (nLenght in template_lengths){
		AAFFTdata_fixedTemplate <- bestAAFFTperformance[(bestAAFFTperformance[[2]]==nLenght),];
		cuFFTdata_fixedTemplate <- bestcuFFTperformance[(bestcuFFTperformance[[2]]==nLenght),];
		
		#set up container
		AAFFTdata_temp <- AAFFTdata_fixedTemplate[(AAFFTdata_fixedTemplate[[3]]==number_of_templates[[3]]),];
		signal_length <- cbind(AAFFTdata_temp[[1]]);
		resultdata <- cbind(AAFFTdata_temp[[1]]);
		#add columns with signal length and time for each signal length
		for (nTemplates in number_of_templates){
			temporaryresults <- signal_length
			AAFFTdata_temp <- AAFFTdata_fixedTemplate[(AAFFTdata_fixedTemplate[[3]]==nTemplates),];
			cuFFTdata_temp <- cuFFTdata_fixedTemplate[(cuFFTdata_fixedTemplate[[3]]==nTemplates),];
			#Creating dataframe with results
			temporaryresults <-merge(temporaryresults, AAFFTdata_temp, by.x='V1', by.y='V1', all.x=TRUE, all.y=FALSE)
			temporaryresults <-merge(temporaryresults, cuFFTdata_temp, by.x='V1', by.y='V1', all.x=TRUE, all.y=FALSE)
			local_results<-cbind(temporaryresults[[3]], temporaryresults[[12]], temporaryresults[[4]], temporaryresults[[12]]/temporaryresults[[4]]);
			resultdata<-cbind(resultdata, local_results);
			rm(temporaryresults);
		}
		#export data
		filename="Results_TitanV_cuFFT_callbacks_R2R_speedup_signal_length";
		filename<-paste(filename,nLenght,sep="_");
		filename<-paste(filename,output_extension,sep="");
		unlink(filename);
		write.table(resultdata, file = filename, append = FALSE, sep = " ", row.names=FALSE, col.names=FALSE, quote = FALSE);
	}
}
