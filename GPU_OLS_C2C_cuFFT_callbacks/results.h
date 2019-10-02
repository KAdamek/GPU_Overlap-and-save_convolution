#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

class Performance_results{
public:
	double GPU_time;
	int nTimesamples;
	int template_length;
	int nTemplates;
	int nRuns;
	int reglim;
	int OaS_conv_size;
	int templates_per_block;
	char filename[200];
	char kernel[10];
	
	Performance_results() {
		GPU_time=0;
	}
	
	void Save(){
		ofstream FILEOUT;
		FILEOUT.open (filename, std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << nTimesamples << " " << template_length << " " << nTemplates << " " << GPU_time << " " << nRuns << " " << reglim << " " << OaS_conv_size << " " << templates_per_block << " " << kernel << endl;
		FILEOUT.close();
	}
	
	void Print(){
		cout << std::fixed << std::setprecision(8) << nTimesamples << " " << template_length << " " << nTemplates << " " << GPU_time << " " << nRuns << " " << reglim << " " << OaS_conv_size << " " << templates_per_block << " " << kernel << endl;
	}
	
	void Assign(int t_nTimesamples, int t_template_length, int t_nTemplates, int t_nRuns, int t_reglim, int t_OaS_conv_size, int t_templates_per_block, char const *t_filename, char const *t_kernel){
		nTimesamples        = t_nTimesamples;
		template_length     = t_template_length;
		nTemplates          = t_nTemplates;
		nRuns               = t_nRuns;
		reglim              = t_reglim;
		OaS_conv_size       = t_OaS_conv_size;
		templates_per_block = t_templates_per_block;
		sprintf(filename,"%s", t_filename);
		sprintf(kernel,"%s",t_kernel);
	}
	
};
