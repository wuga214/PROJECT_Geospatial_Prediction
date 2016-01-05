package evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import regressions.EProblemList;

public class CVOutput {
	public List<List<FoldRecord>> evals;
	public List<EProblemList> problems;
	public CVOutput(){
		evals=new ArrayList<List<FoldRecord>>();
		problems=new ArrayList<EProblemList>();
	}
	public void add(List<FoldRecord> clsrst,EProblemList problemname){
		evals.add(clsrst);
		problems.add(problemname);
	}
	
	public String getCCTable(){
//		\begin{table}[]
//				\centering
//				\caption{Correlation Coefficient Table}
//				\label{table_1}
//				\begin{tabular}{lllll}
//				 &  &  &  &  \\
//				 &  &  &  &  \\
//				 &  &  &  &  \\
//				 &  &  &  & 
//				\end{tabular}
//				\end{table}
		DecimalFormat df = new DecimalFormat("#.00"); 
		String latextable="\\begin{table}[]\n";
		latextable+="\\centering\n\\caption{Correlation Coefficient Table}\n";
		latextable+="\\label{table_1}\n\\begin{tabular}{llllll}\n";
		latextable+="Classifiers& RegressionTree& GaussianProcess& KNN & MSTMean & MSTHyperplate\n";
		latextable+="\\hline\n";
		for(int i=0; i<evals.size();i++){
			List<FoldRecord> problemresult=evals.get(i);
			latextable+="$"+problems.get(i).toString()+"$";
			for(int j=0;j<problemresult.size();j++){
				latextable+="&"+df.format(100*problemresult.get(j).correlation_coefficient);
			}
			latextable+="\\\\\n";
		}
		latextable+="\\end{tabular}\n";
		latextable+="\\end{table}";
		return latextable;
	}
	
	public String getRMSETable(){
//		\begin{table}[]
//				\centering
//				\caption{Correlation Coefficient Table}
//				\label{table_1}
//				\begin{tabular}{lllll}
//				 &  &  &  &  \\
//				 &  &  &  &  \\
//				 &  &  &  &  \\
//				 &  &  &  & 
//				\end{tabular}
//				\end{table}
		DecimalFormat df = new DecimalFormat("#.00"); 
		String latextable="\\begin{table}[]\n";
		latextable+="\\centering\n\\caption{Correlation Coefficient Table}\n";
		latextable+="\\label{table_1}\n\\begin{tabular}{llll1l}\n";
		latextable+="Classifiers& RegressionTree& GaussianProcess& KNN & MSTMean & MSTHyperplate\n";
		latextable+="\\hline\n";
		for(int i=0; i<evals.size();i++){
			List<FoldRecord> problemresult=evals.get(i);
			latextable+="$"+problems.get(i).toString()+"$";
			for(int j=0;j<problemresult.size();j++){
				latextable+="&"+df.format(100*problemresult.get(j).RMSE);
			}
			latextable+="\\\\\n";
		}
		latextable+="\\end{tabular}\n";
		latextable+="\\end{table}";
		return latextable;
	}
	
	public String getCCTableWVariance(){
		DecimalFormat df = new DecimalFormat("#.00"); 
		String latextable="\\begin{table}[]\n";
		latextable+="\\centering\n\\caption{Correlation Coefficient CI95}\n";
		latextable+="\\label{table_1}\n\\begin{tabular}{lllll1}\n";
		latextable+="Classifiers& RegressionTree& GaussianProcess& KNN & MSTMean & MSTHyperplate\n";
		latextable+="\\hline\n";
		for(int i=0; i<evals.size();i++){
			List<FoldRecord> problemresult=evals.get(i);
			latextable+="$"+problems.get(i).toString()+"$";
			for(int j=0;j<problemresult.size();j++){
				latextable+="&"+df.format(100*problemresult.get(j).correlation_coefficient)+"+/-"+df.format(100*problemresult.get(j).VC);
			}
			latextable+="\\\\\n";
		}
		latextable+="\\end{tabular}\n";
		latextable+="\\end{table}";
		return latextable;
	}
	
	public String getRMSETableWVariance(){
		DecimalFormat df = new DecimalFormat("#.00"); 
		String latextable="\\begin{table}[]\n";
		latextable+="\\centering\n\\caption{RMSE CI95}\n";
		latextable+="\\label{table_2}\n\\begin{tabular}{lllll1}\n";
		latextable+="Classifiers& RegressionTree& GaussianProcess& KNN & MSTMean & MSTHyperplate\n";
		latextable+="\\hline\n";
		for(int i=0; i<evals.size();i++){
			List<FoldRecord> problemresult=evals.get(i);
			latextable+="$"+problems.get(i).toString()+"$";
			for(int j=0;j<problemresult.size();j++){
				latextable+="&$"+df.format(100*problemresult.get(j).RMSE)+"+/-"+df.format(100*problemresult.get(j).VR)+"$";
			}
			latextable+="\\\\\n";
		}
		latextable+="\\end{tabular}\n";
		latextable+="\\end{table}";
		return latextable;
	}
	
	public String getMAETableWVariance(){
		DecimalFormat df = new DecimalFormat("#.00"); 
		String latextable="\\begin{table}[]\n";
		latextable+="\\centering\n\\caption{MAE CI95}\n";
		latextable+="\\label{table_3}\n\\begin{tabular}{llll1l}\n";
		latextable+="Classifiers& RegressionTree& GaussianProcess& KNN & MSTMean & MSTHyperplate\n";
		latextable+="\\hline\n";
		for(int i=0; i<evals.size();i++){
			List<FoldRecord> problemresult=evals.get(i);
			latextable+="$"+problems.get(i).toString()+"$";
			for(int j=0;j<problemresult.size();j++){
				latextable+="&$"+df.format(100*problemresult.get(j).MAE)+"+/-"+df.format(100*problemresult.get(j).VM)+"$";
			}
			latextable+="\\\\\n";
		}
		latextable+="\\end{tabular}\n";
		latextable+="\\end{table}";
		return latextable;
	}
	
}
