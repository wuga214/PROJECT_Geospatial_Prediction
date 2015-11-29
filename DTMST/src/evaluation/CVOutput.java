package evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class CVOutput {
	public List<List<FoldRecord>> evals;
	public List<String> problems;
	public CVOutput(){
		evals=new ArrayList<List<FoldRecord>>();
		problems=new ArrayList<String>();
	}
	public void add(List<FoldRecord> clsrst,String problemname){
		evals.add(clsrst);
		problems.add(problemname);
	}
	
	public String getTable(){
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
		latextable+="\\label{table_1}\n\\begin{tabular}{lllll}\n";
		latextable+="Classifiers& RegressionTree& GaussianProcess& KNN & MAPofMerging\n";
		latextable+="\\hline\n";
		for(int i=0; i<evals.size();i++){
			List<FoldRecord> problemresult=evals.get(i);
			latextable+="$"+problems.get(i)+"$";
			for(int j=0;j<problemresult.size();j++){
				latextable+="&"+df.format(100*problemresult.get(j).correlation_coefficient);
			}
			latextable+="\\\\\n";
		}
		latextable+="\\end{tabular}\n";
		latextable+="\\end{table}";
		return latextable;
	}
	
}
