package evaluation;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

import regressions.EObjectiveList;
import regressions.EProblemList;

public class SingleProblemOutput {
	public List<List<FoldRecord>> evals;
	public EProblemList problem;
	
	public SingleProblemOutput(EProblemList problem_name){
		evals=new ArrayList<List<FoldRecord>>();
		problem=problem_name;
	}
	
	public void add(List<FoldRecord> clsrst){
		evals.add(clsrst);
	}
	
	public void writeCSV(EObjectiveList objective) throws FileNotFoundException, UnsupportedEncodingException{
		//Correlation Coefficient or RMSE or MAE
		switch(objective){
			case COEFFICIENT:
				writeCoefficient(objective);
				break;
			case RMSE:
				writeRMSE(objective);
				break;
			case MAE:
				writeMAE(objective);
				break;
		}
	}
	
	public void writeCoefficient(EObjectiveList objective) throws FileNotFoundException, UnsupportedEncodingException{
		int m = evals.size();
		int n = evals.get(0).size();
		PrintWriter writer = new PrintWriter("outputs/trainsize/"+problem.toString()+"_"+objective.toString(), "UTF-8");
		for(int i=0;i<n;i++){
			String line="";
			for(int j=0;j<m;j++){
				line+=evals.get(j).get(i).correlation_coefficient+",";
			}
			writer.println(line.subSequence(0, line.length()-2));
			System.out.println(line.subSequence(0, line.length()-2));
		}
		writer.close();
		System.out.println("File Saved!");
	}
	
	public void writeRMSE(EObjectiveList objective) throws FileNotFoundException, UnsupportedEncodingException{
		int m = evals.size();
		int n = evals.get(0).size();
		PrintWriter writer = new PrintWriter("outputs/trainsize/"+problem.toString()+"_"+objective.toString(), "UTF-8");
		for(int i=0;i<n;i++){
			String line="";
			for(int j=0;j<m;j++){
				line+=evals.get(j).get(i).RMSE+",";
			}
			writer.println(line.subSequence(0, line.length()-2));
			System.out.println(line.subSequence(0, line.length()-2));
		}
		writer.close();
		System.out.println("File Saved!");
	}
	
	public void writeMAE(EObjectiveList objective) throws FileNotFoundException, UnsupportedEncodingException{
		int m = evals.size();
		int n = evals.get(0).size();
		PrintWriter writer = new PrintWriter("outputs/trainsize/"+problem.toString()+"_"+objective.toString()+".csv", "UTF-8");
		for(int i=0;i<n;i++){
			String line="";
			for(int j=0;j<m;j++){
				line+=evals.get(j).get(i).MAE+",";
			}
			writer.println(line.subSequence(0, line.length()-1));
			System.out.println(line.subSequence(0, line.length()-2));
		}
		writer.close();
		System.out.println("File Saved!");
	}
}
