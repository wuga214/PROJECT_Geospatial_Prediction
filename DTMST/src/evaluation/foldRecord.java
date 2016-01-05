package evaluation;

import regressions.ERegressionList;

public class FoldRecord implements Comparable<FoldRecord> {
	public ERegressionList name;
	public String settings;
	public double correlation_coefficient;
	public double RMSE;
	public double MAE;
	public double VC;
	public double VR;
	public double VM;
	
	public FoldRecord(ERegressionList n, String s, double cc, double r){
		name=n;
		settings=s;
		correlation_coefficient=cc;
		RMSE=r;
	}
	
	public FoldRecord(ERegressionList n, String s, double cc, double r, double mae, double varc, double varr,double varm){
		name=n;
		settings=s;
		correlation_coefficient=cc;
		RMSE=r;
		MAE=mae;
		VC=varc;
		VR=varr;
		VM=varm;
	}
	
	public FoldRecord(ERegressionList n, String s){
		name=n;
		settings=s;
	}


	public String toString(){
		String output="";
		output+="Name:"+name+"\n";
		output+="Settings:"+settings+"\n";
		output+="Correlation Coefficient"+correlation_coefficient+"\n";
		output+="Root Mean Square Error"+RMSE+"\n";
		return output;
	}

	@Override
	public int compareTo(FoldRecord arg0) {
		// TODO Auto-generated method stub
		int compare = this.correlation_coefficient > arg0.correlation_coefficient ? -1 : this.correlation_coefficient < arg0.correlation_coefficient ? +1 : 0;
		return compare;
	}
}
