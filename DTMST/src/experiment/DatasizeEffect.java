package experiment;


import evaluation.CrossValidations;
import evaluation.SingleProblemOutput;
import regressions.EObjectiveList;
import regressions.EProblemList;

public class DatasizeEffect {
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			for(EProblemList x:EProblemList.values()){
				SingleProblemOutput spout=CrossValidations.TrainingDataSizeEval(x);
				spout.writeCSV(EObjectiveList.COEFFICIENT);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
