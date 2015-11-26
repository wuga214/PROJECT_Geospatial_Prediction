package evaluation;

import java.io.IOException;

import regressions.MAPofBMA;
import utils.RegressionProblem;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class CrossValidations {
	
	public static double crossValidation(Object classifier, Instances data,int folds) throws Exception{
		Evaluation eval = new Evaluation(data);
		for(int n=0;n<folds;n++){
			Classifier clsCopy = Classifier.makeCopy((Classifier)classifier);
			Instances train=data.trainCV(folds, n);
			Instances valid=data.testCV(folds, n);
			clsCopy.buildClassifier(train);
	        eval.evaluateModel(clsCopy, valid);
	        System.out.println("iteration:"+n);
		}
	      System.out.println();
	      System.out.println("=== Setup run ===");
	      System.out.println("Classifier: " + classifier.getClass().getName() + " " + Utils.joinOptions(((Classifier) classifier).getOptions()));
	      System.out.println("Dataset: " + data.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run ===", false));
	      return eval.correlationCoefficient();
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			RegressionProblem cp = new RegressionProblem("data/tobs-averages.arff");
			Instances data=cp.getData();
			Resample filter=new Resample();
			filter.setOptions(new String[]{"-Z","30","-no-replacement","-S","1"});
			filter.setInputFormat(cp.getData());
			Instances newTrain = Filter.useFilter(cp.getData(), filter);
			MAPofBMA classifier=new MAPofBMA(26,-124,24,70);
			classifier.setOptions(new String[]{"-I","1"});
			double ave=crossValidation(classifier,newTrain,10);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
