package evaluation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import extern.arff.ArfftoCSV;
import regressions.Algorithms;
import regressions.EProblemList;
import regressions.ERegressionList;
import regressions.MAPofBMA;
import regressions.Problems;
import utils.RandomPermutation;
import utils.RegressionProblem;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class CrossValidations {
	
	public static Evaluation crossValidation(Object classifier, Instances data,int folds) throws Exception{
		Evaluation eval = new Evaluation(data);
		for(int n=0;n<folds;n++){
			Classifier clsCopy = Classifier.makeCopy((Classifier)classifier);
			Instances train=data.trainCV(folds, n);
			Instances valid=data.testCV(folds, n);
			clsCopy.buildClassifier(train);
	        eval.evaluateModel(clsCopy, valid);
		}
//	      System.out.println();
//	      System.out.println("=== Setup run ===");
//	      System.out.println("Classifier: " + classifier.getClass().getName() + " " + Utils.joinOptions(((Classifier) classifier).getOptions()));
//	      System.out.println("Dataset: " + data.relationName());
//	      System.out.println("Folds: " + folds);
//	      System.out.println();
//	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run ===", false));
	      return eval;
	}
	
	public static List<FoldRecord> batchCrossValidation(RegressionProblem cp) throws Exception{
		RandomPermutation randPerm=new RandomPermutation();
		randPerm.getRandomPermutation(cp.getData());
		Instances data=randPerm.permutated;
		Resample filter=new Resample();
		filter.setOptions(new String[]{"-Z","20","-no-replacement","-S","1"});
		filter.setInputFormat(data);
		Instances newTrain = Filter.useFilter(data, filter);
		filter.setOptions(new String[]{"-Z","30","-no-replacement","-S","3"});
        Instances newTest = Filter.useFilter(data, filter);
		Algorithms algo=new Algorithms();
		List<FoldRecord> probresults= new ArrayList<FoldRecord>();
		for(ERegressionList name:ERegressionList.values()){
			Map<String, Set<String>> params=algo.getDefaultClassifiersParameters(name);
			List<String> settings=SettingExtender.generateModels(params);
			List<FoldRecord> foldresult=new ArrayList<FoldRecord>();
			for(String setting:settings){
				Classifier classifier=algo.createClassifier(name);
				classifier.setOptions(setting.split("\\s+"));
				Evaluation eval=crossValidation(classifier,newTrain,10);
				FoldRecord record=new FoldRecord(name,setting,eval.correlationCoefficient(),eval.rootMeanSquaredError());
				foldresult.add(record);
			}
			Collections.sort(foldresult);
			FoldRecord bestSetting=foldresult.get(0);
			Classifier classifier=algo.createClassifier(bestSetting.name);
			classifier.setOptions(bestSetting.settings.split("\\s+"));
			classifier.buildClassifier(newTrain);
			Evaluation eval=new Evaluation(newTrain);
			eval.evaluateModel(classifier, newTest);
		      System.out.println();
		      System.out.println("=== Setup run ===");
		      System.out.println("Classifier: " + classifier.getClass().getName() + " " + Utils.joinOptions(((Classifier) classifier).getOptions()));
		      System.out.println("Dataset: " + data.relationName());
		      System.out.println();
		      System.out.println(eval.toSummaryString("=== test dataset result ===", false));
		    probresults.add(new FoldRecord(bestSetting.name,bestSetting.settings));
		}
		probresults=multipleRun(probresults,randPerm.permutated);
		return probresults;
	}
	
	public static List<FoldRecord> multipleRun(List<FoldRecord> records, Instances permutated ) throws Exception{
		int m=records.size();
		int iteration=50;
		double[][] corr=new double[iteration][m];
		double[][] rmse=new double[iteration][m];
		double[][] mae=new double[iteration][m];
		for(int i=0;i<iteration;i++){
			RandomPermutation randPerm=new RandomPermutation();
			randPerm.getRandomPermutation(permutated);
			Instances data=randPerm.permutated;
			Resample filter=new Resample();
			filter.setOptions(new String[]{"-Z","20","-no-replacement","-S","1"});
			filter.setInputFormat(data);
			Instances newTrain = Filter.useFilter(data, filter);
			filter.setOptions(new String[]{"-Z","30","-no-replacement","-S","3"});
	        Instances newTest = Filter.useFilter(data, filter);
	        Algorithms algo=new Algorithms();
	        for(int j=0;j<m;j++){
	        	Classifier classifier=algo.createClassifier(records.get(j).name);
				classifier.setOptions(records.get(j).settings.split("\\s+"));
				classifier.buildClassifier(newTrain);
				Evaluation eval = new Evaluation(newTrain);
				eval.evaluateModel(classifier, newTest);
				corr[i][j]=eval.correlationCoefficient();
				rmse[i][j]=eval.rootMeanSquaredError();
				mae[i][j]=eval.meanAbsoluteError();
	        }
		}
		List<FoldRecord> probresults=new ArrayList<FoldRecord>();
		for(int j=0;j<m;j++){
			double cc=0;
			double rm=0;
			double ma=0;
			double varic=0;
			double varir=0;
			double varim=0;
			for(int i=0;i<iteration;i++){
				cc+=corr[i][j];
				rm+=rmse[i][j];
				ma+=mae[i][j];
			}
			cc=cc/iteration;
			rm=rm/iteration;
			ma=ma/iteration;
			for(int i=0;i<iteration;i++){
				varic+=Math.pow(corr[i][j]-cc, 2);
				varir+=Math.pow(rmse[i][j]-rm, 2);
				varim+=Math.pow(mae[i][j]-ma, 2);
			}
			varic=Math.sqrt(varic/iteration)/Math.sqrt(iteration);
			varir=Math.sqrt(varir/iteration)/Math.sqrt(iteration);
			varim=Math.sqrt(varim/iteration)/Math.sqrt(iteration);
			probresults.add(new FoldRecord(records.get(j).name,records.get(j).settings,cc,rm,ma,varic,varir,varim));
		}
		return probresults;
	}
	
	public static CVOutput autobatchCrossValidation() throws Exception{
		CVOutput cvout=new CVOutput();
		Problems pbs=new Problems();
		for(EProblemList name:EProblemList.values()){
			RegressionProblem cp=pbs.createRegressionProblem(name);
			cvout.add(batchCrossValidation(cp),name);
		}
		return cvout;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
//			RegressionProblem cp = new RegressionProblem("data/tobs-averages.arff");
//			Instances data=cp.getData();
//			Resample filter=new Resample();
//			filter.setOptions(new String[]{"-Z","30","-no-replacement","-S","1"});
//			filter.setInputFormat(cp.getData());
//			Instances newTrain = Filter.useFilter(cp.getData(), filter);
//			MAPofBMA classifier=new MAPofBMA(26,-124,24,70);
//			classifier.setOptions(new String[]{"-I","1"});
//			Evaluation ave=crossValidation(classifier,newTrain,10);
			CVOutput cvout=autobatchCrossValidation();
			System.out.println(cvout.getCCTableWVariance());
			System.out.println(cvout.getRMSETableWVariance());
			System.out.println(cvout.getMAETableWVariance());
			ReConstruct.Labeling(cvout);
			ArfftoCSV.BatchConvert(cvout);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
