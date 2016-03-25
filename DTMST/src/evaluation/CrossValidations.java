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
	      return eval;
	}
	
	public static List<FoldRecord> batchCrossValidation(Instances data, int training_percent) throws Exception{
		int trainSize = (int) Math.round(data.numInstances() * (double)training_percent/ 100);
		int testSize = data.numInstances() - (int) Math.round(data.numInstances() * 0.5);
		Instances newTrain = new Instances(data, 0, trainSize);
		Instances newTest = new Instances(data, testSize, testSize);
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
				System.gc();
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
		System.gc();
		probresults=multipleRun(probresults,data,training_percent);
		return probresults;
	}
	
	public static List<FoldRecord> multipleRun(List<FoldRecord> records, Instances permutated,int training_percent ) throws Exception{
		int m=records.size();
		int iteration=10;
		double[][] corr=new double[iteration][m];
		double[][] rmse=new double[iteration][m];
		double[][] mae=new double[iteration][m];
		for(int i=0;i<iteration;i++){
			RandomPermutation randPerm=new RandomPermutation();
			randPerm.getRandomPermutation(permutated);
			Instances data=randPerm.permutated;
			int trainSize = (int) Math.round(data.numInstances() * (double)training_percent/ 100);
			int testSize = data.numInstances() - (int) Math.round(data.numInstances() * 0.5);
			Instances newTrain = new Instances(data, 0, trainSize);
			Instances newTest = new Instances(data, testSize, testSize);
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
//				rmse[i][j]=eval.rootRelativeSquaredError();
//				mae[i][j]=eval.relativeAbsoluteError();
	        }
	        System.gc();
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
			cp.normalizeData();
			RandomPermutation randPerm=new RandomPermutation();
			randPerm.getRandomPermutation(cp.getData());
			Instances data=randPerm.permutated;
			cvout.add(batchCrossValidation(data,30),name);
		}
		return cvout;
	}
	
	public static SingleProblemOutput TrainingDataSizeEval(EProblemList name) throws Exception{
		SingleProblemOutput spout=new SingleProblemOutput(name);
		Problems pbs=new Problems();
		RegressionProblem cp=pbs.createRegressionProblem(name);
		cp.normalizeData();
		RandomPermutation randPerm=new RandomPermutation();
		randPerm.getRandomPermutation(cp.getData());
		Instances data=randPerm.permutated;
		for(int i=10;i<=50;i=i+10){
			spout.add(batchCrossValidation(data,i));
		}
		return spout;
	}
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
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
