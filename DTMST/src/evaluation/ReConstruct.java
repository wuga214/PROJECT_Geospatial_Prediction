package evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;

import regressions.Algorithms;
import regressions.Problems;
import utils.RegressionProblem;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class ReConstruct {

	public static void Labeling(CVOutput cv) throws Exception{
		Problems pbs=new Problems();
		Algorithms algo=new Algorithms();
		for(int i=0; i<cv.problems.size();i++){
			RegressionProblem cp=pbs.createRegressionProblem(cv.problems.get(i));
			Instances unlabeled=cp.getData();
			// set class attribute
			unlabeled.setClassIndex(unlabeled.numAttributes() - 1);			
			// create copy
			Random rand=new Random();
			unlabeled.randomize(rand);
			Resample filter=new Resample();
			filter.setOptions(new String[]{"-Z","20","-no-replacement"});
			filter.setInputFormat(unlabeled);
			Instances newTrain = Filter.useFilter(unlabeled, filter);
			Instances labeled = new Instances(unlabeled);
			for(int j=0;j<cv.evals.get(i).size();j++){
				AbstractClassifier classifier=algo.createClassifier(cv.evals.get(i).get(j).name);
				classifier.setOptions(cv.evals.get(i).get(j).settings.split("\\s+"));				
				classifier.buildClassifier(newTrain);
				// label instances
				for (int k = 0; k < unlabeled.numInstances(); k++) {
					double clsLabel = classifier.classifyInstance(unlabeled.instance(k));
					labeled.instance(k).setClassValue(clsLabel);
				}
				// save labeled data
				BufferedWriter writer = new BufferedWriter(
						new FileWriter("outputs/"+cv.problems.get(i).toString()+"_"+cv.evals.get(i).get(j).name.toString()+".arff"));
				writer.write(labeled.toString());
				writer.newLine();
				writer.flush();
				writer.close();
			}
		}
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
