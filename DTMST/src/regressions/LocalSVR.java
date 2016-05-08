package regressions;

import java.io.PrintStream;
import java.util.Arrays;

import utils.RandomPermutation;
import utils.RegressionProblem;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class LocalSVR extends AbstractClassifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = -365825347273082530L;
	private int radiusrate;
	private double radius;
	private double stridesize;
	private Instances data;

	public LocalSVR(){
		radiusrate=1;
		radius=1;
		stridesize=1;
	}
	
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		data=arg0;
		AttributeStats x= data.attributeStats(0);
		double x_diff= x.numericStats.max;
		x_diff=x_diff-x.numericStats.min;
		AttributeStats y= data.attributeStats(0);
		double y_diff= y.numericStats.max;
		y_diff=y_diff-y.numericStats.min;
		
		if(x_diff<=y_diff){
			stridesize=x_diff/20;
		}
		else{
			stridesize=y_diff/20;
		}
		computeRadius();
	}
	
	private double computeRadius(){
		radius=radiusrate*stridesize;
		return radius;
	}

	public double distance(Instance inst1, Instance inst2) {
	    double dist = 0.0;

	    for (int i = 0; i < inst1.numAttributes(); i++) {
	        double x = inst1.value(i);
	        double y = inst1.value(i);

	        if (Double.isNaN(x) || Double.isNaN(y)) {
	            continue; // Mark missing attributes ('?') as NaN.
	        }

	        dist += (x-y)*(x-y);
	    }

	    return Math.sqrt(dist);
	}
	
	public double classifyInstance(Instance newInstance) throws Exception {
        Instances activeinstances=new Instances(data,0);
        double averagevalue=0;
        for(int i=0;i<data.numInstances();i++){
        	if(radius>=distance(newInstance, data.instance(i))){
        		activeinstances.add(data.instance(i));
        		averagevalue+=data.instance(i).classValue();
        	}
        }
        
        if(activeinstances.numInstances()==0){
        	 throw new Exception("No data point in predesigned distance!");
        }
        averagevalue=averagevalue/activeinstances.numInstances();
        if(activeinstances.numInstances()>4){
        	SVR classifier=new SVR();
        	classifier.buildClassifier(activeinstances);
        	return classifier.classifyInstance(newInstance);
        }
        else{
        	return averagevalue;
        }
    }
	
    public void setOptions(String[] options) throws Exception {
    	PrintStream console = System.out;
        super.setOptions(options);
        if(Arrays.asList(options).contains("-R")){
        	String iter=Arrays.asList(options).get(Arrays.asList(options).indexOf("-R")+1);
        	radiusrate=Integer.parseInt(iter);
        	computeRadius();
        }
        System.setOut(console);
    }
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			RegressionProblem cp = new RegressionProblem("data/box.arff");
			cp.normalizeData();
			RandomPermutation randPerm=new RandomPermutation();
			randPerm.getRandomPermutation(cp.getData());
			Instances data=new Instances(randPerm.permutated);
			//MAPofBMA classifier=new MAPofBMA(26,-124,24,70);
			long startTime = System.currentTimeMillis();
			LocalSVR classifier=new LocalSVR();
			classifier.setOptions(new String[]{"-R","5"});
			Resample filter=new Resample();
			filter.setOptions(new String[]{"-Z","15","-no-replacement","-S","1"});
			filter.setInputFormat(data);
			Instances newTrain = Filter.useFilter(data, filter);
			classifier.buildClassifier(newTrain);
			filter.setOptions(new String[]{"-Z","15","-no-replacement","-S","2"});
            Instances newTest = Filter.useFilter(data, filter); 
            Evaluation eval = new Evaluation(newTrain);
            eval.evaluateModel(classifier, newTest);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));	
            long endTime   = System.currentTimeMillis();
            long totalTime = endTime - startTime;
            System.out.println(totalTime);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
