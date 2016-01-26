package regressions;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import sampling.Gibbs;
import sampling.SampleManager;
import utils.RegressionProblem;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class GibbsMerging extends Classifier{
	public SampleManager samp;
	public int iteration;
	public int labelRestriction;
	
	public GibbsMerging(){
		samp=new SampleManager();
		iteration=1000;
		labelRestriction=4;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		Gibbs gb=new Gibbs(arg0, arg0, iteration,labelRestriction, samp);
        gb.Sampling(arg0,false);
        samp.createBaggingModel(arg0);
	}
	
    public double classifyInstance(Instance newInstance) throws Exception {
        return samp.predictLabel(newInstance);
    }
    
    public void setOptions(String[] options) throws Exception {
        //work around to avoid the api print trash in the console
        PrintStream console = System.out;
        System.setOut(new PrintStream(new OutputStream() {
                @Override public void write(int b) throws IOException {}
        }));

        super.setOptions(options);
        if(Arrays.asList(options).contains("-I")){
        	String iter=Arrays.asList(options).get(Arrays.asList(options).indexOf("-I")+1);
        	iteration=Integer.parseInt(iter);
        }
        
        if(Arrays.asList(options).contains("-L")){
        	String iter=Arrays.asList(options).get(Arrays.asList(options).indexOf("-L")+1);
        	labelRestriction=Integer.parseInt(iter);
        }
        
        //work around to avoid the api print trash in the console
        System.setOut(console);
    }
    
    public String[] getOptions() {

        String[] options = new String[4];
        options[0] = "-I";//number of merging required!!
        options[1] = Integer.toString(iteration);
        options[2] = "-L";//number of different label restricted
        options[1] = Integer.toString(labelRestriction);
        return options;
    }
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			for(int i=2;i<=10;i++){
			RegressionProblem cp = new RegressionProblem("data/box.arff");
			//MAPofBMA classifier=new MAPofBMA(26,-124,24,70);
			GibbsMerging classifier=new GibbsMerging();
			classifier.setOptions(new String[]{"-I","1000","-L",Integer.toString(i)});
			Resample filter=new Resample();
			filter.setOptions(new String[]{"-Z","20","-no-replacement","-S","1"});
			filter.setInputFormat(cp.getData());
			Instances newTrain = Filter.useFilter(cp.getData(), filter);
			classifier.buildClassifier(newTrain);
			filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","2"});
            Instances newTest = Filter.useFilter(cp.getData(), filter); 
            Evaluation eval = new Evaluation(newTrain);
            eval.evaluateModel(classifier, newTest);
            //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            //System.out.println(i+","+eval.correlationCoefficient());
            System.out.println(i+","+eval.correlationCoefficient());
            }
		
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
