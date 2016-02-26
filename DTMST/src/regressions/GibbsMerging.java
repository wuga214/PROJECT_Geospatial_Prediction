package regressions;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import sampling.Gibbs;
import sampling.GibbsDirLocal;
import sampling.SampleManager;
import utils.RandomPermutation;
import utils.RegressionProblem;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class GibbsMerging extends Classifier{
	/**
	 * 
	 */
	private static final long serialVersionUID = -3126934031726996304L;
	public SampleManager samp;
	public int iteration;
	public int labelRestriction;
	public double alpha;
	
	public GibbsMerging(){
		samp=new SampleManager();
		iteration=1000;
		labelRestriction=0;
		alpha=0.1;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		if(labelRestriction==0){labelRestriction=arg0.numInstances();}
		Gibbs gb=new Gibbs(arg0, arg0, iteration,labelRestriction, alpha, samp);
		//GibbsDirLocal gb=new GibbsDirLocal(arg0, arg0, iteration,labelRestriction, samp);
        gb.Sampling(arg0,false);
        //samp.sampleReport();
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
        
        if(Arrays.asList(options).contains("-A")){
        	String iter=Arrays.asList(options).get(Arrays.asList(options).indexOf("-A")+1);
        	alpha=Double.parseDouble(iter);
        }
        
//        if(Arrays.asList(options).contains("-L")){
//        	String iter=Arrays.asList(options).get(Arrays.asList(options).indexOf("-L")+1);
//        	labelRestriction=Integer.parseInt(iter);
//        }
        
        //work around to avoid the api print trash in the console
        System.setOut(console);
    }
    
    public String[] getOptions() {

        String[] options = new String[4];
        options[0] = "-I";//number of merging required!!
        options[1] = Integer.toString(iteration);
        options[2] = "-A";//number of different label restricted
        options[3] = Double.toString(alpha);;
//        options[3] = Integer.toString(labelRestriction);
        return options;
    }
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			RegressionProblem cp = new RegressionProblem("data/houseprice.arff");
			cp.normalizeData();
			RandomPermutation randPerm=new RandomPermutation();
			randPerm.getRandomPermutation(cp.getData());
			Instances data=new Instances(randPerm.permutated);
			//MAPofBMA classifier=new MAPofBMA(26,-124,24,70);
			long startTime = System.currentTimeMillis();
			GibbsMerging classifier=new GibbsMerging();
			classifier.setOptions(new String[]{"-I","2000","-A","0.001"});
			Resample filter=new Resample();
			filter.setOptions(new String[]{"-Z","15","-no-replacement","-S","1"});
			filter.setInputFormat(data);
			Instances newTrain = Filter.useFilter(data, filter);
			classifier.buildClassifier(newTrain);
			filter.setOptions(new String[]{"-Z","15","-no-replacement","-S","2"});
            Instances newTest = Filter.useFilter(data, filter); 
            Evaluation eval = new Evaluation(newTrain);
            eval.evaluateModel(classifier, newTest);
            //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            //System.out.println(i+","+eval.correlationCoefficient());
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
