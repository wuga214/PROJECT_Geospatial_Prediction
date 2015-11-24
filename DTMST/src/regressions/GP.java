package regressions;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

import utils.RegressionProblem;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.GaussianProcesses;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class GP extends GaussianProcesses {

	public GP(){
		super();
	}

	//method to train a naive bayes classifier
    @Override
    public void buildClassifier(Instances data)throws Exception {
            super.buildClassifier(data);

    }

    //method to classify a new intance using the naive bayes classifyer
    @Override
    public double classifyInstance(Instance newInstance) throws Exception {
            return super.classifyInstance(newInstance);

    }
    /**
	 * Valid options are:
	
	       -D
	        If set, classifier is run in debug mode and
	        may output additional info to the console
	 
	       -L <double>
	        Level of Gaussian Noise. (default 0.1)
	 
	       -M <double>
	        Level of Gaussian Noise for the class. (default 0.1)
	 
	       -N
	        Whether to 0=normalize/1=standardize/2=neither. (default 0=normalize)
	 
	       -K <classname and parameters>
	        The Kernel to use.
	        (default: weka.classifiers.functions.supportVector.PolyKernel)
	 
	 
	       Options specific to kernel weka.classifiers.functions.supportVector.RBFKernel:
	 
	       -D
	        Enables debugging output (if available) to be printed.
	        (default: off)
	 
	       -no-checks
	        Turns off all checks - use with caution!
	        (default: checks on)
	 
	       -C <num>
	        The size of the cache (a prime number).
	        (default: 250007)
	 
	       -G <num>
	        The Gamma parameter.
	        (default: 0.01)
     */
    
    public void setOptions(String[] options) throws Exception {
        //work around to avoid the api print trash in the console
        PrintStream console = System.out;
        System.setOut(new PrintStream(new OutputStream() {
                @Override public void write(int b) throws IOException {}
        }));

        super.setOptions(options);

        //work around to avoid the api print trash in the console
        System.setOut(console);
}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			for(double i=0.01;i<=100;i=i*10){
            RegressionProblem cp = new RegressionProblem("data/tobs-averages.arff");
            GP classifier = new GP();
            classifier.setOptions(new String[]{"-L",Double.toString(i),"-N","1","-K","weka.classifiers.functions.supportVector.RBFKernel"});
            Resample filter=new Resample();
            filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","1"});
            filter.setInputFormat(cp.getData());
            Instances newTrain = Filter.useFilter(cp.getData(), filter); 
            filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","3"});
            Instances newTest = Filter.useFilter(cp.getData(), filter);
            classifier.buildClassifier(newTrain);
//            for (int i=0;i<cp.getData().numInstances();i++) {
//            	Instance instance=cp.getData().instance(i);
//                    System.out.print(classifier.classifyInstance(instance) + ", ");
//            }
            Evaluation eval = new Evaluation(newTrain);
            eval.evaluateModel(classifier, newTest);
            //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(i+","+eval.correlationCoefficient());
			}
            

    } catch (Exception e) {
            e.printStackTrace();
    }
	}

}
