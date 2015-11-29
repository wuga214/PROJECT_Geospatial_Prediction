package regressions;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

import utils.RegressionProblem;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class FRT extends REPTree {
	
	public FRT(){
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

		 -M <minimum number of instances>
		  Set minimum number of instances per leaf (default 2).
		 
		 -V <minimum variance for split>
		  Set minimum numeric class variance proportion
		  of train variance for split (default 1e-3).
		 
		 -N <number of folds>
		  Number of folds for reduced error pruning (default 3).
		 
		 -S <seed>
		  Seed for random data shuffling (default 1).
		 
		 -P
		  No pruning.
		 
		 -L
		  Maximum tree depth (default -1, no maximum)
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
			for(int i=1;i<=10;i++){
            RegressionProblem cp = new RegressionProblem("data/Temperature.arff");
            FRT classifier = new FRT();
            classifier.setOptions(new String[]{"-M",Integer.toString(i),"-V","0.001","-P"});
            Resample filter=new Resample();
            filter.setOptions(new String[]{"-Z","10","-no-replacement"});
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
