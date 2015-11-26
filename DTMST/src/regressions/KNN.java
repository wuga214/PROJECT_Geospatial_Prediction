package regressions;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

import utils.RegressionProblem;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class KNN  extends IBk{
	
	public KNN(){
		super();
	}

	//method to train a knn regression
    @Override
    public void buildClassifier(Instances data)throws Exception {
            super.buildClassifier(data);

    }

    //method to classify a new intance using the knn regression
    @Override
    public double classifyInstance(Instance newInstance) throws Exception {
            return super.classifyInstance(newInstance);

    }
    /*
     * Valid options are:

		 -I
		  Weight neighbours by the inverse of their distance
		  (use when k > 1)
		 -F
		  Weight neighbours by 1 - their distance
		  (use when k > 1)
		 -K <number of neighbors>
		  Number of nearest neighbours (k) used in classification.
		  (Default = 1)
		 -E
		  Minimise mean squared error rather than mean absolute
		  error when using -X option with numeric prediction.
		 -W <window size>
		  Maximum number of training instances maintained.
		  Training instances are dropped FIFO. (Default = no window)
		 -X
		  Select the number of nearest neighbours between 1
		  and the k value specified using hold-one-out evaluation
		  on the training data (use when k > 1)
		 -A
		  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
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
            RegressionProblem cp = new RegressionProblem("data/tobs-averages.arff");
            KNN classifier = new KNN();
            classifier.setOptions(new String[]{"-F","-K",Integer.toString(i)});
            Resample filter=new Resample();
            filter.setOptions(new String[]{"-Z","20","-no-replacement","-S","1"});
            filter.setInputFormat(cp.getData());
            Instances newTrain = Filter.useFilter(cp.getData(), filter); 
            filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","2"});
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
