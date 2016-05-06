package regressions;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

import com.google.common.collect.Sets;

import utils.RegressionProblem;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class SVR extends LibSVM{
	
	static{
	    libsvm.svm.svm_set_print_string_function(new libsvm.svm_print_interface() {
	        @Override
	        public void print(String s) {
	        } // Disables svm output
	    });
	}

	//constructor
	public SVR(){
		this.setSVMType(new SelectedTag(3, LibSVM.TAGS_SVMTYPE));
	}

	@Override
	public void buildClassifier(Instances data)
			throws Exception {

		//set the liblinear to use L2-regularized L2-loss support vector regression (dual)
		this.setSVMType(new SelectedTag(3, LibSVM.TAGS_SVMTYPE));

		//build the classify (= train)
		super.buildClassifier(data);

	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		//set the liblinear to use L2-regularized L2-loss support vector regression (dual)
		this.setSVMType(new SelectedTag(3, LibSVM.TAGS_SVMTYPE));
		//classify the new instance
		return super.classifyInstance(arg0);
	}



	@Override
	public void setOptions(String[] options) throws Exception {

		//work around to avoid the api print trash in the console
		PrintStream console = System.out;
		System.setOut(new PrintStream(new OutputStream() {
			@Override public void write(int b) throws IOException {}
		}));

		super.setOptions(options);

		//set the liblinear to use L2-regularized L2-loss support vector regression (dual)
		this.setSVMType(new SelectedTag(3, LibSVM.TAGS_SVMTYPE));

		//work around to avoid the api print trash in the console
		System.setOut(console);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			for(int i=1;i<=1;i++){
            RegressionProblem cp = new RegressionProblem("data/HousePrice.arff");
            SVR classifier = new SVR();
            classifier.setOptions(new String[]{"-C",Integer.toString(i),"-L","0.1"});
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
