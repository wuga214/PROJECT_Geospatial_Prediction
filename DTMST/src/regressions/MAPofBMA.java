package regressions;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import mst.Kruskal;
import delaunay.BowyerWatson;
import structure.DEdge;
import structure.DPoint;
import utils.InstancesToPoints;
import utils.PointsToInstances;
import utils.RegressionProblem;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class MAPofBMA extends Classifier{
	/**
	 * 
	 */
	private static final long serialVersionUID = 573536156928733812L;
	double height=70;
	double width=24;
	double startx=0;
	double starty=0;
	int iteration=0;
	KNN model;
	
	public MAPofBMA(double sx,double sy,double w,double h){
		startx=sx;
		starty=sy;
		height=h;
		width=w;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// TODO Auto-generated method stub
		ArrayList<DPoint> points=InstancesToPoints.transfer(arg0);
		BowyerWatson bw=new BowyerWatson(startx,starty,width,height,points);
		HashSet<DEdge> full_edges=bw.getPrunEdges();
		//System.out.println(full_edges.size());
		Kruskal k=new Kruskal(points,full_edges,iteration);
		ArrayList<HashSet<DPoint>> components=k.getMerges();
		//System.out.println(components.size());
		for(int i=0; i<components.size();i++){
			double averagevalue=0;
			for(DPoint p:components.get(i)){
				averagevalue+=p.value;
			}
			averagevalue=averagevalue/components.get(i).size();
			for(DPoint p:components.get(i)){
				p.value=averagevalue;
			}
		}
		KNN classifier = new KNN();
		classifier.setOptions(new String[]{"-K","1"});
		classifier.buildClassifier(PointsToInstances.transfer(components));
		model=classifier;
	}
	
    public double classifyInstance(Instance newInstance) throws Exception {
        return model.classifyInstance(newInstance);

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
        //work around to avoid the api print trash in the console
        System.setOut(console);
    }
    
    public String[] getOptions() {

        String[] options = new String[3];
        options[0] = "-I";//number of merging required!!
        options[1] = Integer.toString(iteration);
        options[2] = "-F";//not useful at all, just prevent parameter setting function failure
        return options;
    }

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			for(int i=1;i<=1000;i++){
			RegressionProblem cp = new RegressionProblem("data/Temperature.arff");
			//MAPofBMA classifier=new MAPofBMA(26,-124,24,70);
			MAPofBMA classifier=new MAPofBMA(-200,-200,300,300);
			classifier.setOptions(new String[]{"-I",Integer.toString(i)});
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
