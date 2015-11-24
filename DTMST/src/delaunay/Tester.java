package delaunay;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import javax.swing.JFrame;

import extern.csv.FileOperator;
import mst.Kruskal;
import regressions.GP;
import regressions.KNN;
import structure.DEdge;
import structure.DPoint;
import structure.DTriangle;
import utils.InstancesToPoints;
import utils.Lines;
import utils.Polygons;
import utils.RegressionProblem;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class Tester {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		double height=70;
		double width=24;
		ArrayList<DPoint> points=new ArrayList<DPoint>();
//		points.add(new DPoint(20,20));
//		points.add(new DPoint(20,40));
//		points.add(new DPoint(60,53));
//		points.add(new DPoint(80,70));

//		Random r = new Random();
//		for(int i=0; i<50; i++) 
//		{
//			points.add(new DPoint( width*r.nextDouble(), height*r.nextDouble(),10000*r.nextDouble())); 
//		}
		
//		try {
//			points=FileOperator.getData("data/tobs-averages");
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
        RegressionProblem cp;
		try {
			cp = new RegressionProblem("data/tobs-averages.arff");
	        Resample filter=new Resample();
	        filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","1"});
	        filter.setInputFormat(cp.getData());
	        Instances newTrain = Filter.useFilter(cp.getData(), filter); 
	        filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","2"});
	        Instances newValid = Filter.useFilter(cp.getData(), filter);
	        filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","3"});
	        Instances newTest = Filter.useFilter(cp.getData(), filter);
			points=InstancesToPoints.transfer(newTrain);
			BowyerWatson bw=new BowyerWatson(26,-124,width,height,points);
			DTriangle x=new DTriangle(new DPoint(0,0),new DPoint(100,0),new DPoint(10,10));
			//System.out.println(bw.toString());
		    JFrame window = new JFrame();
		    window.setBounds(0, 0, 510, 525);
		    //window.getContentPane().add(new Polygons(bw.getPolygons()));
		    HashSet<DEdge> full_edges=bw.getPrunEdges();
		    System.out.println(full_edges.size());
		    Kruskal k=new Kruskal(points,full_edges,newValid);
		    System.out.println(points.size());
			window.getContentPane().add(new Lines(full_edges,k.getMST(),20,-124));
			System.out.println(k.getEval().toSummaryString("\nBest Validating Results\n======\n", false));
			Instances besttrain=k.getBestData();
			KNN classifier = new KNN();
	        classifier.setOptions(new String[]{"-K","3"});
	        classifier.buildClassifier(besttrain);
	        Evaluation eval = new Evaluation(besttrain);
	        eval.evaluateModel(classifier, newTest);
			System.out.println(eval.toSummaryString("\nTest Results\n======\n", false));
			 window.setVisible(true);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

}
