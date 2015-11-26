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
		
        RegressionProblem cp;
		try {
			cp = new RegressionProblem("data/tobs-averages.arff");
	        Resample filter=new Resample();
	        filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","1"});
	        filter.setInputFormat(cp.getData());
	        Instances newTrain = Filter.useFilter(cp.getData(), filter); 
			points=InstancesToPoints.transfer(newTrain);
			BowyerWatson bw=new BowyerWatson(26,-124,width,height,points);
			DTriangle x=new DTriangle(new DPoint(0,0),new DPoint(100,0),new DPoint(10,10));
			//System.out.println(bw.toString());
		    JFrame window = new JFrame();
		    window.setBounds(0, 0, 510, 525);
		    //window.getContentPane().add(new Polygons(bw.getPolygons()));
		    HashSet<DEdge> full_edges=bw.getPrunEdges();
		    System.out.println(full_edges.size());
		    Kruskal k=new Kruskal(points,full_edges,full_edges.size());
		    System.out.println(points.size());
			window.getContentPane().add(new Lines(full_edges,k.getMST(),20,-124));
			 window.setVisible(true);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

}
