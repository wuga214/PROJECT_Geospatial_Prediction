package regressions;

import java.util.ArrayList;
import java.util.HashSet;

import mst.Kruskal;
import delaunay.BowyerWatson;
import structure.DEdge;
import structure.DPoint;
import utils.InstancesToPoints;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class MAPofBMA extends Classifier{
	double height=70;
	double width=24;

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// TODO Auto-generated method stub
		ArrayList<DPoint> points=InstancesToPoints.transfer(arg0);
		BowyerWatson bw=new BowyerWatson(26,-124,width,height,points);
		HashSet<DEdge> full_edges=bw.getPrunEdges();
		Kruskal k=new Kruskal(points,full_edges,newValid);
	}

}
