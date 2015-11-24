package mst;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

import regressions.KNN;
import structure.DEdge;
import structure.DPoint;
import utils.PointsToInstances;
import utils.RegressionProblem;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class Kruskal {
	
	public ArrayList<DEdge> edges;
	public ArrayList<DPoint> points;
	public Evaluation evaluation;
	public Instances test;
	public Instances bestdata;
	public Kruskal(ArrayList<DPoint> p, HashSet<DEdge> e,Instances t){
		getEdgesWeights(e);
		sort(e);
		points=p;
		test=t;
	}
	public void getEdgesWeights(HashSet<DEdge> e){
		for(DEdge i:e){
			i.getWeight();
		}
	}
	public void sort(HashSet<DEdge> e){
		edges=new ArrayList<DEdge>(e);
		Collections.sort(edges);
	}
	
	public HashSet<DEdge> getMST() throws Exception{
        double MIN_RSE=1e30;
		ArrayList<HashSet<DPoint>> components=new ArrayList<HashSet<DPoint>>();
		HashSet<DEdge> mst=new HashSet<DEdge>();
		HashSet<DEdge> bestmerge=mst;
		for(DPoint p:points){
			HashSet<DPoint> set=new HashSet<DPoint>();
			set.add(p);
			components.add(set);
		}
		Instances data=null;
		Evaluation irse=eval(test,components,data);
		MIN_RSE=irse.rootMeanSquaredError();
		evaluation=irse;
		bestdata=data;
		bestmerge=(HashSet<DEdge>) mst.clone();
		for(int i=0;i<edges.size();i++){
			DPoint p0=edges.get(i).p[0];
			DPoint p1=edges.get(i).p[1];
			int indexp0=0;
			int indexp1=0;
			for(int j=0;j<components.size();j++){
				if(components.get(j).contains(p0)){indexp0=j;};
				if(components.get(j).contains(p1)){indexp1=j;};
			}
			//Here is a problem!
			if(indexp0!=indexp1){
				mst.add(edges.get(i));
				HashSet<DPoint> merge=components.get(indexp0);
				merge.addAll(components.get(indexp1));
				if(indexp1>indexp0){
					components.remove(indexp1);
					components.remove(indexp0);
				}else{
					components.remove(indexp0);
					components.remove(indexp1);
				}
				components.add(merge);
				Evaluation rse=eval(test,components,data);
				if(MIN_RSE>rse.rootMeanSquaredError()){
					MIN_RSE=rse.rootMeanSquaredError();
					evaluation=rse;
					bestdata=PointsToInstances.transfer(components);
					bestmerge=(HashSet<DEdge>) mst.clone();
				};
			}
		}
		return bestmerge;
	}

	public Evaluation getEval(){
		return evaluation;
	}
	
	public Instances getBestData(){
		return bestdata;
	}
	
	public Evaluation eval(Instances test, ArrayList<HashSet<DPoint>> components,Instances d) throws Exception{
		Instances data=PointsToInstances.transfer(components);
		KNN classifier = new KNN();
        classifier.setOptions(new String[]{"-K","3"});
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(classifier, test);
        System.out.println(eval.correlationCoefficient());
        d=data;
        return eval;
	}
}
