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
	public int iterations;
	public Kruskal(ArrayList<DPoint> p, HashSet<DEdge> e, int k){
		getEdgesWeights(e);
		sort(e);
		points=p;
		iterations=k;
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
		ArrayList<HashSet<DPoint>> components=new ArrayList<HashSet<DPoint>>();
		HashSet<DEdge> mst=new HashSet<DEdge>();
		for(DPoint p:points){
			HashSet<DPoint> set=new HashSet<DPoint>();
			set.add(p);
			components.add(set);
		}
		for(int i=0;i<edges.size();i++){
		//for(int i=0;i<iterations;i++){
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
			}
		}
		return mst;
	}
	
	public ArrayList<HashSet<DPoint>> getMerges() throws Exception{
		ArrayList<HashSet<DPoint>> components=new ArrayList<HashSet<DPoint>>();
		HashSet<DEdge> mst=new HashSet<DEdge>();
		for(DPoint p:points){
			HashSet<DPoint> set=new HashSet<DPoint>();
			set.add(p);
			components.add(set);
		}
		for(int i=0;i<edges.size();i++){
		//for(int i=0;i<iterations;i++){
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
				iterations--;
				if(iterations==0){
					break;
				}
			}
		}
		return components;
	}
	
	// need to change value of each point after merging!! forgot this previously!!!!
	

	public Evaluation getEval(){
		return evaluation;
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
