package utils;

import java.util.ArrayList;
import java.util.HashSet;

import structure.DPoint;
import weka.core.Attribute;
//import weka.core.FastVector;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Instances;

public class PointsToInstances {
	
	public static Instances transfer(ArrayList<HashSet<DPoint>> points){
		Attribute Attribute1 = new Attribute("Latitude");
		Attribute Attribute2 = new Attribute("Longitude");
		Attribute Attribute3 = new Attribute("Value");
//		FastVector fvWekaAttributes = new FastVector(4);
//		 fvWekaAttributes.addElement(Attribute1);
//		 fvWekaAttributes.addElement(Attribute2);
//		 fvWekaAttributes.addElement(Attribute3);
//		Instances instances=new Instances("dataset", fvWekaAttributes, 0);
		ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
		fvWekaAttributes.add(Attribute1);
		fvWekaAttributes.add(Attribute2);
		fvWekaAttributes.add(Attribute3);
		Instances instances=new Instances("dataset", fvWekaAttributes, 0);
		instances.setClassIndex(instances.numAttributes() - 1);
		for(HashSet<DPoint> set:points){
			double value=getAverageValue(set);
			for(DPoint p:set){
				Instance instance=new DenseInstance(3);
				instance.setValue(Attribute1, p.x);
				instance.setValue(Attribute2, p.y);
				instance.setValue(Attribute3, value);
				instances.add(instance);
			}
		}		
		return instances;
		
	}
	
	public static Instances transfer(HashSet<DPoint> points){
		Attribute Attribute1 = new Attribute("Latitude");
		Attribute Attribute2 = new Attribute("Longitude");
		Attribute Attribute3 = new Attribute("Value");
//		FastVector fvWekaAttributes = new FastVector(4);
//		 fvWekaAttributes.addElement(Attribute1);
//		 fvWekaAttributes.addElement(Attribute2);
//		 fvWekaAttributes.addElement(Attribute3);
		ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
		fvWekaAttributes.add(Attribute1);
		fvWekaAttributes.add(Attribute2);
		fvWekaAttributes.add(Attribute3);
		Instances instances=new Instances("dataset", fvWekaAttributes, 0);
		instances.setClassIndex(instances.numAttributes() - 1);
		for(DPoint p:points){
			Instance instance=new DenseInstance(3);
			instance.setValue(Attribute1, p.x);
			instance.setValue(Attribute2, p.y);
			instance.setValue(Attribute3, p.value);
			instances.add(instance);
		}		
		return instances;
		
	}
	
	public static double getAverageValue(HashSet<DPoint> set){
		double counts=0;
		for(DPoint p:set){
			counts+=p.value;
		}
		return counts/set.size();
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
