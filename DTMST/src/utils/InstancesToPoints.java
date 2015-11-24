package utils;

import java.util.ArrayList;
import java.util.HashSet;

import structure.DPoint;
import weka.core.Instance;
import weka.core.Instances;

public class InstancesToPoints {

	public static ArrayList<DPoint> transfer(Instances data){
		ArrayList<DPoint> points=new ArrayList<DPoint>();
		for(int i=0;i<data.numInstances();i++){
			Instance inst=data.instance(i);
			DPoint point=new DPoint(inst.value(0),inst.value(1),inst.value(2));
			points.add(point);
		}
		return points;
	}
}
