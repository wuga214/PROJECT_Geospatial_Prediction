package sampling;

import java.util.ArrayList;
import java.util.List;

public class SampleManager {
	public List<List<Segmentation>> sampleRecords;
	public List<Double> weights;
	
	public SampleManager(){
		sampleRecords=new ArrayList<List<Segmentation>>();
		weights=new ArrayList<Double>();
		
	}
	
	public void addSample(List<Segmentation> sample,double w){
		sampleRecords.add(sample);
		weights.add(w);
	}
	
}
