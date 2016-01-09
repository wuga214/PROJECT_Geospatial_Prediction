package sampling;

import java.util.List;

public class SampleManager {
	public List<List<Segmentation>> sampleRecords;
	public List<Double> weights;
	
	public SampleManager(){
		
	}
	
	public void addSample(List<Segmentation> sample){
		sampleRecords.add(sample);
	}
	
}
