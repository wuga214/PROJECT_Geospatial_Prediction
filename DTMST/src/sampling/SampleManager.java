package sampling;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class SampleManager {
	public List<List<Segmentation>> sampleRecords;
	public Instances adjustedData;
	public NearestNeighbourSearch NNSearcher;
	
	public SampleManager(){
		sampleRecords=new ArrayList<List<Segmentation>>();
		
	}
	
	public void addSample(List<Segmentation> sample){
		sampleRecords.add(sample);
	}
	
	public void showSampleSize(){
		System.out.println("Current Samples Collected:"+sampleRecords.size());
	}
	
	public int findSegmentIndex(int cellID, List<Segmentation> segmentations){
		int segIndex=-1;
		for(int i=0;i<segmentations.size();i++){
			if(segmentations.get(i).contains(cellID)){
				segIndex=i;
			}
		}
		return segIndex;
	}
	
	public void createBaggingModel(Instances instances) throws Exception{
//		weightNormalization();
		Instances newInstances=new Instances(instances);
		for(int i=0;i<newInstances.numInstances();i++){
			double value=0;
			for(int j=0;j<sampleRecords.size();j++){
				int segIndex=findSegmentIndex(i,sampleRecords.get(j));
				value+=sampleRecords.get(j).get(segIndex).EX*(1.0/sampleRecords.size());				
			}
			newInstances.instance(i).setClassValue(value);
		}
		adjustedData=newInstances;
		NNSearcher= new LinearNNSearch();
		NNSearcher.setInstances(adjustedData);
	}
	
//	public void weightNormalization(){
//
//		double largest=-Double.MAX_VALUE;
//		double partition=0;
//		for(int i=0;i<weights.size();i++){
//			if(weights.get(i)>largest){
//				largest=weights.get(i);
//			}
//		}
//		double exps=0;
//		for(int i=0;i<weights.size();i++){
//			exps+=Math.exp(weights.get(i)-largest);
//		}
//		partition=largest+Math.log(exps);
//		//log likelihood now tune into likelihood, even still using name loglikelihood
//		for(int i=0;i<weights.size();i++){
//			weights.set(i,Math.exp(weights.get(i)-partition));
//		}
//	}
	
	public double predictLabel(Instance instance) throws Exception{
		Instance neighbor=NNSearcher.nearestNeighbour(instance);
		return neighbor.classValue();
	}
	
	public void batchPrediction(Instances test) throws Exception{
		Instances labeled = new Instances(test);
		for (int k = 0; k < test.numInstances(); k++) {
			//bug founded here! the training instance value is changed to segmentation index!!!!
			double clsLabel = predictLabel(test.instance(k));
			labeled.instance(k).setClassValue(clsLabel);
		}
		// Modify this function to predict with likelihood
		// save labeled data
		BufferedWriter writer = new BufferedWriter(
				new FileWriter("outputs/Gibbs/BaggingPrediction.arff"));
		writer.write(labeled.toString());
		writer.newLine();
		writer.flush();
		writer.close();
	}
}
