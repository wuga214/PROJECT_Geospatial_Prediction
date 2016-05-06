package sampling.crp;

import java.io.Serializable;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class SampleManager implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7429069846251542457L;
	public List<List<Segmentation>> sampleRecords;
	public Instances adjustedData;
	public NearestNeighbourSearch NNSearcher;
	public NumericToNominal filter;
	
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
	
	public double predictLabel(Instance instance) throws Exception{
		double value=0;
		Instance neighbor=NNSearcher.nearestNeighbour(instance);
		value= neighbor.classValue();
		return value;
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
	
	public void sampleReport(){
		showSampleSize();
		for(int i=0;i<sampleRecords.size();i++){
			System.out.println("SAMPLE ID:"+i);
			System.out.println("===================");
			for(int j=0;j<sampleRecords.get(i).size();j++){
				System.out.println(sampleRecords.get(i).get(j).cells.toString());
			}
			System.out.println("======+end=========\n\n");
			
		}
	}
}
