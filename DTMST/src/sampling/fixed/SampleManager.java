package sampling.fixed;

import java.io.Serializable;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
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
	public boolean isSVM;
	public LibSVM[] SVM;
	public NumericToNominal filter;
	
	public SampleManager(){
		sampleRecords=new ArrayList<List<Segmentation>>();
		isSVM=false;
		
	}
	
	public void setSVM(boolean x){
		isSVM=x;
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
		SVM=new LibSVM[sampleRecords.size()];
		if(isSVM){
			for(int i=0;i<sampleRecords.size();i++){
				Instances newInstances=new Instances(instances);
				for(int j=0;j<newInstances.numInstances();j++){
					int segIndex=findSegmentIndex(j,sampleRecords.get(i));
					newInstances.instance(j).setClassValue(segIndex);
				}
				newInstances.setClassIndex(-1);
				System.out.println("Nominal? "+newInstances.attribute(2).isNominal());
			    filter = new NumericToNominal();
			    filter.setOptions(new String[]{"-R","3"});
			    filter.setInputFormat(newInstances);
			    System.out.println(filter.getAttributeIndices());
			    newInstances=Filter.useFilter(newInstances, filter);
			    System.out.println("Nominal? "+newInstances.attribute(2).isNominal());
			    newInstances.setClassIndex(2);
				SVM[i]=new LibSVM();
				SVM[i].setOptions(new String[]{"-K","2"});
				SVM[i].buildClassifier(newInstances);
			}
		}
		else{
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
	}
	
	public double predictLabel(Instance instance) throws Exception{
		double value=0;
		if(isSVM){
			for(int i=0;i<SVM.length;i++){
				double index=SVM[i].classifyInstance(instance);
				value+=sampleRecords.get(i).get((int)index).EX;
			}
			value=value*(1.0/sampleRecords.size());
		}
		else{
			Instance neighbor=NNSearcher.nearestNeighbour(instance);
			value= neighbor.classValue();
		}
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
