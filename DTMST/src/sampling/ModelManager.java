package sampling;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import regressions.KNN;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddID;

public class ModelManager {
	public List<Segmentation> segmentations;
	public Instances modeledData;
	public KNN model;
	public int[] segmentTracker;
	public int[] trainDataTracker;
	public NearestNeighbourSearch NNSearcher;
	//consider to initial manger with single segmentation, so gibbs can partition it into proper segmentations
	//Change this tonight
	public ModelManager(Instances data) throws Exception{
		modeledData=new Instances(data);
		segmentations=new ArrayList<Segmentation>();
		Segmentation segment=new Segmentation();
		for(int i=0;i<modeledData.numInstances();i++){
			segment.addCell(i, modeledData);
		}
		segmentations.add(segment);
		segmentations.add(new Segmentation());
		segmentTracker=new int[data.numInstances()]; 
		for(int i=0;i<modeledData.numInstances();i++){
			modeledData.instance(i).setClassValue(i);
		}
		buildNNSearcher();
	}
	
	public void buildNNSearcher() throws Exception{
		NearestNeighbourSearch m_NNSearch = new LinearNNSearch();
		m_NNSearch.setInstances(modeledData);
		NNSearcher=m_NNSearch;
	}
	
	public void flipCellAssignment(int cellID, int segID, Instances data){
		for(int i=0;i<segmentations.size();i++){
			if(segmentations.get(i).contains(cellID)){
				segmentations.get(i).removeCell(cellID, data);
				updateModel(segmentations.get(i),i);
			}
			if(i==segID){
				segmentations.get(i).addCell(cellID, data);
				updateModel(segmentations.get(i),i);
			}
		}
	}
	
	public void updateModel(Segmentation seg,int segIndex){
		for(int i:seg.cells){
			segmentTracker[i]=segIndex;
		}
	}
	
	public void removeEmptySegments(){
		for(int i=segmentations.size()-1;i>=0;i--){
			if(segmentations.get(i).cells.isEmpty()){
				segmentations.remove(i);
				for(int j=0;j<segmentTracker.length;j++){
					if(segmentTracker[j]>i){
						segmentTracker[j]=segmentTracker[j]-1;
					}
				}
			}
		}
		//if(segmentations.size()<modeledData.numInstances()){
		if(segmentations.size()<4){
			segmentations.add(new Segmentation());
		}
	}
	
	public int findSegmentIndex(int cellID){
		int segIndex=-1;
		for(int i=0;i<segmentations.size();i++){
			if(segmentations.get(i).contains(cellID)){
				segIndex=i;
			}
		}
		return segIndex;
	}
	
	public void findNearestNeighbour(Instances validating) throws Exception{
		trainDataTracker=new int[validating.numInstances()];
		for(int i=0;i<validating.numInstances();i++){
			Instance neighbor=NNSearcher.nearestNeighbour(validating.instance(i));
			trainDataTracker[i]=(int)neighbor.classValue();
		}
	}
	
	public double classifyInstance(Instance inst) throws Exception{
		Instance neighbor=NNSearcher.nearestNeighbour(inst);
		return segmentations.get(segmentTracker[(int)neighbor.classValue()]).EX;
	}
	
	
	public double getLogLikelihood(Instances validating) throws Exception{
		double logLikelihood=0;
		for(int i=0;i<validating.numInstances();i++){
			double mean=segmentations.get(segmentTracker[trainDataTracker[i]]).EX;
			double var=segmentations.get(segmentTracker[trainDataTracker[i]]).VAR;
			logLikelihood+=-(Math.pow(validating.instance(i).classValue()-mean,2)/var)-0.5*Math.log(2*Math.PI*var);			
		}
		return logLikelihood;
	}
	
	public List<Segmentation> deepCopySegmentations(){
		List<Segmentation> newList=new ArrayList<Segmentation>();
		for(Segmentation p : segmentations) {
		    newList.add(p.clone());
		}
		return newList;
	}
	
	public void writeFile(String name) throws Exception{
		BufferedWriter writer = new BufferedWriter(
				new FileWriter("outputs/"+name+".arff"));
		writer.write(modeledData.toString());
		writer.newLine();
		writer.flush();
		writer.close();
	}
	/*
	 * Using KNN is not good idea here, this is just an temperal method for making Gibbs sampling work.
	 * I expect to use maximum likelihood estimation of mu and sigma instead of only mu here, so that we can use maximum
	 * likelihood to evaluate model weight but not correlation coefficient, which is undesirable.
	 * 
	 * But computing variance when segmentation assignment flipping is expensive, looking for incremental method for this.
	 */

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
