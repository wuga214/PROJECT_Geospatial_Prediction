package sampling;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import regressions.KNN;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ModelManager {
	public List<Segmentation> segmentations;
	public Instances modeledData;
	public KNN model;
	
	//consider to initial manger with single segmentation, so gibbs can partition it into proper segmentations
	//Change this tonight
	public ModelManager(Instances data){
		modeledData=new Instances(data);
		segmentations=new ArrayList<Segmentation>();
		Segmentation segment=new Segmentation();
		for(int i=0;i<modeledData.numInstances();i++){
			segment.addCell(i, modeledData);
		}
		segmentations.add(segment);
		segmentations.add(new Segmentation());
//		for(int i=0;i<modeledData.numInstances();i++){
//			segmentations.add(new Segmentation(i,modeledData));
//		}
	}
	
	public void flipCellAssignment(int cellID, int segID, Instances data){
		for(int i=0;i<segmentations.size();i++){
			if(segmentations.get(i).contains(cellID)){
				segmentations.get(i).removeCell(cellID, data);
				updateModel(segmentations.get(i));
			}
			if(i==segID){
				segmentations.get(i).addCell(cellID, data);
				updateModel(segmentations.get(i));
			}
		}
	}
	
	public void updateModel(Segmentation seg){
		for(int i:seg.cells){
			modeledData.instance(i).setClassValue(seg.value);
		}
	}
	
	public void removeEmptySegments(){
		for(int i=segmentations.size()-1;i>=0;i--){
			if(segmentations.get(i).cells.isEmpty()){
				segmentations.remove(i);
			}
		}
		if(segmentations.size()<modeledData.numInstances()){
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
	
	public void buildClassifier() throws Exception{
		KNN classifier = new KNN();
		classifier.setOptions(new String[]{"-K","1"});
		classifier.buildClassifier(modeledData);
		model= classifier;
	}
	
	public double modelEval(Instances validating) throws Exception{
		 Evaluation eval = new Evaluation(modeledData);
         eval.evaluateModel(model, validating);
         return eval.correlationCoefficient();
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
