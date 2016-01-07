package sampling;

import java.util.ArrayList;
import java.util.List;

import regressions.KNN;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ModelManager {
	public List<Segmentation> segmentations;
	public Instances modeledData;
	public KNN model;
	
	public ModelManager(Instances data){
		modeledData=new Instances(data);
		segmentations=new ArrayList<Segmentation>();
		for(int i=0;i<modeledData.numInstances();i++){
			segmentations.add(new Segmentation(i,modeledData));
		}
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
