package sampling;

import weka.core.Instances;

public class Gibbs {
	public ModelManager Manager;
	public Instances oregData;
	public Instances validating;
	public int iteration;
	public SampleManager samples;
	public double currentSampleWeight;
	
	public Gibbs(Instances data, Instances valid, int iter, SampleManager samp){
		Manager=new ModelManager(data);
		oregData=data;
		validating=valid;
		iteration=iter;
		samples=samp;
		currentSampleWeight=0;
	}
	
	//Gibbs Sampling outer iterations: number of iterations and dimension selection without random selection(tuning around)
	public void Sampling() throws Exception{
		for(int i=0;i<iteration;i++){
			for(int j=0;j<oregData.numInstances();j++){
				int sdsIndex=singleDimensionSampling(j);
				Manager.flipCellAssignment(j, sdsIndex, oregData);
				Manager.removeEmptySegments();
				//only sample 10 models!
				if((iteration-i)<=10&&j==oregData.numInstances()){
					samples.addSample(Manager.deepCopySegmentations());
				}
			}
		}
	}
	
	//Single Dimension Conditional Distribution, Sample value of single dimension given values from other dimensions fixed.
	public int singleDimensionSampling(int dimIndex) throws Exception{
		int segSize=Manager.segmentations.size();
		double[] densities=new double[segSize];
		double sum=0;
		for(int i=0;i<segSize;i++){
			Manager.flipCellAssignment(dimIndex, i, oregData);
			Manager.buildClassifier();
			densities[i]=Manager.modelEval(validating);
			sum+=densities[i];
		}
		
		double rand=Math.random();
		double cumuDensity=0;
		int sampleIndex=-1;
		for(int i=0;i<segSize;i++){
			cumuDensity+=densities[i];
			if(rand<=(cumuDensity/sum)){
				sampleIndex=i;
				break;
			}
		}
		currentSampleWeight=densities[sampleIndex];
		return sampleIndex;
		/*
		 * 1 find sampled segmentation index;
		 * 2 update ModelManger by the selected segmentation index;
		 * 3 clean up empty segmentations
		 */
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
