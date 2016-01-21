package sampling;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import regressions.KNN;
import utils.RegressionProblem;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class Gibbs {
	public ModelManager Manager;
	public Instances oregData;
	public Instances validating;
	public int iteration;
	public SampleManager samples;
	public double currentSampleWeight;
	
	public Gibbs(Instances data, Instances valid, int iter, SampleManager samp) throws Exception{
		Manager=new ModelManager(data);
		oregData=data;
		validating=valid;
		iteration=iter;
		samples=samp;
		currentSampleWeight=0;
		Manager.findNearestNeighbour(validating);
	}
	
	//Gibbs Sampling outer iterations: number of iterations and dimension selection without random selection(tuning around)
	public void Sampling(Instances wholedata) throws Exception{
		System.out.println("Number of training data:"+oregData.numInstances());
		//Gibbs Sampling outer layer: iteration of full dimensions route
		for(int i=0;i<iteration;i++){
			System.out.println("iteration:"+(i+1));
			System.out.println("Number of Segmentations:"+Manager.segmentations.size());
			//Gibbs Samping inner layer: sampling for each dimension
			for(int j=0;j<oregData.numInstances();j++){
				//taking sample value of the dimension through singleDimensionSampling() function
				int sdsIndex=singleDimensionSampling(j);
				Manager.flipCellAssignment(j, sdsIndex, oregData);
				Manager.removeEmptySegments();
				//only sample 10 models!
			}
			if(i%10.0==0.0){
				if((iteration-i)<=200){
					samples.addSample(Manager.deepCopySegmentations(),currentSampleWeight);
				}
				
				Instances labeled = new Instances(wholedata);
				for (int k = 0; k < wholedata.numInstances(); k++) {
					//bug founded here! the training instance value is changed to segmentation index!!!!
					double clsLabel = Manager.classifyInstance(wholedata.instance(k));
					labeled.instance(k).setClassValue(clsLabel);
				}
				// save labeled data
				BufferedWriter writer = new BufferedWriter(
						new FileWriter("outputs/Gibbs/iteration_"+i+".arff"));
				writer.write(labeled.toString());
				writer.newLine();
				writer.flush();
				writer.close();
			}
		}
	}
	
	//Single Dimension Conditional Distribution, Sample value of single dimension given values from other dimensions fixed.
	public int singleDimensionSampling(int dimIndex) throws Exception{
		int segSize=Manager.segmentations.size();
		double[] logLikelihood=new double[segSize];
		double largest=-Double.MAX_VALUE;
		double partition=0;
		for(int i=0;i<segSize;i++){
			Manager.flipCellAssignment(dimIndex, i, oregData);
			logLikelihood[i]=Manager.getLogLikelihood(validating);
			if(logLikelihood[i]>largest){
				largest=logLikelihood[i];
			}
		}
		double exps=0;
		for(int i=0;i<logLikelihood.length;i++){
			exps+=Math.exp(logLikelihood[i]-largest);
		}
		partition=largest+Math.log(exps);
		//log likelihood now tune into likelihood, even still using name loglikelihood
		double[] likeLihood=new double[logLikelihood.length];
		double sum=0;
		for(int i=0;i<likeLihood.length;i++){
			likeLihood[i]=Math.exp(logLikelihood[i]-partition);
			sum+=likeLihood[i];
		}
		
		//Uniformly distributed random value
		//cumulative density larger than this value, then the last candidate added is selected as sample 
		double rand=Math.random();
		double cumuDensity=0;
		int sampleIndex=-1;
		for(int i=0;i<segSize;i++){
			cumuDensity+=likeLihood[i];
			if(rand<=(cumuDensity/sum)){
				sampleIndex=i;
				break;
			}
		}
		currentSampleWeight=logLikelihood[sampleIndex];
		System.out.println("Sample Logged Likelihood:"+currentSampleWeight);
		return sampleIndex;
		/*
		 * 1 find sampled segmentation index;
		 * 2 update ModelManger by the selected segmentation index;
		 * 3 clean up empty segmentations
		 */
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			RegressionProblem cp;
	        SampleManager samp=new SampleManager();
			cp = new RegressionProblem("data/box.arff");
	        Resample filter=new Resample();
	        filter.setOptions(new String[]{"-Z","20","-no-replacement","-S","1"});
	        filter.setInputFormat(cp.getData());
	        Instances newTrain = Filter.useFilter(cp.getData(), filter); 
	        filter.setOptions(new String[]{"-Z","20","-no-replacement","-S","2"});
	        Instances newTest = Filter.useFilter(cp.getData(), filter); 
	        Gibbs gb=new Gibbs(newTrain, newTrain, 10000, samp);
	        gb.Sampling(cp.getData());
	        gb.Manager.writeFile("Gibbs");
	        samp.showSampleSize();
	        samp.normalizeWeights();
	        samp.createBaggingModel(newTrain);
	        samp.batchPrediction(cp.getData());
	        System.out.println("All results are under output/Gibbs folder");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
