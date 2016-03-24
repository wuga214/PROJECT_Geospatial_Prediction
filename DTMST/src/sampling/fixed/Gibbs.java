package sampling.fixed;

import java.io.BufferedWriter;
import java.io.FileWriter;
import utils.RegressionProblem;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

/**
 * @author Wuga
 * Need to embed Gibbs voronoi merging classifier into WEKA Classifier to run cross validation for later convenience
 * Trained model by maximize likelihood, choose model by RMSE and output is also RMSE
 */
public class Gibbs {
	public ModelManager Manager;
	public Instances oregData;
	public Instances validating;
	public int iteration;
	public SampleManager samples;
	
	
	public Gibbs(Instances data, Instances valid, int iter, int lablenum, SampleManager samp) throws Exception{
		Manager=new ModelManager(data,lablenum);
		oregData=data;
		validating=valid;
		iteration=iter;
		samples=samp;
		Manager.findNearestNeighbour(validating);
	}
	
	/**
	 * Here the parameter need is for plot, which is not related to Gibbs Samping!!
	 * @param whole data
	 * @throws Exception
	 */
	public void Sampling(Instances wholedata, boolean debug) throws Exception{
		for(int i=0;i<iteration;i++){
			for(int j=0;j<oregData.numInstances();j++){
				//taking sample value of the dimension through singleDimensionSampling() function
				int sdsIndex=singleDimensionSampling(j);
				Manager.flipCellAssignment(j, sdsIndex, oregData);
				Manager.removeEmptySegments();
			}
			if(i%5.0==0.0){
				if((iteration-i)<=200){
					samples.addSample(Manager.deepCopySegmentations());
				}
				
				if(debug==true){
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
	}
	
	//Single Dimension Conditional Distribution, Sample value of single dimension given values from other dimensions fixed.
	public int singleDimensionSampling(int dimIndex) throws Exception{
		int segSize=Manager.segmentations.size();
		double[] logLikelihood=new double[segSize];
		double largest=-Double.MAX_VALUE;
		double partition=0;
		double n=Manager.segmentTracker.length;
		for(int i=0;i<segSize;i++){
			Manager.flipCellAssignment(dimIndex, i, oregData);
			double ni=Manager.segmentations.get(i).cells.size();
			logLikelihood[i]=Manager.getLogLikelihood(validating,dimIndex,i);
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
		
		return sampleIndex;
		/*
		 * 1 find sampled segmentation index;
		 * 2 update ModelManger by the selected segmentation index;
		 * 3 clean up empty segmentations
		 */
	}
	
	public void setIteration(int iter){
		iteration=iter;
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
	        Gibbs gb=new Gibbs(newTrain, newTrain, 1000, newTrain.numInstances(), samp);
	        gb.Sampling(cp.getData(),true);
	        gb.Manager.writeFile("Gibbs");
	        samp.showSampleSize();
	        samp.createBaggingModel(newTrain);
	        samp.batchPrediction(cp.getData());
	        System.out.println("All results are under output/Gibbs folder");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
