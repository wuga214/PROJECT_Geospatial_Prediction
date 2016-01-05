package extern.arff;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.io.IOException;

import evaluation.CVOutput;
 
public class ArfftoCSV {
  /**
   * takes 2 arguments:
   * - ARFF input file
   * - CSV output file
   */
	public static String csv="data/Circles.csv";
	public static String arff="data/Circles.arff";
	
	public static void BatchConvert(CVOutput cv) throws IOException{
		for(int i=0; i<cv.problems.size();i++){
			for(int j=0;j<cv.evals.get(i).size();j++){
				csv="outputs/"+cv.problems.get(i).toString()+"_"+cv.evals.get(i).get(j).name.toString()+".csv";
				arff="outputs/"+cv.problems.get(i).toString()+"_"+cv.evals.get(i).get(j).name.toString()+".arff";
				ArffLoader loader = new ArffLoader();
			    loader.setSource(new File(arff));
			    Instances data = loader.getDataSet();
			 
			    // save ARFF
			    CSVSaver saver = new CSVSaver();
			    saver.setInstances(data);
			    saver.setFile(new File(csv));
			    //saver.setDestination(new File(arff));
			    saver.writeBatch();
			}
		}
	}
  
	public static void main(String[] args) throws Exception {
 
	// load CSV
		ArffLoader loader = new ArffLoader();
	    loader.setSource(new File(arff));
	    Instances data = loader.getDataSet();
	 
	    // save ARFF
	    CSVSaver saver = new CSVSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(csv));
	    //saver.setDestination(new File(arff));
	    saver.writeBatch();
	}
}
