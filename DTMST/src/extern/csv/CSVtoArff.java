package extern.csv;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;
 
public class CSVtoArff {
  /**
   * takes 2 arguments:
   * - CSV input file
   * - ARFF output file
   */
	public static String csv="data/syntheticdata.csv";
	public static String arff="data/syntheticdata.arff";
  
	public static void main(String[] args) throws Exception {
 
	// load CSV
		CSVLoader loader = new CSVLoader();
	    loader.setSource(new File(csv));
	    Instances data = loader.getDataSet();
	 
	    // save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(arff));
	    saver.setDestination(new File(arff));
	    saver.writeBatch();
	}
}
