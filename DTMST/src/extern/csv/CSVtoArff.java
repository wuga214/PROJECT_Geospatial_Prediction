package extern.csv;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.util.ArrayList;
 
public class CSVtoArff {
  /**
   * takes 2 arguments:
   * - CSV input file
   * - ARFF output file
   */
  
	public static void main(String[] args) throws Exception {
		File folder = new File("data");
		File[] listOfFiles = folder.listFiles();

		for (int i = 0; i < listOfFiles.length; i++) {
		  File file = listOfFiles[i];
		  if (file.isFile() && file.getName().endsWith("-averages")) {
				// load CSV
				CSVLoader loader = new CSVLoader();
			    loader.setSource(file);
			    Instances data = loader.getDataSet();
			 
			    // save ARFF
			    ArffSaver saver = new ArffSaver();
			    saver.setInstances(data);
			    saver.setFile(new File(file.getName().replace("-averages", ".arff")));
			    //saver.setDestination(new File(arff));
			    saver.writeBatch();
		  } 
		}
	}
}
