package extern.arff;

import java.io.File;
import java.io.IOException;

import utils.RandomPermutation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;


public class SampleData {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			File folder = new File("data");
			File[] listOfFiles = folder.listFiles();
	
			for (int i = 0; i < listOfFiles.length; i++) {
			  File file = listOfFiles[i];
			  if (file.isFile() && file.getName().endsWith(".arff")) {
					ArffLoader loader = new ArffLoader();
					loader.setSource(file);
				    Instances data = loader.getDataSet();
				    if(data.numInstances()>500){
					    RandomPermutation randPerm=new RandomPermutation();
						randPerm.getRandomPermutation(data);
						data=randPerm.permutated;
						Instances subset=new Instances(data, 500);
						for(int j=0;j<500;j++){
							subset.add(data.instance(j));
						}
						
					    ArffSaver saver = new ArffSaver();
					    saver.setInstances(subset);
					    saver.setFile(new File(file.getName().replace(".arff", "_subset.arff")));
					    //saver.setDestination(new File(arff));
					    saver.writeBatch();
				    }
			  } 
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
