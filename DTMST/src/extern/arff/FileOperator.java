package extern.arff;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class FileOperator {
	
	public static String path="data/tobs-averages.arff";

	public static Instances getData(String path) throws IOException{
		
		BufferedReader reader = new BufferedReader(new FileReader(path));
		Instances data = new Instances(reader);
		reader.close();
		 // setting class attribute
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) {
		try {
			Instances data=getData(path);
			System.out.println(data);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
