package extern.csv;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import structure.DPoint;
import weka.core.Debug.Random;

public class FileOperator {

	public static ArrayList<DPoint> getData(String path) throws IOException{
		Random r = new Random();
		ArrayList<DPoint> points=new ArrayList<DPoint>();
		Reader in = new FileReader(path);
		Iterable<CSVRecord> records = CSVFormat.EXCEL.withHeader().parse(in);
		 for (CSVRecord csvRecord : records) {
		     DPoint p=new DPoint(Double.parseDouble(csvRecord.get(0))+r.nextDouble()*1e-5,Double.parseDouble(csvRecord.get(1))+r.nextDouble()*1e-5,Double.parseDouble(csvRecord.get(2)));
		     points.add(p);
		 }
		 return points;
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try {
			ArrayList<DPoint> points=getData("data/tobs-averages");
			System.out.println(points.size());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
