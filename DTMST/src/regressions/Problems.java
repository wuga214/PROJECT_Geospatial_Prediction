package regressions;

import java.io.IOException;

import utils.RegressionProblem;
import weka.classifiers.Classifier;

public class Problems {
	private static Problems instance;
	public synchronized static Problems getInstance(){
		if(instance == null)
			instance = new Problems();

		return instance;
	}
	public Problems(){}

	public RegressionProblem createRegressionProblem(EProblemList EType) throws IOException {
		RegressionProblem ret = null;

		switch (EType) {
		case US_TAMPERATURE :
			ret = new RegressionProblem("data/Temperature.arff");
			break;
		case SAN_FRANCISICO_HOUSE_PRICE :
			ret = new RegressionProblem("data/HousePrice.arff");
			break;
		case SYNTHETIC_DATA :
			ret = new RegressionProblem("data/Synthetic.arff");
			break;
		}

		return ret;
	}
}
