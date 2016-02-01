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
//		case USTemperature :
//			ret = new RegressionProblem("data/Temperature.arff");
//			break;
//		case SanFranciscoHousePrice :
//			ret = new RegressionProblem("data/HousePrice.arff");
//			break;
		case SyntheticBox :
			ret = new RegressionProblem("data/box.arff");
			break;
		case SyntheticCircles :
			ret = new RegressionProblem("data/Circles.arff");
			break;
		case SyntheticStair :
			ret = new RegressionProblem("data/Stair.arff");
			break;
		case SyntheticCake :
			ret = new RegressionProblem("data/Cake.arff");
			break;
		}

		return ret;
	}
}
