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
//		case SyntheticBox :
//			ret = new RegressionProblem("data/box.arff");
//			break;
		case SyntheticCircles :
			ret = new RegressionProblem("data/Circles.arff");
			break;
		case SyntheticStair :
			ret = new RegressionProblem("data/Stair.arff");
			break;
		case SyntheticCake :
			ret = new RegressionProblem("data/Cake.arff");
			break;
//		case Queen :
//			ret = new RegressionProblem("data/queens_subset.arff");
//			break;
//		case Statten :
//			ret = new RegressionProblem("data/statten-island_subset.arff");
//			break;
//		case Temperature :
//			ret = new RegressionProblem("data/tobs_subset.arff");
//			break;
//		case USGS :
//			ret = new RegressionProblem("data/usgs_subset.arff");
//			break;
//		case Bronx :
//			ret = new RegressionProblem("data/bronx_subset.arff");
//			break;
//		case Brooklyn :
//			ret = new RegressionProblem("data/brooklyn_subset.arff");
//			break;
//		case Dublin :
//			ret = new RegressionProblem("data/dublin-2010_subset.arff");
//			break;
//		case Manhattan :
//			ret = new RegressionProblem("data/manhattan_subset.arff");
//			break;
		}

		return ret;
	}
}
