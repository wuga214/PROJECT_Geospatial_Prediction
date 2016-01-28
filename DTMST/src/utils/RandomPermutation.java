package utils;

import java.io.IOException;
import java.util.Random;

import weka.core.Instances;

public class RandomPermutation {
	public Instances permutated;

	public RandomPermutation(){
		
	}
	
	public void getRandomPermutation(Instances data){
		Random rand=new Random();
		permutated=new Instances(data, data.numInstances());
		for(int i=data.numInstances();i>0;i--){
			permutated.add(data.instance(rand.nextInt(i)));
		}
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		RegressionProblem cp = new RegressionProblem("data/box.arff");
		RandomPermutation randPerm=new RandomPermutation();
		randPerm.getRandomPermutation(cp.getData());
		System.out.println(randPerm.permutated.numInstances());
	}

}
