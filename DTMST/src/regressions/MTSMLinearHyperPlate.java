package regressions;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import delaunay.BowyerWatson;
import mst.Kruskal;
import structure.DEdge;
import structure.DPoint;
import utils.InstancesToPoints;
import utils.PointsToInstances;
import utils.RegressionProblem;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class MTSMLinearHyperPlate extends AbstractClassifier {

	private static final long serialVersionUID = 573536156928733812L;
	double height=100;
	double width=100;
	double startx=0;
	double starty=0;
	int iteration=0;
	double[][] parameters;
	ArrayList<HashSet<DPoint>> trainData;
	List<double[]> modelWeights;
	Instances orgininalTrainingData;
	
	
	
	public MTSMLinearHyperPlate(double sx,double sy,double w,double h){
		startx=sx;
		starty=sy;
		height=h;
		width=w;
	}
	
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// TODO Auto-generated method stub
		orgininalTrainingData=arg0;
		ArrayList<DPoint> points=InstancesToPoints.transfer(arg0);
		BowyerWatson bw=new BowyerWatson(startx,starty,width,height,points);
		HashSet<DEdge> full_edges=bw.getPrunEdges();
		//System.out.println(full_edges.size());
		Kruskal k=new Kruskal(points,full_edges,iteration);
		ArrayList<HashSet<DPoint>> components=k.getMerges();
		//System.out.println(components.size());
		List<double[]> weights=new ArrayList<double[]>(); 
		for(int i=0; i<components.size();i++){
			if(components.get(i).size()>=3){
				List<Double> vy = new ArrayList<Double>();
				List<Feature[]> vx = new ArrayList<Feature[]>();
				for(DPoint p:components.get(i)){
					vy.add(p.value);
					Feature[] x=new Feature[3];
					x[0]=new FeatureNode(1, p.x);
					x[1]=new FeatureNode(2, p.y);
					vx.add(x);
				}
				Problem problem=constructProblem(vy,vx,2,1);
				SolverType solver = SolverType.L2R_L2LOSS_SVR_DUAL; 
				double C = 1.0;    // cost of constraints violation
				double eps = 0.1; // stopping criteria
				Parameter parameter = new Parameter(solver, C, eps);
				Linear.disableDebugOutput();
				Model model = Linear.train(problem, parameter);
				weights.add(model.getFeatureWeights());
			}else{
				double averagevalue=0;
				for(DPoint p:components.get(i)){
					averagevalue+=p.value;
				}
				averagevalue=averagevalue/components.get(i).size();
				weights.add(new double[]{0,0,averagevalue});
			}
			//System.out.println(model.getFeatureWeights().length);
			for(DPoint p:components.get(i)){
				p.value=i;
			}
		}
		trainData=components;
		modelWeights=weights;
	}
	
	@Override
    public double classifyInstance(Instance newInstance) throws Exception {		
		NearestNeighbourSearch m_NNSearch = new LinearNNSearch();
		m_NNSearch.setInstances(orgininalTrainingData);
		Instance neighbor=m_NNSearch.nearestNeighbour(newInstance);
		DPoint point=new DPoint(neighbor.value(0),neighbor.value(1),neighbor.value(2));
		boolean found=false;
		int modelIndex=0;
		for(int i=0; i<trainData.size();i++){
			for(DPoint p:trainData.get(i)){
				if(p.x==point.x&&p.y==point.y){
					found=true;
					modelIndex=i;
				}
			}
			if(found){break;}
		}
		double[] weights=modelWeights.get(modelIndex);
		return weights[0]*newInstance.value(0)+weights[1]*newInstance.value(1)+weights[2];		
    }
	
    private static Problem constructProblem(List<Double> vy, List<Feature[]> vx, int max_index, double bias) {
        Problem prob = new Problem();
        prob.bias = bias;
        prob.l = vy.size();
        prob.n = max_index;
        if (bias >= 0) {
            prob.n++;
        }
        prob.x = new Feature[prob.l][];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = vx.get(i);

            if (bias >= 0) {
                assert prob.x[i][prob.x[i].length - 1] == null;
                prob.x[i][prob.x[i].length - 1] = new FeatureNode(max_index + 1, bias);
            }
        }

        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.get(i).doubleValue();

        return prob;
    }
    
    public void setOptions(String[] options) throws Exception {
        //work around to avoid the api print trash in the console
        PrintStream console = System.out;
        System.setOut(new PrintStream(new OutputStream() {
                @Override public void write(int b) throws IOException {}
        }));

        super.setOptions(options);
        if(Arrays.asList(options).contains("-I")){
        	String iter=Arrays.asList(options).get(Arrays.asList(options).indexOf("-I")+1);
        	iteration=Integer.parseInt(iter);
        }
        //work around to avoid the api print trash in the console
        System.setOut(console);
    }
    
    public String[] getOptions() {

        String[] options = new String[3];
        options[0] = "-I";//number of merging required!!
        options[1] = Integer.toString(iteration);
        options[2] = "-F";//not useful at all, just prevent parameter setting function failure
        return options;
    }

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		RegressionProblem cp;
		try {
			cp = new RegressionProblem("data/Stair.arff");
			MTSMLinearHyperPlate classifier=new MTSMLinearHyperPlate(-200,-200,300,300);
			classifier.setOptions(new String[]{"-I","70"});
			Resample filter=new Resample();
			filter.setOptions(new String[]{"-Z","20","-no-replacement","-S","1"});
			filter.setInputFormat(cp.getData());
			Instances newTrain = Filter.useFilter(cp.getData(), filter);
			classifier.buildClassifier(newTrain);
			filter.setOptions(new String[]{"-Z","10","-no-replacement","-S","2"});
            Instances newTest = Filter.useFilter(cp.getData(), filter); 
            Evaluation eval = new Evaluation(newTrain);
            eval.evaluateModel(classifier, newTest);
            //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            //System.out.println(i+","+eval.correlationCoefficient());
            System.out.println(100+","+eval.correlationCoefficient());
            System.out.println(100+","+eval.rootMeanSquaredError());

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//MAPofBMA classifier=new MAPofBMA(26,-124,24,70);
 catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
