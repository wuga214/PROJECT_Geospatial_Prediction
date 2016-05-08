package sampling.proper;

import java.util.HashSet;
import java.io.Serializable;

import weka.core.Instances;

public class Segmentation implements Serializable{/**
	 * 
	 */
	private static final long serialVersionUID = -8765348527663816963L;

	public HashSet<Integer> cells;
	public double EX;
	public double EXS;
	public double VAR;
	
	public Segmentation(HashSet<Integer> c,Instances data){
		cells=c;
		computeEX(data);
		computeEXS(data);
		updateVar();
	}
	
	public Segmentation(int i, Instances data){
		cells=new HashSet<Integer>();
		cells.add(i);
		EX=data.instance(i).classValue();
		EXS=Math.pow(data.instance(i).classValue(),2);
		useDefaultVar();

	}
	
	public Segmentation(){
		cells=new HashSet<Integer>();
		EX=0;
		EXS=0;
		VAR=0;
	}
	
	private void computeEX(Instances data){
		for(int index:cells){
			EX+=data.instance(index).classValue();
		}
		EX=EX/cells.size();
	}
	
	private void computeEXS(Instances data){
		for(int index:cells){
			EXS+=Math.pow(data.instance(index).classValue(),2);
		}
		EXS=EXS/cells.size();
	}
	
	public void addCell(int i, Instances data){
		EX=EX+(data.instance(i).classValue()-EX)/(cells.size()+1);
		EXS=EXS+(Math.pow(data.instance(i).classValue(),2)-EXS)/(cells.size()+1);
		cells.add(i);
		updateVar();
		if(VAR==0 || VAR<0.001){
			useDefaultVar();
		}
	}
	
	public void removeCell(int i, Instances data){
		if(cells.size()==1){
			cells.remove(i);
		}else{
			EX=(EX*cells.size()-data.instance(i).classValue())/(cells.size()-1);
			EXS=(EXS*cells.size()-Math.pow(data.instance(i).classValue(),2))/(cells.size()-1);
			cells.remove(i);
		}
		updateVar();
		if(VAR==0|| VAR<0.001||cells.size()==1){
			useDefaultVar();
		}
	}
	
	public void useDefaultVar(){
		VAR=10;
	}
	
	public void updateVar(){
		VAR=EXS-Math.pow(EX, 2);
	}
	
	@SuppressWarnings("unchecked")
	public Segmentation clone(){
		Segmentation newseg=new Segmentation();
		newseg.cells=(HashSet<Integer>) this.cells.clone();
		newseg.EX=this.EX;
		newseg.EXS=this.EXS;
		newseg.VAR=this.VAR;
		return newseg;

	}
	
	public boolean isEmpty(){
		return cells.isEmpty();
	}
	
	public boolean contains(int i){
		return cells.contains(i);
	}
}
