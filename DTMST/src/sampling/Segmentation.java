package sampling;

import java.util.HashSet;

import weka.core.Instances;

public class Segmentation {
	public HashSet<Integer> cells;
	public double value;
	
	public Segmentation(HashSet<Integer> c,Instances data){
		cells=c;
		computeValue(data);
	}
	
	public Segmentation(int i, Instances data){
		cells=new HashSet<Integer>();
		cells.add(i);
		value=data.instance(i).classValue();
	}
	
	public Segmentation(){
		cells=new HashSet<Integer>();
		value=0;
	}
	
	private void computeValue(Instances data){
		for(int index:cells){
			value+=data.instance(index).classValue();
		}
		value=value/cells.size();
	}
	
	public void addCell(int i, Instances data){
		value=value+(data.instance(i).classValue()-value)/(cells.size()+1);
		cells.add(i);
	}
	
	public void removeCell(int i, Instances data){
		if(cells.size()==1){
			cells.remove(i);
		}else{
			value=(value*cells.size()-data.instance(i).classValue())/(cells.size()-1);
			cells.remove(i);
		}
	}
	
	@SuppressWarnings("unchecked")
	public Segmentation clone(){
		Segmentation newseg=new Segmentation();
		newseg.cells=(HashSet<Integer>) this.cells.clone();
		newseg.value=this.value;
		return newseg;

	}
	
	public boolean isEmpty(){
		return cells.isEmpty();
	}
	
	public boolean contains(int i){
		return cells.contains(i);
	}
}
