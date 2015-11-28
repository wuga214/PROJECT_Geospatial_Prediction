package evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class CVOutput {
	public List<List<FoldRecord>> evals;

	public CVOutput(){
		evals=new ArrayList<List<FoldRecord>>();
	}
	public void add(List<FoldRecord> clsrst){
		evals.add(clsrst);
	}
	
}
