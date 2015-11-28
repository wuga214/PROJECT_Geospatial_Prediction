package evaluation;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

public class SettingExtender {

	public static List<String> generateModels(Map<String, Set<String>> params){
		List<Set<String>> paramsList =Lists.newArrayList(params.values());
		List<String> modelsStringSetting = null;
		final Joiner joiner = Joiner.on(" ").skipNulls();
		Function<List<String>, String> buildModelSettingString = new Function<List<String>, String>() {
			public String apply(List<String> params) {
				return joiner.join(params);
			}
		};

		if(paramsList != null && paramsList.size() > 1){
			Set<List<String>> cartesianProd = Sets.cartesianProduct(paramsList);

			if(cartesianProd != null){
				modelsStringSetting = Lists.newArrayList( Iterables.transform(cartesianProd, buildModelSettingString));
			}
		}

		return modelsStringSetting;
	}
	public static void main(String[] args) {
		HashMap<String,Set<String>> param = new HashMap<String,Set<String>>();
		param.put("-I",Sets.newHashSet("-I 30","-I 60","-I 100","-I 130","-I 160","-I 200","-I 230","-I 260","-I 300","-I 330","-I 360","-I 400","-I 430","-I 460","-I 500"));
		param.put("-F",Sets.newHashSet("-F"));
		List<String> settings=generateModels(param);
		for(String setting:settings){
			System.out.println(setting);
		}
	}
}
