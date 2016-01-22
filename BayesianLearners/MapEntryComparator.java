import java.util.*;
import java.util.Map.Entry;

public class MapEntryComparator implements Comparator<Map.Entry<Integer, List<Integer>>>
	{

	@Override
	public int compare(Entry<Integer, List<Integer>> o1, Entry<Integer, List<Integer>> o2)
		{
		return o1.getValue().size() - o2.getValue().size();
		}

	}
