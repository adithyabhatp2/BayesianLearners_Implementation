import weka.core.*;
import java.util.*;

public class StatisticsHelper
	{
	static int debugLevel = 4;

	static HashMap<Integer, Integer> originalPosFoldMap; // works - filled
	static HashMap<Integer, Instance> originalPosInstMap; 
	static HashMap<Instance, Integer> originalInstPosMap; // works - filled
	
	// TODO : Check stratified definition

	public static LinkedList<LinkedList<Instance>> getStratifiedFolds_Lists(Instances insts, int k)
	{
	originalPosFoldMap = new HashMap<Integer, Integer>();
	LinkedList<LinkedList<Instance>> folds = new LinkedList<LinkedList<Instance>>();
	// Instances instsSet = new Instances(insts); - keeping originals so that we can lookup in Map
	Instances instsSet = insts;
	setOriginalPosMap(instsSet);
	instsSet.randomize(new Random());
	AttributeStats classAttrStats = insts.attributeStats(insts.classIndex());
	int numNeg = classAttrStats.nominalCounts[0];
	int numPos = classAttrStats.nominalCounts[1];

	int residualN = numNeg % k;
	int residualP = numPos % k;

	int i;
	for (i = 0;i < k;i++)
		{
		int nN = numNeg / k; // per fold
		int nP = numPos / k; // per fold

		if(folds.size() <= i)
			{
			folds.add(new LinkedList<Instance>());
			}

		LinkedList<Instance> fold = folds.get(i);

		if(debugLevel <= 2)
			{
			System.out.println("numNeg: " + numNeg + " numPos: " + numPos + " nN: " + nN + " nP: " + nP + " resN: " + residualN + " resP: " + residualP + " k: " + k);
			}

		while (nN > 0)
			{
			Instance inst = removeFirstInstanceOfClass(instsSet, 0); 
			fold.add(inst);
			originalPosFoldMap.put(originalInstPosMap.get(inst),i);
			nN--;
			if(debugLevel<=3)
				{
				System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
				Instance inst2 = fold.getLast();
				System.out.println("Same - Adding inst at index: "+originalInstPosMap.get(inst2)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst2)));
				}
			}
		while (nP > 0)
			{
			Instance inst =removeFirstInstanceOfClass(instsSet, 1); 
			fold.add(inst);
			originalPosFoldMap.put(originalInstPosMap.get(inst),i);
			nP--;
			if(debugLevel<=3)
				{
				System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
				Instance inst2 = fold.getLast();
				System.out.println("Same - Adding inst at index: "+originalInstPosMap.get(inst2)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst2)));
				}
			}

		if(residualN > 0)
			{
			Instance inst = removeFirstInstanceOfClass(instsSet, 0); 
			fold.add(inst);
			originalPosFoldMap.put(originalInstPosMap.get(inst),i);
			residualN--;
			if(debugLevel<=3)
				{
				System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
				}
			}
		if(residualP > 0)
			{
			Instance inst =removeFirstInstanceOfClass(instsSet, 1); 
			fold.add(inst);
			originalPosFoldMap.put(originalInstPosMap.get(inst),i);
			residualP--;
			if(debugLevel<=3)
				{
				System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
				}
			}
		}

	return folds;
	}

	
	
	
	public static List<Instances> getStratifiedFolds(Instances insts, int k)
		{
		originalPosFoldMap = new HashMap<Integer, Integer>();
		LinkedList<Instances> folds = new LinkedList<Instances>();
		// Instances instsSet = new Instances(insts); - keeping originals so that we can lookup in Map
		Instances instsSet = insts;
		setOriginalPosMap(instsSet);
		//instsSet.randomize(new Random());
		AttributeStats classAttrStats = insts.attributeStats(insts.classIndex());
		int numNeg = classAttrStats.nominalCounts[0];
		int numPos = classAttrStats.nominalCounts[1];

		int residualN = numNeg % k;
		int residualP = numPos % k;

		int i;
		for (i = 0;i < k;i++)
			{
			int nN = numNeg / k; // per fold
			int nP = numPos / k; // per fold

			if(folds.size() <= i)
				{
				folds.add(new Instances(insts, ((numPos+numNeg) / k) + 1));
				}

			Instances fold = folds.get(i);

			if(debugLevel <= 2)
				{
				System.out.println("numNeg: " + numNeg + " numPos: " + numPos + " nN: " + nN + " nP: " + nP + " resN: " + residualN + " resP: " + residualP + " k: " + k);
				}

			while (nN > 0)
				{
				Instance inst = removeFirstInstanceOfClass(instsSet, 0); 
				fold.add(inst);
				originalPosFoldMap.put(originalInstPosMap.get(inst),i);
				nN--;
				if(debugLevel<=3)
					{
					System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
					Instance inst2 = fold.lastInstance();
					System.out.println("Same - Adding inst at index: "+originalInstPosMap.get(inst2)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst2)));
					}
				}
			while (nP > 0)
				{
				Instance inst =removeFirstInstanceOfClass(instsSet, 1); 
				fold.add(inst);
				originalPosFoldMap.put(originalInstPosMap.get(inst),i);
				nP--;
				if(debugLevel<=3)
					{
					System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
					Instance inst2 = fold.lastInstance();
					System.out.println("Same - Adding inst at index: "+originalInstPosMap.get(inst2)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst2)));
					}
				}

			if(residualN > 0)
				{
				Instance inst = removeFirstInstanceOfClass(instsSet, 0); 
				fold.add(inst);
				originalPosFoldMap.put(originalInstPosMap.get(inst),i);
				residualN--;
				if(debugLevel<=3)
					{
					System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
					}
				}
			if(residualP > 0)
				{
				Instance inst =removeFirstInstanceOfClass(instsSet, 1); 
				fold.add(inst);
				originalPosFoldMap.put(originalInstPosMap.get(inst),i);
				residualP--;
				if(debugLevel<=3)
					{
					System.out.println("Adding inst at index: "+originalInstPosMap.get(inst)+" to fold: "+originalPosFoldMap.get(originalInstPosMap.get(inst)));
					}
				}
			}

		return folds;
		}

	private static void setOriginalPosMap(Instances instsSet)
		{
		originalPosInstMap = new HashMap<Integer, Instance>();
		originalInstPosMap = new HashMap<Instance, Integer>();
		
		for(int i=0;i<instsSet.numInstances();i++)
			{
			originalPosInstMap.put(i, instsSet.instance(i));
			originalInstPosMap.put(instsSet.instance(i), i);
			}
		}

	public static Instance removeFirstInstanceOfClass(Instances instsSet, int classInd)
		{
		Instance inst = null;
		for (int i = 0;i < instsSet.numInstances();i++)
			{
			if(instsSet.instance(i).classValue() == classInd)
				{
				inst = instsSet.instance(i);
				instsSet.delete(i);
				return inst;
				}
			}
		System.out.println("\nNOPE. \nNOPE. \nNOPE. - StatsHelper.removeInstOfCls");
		return null;
		}

	}
