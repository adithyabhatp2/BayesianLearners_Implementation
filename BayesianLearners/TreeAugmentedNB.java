import java.util.*;
import weka.core.*;

public class TreeAugmentedNB
	{

	NaiveBayesLearner nbl;
	Instances insts;

	// CC : Class conditional | M : Mutual | C : Count | P : Probability | I :
	// Information
	HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>> ccfvc;
	HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> ccfvp;

	// ccmc[class][attrInd1][attr1Val][attrInd2][attr2Val]
	HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>>> classConditionedMutualCounts;
	HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>>> classConditionedMutualProb;
	HashMap<Integer, HashMap<Integer, Double>> classConditionedMutualInfo;

	HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>>> ccmc;
	HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>>> ccmp;
	HashMap<Integer, HashMap<Integer, Double>> ccmi;

	double[][] mutualInfoMatrix;
	LinkedHashMap<Integer, ArrayList<Integer>> parentToKidsMap;
	LinkedHashMap<Integer, ArrayList<Integer>> parentsMap;

	HashMap<Integer, Object> condProbTables;

	public TreeAugmentedNB(Instances insts)
		{
		this.insts = insts;
		this.nbl = new NaiveBayesLearner();
		nbl.computeClassProbabilities(insts);
		nbl.computeClassCondInputProbabilities(insts);
		this.ccfvc = nbl.classConditionedFeatureValCounts;
		this.ccfvp = nbl.classConditionedFeatureValProbabilities;
		}

	public void printNetworkStructure()
		{
		for (int attrInd = 0;attrInd < this.insts.numAttributes();attrInd++)
			{
			if(attrInd == this.insts.classIndex())
				continue;
			Attribute attr = this.insts.attribute(attrInd);
			System.out.print(attr.name());
			ArrayList<Integer> parentInds = parentsMap.get(attrInd);
			for (int parentNum = 0;parentNum < parentInds.size();parentNum++)
				{
				int parentId = parentInds.get(parentNum);
				if(parentId == this.insts.classIndex())
					continue;
				System.out.print(" " + this.insts.attribute(parentId).name());
				}
			System.out.println(" " + this.insts.classAttribute().name());
			}
		System.out.println();
		}

	public int predictClass(Instance inst)
		{

		// P(Y=yi | X) = [P(Y=yi)*P(X|yi)]/ Pi over all y [P(Y=y)*P(X|y)]
		// P(X|yi) = Pi over all attributes xi P(xi|Parents(xi)) (lookup)

		HashMap<Integer, Double> yval_probabilities = new HashMap<Integer, Double>();

		double denom = 1.0;
		for (int classId = 0;classId < inst.classAttribute().numValues();classId++)
			{
			double probability = 1.0;
			double numerator = this.nbl.classProbabilities.get(classId);

			for (int attrInd = 0;attrInd < inst.numAttributes();attrInd++)
				{
				if(attrInd == inst.classIndex())
					continue;
				int attrVal = (int) inst.value(attrInd);
				ArrayList<Integer> parentInds = parentsMap.get(attrInd);

				if(parentInds.size() == 1)
					{
					int parentId1 = parentInds.get(0);
					int parent1Val = classId;

					HashMap<Integer, HashMap<Integer, Double>> cpt = (HashMap<Integer, HashMap<Integer, Double>>) condProbTables.get(attrInd);
					numerator *= cpt.get(parent1Val).get(attrVal);
					denom *= numerator;
					}
				else if(parentInds.size() == 2)
					{
					int parentId1 = (parentInds.get(0) == this.insts.classIndex())? parentInds.get(0) : parentInds.get(1);
					int parentId2 = (parentInds.get(0) == this.insts.classIndex())? parentInds.get(1) : parentInds.get(0);

					int parent1Val = classId;
					int parent2Val = (int) inst.value(parentId2);

					HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> cpt = (HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>) condProbTables.get(attrInd);

					numerator *= cpt.get(parent1Val).get(parent2Val).get(attrVal);
					denom *= numerator;
					}
				}
			probability = numerator;
			yval_probabilities.put(classId, probability);
			}

		double max = -1;
		int argmax = -1;
		double sum = 0.0;
		for (Map.Entry<Integer, Double> entry : yval_probabilities.entrySet())
			{
			sum += entry.getValue();
			if(entry.getValue() > max)
				{
				max = entry.getValue();
				argmax = entry.getKey();
				}
			}

		String predictedClass = inst.classAttribute().value(argmax);
		String actualClass = inst.stringValue(inst.classAttribute());

		double prob = max / sum;
		// prob = (double) Math.round(prob * 1000000) / 1000000.0;
		System.out.println(predictedClass + " " + actualClass + " " + prob);
		return argmax;

		}

	public void generateConditionalProbabilityTables()
		{
		// for each attr, use cond prob of parents in TAN
		HashSet<Integer> traversedSet = new HashSet<Integer>();
		for (Map.Entry<Integer, ArrayList<Integer>> parentsOfAttr : parentsMap.entrySet())
			{
			if(traversedSet.contains(parentsOfAttr.getKey()))
				continue;
			for (Integer parent : parentsOfAttr.getValue())
				{
				if(!traversedSet.contains(parent))
					System.out.println("Tree not ordered correctly??");
				}

			// compute CPT
			int attrInd = parentsOfAttr.getKey();
			ArrayList<Integer> parentInds = parentsOfAttr.getValue();

			if(parentInds.size() == 0) // Class Attr only
				{
				condProbTables.put(attrInd, new HashMap<Integer, Double>());
				HashMap<Integer, Double> cpt = (HashMap<Integer, Double>) condProbTables.get(attrInd);
				for (int attrVal = 0;attrVal < insts.attribute(attrInd).numValues();attrVal++)
					{
					cpt.put(attrVal, this.nbl.classProbabilities.get(attrVal));
					}
				traversedSet.add(attrInd);
				}

			else if(parentInds.size() == 1) // Root Attr (0) only
				{
				int parentId1 = parentInds.get(0);
				condProbTables.put(attrInd, new HashMap<Integer, HashMap<Integer, Double>>());
				HashMap<Integer, HashMap<Integer, Double>> cpt = (HashMap<Integer, HashMap<Integer, Double>>) condProbTables.get(attrInd);

				for (int parent1Val = 0;parent1Val < insts.attribute(parentId1).numValues();parent1Val++)
					{
					cpt.put(parent1Val, new HashMap<Integer, Double>());
					for (int attrVal = 0;attrVal < insts.attribute(attrInd).numValues();attrVal++)
						{
						double condProb = this.ccfvp.get(parent1Val).get(attrInd).get(attrVal);
						cpt.get(parent1Val).put(attrVal, condProb);
						}
					}
				traversedSet.add(attrInd);
				}
			else if((parentInds.size() == 2))
				{
				int parentId1 = (parentInds.get(0) == this.insts.classIndex())? parentInds.get(0) : parentInds.get(1);
				int parentId2 = (parentInds.get(0) == this.insts.classIndex())? parentInds.get(1) : parentInds.get(0);

				condProbTables.put(attrInd, new HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>());
				HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> cpt = (HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>) condProbTables.get(attrInd);

				for (int parent1Val = 0;parent1Val < insts.attribute(parentId1).numValues();parent1Val++)
					{
					cpt.put(parent1Val, new HashMap<Integer, HashMap<Integer, Double>>());
					for (int parent2Val = 0;parent2Val < insts.attribute(parentId2).numValues();parent2Val++)
						{
						cpt.get(parent1Val).put(parent2Val, new HashMap<Integer, Double>());
						for (int attrVal = 0;attrVal < insts.attribute(attrInd).numValues();attrVal++)
							{

							int numerator = this.ccmc.get(parent1Val).get(parentId2).get(parent2Val).get(attrInd).get(attrVal) - 1 + 1; // Laplace
							int denom = this.ccfvc.get(parent1Val).get(parentId2).get(parent2Val) - 1 + insts.attribute(attrInd).numValues(); // Laplace

							double condProb = (double) numerator / (double) denom;
							cpt.get(parent1Val).get(parent2Val).put(attrVal, condProb);
							}
						}
					}
				traversedSet.add(attrInd);
				}
			else
				{
				System.out.println("TAN cannot have more than 2 parents to a node");
				}
			}
		// System.out.println();
		}

	public void generateMST_Prims()
		{
		HashSet<Integer> sourceSet = new HashSet<Integer>();

		sourceSet.add(0);
		parentsMap.put(0, new ArrayList<Integer>());

		while (sourceSet.size() < insts.numAttributes() - 1)
			{
			int maxEdgeSrc = -1;
			int maxEdgeDest = -1;
			double maxEdgeWt = -1.0;
			for (Integer sourceInd : sourceSet)
				{
				for (int attrInd2 = 0;attrInd2 < mutualInfoMatrix[0].length;attrInd2++)
					{
					if(sourceSet.contains(attrInd2))
						continue;
					if(mutualInfoMatrix[sourceInd][attrInd2] > maxEdgeWt)
						{
						maxEdgeSrc = sourceInd;
						maxEdgeDest = attrInd2;
						maxEdgeWt = mutualInfoMatrix[sourceInd][attrInd2];
						}
					else if(mutualInfoMatrix[sourceInd][attrInd2] == maxEdgeWt)
						{
						if(sourceInd < maxEdgeSrc)
							{
							maxEdgeSrc = sourceInd;
							maxEdgeDest = attrInd2;
							maxEdgeWt = mutualInfoMatrix[sourceInd][attrInd2];
							}
						else if(sourceInd == maxEdgeSrc)
							{
							if(attrInd2 < maxEdgeDest)
								{
								maxEdgeSrc = sourceInd;
								maxEdgeDest = attrInd2;
								maxEdgeWt = mutualInfoMatrix[sourceInd][attrInd2];
								}
							}
						}
					}
				}
			if(maxEdgeWt <= -1.0)
				{
				System.out.println("WUT?? ");
				}

			if(!parentToKidsMap.containsKey(maxEdgeSrc))
				parentToKidsMap.put(maxEdgeSrc, new ArrayList<Integer>());
			if(parentsMap.containsKey(maxEdgeDest))
				System.out.println("More than one parent??");
			parentsMap.put(maxEdgeDest, new ArrayList<Integer>());

			parentToKidsMap.get(maxEdgeSrc).add(maxEdgeDest);
			parentsMap.get(maxEdgeDest).add(maxEdgeSrc);
			sourceSet.add(maxEdgeSrc);
			sourceSet.add(maxEdgeDest);
			}

		for (int attrInd = 0;attrInd < insts.numAttributes();attrInd++)
			{
			if(attrInd == insts.classIndex())
				continue;
			parentToKidsMap.get(insts.classIndex()).add(attrInd);
			parentsMap.get(attrInd).add(insts.classIndex());
			}

		// System.out.println();
		}

	public void intializeStuff()
		{

		condProbTables = new HashMap<Integer, Object>();

		parentToKidsMap = new LinkedHashMap<Integer, ArrayList<Integer>>();
		parentsMap = new LinkedHashMap<Integer, ArrayList<Integer>>();
		mutualInfoMatrix = new double[insts.numAttributes() - 1][insts.numAttributes() - 1];
		parentToKidsMap.put(insts.classIndex(), new ArrayList<Integer>());
		parentsMap.put(insts.classIndex(), new ArrayList<Integer>());

		classConditionedMutualCounts = new HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>>>();
		classConditionedMutualProb = new HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>>>();
		classConditionedMutualInfo = new HashMap<Integer, HashMap<Integer, Double>>();

		ccmc = classConditionedMutualCounts;
		ccmp = classConditionedMutualProb;
		ccmi = classConditionedMutualInfo;

		for (int classId = 0;classId < insts.numClasses();classId++)
			{
			ccmc.put(classId, new HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>>());
			ccmp.put(classId, new HashMap<Integer, HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>>());

			// AttrInd1
			for (int attrInd1 = 0;attrInd1 < insts.numAttributes();attrInd1++)
				{
				if(attrInd1 == insts.classIndex())
					continue;

				ccmc.get(classId).put(attrInd1, new HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>());
				ccmp.get(classId).put(attrInd1, new HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>());
				ccmi.put(attrInd1, new HashMap<Integer, Double>());

				// AttrVal1
				for (int attrVal1 = 0;attrVal1 < insts.attribute(attrInd1).numValues();attrVal1++)
					{
					ccmc.get(classId).get(attrInd1).put(attrVal1, new HashMap<Integer, HashMap<Integer, Integer>>());
					ccmp.get(classId).get(attrInd1).put(attrVal1, new HashMap<Integer, HashMap<Integer, Double>>());

					// AttrInd2
					for (int attrInd2 = 0;attrInd2 < insts.numAttributes();attrInd2++)
						{
						if(attrInd2 == insts.classIndex())
							continue;

						ccmc.get(classId).get(attrInd1).get(attrVal1).put(attrInd2, new HashMap<Integer, Integer>());
						ccmp.get(classId).get(attrInd1).get(attrVal1).put(attrInd2, new HashMap<Integer, Double>());
						ccmi.get(attrInd1).put(attrInd2, new Double(0.0));

						// AttrVal2
						for (int attrVal2 = 0;attrVal2 < insts.attribute(attrInd2).numValues();attrVal2++)
							{
							ccmc.get(classId).get(attrInd1).get(attrVal1).get(attrInd2).put(attrVal2, 1); // Laplace!
							}
						}
					}
				}
			}
		}

	public void computeClassCondMutualProbabilitiesAndInformation()
		{

		// MI (attr1, attr2) = Exi Exj Ey P(xi, xj, y) * log
		// [(P(xi,xj|y)/P(xi|y)*P(xj|y)]
		// use ccfvc and ccfvp to compute. we already have P(xi|y)

		int numClasses = insts.classAttribute().numValues();
		AttributeStats classStats = insts.attributeStats(insts.classIndex());

		for (int attrInd1 = 0;attrInd1 < insts.numAttributes();attrInd1++)
			{
			if(attrInd1 == insts.classIndex())
				continue;

			for (int attrInd2 = 0;attrInd2 < insts.numAttributes();attrInd2++)
				{
				if(attrInd2 == insts.classIndex())
					continue;

				double mutualInfo = 0.0;

				for (int attrVal1 = 0;attrVal1 < insts.attribute(attrInd1).numValues();attrVal1++)
					{
					for (int attrVal2 = 0;attrVal2 < insts.attribute(attrInd2).numValues();attrVal2++)
						{
						for (int classId = 0;classId < numClasses;classId++)
							{
							int ccmc_value = ccmc.get(classId).get(attrInd1).get(attrVal1).get(attrInd2).get(attrVal2);

							double ccmp_value_numerator = (double) ccmc_value;
							double ccmp_value_denom = (double) classStats.nominalCounts[classId];
							ccmp_value_denom += (insts.attribute(attrInd1).numValues() * insts.attribute(attrInd2).numValues()); // Laplace
							double ccmp_value = ccmp_value_numerator / ccmp_value_denom;

							if(ccmp.get(classId).get(attrInd1).get(attrVal1).get(attrInd2).containsKey(attrVal2))
								{
								System.out.println("SOMETHING WRONG!!! Overwriting in ccmp");
								}
							ccmp.get(classId).get(attrInd1).get(attrVal1).get(attrInd2).put(attrVal2, ccmp_value);

							double all3_prob_num = ccmc_value; // Laplace
																// already done
							double all3_prob_denom = insts.numInstances();
							all3_prob_denom += (insts.attribute(attrInd1).numValues() * insts.attribute(attrInd2).numValues() * insts.classAttribute().numValues());
							double all3_prob = all3_prob_num / all3_prob_denom;

							double two_prob = ccmp_value;

							// need to iterate through to get this..
							double attr1_condprob = ccfvp.get(classId).get(attrInd1).get(attrVal1);
							double attr2_condprob = ccfvp.get(classId).get(attrInd2).get(attrVal2);

							double local_mutualInfo = all3_prob;
							local_mutualInfo = local_mutualInfo * (Math.log(two_prob / (attr1_condprob * attr2_condprob)) / Math.log(2));
							mutualInfo += local_mutualInfo;

							}
						}
					}
				ccmi.get(attrInd1).put(attrInd2, mutualInfo);
				mutualInfoMatrix[attrInd1][attrInd2] = (double) Math.round(mutualInfo * 1000000.0) / 1000000.0;
				if(attrInd1 == attrInd2)
					{
					mutualInfoMatrix[attrInd1][attrInd2] = -1.0;
					}
				}
			}
		// System.out.println();
		}

	public void computeClassCondMutualCounts(Instances instances)
		{
		ccmc = classConditionedMutualCounts;

		for (int i = 0;i < instances.numInstances();i++)
			{
			Instance inst = instances.instance(i);
			// classId
			int classId = (int) inst.classValue();

			// AttrInd1
			for (int attrInd1 = 0;attrInd1 < insts.numAttributes();attrInd1++)
				{
				if(attrInd1 == insts.classIndex())
					continue;

				// AttrVal1
				int attrVal1 = (int) inst.value(attrInd1);

				// AttrInd2
				for (int attrInd2 = 0;attrInd2 < insts.numAttributes();attrInd2++)
					{
					if(attrInd2 == insts.classIndex())
						continue;

					int attrVal2 = (int) inst.value(attrInd2);
					// AttrVal2

					// Laplace already done in init..
					int oldCount = ccmc.get(classId).get(attrInd1).get(attrVal1).get(attrInd2).get(attrVal2);
					ccmc.get(classId).get(attrInd1).get(attrVal1).get(attrInd2).remove(attrVal2);
					ccmc.get(classId).get(attrInd1).get(attrVal1).get(attrInd2).put(attrVal2, oldCount + 1);

					}
				}
			}
		}

	}