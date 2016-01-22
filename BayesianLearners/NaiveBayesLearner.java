import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Map;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveBayesLearner
	{
	int debug_nbl = 0;

	HashMap<Integer, Double> classProbabilities;
	HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> classConditionedFeatureValProbabilities;
	HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>> classConditionedFeatureValCounts;
	// ccfvp/ccfvc[classId][attIndex][value] = count / prob

	public void computeClassCondInputProbabilities(Instances instances)
		{
		// Initialize Counts and Probabilities
		HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>> ccfvc = classConditionedFeatureValCounts;
		HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> ccfvp = classConditionedFeatureValProbabilities;

		if(classConditionedFeatureValCounts == null)
			{
			classConditionedFeatureValCounts = new HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>();
			classConditionedFeatureValProbabilities = new HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>();

			ccfvc = classConditionedFeatureValCounts;
			ccfvp = classConditionedFeatureValProbabilities;

			for (int i = 0;i < instances.classAttribute().numValues();i++)
				{
				ccfvc.put(i, new HashMap<Integer, HashMap<Integer, Integer>>());
				ccfvp.put(i, new HashMap<Integer, HashMap<Integer, Double>>());
				}
			}

		// Fill in occurrence counts
		for (int i = 0;i < instances.numInstances();i++)
			{
			Instance inst = instances.instance(i);
			Integer classId = (int) inst.value(inst.classAttribute());

			for (int attrInd = 0;attrInd < inst.numAttributes();attrInd++)
				{
				if(attrInd == instances.classIndex())
					continue;

				HashMap<Integer, HashMap<Integer, Integer>> attrId_xval_counts = ccfvc.get(classId);

				if(!attrId_xval_counts.containsKey(attrInd))
					attrId_xval_counts.put(attrInd, new HashMap<Integer, Integer>());

				HashMap<Integer, Integer> xval_counts = attrId_xval_counts.get(attrInd);

				int xval = (int) inst.value(attrInd);
				if(!xval_counts.containsKey(xval))
					{
					// TODO: NOTE : LAPLACE!!!
					xval_counts.put(xval, 1);
					}

				int xval_count = xval_counts.get(xval);
				xval_counts.remove(xval);
				xval_counts.put(xval, xval_count + 1);
				}
			}

		if(debug_nbl >= 1)
			{
			System.out.println("CCIC : " + ccfvc);
			}

		// Compute Probability from counts
		for (int classId = 0;classId < instances.classAttribute().numValues();classId++)
			{
			HashMap<Integer, HashMap<Integer, Integer>> attrId_xval_counts = ccfvc.get(classId);

			for (int attrInd = 0;attrInd < instances.numAttributes();attrInd++)
				{
				if(attrInd == instances.classIndex())
					continue;

				if(!ccfvp.get(classId).containsKey(attrInd))
					{
					ccfvp.get(classId).put(attrInd, new HashMap<Integer, Double>());
					}
				else
					{
					System.out.println("Error.. overwriting.");
					}

				int numAttrVals = 0;
				for (int xval = 0;xval < instances.attribute(attrInd).numValues();xval++)
					{
					if(!attrId_xval_counts.get(attrInd).containsKey(xval))
						{
						// LAPLACE!!!
						attrId_xval_counts.get(attrInd).put(xval, 1);
						}

					int xval_count = attrId_xval_counts.get(attrInd).get(xval);
					numAttrVals += xval_count;
					}

				for (int xval = 0;xval < instances.attribute(attrInd).numValues();xval++)
					{
					int xval_count = attrId_xval_counts.get(attrInd).get(xval);
					double xval_prob = (double) xval_count / (double) (numAttrVals);
					// xval_prob = (double) Math.round(xval_prob * 1000000) /
					// 1000000.0;

					ccfvp.get(classId).get(attrInd).put(xval, xval_prob);

					if(debug_nbl >= 2)
						{
						System.out.println("classid: " + classId + "\tAttrInd: " + attrInd + "\txval: " + xval + "\tcount: " + xval_count + "\ttot: " + numAttrVals + "\tprob: " + xval_prob);
						}
					}
				}
			}

		if(debug_nbl >= 1)
			{
			System.out.println("CC-FVC : " + ccfvc);
			System.out.println("CC-FVP : " + ccfvp);
			}
		}

	public void computeClassProbabilities(Instances instances)
		{
		if(this.classProbabilities == null)
			{
			this.classProbabilities = new HashMap<Integer, Double>();
			}
		AttributeStats classStats = instances.attributeStats(instances.classIndex());
		int numInstances = classStats.totalCount;
		if(numInstances != instances.numInstances())
			{
			System.out.println("ATTRIBUTE STATS incorrect???");
			}

		for (int i = 0;i < instances.classAttribute().numValues();i++)
			{
			double classProb = ((double) (classStats.nominalCounts[i] + 1) / (double) (numInstances + 2)); // Laplace!!
			// classProb = (double) Math.round(classProb * 1000000) / 1000000.0;
			classProbabilities.put(i, classProb);
			}
		if(debug_nbl >= 1)
			{
			System.out.println("Class counts : " + Arrays.toString(classStats.nominalCounts));
			System.out.println("Class Probabilities: " + classProbabilities);
			}
		}

	public int predictClass(Instance inst)
		{

		HashMap<Integer, Double> yval_probabilities = new HashMap<Integer, Double>();

		double probability = 1.0;

		for (int classId = 0;classId < inst.classAttribute().numValues();classId++)
			{
			probability = classProbabilities.get(classId);
			for (int attrInd = 0;attrInd < inst.numAttributes();attrInd++)
				{
				if(attrInd == inst.classIndex())
					continue;
				int xval = (int) inst.value(attrInd);
				probability *= classConditionedFeatureValProbabilities.get(classId).get(attrInd).get(xval);
				}
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

	public void printNetworkStructure(Instances insts)
		{
		for (int attrInd = 0;attrInd < insts.numAttributes();attrInd++)
			{
			if(attrInd == insts.classIndex())
				continue;
			Attribute attr = insts.attribute(attrInd);
			System.out.print(attr.name());
			System.out.println(" " + insts.classAttribute().name());
			}
		System.out.println();
		}

	}
