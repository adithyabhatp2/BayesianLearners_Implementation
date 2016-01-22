import java.io.PrintStream;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import weka.core.Instance;
import weka.core.Instances;

public class Main
	{

	static int debugLevel = 5;
	static LinkedList<String> predictStats;
	static HashMap<Integer, String> outputLines;

	public static void main(String[] args)
		{

		String arffTrainFilePath = null;
		String arffTestFilePath = null;

		Instances trainInstances;
		char naiveOrTan = 0;
		if(args.length == 3)
			{
			arffTrainFilePath = args[0];
			arffTestFilePath = args[1];
			naiveOrTan = args[2].charAt(0);
			}

		else if(args.length == 0)
			{
			System.out.println("Usage: bayes <train-set-file> <test-set-file> <n|t>");
			String dataSet = "lymph";
			arffTrainFilePath = "input/" + dataSet + "_train.arff";
			arffTestFilePath = "input/" + dataSet + "_test.arff";
			}
		else
			{
			System.out.println("Usage: bayes <train-set-file> <test-set-file> <n|t>");
			System.exit(0);
			}

		Instances allInstances = ArffHelper.readInputFromArff(arffTrainFilePath);
		Instances testInstances = ArffHelper.readInputFromArff(arffTestFilePath);

		trainInstances = allInstances;

		NaiveBayesLearner naiveBayes = null;
		TreeAugmentedNB tan = null;

		if(naiveOrTan == 'n')
			{
			naiveBayes = new NaiveBayesLearner();
			naiveBayes.computeClassProbabilities(trainInstances);
			naiveBayes.computeClassCondInputProbabilities(trainInstances);
			naiveBayes.printNetworkStructure(trainInstances);
			}
		else if(naiveOrTan == 't')
			{
			tan = new TreeAugmentedNB(trainInstances);

			tan.intializeStuff();
			tan.computeClassCondMutualCounts(tan.insts);
			tan.computeClassCondMutualProbabilitiesAndInformation();
			tan.generateMST_Prims();
			tan.printNetworkStructure();
			tan.generateConditionalProbabilityTables();
			}
		else
			{
			System.out.println("Wrong value given for <t|n>");
			System.exit(0);
			}

		// Single Run thru all instances in order - for training

		int correctlyPredicted = 0;
		for (int i = 0;i < testInstances.numInstances();i++)
			{
			Instance inst = testInstances.instance(i);
			double predictedClassValue;
			if(naiveOrTan == 'n')
				predictedClassValue = naiveBayes.predictClass(inst);
			else
				predictedClassValue = tan.predictClass(inst);

			String predictedClass = inst.classAttribute().value((int) predictedClassValue);
			String actualClass = inst.stringValue(inst.classAttribute());

			if(predictedClass == actualClass)
				correctlyPredicted++;

			// System.out.println("Predicted: " + predictedClass + "\tActual: " + actualClass);
			}
		System.out.println("\n" + correctlyPredicted);
		// System.out.println("Correct: " + correctlyPredicted + " out of: " + testInstances.numInstances() + " Accuracy : " + (double) correctlyPredicted / (double)
		// testInstances.numInstances());;
		// */
		}

	public void temp()
		{
		// if(args.length == 3)
		// {
		// String arffTrainFilePath = args[0];
		// String arffTestFilePath = args[1];
		// char naiveOrTan = args[2].charAt(0);
		// outputLines = new HashMap<Integer, String>();
		// Instances allInstances =
		// ArffHelper.readInputFromArff(arffTrainFilePath);
		// }
		// else
		// {
		// System.out.println("Usage: bayes <train-set-file> <test-set-file>
		// <n|t>");
		// }
		//
		// if(args.length == 0)
		// {
		}
	}
