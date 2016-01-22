import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class ArffHelper
	{
	/*
	 * Read and mark last attribute as class attribute
	 */
	public static Instances readInputFromArff(String arffInFilePath)
		{
		Instances data = null;

		try
			{
			BufferedReader reader = new BufferedReader(new FileReader(arffInFilePath));
			ArffReader arff = new ArffReader(reader);

			data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);

			reader.close();
			}
		catch(IOException e)
			{
			e.printStackTrace();
			}
		finally
			{

			}
		return data;
		}

	/*
	 * Randomly sample 'size' instances
	 */
	public static Instances createRandomSample(Instances sourceInsts, int size)
		{
		System.out.println("Beginning Sampling : " + size);
		Instances sampled = new Instances(sourceInsts, size);
		Random r = new Random();
		for (int i = 0;i < size;i++)
			{
			int ind = r.nextInt(sourceInsts.numInstances());
			sampled.add(sourceInsts.instance(ind));
			sourceInsts.delete(ind);
			}

		return sampled;
		}

	}
