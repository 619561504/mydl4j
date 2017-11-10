import com.google.common.collect.Maps;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Created by hanjiewu on 2017/11/1.
 */
public class GravesLSTMCharModellingUseExample {

	public static void main(String[] args) throws IOException {
		File locationFileToSave = new File("GravesLSTMModel.zip");
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationFileToSave);
model.init();
		Layer[] layers = model.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		Random random = new Random(12345);

		char[] validCharacters = getMinimalCharacterSet();
		int nSamplesToGenerate = 4;
		int nCharactersToSample = 300;
		System.out.println(random.nextDouble() * validCharacters.length);
		String initialization = "W";

		System.out.println("Initialzation: " + initialization);

		INDArray initializationInput = Nd4j.zeros(nSamplesToGenerate, validCharacters.length, initialization.length());
		Map<Character, Integer> charToIdxMap = Maps.newHashMap();
		for (int i = 0; i < validCharacters.length; i++) charToIdxMap.put(validCharacters[i], i);

		char[] init = initialization.toCharArray();

		for (int i = 0; i < init.length; i++) {
			int idx = charToIdxMap.get(init[i]);

			for (int j = 0; j < nSamplesToGenerate; j++) {
				initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
			}
		}

		StringBuilder[] sb = new StringBuilder[nSamplesToGenerate];
		for (int i = 0; i < nSamplesToGenerate; i++) sb[i] = new StringBuilder(initialization);

		model.rnnClearPreviousState();
		INDArray output = model.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2) - 1, 0, 1);

		for (int i = 0; i < nCharactersToSample; i++) {
			INDArray nextInput = Nd4j.zeros(nSamplesToGenerate, validCharacters.length);
			for (int s = 0; s < nSamplesToGenerate; s++) {
				double[] outputProbDistribution = new double[validCharacters.length];
				for (int j = 0; j < outputProbDistribution.length; j++) {
					outputProbDistribution[j] = output.getDouble(s, j);
				}
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, random);
				nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);
				sb[s].append(validCharacters[sampledCharacterIdx]);
			}
			output = model.rnnTimeStep(nextInput);
		}

		for (int i = 0; i < nSamplesToGenerate; i++) {
			System.out.println(sb[i].toString());
		}
	}

	public static int sampleFromDistribution(double[] distribution, Random rng) {
		double d = 0.0;
		double sum = 0.0;
		for (int t = 0; t < 10; t++) {
			d = rng.nextDouble();
			sum = 0.0;
			for (int i = 0; i < distribution.length; i++) {
				sum += distribution[i];
				if (d <= sum) return i;
			}
			//If we haven't found the right index yet, maybe the sum is slightly
			//lower than 1 due to rounding error, so try again.
		}
		//Should be extremely unlikely to happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
	}

	public static char[] getMinimalCharacterSet() {
		List<Character> validChars = new LinkedList<>();
		for (char c = 'a'; c <= 'z'; c++) validChars.add(c);
		for (char c = 'A'; c <= 'Z'; c++) validChars.add(c);
		for (char c = '0'; c <= '9'; c++) validChars.add(c);
		char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
		for (char c : temp) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i = 0;
		for (Character c : validChars) out[i++] = c;
		return out;
	}
}
