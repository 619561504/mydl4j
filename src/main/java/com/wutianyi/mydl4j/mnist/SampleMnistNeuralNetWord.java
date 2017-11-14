package com.wutianyi.mydl4j.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.io.IOException;
import java.util.Random;

public class SampleMnistNeuralNetWord {

	public static void main(String[] args) throws IOException {

		final int numRows = 28;
		final int numColumns = 28;
		int outputNum = 10;
		int batchSize = 10;
		int hiddenNum = 30;
		int seed = 123456;
		int numEpochs = 100;
		//double rate = 3.0d;
		double rate = 0.5;

		MnistDataSetIterator trainItr = new MnistDataSetIterator(batchSize, true, seed);
		MnistDataSetIterator testItr = new MnistDataSetIterator(batchSize, false, seed);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1)
				.activation(Activation.SIGMOID)
				.learningRate(rate)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numRows * numColumns).nOut(hiddenNum).build())
				.layer(1, new DenseLayer.Builder().nIn(hiddenNum).nOut(hiddenNum).build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
						.nIn(hiddenNum).nOut(outputNum).activation(Activation.SIGMOID).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		for (int i = 0; i < numEpochs; i++) {
			model.fit(trainItr);
			Evaluation evaluation = new Evaluation(outputNum);
			while (testItr.hasNext()) {
				DataSet dataSet = testItr.next();
				INDArray output = model.output(dataSet.getFeatureMatrix());
				evaluation.eval(dataSet.getLabels(), output);
			}
			System.out.println(evaluation.stats());
			testItr.reset();
			trainItr.reset();
		}
	}
}
