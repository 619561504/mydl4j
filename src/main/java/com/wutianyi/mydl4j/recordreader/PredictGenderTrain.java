package com.wutianyi.mydl4j.recordreader;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class PredictGenderTrain {

	public String filePath;

	public static void main(String[] args) {
		PredictGenderTrain dg = new PredictGenderTrain();
		dg.filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\PredictGender\\Data\\";

		dg.train();
	}

	public void train() {
		int seed = 123456;

		double learningRate = 0.005;
		int batchSize = 100;
		int nEpochs = 100;
		int numInputs = 0;
		int numOutputs = 0;
		int numHiddenNodes = 0;

		try (GenderRecordReader rr = new GenderRecordReader(new ArrayList<String>() {{
			add("M");
			add("F");
		}})) {
			long st = System.currentTimeMillis();
			rr.initialize(new FileSplit(new File(filePath)));
			long et = System.currentTimeMillis();
			System.out.println("Preprocessing end time : " + et);
			System.out.println("time taken to process data : " + (et - st) + " ms");

			numInputs = rr.maxLengthName * 5;
			numOutputs = 2;
			numHiddenNodes = 2 * numInputs + numOutputs;

			DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, numInputs, numOutputs);

			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.seed(seed)
					.iterations(1)
					.biasInit(1)
					.regularization(true).l2(1e-4)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.learningRate(learningRate)
					.updater(Updater.NESTEROVS)
					.list()
					.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
							.weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
					.layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
							.weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
					.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(numHiddenNodes)
							.nOut(numOutputs).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
					.pretrain(false).backprop(true).build();
			MultiLayerNetwork model = new MultiLayerNetwork(conf);
			model.init();
			model.setListeners(new ScoreIterationListener());
			for (int i = 0; i < nEpochs; i++) {
				while (trainIter.hasNext()) {
					model.fit(trainIter.next(batchSize));
				}
				trainIter.reset();
			}
			GenderRecordReader rr1 = new GenderRecordReader(new ArrayList<String>() {{
				add("M");
				add("F");
			}});
			rr1.initialize(new FileSplit(new File(filePath)));
			DataSetIterator testIter = new RecordReaderDataSetIterator(rr1, batchSize, numInputs, numOutputs);

			Evaluation eval = new Evaluation(numOutputs);
			while (testIter.hasNext()) {
				DataSet dataSet = testIter.next();
				INDArray features = dataSet.getFeatures();
				INDArray labels = dataSet.getLabels();

				INDArray predicted = model.output(features, false);
				eval.eval(labels, predicted);
			}
			System.out.println(eval.stats());
			ModelSerializer.writeModel(model, this.filePath + "PredictGender.net", true);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}
