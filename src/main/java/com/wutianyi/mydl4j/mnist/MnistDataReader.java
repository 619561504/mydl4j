package com.wutianyi.mydl4j.mnist;


import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.io.*;

public class MnistDataReader {

	public static void main(String[] args) throws IOException, InterruptedException {

		MnistDataSetIterator iter = new MnistDataSetIterator(60, 60000);
		DataSet dataSet = null;
		if (iter.hasNext()) {
			dataSet = iter.next();
			for (int i = 0; i < dataSet.numExamples(); i++) {
				INDArray indArray = dataSet.get(i).getFeatureMatrix().mul(255);
				DrawReconstruction d = new DrawReconstruction(indArray);
				d.draw();

				Thread.sleep(10000L);
				d.frame.dispose();
			}
		}
	}
}
