package com.wutianyi.mydl4j.nd4j.example;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.junit.Test;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.List;

/**
 * Created by hanjiewu on 2017/10/19.
 */
public class ND4JExample {

    @Test
    public void testReadCSV() throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader(0, ",");
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 150, 4, 3);
        DataSet dataset = dataSetIterator.next();

    }

    @Test
    public void readIris() throws IOException {
        List<String> contents = FileUtils.readLines(new ClassPathResource("iris.txt").getFile());

        List<double[]> lists = Lists.newArrayList();
        for (String content : contents) {
            String[] cs = content.split(",");
            double[] vs = new double[cs.length];
            int i = 0;
            for (String c : cs) {
                vs[i] = NumberUtils.toDouble(c);
                ++i;
            }
            lists.add(vs);
        }
        INDArray iris = Nd4j.create(lists.toArray(new double[0][0]));
        System.out.println(iris.length());
        System.out.println(iris);
        System.out.println(iris.cumsum(0));
    }

    @Test
    public void example2() {
        INDArray tt = Nd4j.randn(3, 4);
        System.out.println(tt);
        tt = Nd4j.getExecutioner().execAndReturn(new Tanh(tt));
        System.out.println(tt);
    }

    @Test
    public void example1() {
        INDArray tens = Nd4j.zeros(3, 5).addi(10);
        System.out.println(tens);

        Nd4j.getRandom().setSeed(10);
        System.out.println(Nd4j.randn(3, 4));
        System.out.println(Nd4j.eye(3));

        NdIndexIterator iter = new NdIndexIterator(tens.shape());
        while (iter.hasNext()) {
            int[] nextIndex = iter.next();
            System.out.println(tens.getInt(nextIndex));
        }
        System.out.println(NDArrayIndex.all());


    }
}
