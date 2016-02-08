package com.flipkart.mlapi;

import com.fasterxml.jackson.core.JsonProcessingException;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;


import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;


import org.apache.spark.SparkContext;


public class RandomForestBridgeTest extends TestCase {

    public void testRandomForestBridgeClassification() throws IOException {

        SparkConf sparkConf = new SparkConf();
        String master = "local[1]";
        sparkConf.setMaster(master);
        sparkConf.setAppName("Local Spark Unit Test");
        JavaSparkContext sc = new JavaSparkContext(new SparkContext(sparkConf));

        Integer numClasses = 7;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

        String datapath = "src/test/resources/classification_test.libsvm";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        RandomForestModel model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees,
                featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        RandomForestBridge randomForestBridgeIn = new RandomForestBridge();
        String modelDump = randomForestBridgeIn.export(model);

        RandomForestBridge randomForestBridgeOut = new RandomForestBridge();
        randomForestBridgeOut.load(modelDump);

        List<LabeledPoint> testPoints = data.take(10);
        for(LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = model.predict(v);
            double predicted = randomForestBridgeOut.predict(v.toArray());
            System.out.println(actual + "  -- " + predicted);
            assertEquals(actual, predicted);
        }

        categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        impurity = "variance";
        numTrees = 3;
        maxDepth = 4;
        maxBins = 32;
        featureSubsetStrategy = "auto";
        seed = 12345;

        datapath = "src/test/resources/regression_test.libsvm";

        data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        model = RandomForest.trainRegressor(data,categoricalFeaturesInfo,numTrees,featureSubsetStrategy,impurity,maxDepth,maxBins,seed);

        randomForestBridgeIn = new RandomForestBridge();
        modelDump = randomForestBridgeIn.export(model);

        randomForestBridgeOut = new RandomForestBridge();
        randomForestBridgeOut.load(modelDump);

        testPoints = data.collect();
        for(LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = model.predict(v);
            double predicted = randomForestBridgeOut.predict(v.toArray());
            System.out.println(actual + "  -- " + predicted);
            assertEquals(actual, predicted);
        }

        datapath = "src/test/resources/binary_classification_test.libsvm";
        data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());
        LogisticRegressionBridge logisticRegressionBridgeIn = new LogisticRegressionBridge();

        modelDump = logisticRegressionBridgeIn.export(lrmodel);

        LogisticRegressionBridge logisticRegressionBridgeOut = new LogisticRegressionBridge();
        logisticRegressionBridgeOut.load(modelDump);

        lrmodel.clearThreshold();
        testPoints = data.collect();
        for(LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = lrmodel.predict(v);
            double predicted = logisticRegressionBridgeOut.predict(v.toArray());
            System.out.println(actual + "  -- " + predicted);
            assertEquals(actual, predicted);
        }

    }
}
