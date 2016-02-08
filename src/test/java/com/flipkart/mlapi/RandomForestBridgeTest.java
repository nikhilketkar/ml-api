package com.flipkart.mlapi;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.junit.Test;

import com.flipkart.fdp.ml.model.LRBridge;
import com.flipkart.fdp.ml.model.LRModel;
import com.flipkart.fdp.ml.model.bridge.RandomForestBridge;

public class RandomForestBridgeTest extends SparkTestBase {

	@Test
	public void testRandomForestBridgeClassification() throws IOException {
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
		com.flipkart.fdp.ml.model.RandomForest modelDump = randomForestBridgeIn.transform(model);

		List<LabeledPoint> testPoints = data.take(10);
		for (LabeledPoint i : testPoints) {
			Vector v = i.features();
			double actual = model.predict(v);
			double predicted = modelDump.predict(v.toArray());
			System.out.println(actual + "  -- " + predicted);
			assertEquals(actual, predicted, 0.01);
		}
	}

	private class RFTraining {
		private String dataPath;
		private String impurity;
		private int numTrees;
		private int maxDepth;
		private int maxBins;
		private String featureSubsetStrategy;
		private int seed;
	}

	@Test
	public void testRFRegression() {
		String datapath = "src/test/resources/regression_test.libsvm";
		testRegressor("variance", 3, 4, 32, "auto", 12345, datapath);
	}

	private void testRegressor(String impurity, int numTrees, int maxDepth, int maxBins, String featureSubsetStrategy,
			int seed, String datapath) {
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
		RandomForestModel model = RandomForest.trainRegressor(data, categoricalFeaturesInfo, numTrees,
				featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

		RandomForestBridge randomForestBridgeIn = new RandomForestBridge();
		com.flipkart.fdp.ml.model.RandomForest modelDump = randomForestBridgeIn.transform(model);

		List<LabeledPoint> testPoints = data.collect();
		for (LabeledPoint i : testPoints) {
			Vector v = i.features();
			double actual = model.predict(v);
			double predicted = modelDump.predict(v.toArray());
			System.out.println(actual + "  -- " + predicted);
			assertEquals(actual, predicted, 0.01);
		}
	}

	@Test
	public void testLR() {
		String datapath = "src/test/resources/binary_classification_test.libsvm";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
		LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());
		LRBridge logisticRegressionBridgeIn = new LRBridge();

		LRModel lrModel = logisticRegressionBridgeIn.transform(lrmodel);

		lrmodel.clearThreshold();
		List<LabeledPoint> testPoints = data.collect();
		for (LabeledPoint i : testPoints) {
			Vector v = i.features();
			double actual = lrmodel.predict(v);
			double predicted = lrModel.predict(v.toArray());
			System.out.println(actual + "  -- " + predicted);
			assertEquals(actual, predicted, 0.01);
		}
	}
}
