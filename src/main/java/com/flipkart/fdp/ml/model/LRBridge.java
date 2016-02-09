package com.flipkart.fdp.ml.model;

import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.flipkart.fdp.ml.model.bridge.ModelBridge;

public class LRBridge implements ModelBridge<LogisticRegressionModel, LRModel> {
	private static final Logger LOG = LoggerFactory.getLogger(LRBridge.class);

	@Override
	public LRModel transform(LogisticRegressionModel sparkLRModel) {
		LRModel lrModel = new LRModel();
		lrModel.weights = sparkLRModel.weights().toArray();
		lrModel.intercept = sparkLRModel.intercept();
		lrModel.numClasses = sparkLRModel.numClasses();
		lrModel.numFeatures = sparkLRModel.numFeatures();
		return lrModel;
	}

}
