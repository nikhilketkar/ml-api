package com.flipkart.fdp.ml.model.adapter;

import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.flipkart.fdp.ml.model.LRModel;

public class LRAdapter implements ModelAdapter<LogisticRegressionModel, LRModel> {
	private static final Logger LOG = LoggerFactory.getLogger(LRAdapter.class);

	@Override
	public LRModel transform(LogisticRegressionModel sparkLRModel) {
		LRModel lrModel = new LRModel();
		lrModel.weights = sparkLRModel.weights().toArray();
		lrModel.intercept = sparkLRModel.intercept();
		lrModel.numClasses = sparkLRModel.numClasses();
		lrModel.numFeatures = sparkLRModel.numFeatures();
		return lrModel;
	}

	@Override
	public Class<LogisticRegressionModel> getSource() {
		return LogisticRegressionModel.class;
	}

	@Override
	public Class<LRModel> getTarget() {
		return LRModel.class;
	}

}
