package com.flipkart.fdp.ml.predictors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.flipkart.fdp.ml.model.LRModel;

public class LRPredictor implements Predictor<LRModel> {
	private static final Logger LOG = LoggerFactory.getLogger(LRPredictor.class);
	private LRModel model;

	public LRPredictor(LRModel model) {
		this.model = model;
	}

	public double predict(double[] input) {
		double dotProduct = 0.0;
		for (int i = 0; i < input.length; i++) {
			dotProduct += input[i] * model.weights[i];
		}
		double margin = dotProduct + model.intercept;
		return 1.0 / (1.0 + Math.exp(-margin));
	}
}
