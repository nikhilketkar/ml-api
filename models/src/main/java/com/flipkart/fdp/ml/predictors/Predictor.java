package com.flipkart.fdp.ml.predictors;

public interface Predictor<T> {
	double predict(double[] input);
}
