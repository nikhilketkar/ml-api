package com.flipkart.mlapi;

public class LRModel {
	public double[] weights;
	public double intercept;
	public int numClasses;
	public int numFeatures;

	public double predict(double[] input) {
		double dotProduct = 0.0;
		for (int i = 0; i < input.length; i++) {
			dotProduct += input[i] * weights[i];
		}
		double margin = dotProduct + intercept;
		return 1.0 / (1.0 + Math.exp(-margin));
	}
}