package com.flipkart.fdp.ml.model;

import java.util.ArrayList;

public class RandomForest implements Model {
	public String algorithm;
	public ArrayList<DecisionTree> trees = new ArrayList<DecisionTree>();

	public RandomForest() {
	}


}