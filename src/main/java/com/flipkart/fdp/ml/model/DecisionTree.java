package com.flipkart.fdp.ml.model;

import java.util.ArrayList;
import java.util.HashMap;

public class DecisionTree {
	public int topNodeID;
	public HashMap<Integer, Integer> leftChildMap = new HashMap<Integer, Integer>();
	public HashMap<Integer, Integer> rightChildMap = new HashMap<Integer, Integer>();
	public HashMap<Integer, DecisionNode> nodeInfo = new HashMap<Integer, DecisionNode>();

	public DecisionTree() {

	}

	static class DecisionNode {
		public int id;
		public int feature;
		public boolean isLeaf;
		public String featureType;
		public double threshold;
		public double predict;
		public double probability;
		public ArrayList<Double> categories;

		public DecisionNode() {
		}

		private boolean visitLeft(double val) {
			return featureType.equals("Continuous") ? val <= threshold : categories.contains(val);
		}

	}

	private double predict(DecisionNode node, double[] input) {
		if (node.isLeaf)
			return node.predict;
		else {
			boolean visitLeft = node.visitLeft(input[node.feature]);
			if (visitLeft) {
				DecisionNode leftChild = nodeInfo.get(leftChildMap.get(node.id));
				return predict(leftChild, input);
			} else {
				DecisionNode rightChild = nodeInfo.get(rightChildMap.get(node.id));
				return predict(rightChild, input);
			}
		}
	}

	public double predict(double[] input) {
		DecisionNode node = nodeInfo.get(topNodeID);
		return predict(node, input);
	}
}