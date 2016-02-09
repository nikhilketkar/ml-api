package com.flipkart.fdp.ml.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RandomForest {
	public String algorithm;
	public ArrayList<DecisionTree> trees = new ArrayList<DecisionTree>();

	public RandomForest() {
	}

	public double predict(double[] input) {
		return predictForest(input);
	}

	private double predictForest(double[] input) {
		if (algorithm.equals("Classification")) {
			return classify(input);
		} else {
			return regression(input);
		}
	}

	private double regression(double[] input) {
		double total = 0;
		for (DecisionTree i : trees) {
			total += i.predict(input);
		}
		return total / trees.size();
	}

	private double classify(double[] input) {
		Map<Double, Integer> votes = new HashMap<Double, Integer>();
		int maxVote = 0;
		double maxVoteKey = -1;
		for (DecisionTree i : trees) {
			double label = i.predict(input);

			Integer existingCount = votes.get(label);
			if (existingCount == null) {
				existingCount = 0;
			}

			int newCount = existingCount + 1;
			votes.put(label, newCount);
			if (maxVote < newCount) {
				
			}
		}

		int maxVotes = 0;
		double maxVotesCandidate = 0;
		for (Map.Entry<Double, Integer> entry : votes.entrySet()) {
			if (entry.getValue() >= maxVotes) {
				maxVotes = entry.getValue();
				maxVotesCandidate = entry.getKey();
			}
		}
		return maxVotesCandidate;
	}

}