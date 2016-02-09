package com.flipkart.fdp.ml.predictors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.flipkart.fdp.ml.model.DecisionTree;
import com.flipkart.fdp.ml.model.RandomForest;

public class RFPredictor implements Predictor<RandomForest> {
	private static final Logger LOG = LoggerFactory.getLogger(RFPredictor.class);
	private final RandomForest forest;
	private final List<Predictor<DecisionTree>> subPredictors;

	public RFPredictor(RandomForest forest) {
		this.forest = forest;
		this.subPredictors = new ArrayList<>(forest.trees.size());
		for (DecisionTree tree : forest.trees) {
			subPredictors.add(new DTreePredictor(tree));
		}
	}

	public double predict(double[] input) {
		return predictForest(input);
	}

	private double predictForest(double[] input) {
		if (forest.algorithm.equals("Classification")) {
			return classify(input);
		} else {
			return regression(input);
		}
	}

	private double regression(double[] input) {
		double total = 0;
		for (Predictor<DecisionTree> i : subPredictors) {
			total += i.predict(input);
		}
		return total / subPredictors.size();
	}

	private double classify(double[] input) {
		Map<Double, Integer> votes = new HashMap<Double, Integer>();
		for (Predictor<DecisionTree> i : subPredictors) {
			double label = i.predict(input);

			Integer existingCount = votes.get(label);
			if (existingCount == null) {
				existingCount = 0;
			}

			int newCount = existingCount + 1;
			votes.put(label, newCount);
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
