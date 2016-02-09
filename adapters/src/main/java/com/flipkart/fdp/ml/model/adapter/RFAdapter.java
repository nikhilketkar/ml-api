package com.flipkart.fdp.ml.model.adapter;

import org.apache.spark.mllib.tree.configuration.Algo;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import com.flipkart.fdp.ml.model.DecisionTree;
import com.flipkart.fdp.ml.model.RandomForest;

public class RFAdapter implements ModelAdapter<RandomForestModel, RandomForest> {

	private DTreeAdapter bridge = new DTreeAdapter();

	private RandomForest visitForest(RandomForestModel randomForestModel) {
		RandomForest randomForest = new RandomForest();
		if (randomForestModel.algo().equals(Algo.Classification())) {
			randomForest.algorithm = "Classification";
		}
		if (randomForestModel.algo().equals(Algo.Regression())) {
			randomForest.algorithm = "Regression";
		}

		DecisionTreeModel[] decisionTreeModels = randomForestModel.trees();
		for (DecisionTreeModel i : decisionTreeModels) {
			DecisionTree tree = bridge.transform(i);
			randomForest.trees.add(tree);
		}
		return randomForest;
	}

	@Override
	public RandomForest transform(RandomForestModel from) {
		RandomForest randomForestInfo = visitForest(from);
		return randomForestInfo;
	}
}
