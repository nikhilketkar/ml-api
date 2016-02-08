package com.flipkart.mlapi;

import java.util.List;
import java.util.Stack;

import org.apache.spark.mllib.tree.configuration.FeatureType;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.Node;
import org.apache.spark.mllib.tree.model.Split;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.flipkart.mlapi.DecisionTree.DecisionNode;

import scala.collection.JavaConversions;

public class DecisionTreeBridge implements ModelBridge<DecisionTreeModel, DecisionTree> {
	private static final Logger LOG = LoggerFactory.getLogger(DecisionTreeBridge.class);

	private void visit(Node node, Stack<Node> nodesToVisit, DecisionTree treeInfo) {
		DecisionNode nodeInfo = new DecisionNode();
		nodeInfo.id = node.id();
		nodeInfo.isLeaf = node.isLeaf();
		if (node.split().nonEmpty()) {
			Split split = node.split().get();
			nodeInfo.feature = split.feature();
			nodeInfo.threshold = split.threshold();
			if (split.featureType().equals(FeatureType.Categorical())) {
				nodeInfo.featureType = "Categorical";
			}
			if (split.featureType().equals(FeatureType.Continuous())) {
				nodeInfo.featureType = "Continuous";
			}

			List<Double> categories = (List<Double>) (Object) JavaConversions.seqAsJavaList(split.categories());

		}
		nodeInfo.predict = node.predict().predict();
		nodeInfo.probability = node.predict().prob();
		treeInfo.nodeInfo.put(node.id(), nodeInfo);
		if (node.rightNode().nonEmpty()) {
			Node right = node.rightNode().get();
			treeInfo.rightChildMap.put(node.id(), right.id());
			nodesToVisit.push(right);
		}
		if (node.leftNode().nonEmpty()) {
			Node left = node.leftNode().get();
			treeInfo.leftChildMap.put(node.id(), left.id());
			nodesToVisit.push(left);
		}
	}

	public DecisionTree transform(DecisionTreeModel decisionTreeModel) {
		DecisionTree treeInfo = new DecisionTree();
		Node node = decisionTreeModel.topNode();
		treeInfo.topNodeID = node.id();
		Stack<Node> nodesToVisit = new Stack<>();
		nodesToVisit.push(node);
		while (!nodesToVisit.empty()) {
			Node curr = nodesToVisit.pop();
			visit(curr, nodesToVisit, treeInfo);
		}
		return treeInfo;
	}
}
