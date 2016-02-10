package com.flipkart.export;

import java.io.IOException;
import java.util.Stack;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

import com.flipkart.common.NodeInfo;
import com.flipkart.common.RandomForestInfo;
import com.flipkart.common.TreeInfo;
import com.google.gson.Gson;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.Node;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.configuration.Algo;
import org.apache.spark.mllib.tree.configuration.FeatureType;
import org.apache.spark.mllib.tree.model.Split;

import com.fasterxml.jackson.databind.ObjectMapper;    
import com.fasterxml.jackson.core.JsonProcessingException;
import scala.collection.JavaConversions;

public class RandomForestExporter {

    private void visit(Node node, Stack<Node> nodesToVisit, TreeInfo treeInfo) {
        NodeInfo nodeInfo = new NodeInfo();
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
        if(node.rightNode().nonEmpty()) {
            Node right = node.rightNode().get();
            treeInfo.rightChildMap.put(node.id(), right.id());
            nodesToVisit.push(right);
        }
        if(node.leftNode().nonEmpty()) {
            Node left = node.leftNode().get();
            treeInfo.leftChildMap.put(node.id(), left.id());
            nodesToVisit.push(left);
        }
    }

    private TreeInfo visitTree(DecisionTreeModel decisionTreeModel) {
        TreeInfo treeInfo = new TreeInfo();
        Node node = decisionTreeModel.topNode();
        treeInfo.topNodeID = node.id();
        Stack<Node> nodesToVisit = new Stack();
        nodesToVisit.push(node);
        while (! nodesToVisit.empty()) {
            Node curr = nodesToVisit.pop();
            visit(curr, nodesToVisit, treeInfo);
        }
        return treeInfo;
    }

    private RandomForestInfo visitForest(RandomForestModel randomForestModel) {
        RandomForestInfo randomForestInfo = new RandomForestInfo();
        if ( randomForestModel.algo().equals(Algo.Classification())) {
            randomForestInfo.algorithm = "Classification";
        }
        if ( randomForestModel.algo().equals(Algo.Regression())) {
            randomForestInfo.algorithm = "Regression";
        }

        DecisionTreeModel[] decisionTreeModels = randomForestModel.trees();
        for(DecisionTreeModel i : decisionTreeModels) {
            randomForestInfo.trees.add(visitTree(i));
        }
        return randomForestInfo;
    }

    public String export(RandomForestModel randomForestModel) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        RandomForestInfo randomForestInfo = visitForest(randomForestModel);
        Gson gson = new Gson();
        return gson.toJson(randomForestInfo);
    }

}

