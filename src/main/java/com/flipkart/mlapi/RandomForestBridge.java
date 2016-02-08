package com.flipkart.mlapi;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Stack;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

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

public class RandomForestBridge {

    private class NodeInfo {
        public int id;
        public int feature;
        public boolean isLeaf;
        public String featureType;
        public double threshold;
        public double predict;
        public double probability;
        public ArrayList<Double> categories;
        public NodeInfo() {

        }
    }

    private class TreeInfo {
        public int topNodeID;
        public HashMap<Integer, Integer> leftChildMap = new HashMap<Integer, Integer>();
        public HashMap<Integer, Integer> rightChildMap = new HashMap<Integer, Integer>();
        public HashMap<Integer, NodeInfo> nodeInfo = new HashMap<Integer, NodeInfo>();
        public TreeInfo() {

        }
    }

    private class RandomForestInfo {
        public String algorithm;
        public ArrayList<TreeInfo> trees = new ArrayList<TreeInfo>();
        public RandomForestInfo() {

        }
    }

    RandomForestInfo currRandomForestInfo;
    
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

    private double predictNode(NodeInfo nodeInfo, double[] input, TreeInfo treeInfo) {
        if (nodeInfo.isLeaf) 
            return nodeInfo.predict;
        else {
            if (nodeInfo.featureType.equals("Continuous")) {
                if (input[nodeInfo.feature] <= nodeInfo.threshold)
                    return predictNode(treeInfo.nodeInfo.get(treeInfo.leftChildMap.get(nodeInfo.id)), input, treeInfo);
                else 
                    return predictNode(treeInfo.nodeInfo.get(treeInfo.rightChildMap.get(nodeInfo.id)), input, treeInfo);
            }
            else {
                if (nodeInfo.categories.contains(input[nodeInfo.feature])) 
                    return predictNode(treeInfo.nodeInfo.get(treeInfo.leftChildMap.get(nodeInfo.id)), input, treeInfo);
                else
                    return predictNode(treeInfo.nodeInfo.get(treeInfo.rightChildMap.get(nodeInfo.id)), input, treeInfo);
            }
        }
    }

    private double predictTree(TreeInfo treeInfo, double[] input) {
        return predictNode(treeInfo.nodeInfo.get(treeInfo.topNodeID), input, treeInfo);
    }

    private double predictForest(RandomForestInfo randomForestInfo, double[] input) {
        if (randomForestInfo.algorithm.equals("Classification")) {
            HashMap<Double, Integer> votes = new HashMap<Double, Integer>();
            for(TreeInfo i : randomForestInfo.trees) {
                double candidate = predictTree(i, input);
                int vote_count = 0;                
                if (votes.get(candidate) != null)
                    vote_count = votes.get(candidate);
                votes.put(candidate, vote_count + 1);
            }
            int maxVotes = 0;
            double maxVotesCandidate = 0;
            for(Map.Entry<Double, Integer> entry : votes.entrySet()) {
                if (entry.getValue() >= maxVotes) {
                    maxVotes = entry.getValue();
                    maxVotesCandidate = entry.getKey();
                }
            }
            return maxVotesCandidate;
        }
        else {
            double total = 0;
            for(int i = 0; i < randomForestInfo.trees.size(); i++)
                total += predictTree(randomForestInfo.trees.get(i), input);
            return total/randomForestInfo.trees.size();
        }
    }

    public String export(RandomForestModel randomForestModel) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        RandomForestInfo randomForestInfo = visitForest(randomForestModel);
        Gson gson = new Gson();
        return gson.toJson(randomForestInfo);
    }
    
    public void load(String rep) throws IOException {
        Gson gson = new Gson();
        currRandomForestInfo = gson.fromJson(rep, RandomForestInfo.class);
    }
    
    public double predict(double[] input) {
        return predictForest(currRandomForestInfo, input);
    }
}

