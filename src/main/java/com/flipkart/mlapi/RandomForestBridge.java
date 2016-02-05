package com.flipkart.mlapi;

import java.util.Stack;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Map;
import java.io.IOException;

import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.Node;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.configuration.Algo;
import org.apache.spark.mllib.tree.configuration.FeatureType;
import org.apache.spark.mllib.tree.model.Split;
import scala.collection.immutable.List;

import com.fasterxml.jackson.databind.ObjectMapper;    
import com.fasterxml.jackson.core.JsonProcessingException;


public class RandomForestBridge {

    private class NodeInfo {
        public int id;
        public int feature;
        public boolean isLeaf;
        public String featureType;
        public double threshold;
        public double predict;
        public double probability;
        public List<Double> categories;
    }

    private class TreeInfo {
        public int topNodeID;
        public HashMap<Integer, Integer> leftChildMap = new HashMap<Integer, Integer>();
        public HashMap<Integer, Integer> rightChildMap = new HashMap<Integer, Integer>();
        public HashMap<Integer, NodeInfo> nodeInfo = new HashMap<Integer, NodeInfo>();
    }

    private class RandomForestInfo {
        public String algorithm;
        public ArrayList<TreeInfo> trees = new ArrayList<TreeInfo>();
    }

    RandomForestInfo currRandomForestInfo;
    
    private void visit(Node node, Stack<Node> nodesToVisit, TreeInfo treeInfo) {
        NodeInfo nodeInfo = new NodeInfo();
        nodeInfo.id = node.id();
        nodeInfo.isLeaf = node.isLeaf();
        if (node.split().nonEmpty()) {
            Split split = node.split().get();
            nodeInfo.feature = split.feature();
            if (split.featureType().equals(FeatureType.Categorical())) {
                nodeInfo.featureType = "Categorical";
            }
            if (split.featureType().equals(FeatureType.Continuous())) {
                nodeInfo.featureType = "Continuous";
            }
            nodeInfo.categories = (List<Double>) (Object) split.categories();        
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
        for(int i = 0; i < decisionTreeModels.length; i++) {
            randomForestInfo.trees.add(visitTree(decisionTreeModels[i]));
        }
        return randomForestInfo;
    }

    private double predictNode(NodeInfo nodeInfo, double[] input, TreeInfo treeInfo) {
        if (nodeInfo.isLeaf) 
            return nodeInfo.predict;
        else {
            if (nodeInfo.featureType == "Continuous") {
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
        if (randomForestInfo.algorithm == "Classification") {
            HashMap<Double, Integer> votes = new HashMap<Double, Integer>();
            for(int i = 0; i < randomForestInfo.trees.size(); i++) {
                double candidate = predictTree(randomForestInfo.trees.get(i), input);
                int vote_count = 0;                
                if (votes.get(candidate) != null)
                    vote_count = votes.get(candidate);
                votes.put(candidate, vote_count + 1);
            }
            int maxVotes = 0;
            double maxVotesCandidate = 0;
            for(Map.Entry<Double, Integer> entry : votes.entrySet()) {
                if (entry.getValue() > maxVotes) {
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
        return mapper.writeValueAsString(randomForestInfo);
    }
    
    public void load(String rep) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        currRandomForestInfo = mapper.readValue(rep, RandomForestInfo.class);
    }
    
    public double predict(double[] input) {
        return predictForest(currRandomForestInfo, input);
    }
}

