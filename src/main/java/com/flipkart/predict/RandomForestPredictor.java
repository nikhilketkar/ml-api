package com.flipkart.predict;

import com.flipkart.common.NodeInfo;
import com.flipkart.common.RandomForestInfo;
import com.flipkart.common.TreeInfo;
import com.google.gson.Gson;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class RandomForestPredictor {

    RandomForestInfo currRandomForestInfo;

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

    public void load(String rep) throws IOException {
        Gson gson = new Gson();
        currRandomForestInfo = gson.fromJson(rep, RandomForestInfo.class);
    }

    public double predict(double[] input) {
        return predictForest(currRandomForestInfo, input);
    }
}
