package com.flipkart.mlapi;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Stack;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

import java.lang.Math;

import com.google.gson.Gson;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.Node;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.configuration.Algo;
import org.apache.spark.mllib.tree.configuration.FeatureType;
import org.apache.spark.mllib.tree.model.Split;
import org.codehaus.jackson.annotate.JsonCreator;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import scala.collection.immutable.Range;
import scala.collection.JavaConversions;

public class LogisticRegressionBridge {

    private class LogisticRegressionInfo {
        public double[] weights;
        public double intercept;
        public int numClasses;
        public int numFeatures;
    }

    private LogisticRegressionInfo currLogisticRegressionInfo;

    public String export(LogisticRegressionModel logisticRegressionModel) {
        LogisticRegressionInfo logisticRegressionInfo = new LogisticRegressionInfo();
        logisticRegressionInfo.weights = logisticRegressionModel.weights().toArray();
        logisticRegressionInfo.intercept = logisticRegressionModel.intercept();
        logisticRegressionInfo.numClasses = logisticRegressionModel.numClasses();
        logisticRegressionInfo.numFeatures = logisticRegressionModel.numFeatures();
        Gson gson = new Gson();
        return gson.toJson(logisticRegressionInfo);
    }

    public void load(String rep) throws IOException {
        Gson gson = new Gson();
        currLogisticRegressionInfo = gson.fromJson(rep, LogisticRegressionInfo.class);
    }

    public double predict(double[] input) {
        double dotProduct = 0.0;
        for(int i = 0; i < input.length; i++) {
            dotProduct += input[i] * currLogisticRegressionInfo.weights[i];
        }
        double margin = dotProduct + currLogisticRegressionInfo.intercept;
        return 1.0/(1.0 + Math.exp(-margin));
    }
}
