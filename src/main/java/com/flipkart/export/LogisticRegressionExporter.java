package com.flipkart.export;

import java.io.IOException;
import java.lang.Math;

import com.flipkart.common.LogisticRegressionInfo;
import com.google.gson.Gson;
import org.apache.spark.mllib.classification.LogisticRegressionModel;

public class LogisticRegressionExporter {

    public String export(LogisticRegressionModel logisticRegressionModel) {
        LogisticRegressionInfo logisticRegressionInfo = new LogisticRegressionInfo();
        logisticRegressionInfo.weights = logisticRegressionModel.weights().toArray();
        logisticRegressionInfo.intercept = logisticRegressionModel.intercept();
        logisticRegressionInfo.numClasses = logisticRegressionModel.numClasses();
        logisticRegressionInfo.numFeatures = logisticRegressionModel.numFeatures();
        Gson gson = new Gson();
        return gson.toJson(logisticRegressionInfo);
    }
}
