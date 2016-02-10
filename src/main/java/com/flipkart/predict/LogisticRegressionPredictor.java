package com.flipkart.predict;

import com.flipkart.common.LogisticRegressionInfo;
import com.google.gson.Gson;

import java.io.IOException;

public class LogisticRegressionPredictor {

    private LogisticRegressionInfo currLogisticRegressionInfo;

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
