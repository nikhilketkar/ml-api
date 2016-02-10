package com.flipkart.export;

import com.google.gson.Gson;
import org.apache.spark.ml.feature.StringIndexerModel;

import java.util.HashMap;

public class StringIndexerBridge {

    public String export(StringIndexerModel stringIndexerModel) {

        String[] labels = stringIndexerModel.labels();
        HashMap<String, Integer> result = new HashMap<String, Integer>();
        for(int i = 0; i < labels.length; i++) {
            result.put(labels[i], i);
        }
        Gson gson = new Gson();
        return gson.toJson(result);
    }
}
