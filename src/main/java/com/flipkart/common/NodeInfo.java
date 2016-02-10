package com.flipkart.common;

import java.util.ArrayList;

public class NodeInfo {
        public int id;
        public int feature;
        public boolean isLeaf;
        public String featureType;
        public double threshold;
        public double predict;
        public double probability;
        public ArrayList<Double> categories;
}
