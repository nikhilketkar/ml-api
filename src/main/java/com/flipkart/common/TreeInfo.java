package com.flipkart.common;

import java.util.HashMap;

public class TreeInfo {
    public int topNodeID;
    public HashMap<Integer, Integer> leftChildMap = new HashMap<Integer, Integer>();
    public HashMap<Integer, Integer> rightChildMap = new HashMap<Integer, Integer>();
    public HashMap<Integer, NodeInfo> nodeInfo = new HashMap<Integer, NodeInfo>();

}
