package com.flipkart.mlapi;

public interface ModelBridge<F, T> {

	T transform(F from);
	
}
