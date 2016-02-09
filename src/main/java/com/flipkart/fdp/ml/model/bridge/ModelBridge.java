package com.flipkart.fdp.ml.model.bridge;

public interface ModelBridge<F, T> {

	T transform(F from);
	
}
