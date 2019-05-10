package com.mamcose.nlp.ActivationFunctions;

public interface ActivationFunction {
    double execute(double y);
    double derivative(double y);
}
