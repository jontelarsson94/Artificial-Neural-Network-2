//
//  main.cpp
//  ANN
//
//  Created by Jonathan Larsson on 2017-02-22.
//  Copyright © 2017 Jonathan Larsson. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <fstream>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <cmath>
#include <unistd.h>
using namespace std;

int numOfNeurons = 1500;
float input[2200][3];
float weightsToHidden[3][3];
float hiddenOutputs[3];
float weightsToOutput[5];
float expectedOutput[2200];
float outputOutput;
float outputError;
float hiddenError[3];

const float learningRate = 0.1;

void readData(string fileName)
{
    ifstream infile(fileName);
    float x1, x2, x3, output;
    char afterFirst, afterSecond, afterThird;
    int i = 0;
    while (infile >> x1 >> afterFirst >> x2 >> afterSecond >> x3 >> afterThird >> output)
    {
        if (i < numOfNeurons) {
            input[i][0] = x1;
            input[i][1] = x2;
            input[i][2] = x3;
            if(output < 0)
                expectedOutput[i] = 0.25;
            else
                expectedOutput[i] = 0.75;
        } else {
            input[i][0] = x1;
            input[i][1] = x2;
            input[i][2] = x3;
            expectedOutput[i] = output;
        }
        i++;
    }
    
}

float RandomNumber(float Min, float Max)
{
    return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

void initializeWeights()
{
    weightsToHidden[0][0] = RandomNumber(0.0, 1.0);
    weightsToHidden[0][1] = RandomNumber(0.0, 1.0);
    weightsToHidden[0][2] = RandomNumber(0.0, 1.0);
    weightsToHidden[1][0] = RandomNumber(0.0, 1.0);
    weightsToHidden[1][1] = RandomNumber(0.0, 1.0);
    weightsToHidden[1][2] = RandomNumber(0.0, 1.0);
    weightsToHidden[2][0] = RandomNumber(0.0, 1.0);
    weightsToHidden[2][1] = RandomNumber(0.0, 1.0);
    weightsToHidden[2][2] = RandomNumber(0.0, 1.0);
    weightsToOutput[0] = RandomNumber(0.0, 1.0);
    weightsToOutput[1] = RandomNumber(0.0, 1.0);
    weightsToOutput[2] = RandomNumber(0.0, 1.0);
}

float sigmoid(float net)
{
    return 1 / (1 + exp(-net));
}

float calculateHiddenOutput(int i, int j)
{
    return sigmoid(input[i][0] * weightsToHidden[0][j] + input[i][1] * weightsToHidden[1][j] + input[i][2] * weightsToHidden[2][j]);
}

float calculateOutputOutput()
{
    return sigmoid(hiddenOutputs[0] * weightsToOutput[0] + hiddenOutputs[1] * weightsToOutput[1] + hiddenOutputs[2] * weightsToOutput[2]);
}

float calculateOutputError(int i)
{
    return outputOutput * (1 - outputOutput) * (expectedOutput[i] - outputOutput);
}

float calculateHiddenError(int j)
{
    return hiddenOutputs[j] * (1 - hiddenOutputs[j]) * weightsToOutput[j] * outputError;
}

float calculateWeightToOutput(int j)
{
    return learningRate * outputError * hiddenOutputs[j];
}

float calculateWeightToHidden(int i, int inputPos, int hiddenPos)
{
    return learningRate * hiddenError[hiddenPos] * input[i][inputPos];
}

int main(int argc, const char * argv[]) {
    
    srand(NULL);
    readData("titanic.txt");
    initializeWeights();
    float totalError = 0.0;
    int epok = 0;
    while(epok < 100000){
        epok++;
        cout << "Error: " << totalError/1500 << " For epok: " << epok << endl;
        totalError = 0.0;
        for(int i = 0; i < 1500; i++) {
            
            //Calculate output in hidden layer
            for(int j = 0; j < 3; j++){
                hiddenOutputs[j] = calculateHiddenOutput(i, j);
            }
            
            //Calculate output for output layer
            outputOutput = calculateOutputOutput();
            
            //Calculate error in output layer
            outputError = calculateOutputError(i);
            totalError += abs(outputError);
            
            //Calculate error in hidden layer
            for(int j = 0; j < 3; j++) {
                hiddenError[j] = calculateHiddenError(j);
            }
            
            //Calculate new weights from hidden to output
            for(int j = 0; j < 3; j++){
                weightsToOutput[j] += calculateWeightToOutput(j);
            }
            
            //Calculate weights from input to hidden layer
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < 3; k++){
                    weightsToHidden[k][j] += calculateWeightToHidden(i, k, j);
                }
            }
        }
    }
    
    int correct = 0;
    int uncorrect = 0;
    
    for(int i = 1500; i < 2200; i++) {
        
        //Calculate output in hidden layer
        for(int j = 0; j < 3; j++){
            hiddenOutputs[j] = calculateHiddenOutput(i, j);
        }
        
        //Calculate output for output layer
        outputOutput = calculateOutputOutput();
        cout << outputOutput << endl;
        if(expectedOutput[i] == 1.0){
            if(outputOutput >= 0.5)
                correct++;
            else
                uncorrect++;
        } else{
            if(outputOutput < 0.5)
                correct++;
            else
                uncorrect++;
        }
        
    }
    
    cout << "Correct: " << correct << endl;
    cout << "Uncorrect: " << uncorrect << endl;
    
    
    
    
    
    return 0;
}
