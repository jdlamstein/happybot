# FACS_recognition
This repository applies a neural network to frontal face image sequences to find facial Action Units (AUs). AUs, part of the Facial Action Coding System ([FACS](https://en.wikipedia.org/wiki/Facial_Action_Coding_System)), are a compelling way to analyze facial expressions, as the basic emotions (happiness, sadness, anger, fear, surprise, disgust, and contempt) have AUs in common. Paul Ekman, who established FACS, ran studies showing that people in isolated, rural communities recognize facial expressions just as people in cities recognize facial expressions. The [research](http://psycnet.apa.org/record/1971-07999-001) suggests some facial expressions are universal. 

## Motivation
Many businesses want to provide something to a customer that will make them happier, but we don't have good ways to measure how happy a person is. I wanted to get a taste of how computer vision could identify a person's feeling. Computer vision is a deep field, and I finish my ambitious project wiser and hopeful. 

I was happy that my background in physics and programming for lab provided me with sufficient tools to learn the novel aspects of this project. Many algorithms were new to me and forums, books, and papers guided me in the right direction. 

## Methods and Background
I used the [CK+ Dataset](http://www.pitt.edu/~emotion/ck-spread.htm) because it was available and had labelled AUs and some labelled basic emotions. The CK+ Dataset consists of 593 image sequences of 123 subjects. All sequences start at neutral facial expression and end at peak facial expression. All sequences are coded with AUs and corresponding intensities. 327 sequences are labelled for basic emotions. 



## Evaluating Accuracy
