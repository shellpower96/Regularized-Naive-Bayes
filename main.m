addpath(genpath('./lib'));
clear all;
close all
%% load data
name = './datasets/iris.csv';
disp(name);

dataset = load(name);
%% get features and class label
feat = dataset(:,1:end-1);
label = dataset(:,end);

rest = RNB(feat,label);