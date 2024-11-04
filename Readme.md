the project includes four parts:
the first part is data process: extract the 3300 dataset according to RandomForest doing binary classification:every disease and control group
, then compute the union of the diseases of 3300 as the columns of 3300 dataset from 10477 dataset to train NFM,MLP,ConvMLP
:file extract..., file generate... are related to RandomForest and finally generate the features of 3300


the second part is model trained based on the dataset 3300 dataset


the third part is to compute the top K genes for every model according to the improved simplified model
in this project,we compute top 20,30 and 50 ,and discover when K equals to 30, NFM reach the best, and to other models, k equals 30
:file X_MLP_800..,X_NFM_800.., Y_.__800.. etc., are these models. they includes the functions of the second and the third


the fourth part is compute intersection and evaluate it: we compute the intersection 20, 50, 20_30(NFM:20, OTHERS:20)
: X_intersection_... to compute intersection and evaluate them

the project is developed by jupyter notebook, you can open the file according to the filename, and click every cell from one cell to cell
if you have some questions, please contect me: alice20110324@126.com

the project is from the manuscript: Fang Zheng, Jianbo Qing, Yan Qiang, Yafeng Li, Zihang Yuan, Yaheng Li, Yan Geng, Juanjuan Zhao. Target Identification Using an Improved Interpretable LRP Model and Intersection Selection Idea. 

# target-identification-LRP-intersection-pytorch
# target-identification-LRP
