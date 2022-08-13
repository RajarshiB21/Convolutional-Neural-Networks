#Importing the libraries
import torchmetrics

##Calculating Metrics
def calc_metrics(targets, predictions):
    #We will be using the pytorch metrics library to compute the precision, recall and f1_score
    #The functions take the predictions and the targets, with various other parameters
    #multiclass = True since we have multiple classes in our dataset

    #Precision
    #TP / TP + FP
    #Micro = Calculate the metric globally, across all samples and classes

    precision = torchmetrics.functional.precision(predictions, targets, average= 'micro', multiclass = True )

    #Recall
    #TP / TP + FN
    # Micro = Calculate the metric globally, across all samples and classes
    recall = torchmetrics.functional.recall(predictions, targets, average= 'micro', multiclass = True)

    #F1 Score
    #Threshold for transforming probability or logit predictions to binary
    #Micro =  Calculate the metric globally, across all samples and classes
    f1_score = torchmetrics.functional.f1_score(predictions, targets, threshold=0.5, average= 'micro', multiclass = True)
    return precision, recall, f1_score