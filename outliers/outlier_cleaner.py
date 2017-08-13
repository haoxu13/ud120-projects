#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    ### return error tuple (age, net_worth, error)
    ### then discard 10% of the largest residual error
    residual_error = map(lambda (x1, x2, x3):(x1, x2, (x2-x3)**2), zip(ages, net_worths, predictions))
    residual_error.sort(key = lambda x: x[2])

    cleaned_data = residual_error[:(int)(round(0.9*len(predictions)))]
    return cleaned_data

