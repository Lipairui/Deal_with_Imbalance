from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek

def under_sampling(X,y,method):
  if method=='ClusterCentroids':
    model = ClusterCentroids()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='RandomUnderSampler':
    model = RandomUnderSampler()
    X_resampled, y_resampled = model.fit_resample(X, y)  
  elif method=='NearMiss':
    model = NearMiss()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='EditedNearestNeighbours':
    model = EditedNearestNeighbours()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='RepeatedEditedNearestNeighbours':
    model = RepeatedEditedNearestNeighbours()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='AllKNN':
    model = AllKNN()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='NeighbourhoodCleaningRule':
    model = NeighbourhoodCleaningRule()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='OneSidedSelection':
    model = OneSidedSelection()
    X_resampled, y_resampled = model.fit_resample(X, y)
  return X_resampled, y_resampled
  
def over_sampling(X,y,method):
  if method=='SMOTE':
    model = SMOTE()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='ADASYN':
    model = ADASYN()
    X_resampled, y_resampled = model.fit_resample(X, y)  
  elif method=='RandomOverSampler':
    model = RandomOverSampler()
    X_resampled, y_resampled = model.fit_resample(X, y) 
  return X_resampled, y_resampled
  
def combine_sampling(X,y,method):
  if method=='SMOTEENN':
    model = SMOTE()
    X_resampled, y_resampled = model.fit_resample(X, y)
  elif method=='SMOTETomek':
    model = SMOTETomek()
    X_resampled, y_resampled = model.fit_resample(X, y)  
  return X_resampled, y_resampled   
