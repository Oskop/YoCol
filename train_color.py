from utils.color_recognition_api import color_histogram_feature_extraction as color_extractor
from utils.color_recognition_api import knn_classifier as knn
print('color training data is being created....')
open('./data/training.data', 'w')
color_extractor.training()
print('color training data is ready')
