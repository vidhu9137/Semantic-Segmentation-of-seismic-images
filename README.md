# Semantic-Segmentation-of-seismic-images
Using U-net to semantically segment the pixels that corresponds to salt from the seismic images. Salt domes are highly pressured geographical structures that are needed to be avoided or taken well care off, while drilling. Accurate positions of salt domes will assist field personnels while drilling operations.
The dataset has been taken from Kaggle that contains 4000 seismic images and their masks (which shows regions of salt as white and remaining as black).
U-net model is trained on 90% of dataset and it predicted for remaining 10% of images. Owing to large test size of images, results for some of them are displayed.
During training, val_loss and accuracy is calculated to make sure that model doesn't underfit or overfit. 
