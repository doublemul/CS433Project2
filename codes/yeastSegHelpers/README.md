# `quality_measures`
For a given segmented file and the true segmentation calculates a variety of statistics. Notably, 
- Number Fusions:     How many times are multiple true cells predicted as a single cell?
- Number Splits:      How many times is a single true cell split into multiple predicted cell?
- Av. Overshoot:      How many pixels are wrongly predicted to belong to the cell on average
- Av. Undershoot:     How many pixels are wrongly predicted to not belong to the cell on average
- Av. true area:      Average area of cells of truth that are neither split nor fused
- Av. pred area:      Average area of predicted cells that are neither split nor fused
- Nb considered cells:Number of cells that are neither split nor fused

# `segment`

Takes thresholded image, calculates distance transform, uses minima of distance transform within `min_distance` from each other as seeds for watershed algorithm, then runs the watershed algorithm on the specified topology.

The default topology is the negative distance transform, but a custom topology can be given either as array (such as the predictions or a smoothed version thereof), or as a function of the distance transform (can be used to perform watershed on smoothed distance transform which may give good results).

# `tile_helpers`

`tile_image` takes as argument a 2D image and a tilesize, and subdivides the image into tiles. The tiles are created with a large overlap of half the size of the tiles. Hence tiling a 512x256 image into 256 tiles will create 3 images, a 1024x256 image will create 7 images, etc. 

`untile_image` takes as argument a list of tiles and the shape of the original image. It then sticks the tiles together into a single image. This is done such that only the pixel values at the center of every tile is taken into the final image.

This leads to hard jumps from one tile to another, which can give problems. If this is the case, the tile borders will have to be smoothed. 

