# u-net-sentinel

## Intro

The main goal of this project is to facilitate the semantic segmentation
of Sentinel-2 satelite imagery.

## Architecture

We use the U-Net architecture, backed by the VGG architecture.

## Running the code

1. Clone the code
2. Ensure you have the `/data` folder populated with master `image.npy` and `mask.npy`
   (this should be the case if you cloned the repo)
3. Create a `venv` by `python3 -m venv venv` and activate it by `source venv/bin/activate`
4. Install necessary dependencies by running `pip install -r requrements.txt`
5. Run the `python train.py` to start the training of the U-Net network

## Further work

The code currently present in this repo is just an introductory step to get familiar with
U-Net architecture in the context of remote sensing. To further improve these solutions
the following is necessary:

1. Pre-train the network by using the VGG weights obtained from Sentinel-2 classification
2. Optimize the code to run on GPU, TPU and other similar (non-CPU) environments, in
   order to facilitate the large volume of training data
3. Explore different ground-truth segmentation masks combined with Sentinel-2 images
