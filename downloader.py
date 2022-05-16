# encoding: utf-8
"""Home of the code for downloading Slovenia LULC patches and Sentinel MS images."""

import gzip
import os
import pickle
import shutil

import cv2
import numpy as np
from sentinelhub import (
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from tqdm import tqdm
from wget import download

from patches import patches as PATCHES

shc = SHConfig()

shc.sh_client_id = "24550f2c-330b-4dff-ab0f-d4f095b855ac"
shc.sh_client_secret = "|g>;cD:N3CfP<hkq-CPh6n2*xC<}57*xlr2!C?e~"
shc.save()


EVALSCRIPT_ALL_BANDS = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: [
                    "B01", "B02", "B03", "B04", "B05",
                    "B06", "B07", "B08", "B09", "B10",
                    "B11", "B12", "B8A"
                ],
                units: "DN"
            }],
            output: {
                bands: 13,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [
            sample.B01, sample.B02, sample.B03, sample.B04, sample.B05,
            sample.B06, sample.B07, sample.B08, sample.B09, sample.B10,
            sample.B11, sample.B12, sample.B8A
        ];
    }
"""

FILENAME_BBOX_GZ = "bbox.pkl.gz"
FILENAME_LULC_GZ = "mask_timeless/LULC.npy.gz"
FILENAME_LULC = "LULC.npy"
URL = (
    "http://eo-learn.sentinel-hub.com."
    "s3.eu-central-1.amazonaws.com/eopatches_slovenia_2019/"
)


class Downloader:
    def __init__(self, year, resolution=10):
        self._year = year
        self._resolution = resolution

    def bar_progress(current, total, width=80):
        """Downloading process monitor"""
        print(
            "Downloading: %d%% [%d / %d] bytes"
            % (current / total * 100, current, total)
        )

    def save_image(self, location, bbox, start_date, end_date):
        id_size = bbox_to_dimensions(bbox, resolution=self._resolution)
        request_all_bands = SentinelHubRequest(
            evalscript=EVALSCRIPT_ALL_BANDS,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=(start_date, end_date),
                    mosaicking_order="leastCC",
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=id_size,
            config=shc,
        )
        img = request_all_bands.get_data()[0]
        month = start_date.split("-")[1]
        np.save(f"{location}/{month}-img.npy", img)
        tc_img = np.array(img[:, :, 1:4] * 3.5 / 1e4 * 255, dtype="uint8")
        cv2.imwrite(f"{location}/{month}-img.png", tc_img)

    def download_patches(self):
        """Creates a folder and paths for downloading files bbox and lulc"""

        for patch in tqdm(PATCHES):
            location = f"./data/{patch}"
            if os.path.exists(location):
                continue

            # ---Download resources from web---
            bbox_url = os.path.join(URL, patch, FILENAME_BBOX_GZ)
            lulc_url = os.path.join(URL, patch, FILENAME_LULC_GZ)
            os.makedirs(location, mode=0o777)
            download(bbox_url, out=location, bar=None)
            download(lulc_url, out=location, bar=None)

            # ---Extract LULC file and delete the original GZ---
            lulc_gzip_filename = os.path.join(location, f"{FILENAME_LULC}.gz")
            with gzip.open(lulc_gzip_filename, "rb") as f:
                lulc = np.load(f)

            # ---Only save the relevant LULC data, with enough pixels defined
            if np.sum(np.squeeze(lulc) == 0) > np.prod(lulc.shape) // 2:
                shutil.rmtree(location)
                continue

            np.save(os.path.join(location, FILENAME_LULC), lulc)
            os.remove(lulc_gzip_filename)

            # ---Fetch MS image and save as MS and TC---
            bbox_filename = os.path.join(location, FILENAME_BBOX_GZ)
            with gzip.open(bbox_filename, "rb") as f:
                bbox = pickle.load(f)

            for i in range(12):
                month = i + 1
                start_date = f"{self._year}-{month}-1"
                if month != 12:
                    end_date = f"{self._year}-{month+1}-1"
                else:
                    end_date = f"{self._year + 1}-1-1"
                self.save_image(location, bbox, start_date, end_date)


if __name__ == "__main__":
    year = 2019
    downloader = Downloader(year)
    downloader.download_patches()
