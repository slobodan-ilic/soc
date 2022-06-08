# encoding: utf-8
"""Home of the code for downloading Slovenia LULC patches and Sentinel MS images."""

from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
    geo_utils as gu,
)

# from tqdm import tqdm


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


class SentinelDownloader:
    def __init__(self, lon, lat, start_date, end_date, resolution=10, size=64):
        self._lon = lon
        self._lat = lat
        self._start_date = start_date
        self._end_date = end_date
        self._resolution = resolution
        self._size = size
        self._crs = gu.get_utm_crs(lon, lat)

    def download(self):
        half = self._size / 2
        res = self._resolution
        lon, lat = gu.wgs84_to_utm(self._lon, self._lat, self._crs)
        bbox = [
            lon - (half - 1) * res,
            lat - (half - 1) * res,
            lon + (half + 1) * res,
            lat + (half + 1) * res,
        ]
        bbox = BBox(bbox=bbox, crs=self._crs)
        size = bbox_to_dimensions(bbox, resolution=self._resolution)
        request_all_bands = SentinelHubRequest(
            evalscript=EVALSCRIPT_ALL_BANDS,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=(self._start_date, self._end_date),
                    mosaicking_order="leastCC",
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=shc,
        )
        all_bands_response = request_all_bands.get_data()
        return all_bands_response[0]


if __name__ == "__main__":
    sd = SentinelDownloader(
        13.324244526381801, 46.095319660427400, "2020-06-12", "2020-07-01"
    )
    sd.download()
