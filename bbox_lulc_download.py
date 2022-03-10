import os
from wget import download
from patches import patches


class Downloader:
    def __init__(self, url, file_bbox, file_lulc, patches):
        self.url = url
        self.file_bbox = file_bbox
        self.file_lulc = file_lulc
        self.patches = patches

    def bar_progress(current, total, width=80):
        '''Downloading process monitor'''
        print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))

    def download_file(self, url, location=""):
        '''Download files with progress_bar. download_file method accepts the url and the file storage location. File storage location is an empty string by default.'''
        download(url, out = location, bar=Downloader.bar_progress)

    def download_patches(self):
        '''Creates a folder and paths for downloading files bbox and lulc '''

        for patch in self.patches:
            bbox = os.path.join(self.url, patch, self.file_bbox)
            lulc = os.path.join(self.url, patch, self.file_lulc)

            if not os.path.exists(patch):
            
                try:
                    os.makedirs(patch, mode=0o777)
                    self.download_file(bbox, patch)
                    self.download_file(lulc, patch)

                except FileExistsError:
                    pass
        
if __name__ == "__main__":
    file_bbox = "bbox.pkl.gz"
    file_lulc = "mask_timeless/LULC.npy.gz"
    url = "http://eo-learn.sentinel-hub.com.s3.eu-central-1.amazonaws.com/eopatches_slovenia_2019/"

    download_object = Downloader(url, file_bbox, file_lulc, patches)
    download_object.download_patches()
    