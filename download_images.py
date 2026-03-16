from bing_image_downloader import downloader

keywords = ["building workers with helmet", "building workers without helmet"]

for kw in keywords:
    downloader.download(kw, limit=50,  output_dir='dataset_new', adult_filter_off=True, force_replace=False, timeout=60)