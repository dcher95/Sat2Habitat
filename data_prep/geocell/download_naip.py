import os
import errno
import urllib.request
from multiprocessing import Pool
import csv
import sys
import signal
import code
import tqdm
#
# download aerial images
#
def timeout_handler(signum, frame):
    raise TimeoutError("Program timed out")

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i==0:
                continue
            longitude = float(row[1])
            latitude = float(row[2])
            index = row[0]
            data.append([latitude, longitude, index])
    return data

def ensure_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise

def download(job):
  url = job[0]
  out_file = job[1]

  if not os.path.isfile(out_file):
    ensure_dir(out_file)
    try:
      urllib.request.urlretrieve(url, out_file)
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code)
        raise
    except Exception as e:
        print("Other Error:", e)



if __name__ == '__main__':
  signal.signal(signal.SIGALRM, timeout_handler)
  signal.alarm(16500)

  out_dir = "/data/cher/universe7/herbarium/data/naip"

  # input_file = f"/data/cher/universe7/herbarium/data/geocell/grid_test.csv"
  input_file = f"/data/cher/universe7/herbarium/data/geocell/grid_0.01deg_pt.csv"
  data = read_csv_file(input_file)

  jobs = []
  #template_url = "https://basemap.nationalmap.gov/arcgis/services/USGSImageryOnly/MapServer/WMSServer?service=WMS&version=1.1.1&request=GetMap&layers=0&styles=&width=512&height=512&srs=EPSG:4326&bbox=-90.00,40.00,-89.99,40.01&format=image/png"
  #template_url = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/%%s/%d?mapSize=%s&key=%s" % (zoom_level, im_size, api_key)

  print(data)
  for lat, lon, index in tqdm.tqdm(data):
    print(lat, lon, index)
    image_url = f"https://basemap.nationalmap.gov/arcgis/services/USGSImageryOnly/MapServer/WMSServer?service=WMS&version=1.1.1&request=GetMap&layers=0&styles=&width=256&height=256&srs=EPSG:4326&bbox={lon-0.005},{lat-0.005},{lon+0.005},{lat+0.005}&format=image/png"
    print(image_url)
    out_file = f'{out_dir}/{index}.png'
    jobs.append((image_url, out_file))

  p = Pool(1)
  p.map(download, jobs)