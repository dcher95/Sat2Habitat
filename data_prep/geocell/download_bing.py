import os
import errno
import urllib.request
from multiprocessing import Pool
import csv
import sys
import code
import signal
import time
from datetime import datetime

#
# download aerial images
# 

# Define the timeout handler function
def timeout_handler(signum, frame):
    print("Timeout reached! Signal received:", signum)
    raise TimeoutError("Process exceeded time limit")

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
  print(datetime.fromtimestamp(time.time()))
  # Set the signal handler for SIGALRM
  signal.signal(signal.SIGALRM, timeout_handler)

  # Set an alarm for 50000 seconds (approximately 13.9 hours)
  signal.alarm(2100)

  out_dir = "/data/cher/Sat2Habitat/data/patch-imagery/bing_train/"

  input_file = "/data/cher/Sat2Habitat/data/cluster-data-split/train_imagery.csv"  
  data = read_csv_file(input_file)
  zoom_level=18
  # settings
  im_size = "256,256"
  api_key = ""

  jobs = []

  template_url = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/%%s/%d?mapSize=%s&key=%s" % (zoom_level, im_size, api_key)
  template_fn = "%s%d/%%d/%%d/%%s_%%s.jpg" % (out_dir, zoom_level)
  for lat, lon, key in data[36000:48000]:
    # if already exists -- skip
    if f'{key}.jpg' in os.listdir(out_dir):
      continue

    tmp_loc = "%s,%s" % (lat, lon)
    image_url = template_url % (tmp_loc)
    out_file = f'{out_dir}/{key}.jpg'
    jobs.append((image_url, out_file))

  print(datetime.fromtimestamp(time.time()))
  p = Pool(1)
  p.map(download, jobs)
  print(datetime.fromtimestamp(time.time()))