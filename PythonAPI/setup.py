from distutils.core import setup
import os
import urllib2

setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir = {'pycocotools': 'pycocotools'},
      version='1.0',
      )


# download data for Microsoft coco dataset
def download(url, dst):
   '''
   command line progress bar tool
   :param url: url link for the file to download
   :param dst: dst file name
   :return:
   '''
   if os.path.exists(dst):
       print 'File downloaded'
       return 1

   u = urllib2.urlopen(url)
   f = open(dst, 'wb')
   meta = u.info()
   file_size = int(meta.getheaders("Content-Length")[0])
   print "Downloading: %s Bytes: %s " % (dst, file_size)

   file_size_dl = 0
   block_sz = 8192
   while True:
       buffer = u.read(block_sz)
       if not buffer:
           break

       file_size_dl += len(buffer)
       f.write(buffer)
       status = r"%10d  [%3.2f%%] " % (file_size_dl, file_size_dl * 100. / file_size)
       status = status + chr(8)*(len(status)+1)
       print status,
   f.close()

def query_yes_no(question, default="yes"):
   """Ask a yes/no question via raw_input() and return their answer.

   "question" is a string that is presented to the user.
   "default" is the presumed answer if the user just hits <Enter>.
       It must be "yes" (the default), "no" or None (meaning
       an answer is required of the user).

   The "answer" return value is one of "yes" or "no".
   """
   valid = {"yes": True, "y": True, "ye": True,
            "no": False, "n": False}
   if default is None:
       prompt = " [y/n] "
   elif default == "yes":
       prompt = " [Y/n] "
   elif default == "no":
       prompt = " [y/N] "
   else:
       raise ValueError("invalid default answer: '%s'" % default)

   while True:
       print question + prompt
       choice = raw_input().lower()
       if default is not None and choice == '':
           return valid[default]
       elif choice in valid:
           return valid[choice]
       else:
           print  "Please respond with 'yes' or 'no' (or 'y' or 'n')."

# create directory folder
if not os.path.exists('../images'):
   print 'creating ../images to host images in Microsoft COCO dataset...'
   os.mkdir('../images')
   print 'done'
if not os.path.exists('../annotations'):
   print 'creating ../annotations to host annotations in Microsoft COCO dataset'
   os.mkdir('../annotations')
   print 'done'


print "The following steps help you download images and annotations."
print "Given the size of zipped image files, manual download is recommended at http://mscoco.org/download"
# download train images
if query_yes_no("Do you want to download zipped training images [13GB] under ./images?", default='no'):
   url = 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
   download(url, '../images/train2014.zip')

# download val images
if query_yes_no("Do you want to download zipped validation images [6.2GB] under ./images?", default='no'):
   url = 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'
   download(url, '../images/val2014.zip')

# download annotations
for split in ['train', 'val']:
   for anno in ['instances', 'captions']:
       # download annotations
       if   split == 'train'    and anno == 'instances':
           size = '367'
       elif split == 'val'      and anno == 'instances':
           size = '178'
       elif split == 'train'    and anno == 'captions':
           size = '61'
       elif split == 'val'      and anno == 'captions':
           size = '30'

       if query_yes_no("Do you want to download %s split for %s annotations [%sMB] under ./annotations?"%(split, anno, size), default='yes'):
           fname = '../annotations/%s_%s2014.json'%(anno, split)
           url = 'http://msvocds.blob.core.windows.net/annotations-1-0-2/%s_%s2014.json'%(anno, split)
           download(url, fname)
