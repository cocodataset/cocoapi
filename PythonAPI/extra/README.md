


# MASKS_PARSER.PY



## Notes:

- This program performs a .JSON segmentation masks parsing and applies conversion formats: RLE <-> Polygon.
- Note #1: Conversion only applied to COCO-format annotations "iscrowd==1"
- Note #2: Script only tested over COCO dataset formats
- Note #3: This script overwrites input dataset with same name as input (to avoid redundancy)
- Note #4: Attached sample dataset is a sub-sample of COCO Dataset 2017 for 1 single image with one (only) multi-object annotation (is_crowd= 1). See:  http://images.cocodataset.org/zips/val2017.zip
- Author: Aleix Figueres (afigueres@fluendo.com)

 

## Instructions:

1. Go to "extra" (source):

  	`cd ./extra
2. Modify .JSON dataset input path on "main" function:

  	 ` # Input dataset path
  	 DATASET_PATH = './samples/instances_val2017_1img-sample.json'`

3. Comment/uncomment desired conversion format:

  	` # SAMPLE #1: Convert RLE to Polygon
  	RLE2poly(DATASET_PATH) `<br>
  	` # SAMPLE #2: Convert Polygon to RLE
  	poly2RLE(DATASET_PATH) `

- Note 5: when running both (RLE2poly + poly2RLe) output will not be identical (bit-a-bit) to input dataset due to encoding data loosing but after comparing output vs input masks, quite-identical shapes got obtained

3. Run:

    `python masks_parser.py`

4. Check outputs on "DATASET_PATH"

