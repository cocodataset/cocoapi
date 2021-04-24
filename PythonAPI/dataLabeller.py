from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
import cv2 as cv


# Import annotations (check)
# Create loop to loop through images (check)
# Show image (check)
# Allow going backwards (check)
# Modify label category (check)
# Save modified annotations (check)
# Save progress
# Save weird images
# Print progress (check)

def box_xywh_to_xyxy(x):
    # Converts bounding boxes to (x1, y1, x2, y2) coordinates of top left and bottom right corners
    x_c, y_c, w, h = x
    b = [(x_c), (y_c),
        (x_c + w), (y_c + h)]
    box = list(map(int, map(round, b)))
    return box

def save_dataset(original_filepath, target_filepath, anns, cats):

    # Load dataset val to get structure
    orig_file = json.load(open(original_filepath, 'r'))

    # Make final dictionary
    dataset = dict.fromkeys(orig_file.keys())
    dataset['info'] = orig_file['info']
    dataset['licenses'] = orig_file['licenses']
    dataset['categories'] = cats
    dataset['annotations'] = anns
    dataset['images'] = orig_file['images']

    with open(target_filepath + '.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print('Saved dataset {}.json to disk!'.format(target_filepath))


if __name__ == "__main__":
    cat_show = [10, 92, 93, 94]  # Categories ids that you want shown and relabelled

    # Annotations file
    dataDir = "."
    dataType = "valTraffic"
    annDir = "./annotations/"
    annFile = "{}instances_{}.json".format(annDir, dataType)

    # Save file
    saveName = "instances_valTrafficRelabelled"
    saveFile = annDir + saveName

    # Images folder
    imgDir = "images/valTraffic/"

    # Import from annotations file
    coco=COCO(annFile)

    cats = coco.loadCats(coco.getCatIds())

    if len(cats) == 80:
        cats.append({'supercategory': 'outdoor', 'id': 92, 'name': 'traffic_light_red'})
        cats.append({'supercategory': 'outdoor', 'id': 93, 'name': 'traffic_light_green'})
        cats.append({'supercategory': 'outdoor', 'id': 94, 'name': 'traffic_light_other'})
    else:
        if cats[80]['name'] != 'traffic_light_red' or \
            cats[81]['name'] != 'traffic_light_green' or \
            cats[82]['name'] != 'traffic_light_other':
            raise Exception("Error: Categories mismatched. Check categories to make sure the 80th category is traffic_light_red")
    
    nms = [cat['name'] for cat in cats]
    catId_to_catName = {cats[x]['id']: cats[x]['name'] for x in range(len(cats))}

    # Load image Ids
    catIds = coco.getCatIds(catNms=nms)
    imgIdsAll = coco.getImgIds()
    imgIds = imgIdsAll
    print('Number of images: ' + str(len(imgIds)))

    # Load annotations
    annIds = coco.getAnnIds(imgIds)
    anns = coco.loadAnns(annIds)

    #save_dataset(annFile, "blah", anns)

    # Show image
    annId_i = 0
    go_backwards = False
    ann_counter = 0 # To tell user how many annotations are left
    
    while annId_i < len(annIds):

        if annId_i < 0:
            print("You have reached the beginning")
            annId_i = 0
            go_backwards = False

        ann = anns[annId_i]
        imgId = ann['image_id']

        if ann['category_id'] in cat_show:
            go_backwards = False
            ann_counter += 1
            print()

            image = cv.imread(imgDir + (str(imgId)+'.jpg').zfill(16))
            if image is None:
                raise Exception("Error: Cannot find image {}".format(imgDir + (str(imgId)+'.jpg').zfill(16)))
            
            # Give progress status
            if ann_counter >= 50:
                print("You are on annotation {} / {}".format(str(annId_i + 1), str(len(anns))))
                ann_counter = 0

            b = ann['bbox']
            
            # Convert bounding boxes to (x1, y1, x2, y2)
            box = box_xywh_to_xyxy(b)
            print("Current label: " + catId_to_catName[ann['category_id']])

            # Display bounding box
            image_bboxed = cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (252, 3, 219), 2)
            cv.imshow((str(imgId)+'.jpg'), image)
            if cv.waitKey(1) == ord("q"):
                break
            
            # Ask user for command
            while True:
                inp = str(input("Input q to quit, z to go backwards, and nothing to skip\n")).rstrip().lower()
                if inp == "":
                    break
                elif inp == "q":
                    exit()
                elif inp == "z":
                    go_backwards = True
                    break
                elif inp == "save":
                    save_dataset(annFile, saveFile, anns, cats)
                elif (inp == "r") or (inp == "1"):
                    print("Changed category id to traffic_light_red")
                    anns[annId_i]['category_id'] = 92
                    break
                elif (inp == "g") or (inp == "2"):
                    print("Changed category id to traffic_light_green")
                    anns[annId_i]['category_id'] = 93
                    break
                elif (inp == "o") or (inp == "3"):
                    print("Changed category id to traffic_light_other")
                    anns[annId_i]['category_id'] = 94
                    break
                else:
                    print("Invalid command")

            cv.destroyAllWindows()


        # Update image and annotation indices
        if go_backwards == False:
            annId_i += 1
        else:
            annId_i -= 1


    print("Completed image labelling")
    
    while True:
        inp = str(input("Save?\n")).rstrip().lower()
        if inp in ['yes', 'y']:
            save_dataset(annFile, "blah", anns, cats)
            print("Labels Saved")
            exit()
        elif inp in ['no', 'n']:
            inp = str(input("Are you sure?\n")).rstrip().lower()
            if inp in ['yes', 'y']:
                print("Labels not saved")
                exit()


