from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
import logging, json

'''
Adapted from
https://github.com/ashnair1/COCO-Assistant
'''

class CatRemapper:
    def __init__(self, cat1, cat2):
        """
        Target category list needs to be remapped to
        reference category list.
        The following are the three cases to be considered:
        === "Case 1"
            When cat1 and cat2 overlap and cat2 has no new categories
            ```
            cat1: [{'id': 1, 'name': 'A'},
                   {'id': 2, 'name': 'B'},
                   {'id': 3, 'name': 'C'}]
            cat2: [{'id': 1, 'name': 'B'},
                   {'id': 2, 'name': 'C'},
                   {'id': 3, 'name': 'A'}]
            result: [{'id': 1, 'name': 'A'},
                     {'id': 2, 'name': 'B'},
                     {'id': 3, 'name': 'C'}]
            with
            overlap: {1:2, 2:3, 3:1}
            newcat : {}
            ```
        === "Case 2"
            When cat1 and cat2 overlap & cat2 has new categories.
            ```
            cat1: [{'id': 1, 'name': 'A'},
                   {'id': 2, 'name': 'B'},
                   {'id': 3, 'name': 'C'}]
            cat2: [{'id': 1, 'name': 'B'},
                   {'id': 2, 'name': 'A'},
                   {'id': 3, 'name': 'F'}]
            result: [{'id': 1, 'name': 'A'},
                     {'id': 2, 'name': 'B'},
                     {'id': 3, 'name': 'C'},
                     {'id': 4, 'name': 'F'}]
            with
            overlap: {1:2, 2:1}
            newcat : {3:4}
            ```
        === "Case 3"
            When cat1 and cat2 don't overlap
            ```
            cat1: [{'id': 1, 'name': 'A'},
                   {'id': 2, 'name': 'B'},
                   {'id': 3, 'name': 'C'}]
            cat2: [{'id': 1, 'name': 'D'},
                   {'id': 2, 'name': 'E'},
                   {'id': 3, 'name': 'F'}]
            result: [{'id': 1, 'name': 'A'},
                     {'id': 2, 'name': 'B'},
                     {'id': 3, 'name': 'C'},
                     {'id': 4, 'name': 'D'},
                     {'id': 5, 'name': 'E'},
                     {'id': 6, 'name': 'F'}]
            with
            overlap: {}
            newcat : {1:4, 2:5, 3:6}
            ```
        Args:
            cat1 (list[dict]): Reference category list of dicts
            cat2 (list[dict]): Target category list of dicts
        """

        self.refcat = cat1
        for c in self.refcat:
            c["name"] = c["name"].lower()

        self.cat1 = self.result = self.generate_id_map(cat1)
        self.cat2 = self.generate_id_map(cat2)

    def remap(self, ann):
        rescat, overlap_dict, newcat_dict = self.remap_cats()
        # Modify annotations
        new_ann = self.remap_annotations(ann, overlap_dict, newcat_dict)
        return rescat, new_ann

    def remap_cats(self):
        """
        Check if the reference and target category lists overlap
        and remap them as necessary.
        Returns:
            tuple:
            - result (list[dict]): New category list
            - overlap_dict (dict): Mapping of overlapping categories
            - newcat_dict (dict): Mapping of new categories
        """
        c1 = list(self.cat1.keys())
        c2 = list(self.cat2.keys())
        diff = sorted(list(set(c2) - set(c1)))

        newcat_dict = {}
        overlap_dict = {}
        res = []
        if diff:
            # New categories
            for i, d in enumerate(diff):
                newcat_dict[self.cat2[d]] = len(c1) + i + 1
                self.result[d] = len(c1) + i + 1
                del self.cat2[d]

            for k, v in self.result.items():
                res.append({"id": v, "name": k})

        for k, v in self.cat1.items():
            try:
                overlap_dict[self.cat2[k]] = v
            except KeyError:
                continue

        result = res or self.refcat
        return result, overlap_dict, newcat_dict

    def remap_annotations(self, ann, overlaps, new_cats):
        """
        Remaps the annotation where overlapping category ids
        are mapped to reference category ids and new categories
        are given new ids.
        Args:
            ann (dict): Annotation being modified
            overlaps (dict): Mapping of overlapping categories
            new_cats (dict): Mapping of new categories
        Returns:
            ann: Remapped annotation
        """

        for a in ann:
            cat_id = a["category_id"]
            # The category is either new or exists in the reference annotation
            if cat_id in new_cats:
                a["category_id"] = new_cats[a["category_id"]]
            else:
                a["category_id"] = overlaps[a["category_id"]]
        return ann

    def generate_id_map(self, cat):
        """Generate category id name mapping
        Args:
            cat (list[dict]): Category list
        Returns:
            dict: Category id name mapping
        """
        c = [[i["name"].lower(), i["id"]] for i in cat]
        keys = [i[0] for i in c]
        values = [i[1] for i in c]
        return dict(zip(keys, values))

class COCO_Combinator:
    '''
    TODO: 
    AssertionError: Results do not correspond to current coco set
    Probably because same images are still processed as separate entities, so things screw up
    '''
    def __init__(self, ann_dir: str, merge_name: str=None):
        self.ann_dir = Path(ann_dir)
        self.annfiles = [COCO(i) for i in list(set(self.ann_dir.glob('*.json')) - set(self.ann_dir.glob('exclude_*.json')))]
        self.anndict = dict(enumerate(self.annfiles, 0))
        self.merge_name = merge_name

    def merge(self):
        """
        Merge multiple coco datasets
        """

        # resann_dir = self.dh.create("merged/annotations")

        cann = {"images": [], "annotations": [], "info": None, "licenses": None, "categories": None}

        # logging.debug("Merging Annotations...")

        dst_ann = self.ann_dir / "exclude_merged.json" if not self.merge_name else "exclude_" + self.merge_name

        print("Merging annotations")
        for ind in tqdm(self.anndict):
            cj = self.anndict[ind].dataset

            # Check if this is the 1st annotation.
            # If it is, continue else modify current annotation
            if ind == 0:
                cann["images"] = cann["images"] + cj["images"]
                cann["annotations"] = cann["annotations"] + cj["annotations"]
                if "info" in list(cj.keys()):
                    cann["info"] = cj["info"]
                if "licenses" in list(cj.keys()):
                    cann["licenses"] = cj["licenses"]
                cann["categories"] = sorted(cj["categories"], key=lambda i: i["id"])

                last_imid = cann["images"][-1]["id"]
                last_annid = cann["annotations"][-1]["id"]

                # If last imid or last_annid is a str, convert it to int
                if isinstance(last_imid, str) or isinstance(last_annid, str):
                    logging.debug("String Ids detected. Converting to int")
                    id_dict = {}
                    # Change image id in images field
                    for i, im in enumerate(cann["images"]):
                        id_dict[im["id"]] = i
                        im["id"] = i

                    # Change annotation id & image id in annotations field
                    for i, im in enumerate(cann["annotations"]):
                        im["id"] = i
                        if isinstance(last_imid, str):
                            im["image_id"] = id_dict[im["image_id"]]

                last_imid = max(im["id"] for im in cann["images"])
                last_annid = max(ann["id"] for ann in cann["annotations"])

            else:

                id_dict = {}
                # Change image id in images field
                for i, im in enumerate(cj["images"]):
                    id_dict[im["id"]] = last_imid + i + 1
                    im["id"] = last_imid + i + 1

                # Change annotation and image ids in annotations field
                for i, ann in enumerate(cj["annotations"]):
                    ann["id"] = last_annid + i + 1
                    ann["image_id"] = id_dict[ann["image_id"]]

                # Remap categories
                cmapper = CatRemapper(cann["categories"], cj["categories"])
                cann["categories"], cj["annotations"] = cmapper.remap(cj["annotations"])

                cann["images"] = cann["images"] + cj["images"]
                cann["annotations"] = cann["annotations"] + cj["annotations"]
                if "info" in list(cj.keys()):
                    cann["info"] = cj["info"]
                if "licenses" in list(cj.keys()):
                    cann["licenses"] = cj["licenses"]

                last_imid = cann["images"][-1]["id"]
                last_annid = cann["annotations"][-1]["id"]

        with open(dst_ann, "w") as aw:
            json.dump(cann, aw)
        
        return dst_ann


# test = COCO_Combinator('.')
# test.merge()
