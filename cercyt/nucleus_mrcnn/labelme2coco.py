import collections
import argparse
import glob
import os
import sys
import os.path as osp
import datetime
import json
import numpy as np
import PIL.Image
import PIL.ImageDraw
import math
import pycocotools.mask


def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', default=r'Y:\Users\Jie\CerCyt\BD_cytology_25\nuclei_segmentation\normal-abnormal', help='input annotated directory')
    parser.add_argument('--output_dir', default=r'Y:\Users\Jie\CerCyt\BD_cytology_25\nuclei_segmentation\normal-abnormal_coco', help='output dataset directory')
    parser.add_argument('--labels', default=r'Y:\Users\Jie\CerCyt\BD_cytology_25\nuclei_segmentation\labels.txt', help='labels file')
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    print('Creating dataset:', args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    out_ann_file = osp.join(args.output_dir, 'annotations.json')
    label_files = glob.glob(osp.join(args.input_dir, '*.json'))
    for image_id, label_file in enumerate(label_files):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = osp.join(
            args.output_dir, 'JPEGImages', base + '.jpg'
        )

        img_file = osp.join(
            osp.dirname(label_file), label_data['imagePath']
        )
        img = np.asarray(PIL.Image.open(img_file))
        PIL.Image.fromarray(img).save(out_img_file)
        data['images'].append(dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=image_id,
        ))

        for shape in label_data['shapes']:
            segmentations = collections.defaultdict(list)  # for segmentation

            points = shape['points']
            label = shape['label']
            shape_type = shape.get('shape_type', None)
            mask = shape_to_mask(img.shape[:2], points, shape_type)

            points = np.asarray(points).flatten().tolist()
            segmentations[label].append(points)

            cls_name = label.split('-')[0]
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data['annotations'].append(dict(
                id=len(data['annotations']),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[label],
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
