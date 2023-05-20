# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script exports inlier matches from a COLMAP database to a text file.

import os
import argparse
import sqlite3
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="/mnt/data/colmap/sample_140_p1/database.db", required=False)
    parser.add_argument("--output_path", type=str, default="/mnt/data/colmap/sample_140_p1/matchs.txt", required=False)
    parser.add_argument("--min_num_matches", type=int, default=15)
    args = parser.parse_args()
    return args


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def main():
    root_path = "/mnt/data/colmap/sample_140_p1/"
    args = parse_args()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    images = {}
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        image_name = row[2]
        images[image_id] = image_name

    with open(os.path.join(args.output_path), "w") as fid:
        cursor.execute("SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;", (args.min_num_matches,))
        image_id1_pairs = []
        image_id2_pairs = []
        matchs_pairs      = []
        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_id1_pairs.append(image_id1)
            image_id2_pairs.append(image_id2)
            matchs_pairs.append(inlier_matches)
    for i, id1 in enumerate(image_id1_pairs):
        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;", (id1,))
        keypoints1 = next(cursor)
        keypoints1 = np.fromstring(keypoints1[0], dtype=np.float32).reshape(-1, 6)
        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;", (image_id2_pairs[i],))
        keypoints2 = next(cursor)
        keypoints2 = np.fromstring(keypoints2[0], dtype=np.float32).reshape(-1, 6)
        match_pair = matchs_pairs[i]
        # show match points
        img1 = cv2.imread(os.path.join(root_path, "images", images[id1]))
        img2 = cv2.imread(os.path.join(root_path, "images", images[image_id2_pairs[i]]))
        img_show = np.concatenate((img1, img2), axis=1)
        for pair in matchs_pairs[i]:
            p1 = keypoints1[pair[0], 0:2]
            p2 = keypoints2[pair[1], 0:2]
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0])+1920, int(p2[1])
            cv2.circle(img_show, (x1, y1), 5, (255,0,0), 2)
            cv2.circle(img_show, (x2, y2), 5, (0,0,255), 2)
            cv2.line(img_show, (x1,y1), (x2,y2), (0,255,0), 3)
            # cv2.imwrite("img_show.jpg", img_show)
        print("idx:{} img1:{} img2:{}".format(i, images[id1], images[image_id2_pairs[i]]))
        cv2.imwrite(os.path.join("output", images[id1]+"_"+images[image_id2_pairs[i]]+".jpg"), img_show)

    #     keypoints1 = 
    for key in images.keys():
        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;", (key,))
        keypoints1 = next(cursor)
        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;", (key,))

        for row in cursor:
            print(row)

        

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
