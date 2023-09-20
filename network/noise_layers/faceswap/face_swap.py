import cv2
import dlib
import numpy as np
import os
import torch
import torch.nn as nn
import random, string
from PIL import Image
from torchvision import transforms

class FaceSwap(nn.Module):

    def __init__(self, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(FaceSwap, self).__init__()
        self.temp = temp
        self.target = target
        if not os.path.exists(temp): os.mkdir(temp)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def get_temp(self):
        return self.temp + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png" # png or jpg?

    def get_target(self):
        idx = random.randint(162771, 182637)
        return self.target + str(idx).zfill(6) + ".png"

    def forward(self, image_cover_mask):
        image, cover_image = image_cover_mask[0], image_cover_mask[1]

        noised_image = torch.zeros_like(image)

        for i in range(image.shape[0]):
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            file = self.get_temp()
            while os.path.exists(file):
                file = self.get_temp()

            try:
                self.swap_faces(self.get_target(), im, file)
                fake = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)
                noised_image[i] = self.transform(fake).unsqueeze(0).to(image.device)
            except:
                print("Error in shallowfakes")
                noised_image[i] = image[i]

        return noised_image

    # https://github.com/guipleite/CV2-Face-Swap
    def swap_faces(self, face_image_path, body_image, output_path):
        face = cv2.imread(face_image_path)
        body = np.array(body_image)[:, :, ::-1]

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        body_gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

        # Create empty matrices in the images' shapes
        height, width = face_gray.shape
        mask = np.zeros((height, width), np.uint8)

        height, width, channels = body.shape

        # Loading models and predictors of the dlib library to detect landmarks in both faces
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("facialmodels/shape_predictor_68_face_landmarks.dat")

        # Getting landmarks for the face that will be swapped into to the body
        rect = detector(face_gray)[0]

        # This creates a with 68 pairs of integer values — these values are the (x, y)-coordinates of the facial structures
        landmarks = predictor(face_gray, rect)
        landmarks_points = []

        def get_landmarks(landmarks, landmarks_points):
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

        get_landmarks(landmarks, landmarks_points)

        points = np.array(landmarks_points, np.int32)

        convexhull = cv2.convexHull(points)

        face_cp = face.copy()

        face_image_1 = cv2.bitwise_and(face, face, mask=mask)

        rect = cv2.boundingRect(convexhull)

        subdiv = cv2.Subdiv2D(rect)  # Creates an instance of Subdiv2D
        subdiv.insert(landmarks_points)  # Insert points into subdiv
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        face_cp = face.copy()

        def get_index(arr):
            index = 0
            if arr[0].any():
                index = arr[0][0]
            return index

        for triangle in triangles:

            # Gets the vertex of the triangle
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])

            # Draws a line for each side of the triangle
            cv2.line(face_cp, pt1, pt2, (255, 255, 255), 3, 0)
            cv2.line(face_cp, pt2, pt3, (255, 255, 255), 3, 0)
            cv2.line(face_cp, pt3, pt1, (255, 255, 255), 3, 0)

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = get_index(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = get_index(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = get_index(index_pt3)

            # Saves coordinates if the triangle exists and has 3 vertices
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                vertices = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(vertices)

        # Getting landmarks for the face that will have the first one swapped into
        rect2 = detector(body_gray)[0]

        # This creates a with 68 pairs of integer values — these values are the (x, y)-coordinates of the facial structures
        landmarks_2 = predictor(body_gray, rect2)
        landmarks_points2 = []

        # Uses the function declared previously to get a list of the landmark coordinates
        get_landmarks(landmarks_2, landmarks_points2)

        # Generates a convex hull for the second person
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

        body_cp = body.copy()

        lines_space_new_face = np.zeros((height, width, channels), np.uint8)
        body_new_face = np.zeros((height, width, channels), np.uint8)

        height, width = face_gray.shape
        lines_space_mask = np.zeros((height, width), np.uint8)

        for triangle in indexes_triangles:
            # Coordinates of the first person's delaunay triangles
            pt1 = landmarks_points[triangle[0]]
            pt2 = landmarks_points[triangle[1]]
            pt3 = landmarks_points[triangle[2]]

            # Gets the delaunay triangles
            (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
            cropped_triangle = face[y: y + height, x: x + widht]
            cropped_mask = np.zeros((height, widht), np.uint8)

            # Fills triangle to generate the mask
            points = np.array([[pt1[0] - x, pt1[1] - y], [pt2[0] - x, pt2[1] - y], [pt3[0] - x, pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_mask, points, 255)

            # Draws lines for the triangles
            cv2.line(lines_space_mask, pt1, pt2, 255)
            cv2.line(lines_space_mask, pt2, pt3, 255)
            cv2.line(lines_space_mask, pt1, pt3, 255)

            lines_space = cv2.bitwise_and(face, face, mask=lines_space_mask)

            # Calculates the delaunay triangles of the second person's face

            # Coordinates of the first person's delaunay triangles
            pt1 = landmarks_points2[triangle[0]]
            pt2 = landmarks_points2[triangle[1]]
            pt3 = landmarks_points2[triangle[2]]

            # Gets the delaunay triangles
            (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
            cropped_mask2 = np.zeros((height, widht), np.uint8)

            # Fills triangle to generate the mask
            points2 = np.array([[pt1[0] - x, pt1[1] - y], [pt2[0] - x, pt2[1] - y], [pt3[0] - x, pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_mask2, points2, 255)

            # Deforms the triangles to fit the subject's face : https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points,
                                       points2)  # Warps the content of the first triangle to fit in the second one
            dist_triangle = cv2.warpAffine(cropped_triangle, M, (widht, height))
            dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)

            # Joins all the distorted triangles to make the face mask to fit in the second person's features
            body_new_face_rect_area = body_new_face[y: y + height, x: x + widht]
            body_new_face_rect_area_gray = cv2.cvtColor(body_new_face_rect_area, cv2.COLOR_BGR2GRAY)

            # Creates a mask
            masked_triangle = cv2.threshold(body_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])

            # Adds the piece to the face mask
            body_new_face_rect_area = cv2.add(body_new_face_rect_area, dist_triangle)
            body_new_face[y: y + height, x: x + widht] = body_new_face_rect_area

        body_face_mask = np.zeros_like(body_gray)
        body_head_mask = cv2.fillConvexPoly(body_face_mask, convexhull2, 255)
        body_face_mask = cv2.bitwise_not(body_head_mask)

        body_maskless = cv2.bitwise_and(body, body, mask=body_face_mask)
        result = cv2.add(body_maskless, body_new_face)

        # Gets the center of the face for the body
        (x, y, widht, height) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + widht) / 2), int((y + y + height) / 2))

        seamlessclone = cv2.seamlessClone(result, body, body_head_mask, center_face2, cv2.NORMAL_CLONE)

        cv2.imwrite(output_path, seamlessclone)
