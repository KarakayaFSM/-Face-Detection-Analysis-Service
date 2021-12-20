# coding: utf-8
# -*- coding: utf-8 -*-
import io
import math
import os
import random
import shutil
import zipfile
from pathlib import Path

import torch
import torchvision.datasets as datasets
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, jsonify, request
from torch.utils.data import DataLoader

import MinioService
import constants

# execution flow:
# initialize_mtcnn_and_resnet()
#
# extract_faces_from_family_photo(family_photo_path)
# make_image_folder(path)
# generate_cropped_images_data()
# face_match(given_img_path, saved_data_path, distance_metric=0)
# print('This Person', get_result_message())

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

DEFAULT_CROPPED_IMAGES_PATH = './images/kisiler_cropped'
app = Flask(__name__)


def make_image_folder(path):
    folder_count = 1
    for dirpath, dirnames, file_names in os.walk(path):
        for file_name in file_names:
            sub_dir = f'{dirpath}/kisi_{folder_count}'
            os.mkdir(sub_dir)

            src = f'{dirpath}/{file_name}'
            dest = f'{sub_dir}/{file_name}'

            os.replace(src=src, dst=dest)
            folder_count += 1
        # go down only once
        break


def crop_images_in(images_parent_path):
    for dirpath, dirnames, file_names in os.walk(images_parent_path):
        for sub_dir_path, sub_dir_names, sub_file_names in os.walk(dirpath):
            for sub_file_name in sub_file_names:
                image_path = f'{sub_dir_path}/{sub_file_name}'
                mtcnn(Image.open(Path(image_path)), image_path)
            break


def prepare_metadata(dataset):
    cropped_images_loader = DataLoader(dataset, collate_fn=lambda a: a[0])

    name_list = []  # list of names corresponding to cropped photos
    embedding_list = []  # embeddings for cropped images produced by resnet

    def swap_cls_and_idx():
        return {index: cls for cls, index in dataset.class_to_idx.items()}

    idx_to_cls = swap_cls_and_idx()

    for img, idx in cropped_images_loader:
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob >= 0.9:
            embedding = resnet(face.unsqueeze(0))
            embedding_list.append(embedding.detach())
            name = idx_to_cls[idx]
            name_list.append(name)
        else:
            print('this image does not include a face. skipped:', idx_to_cls[idx], 'prob:', prob)
    return embedding_list, name_list


def get_distances(given_emb, saved_embeddings):
    dist_list = []
    for saved_emb in saved_embeddings:
        distance = torch.dist(given_emb, saved_emb).item()
        dist_list.append(distance)
    return dist_list


def get_dist_by_metric(target, member_embeddings, distance_metric=0):
    distances = []
    for saved in member_embeddings:
        if distance_metric == 0:
            # Euclidian distance
            diff = torch.subtract(target, saved)
            dist = torch.sum(torch.square(diff), 1).item()
            distances.append(dist)
        elif distance_metric == 1:
            # Distance based on cosine similarity
            dot = torch.sum(torch.multiply(target, saved), dim=1)
            norm = torch.linalg.norm(target, dim=1) * torch.linalg.norm(saved, dim=1)
            similarity = dot / norm
            dist = (torch.arccos(similarity) / math.pi).item()
            distances.append(dist)
        else:
            raise 'Undefined distance metric %d' % distance_metric
    return distances


def face_match(target_photo, group_data_path, distance_metric=0):
    target_img = Image.open(target_photo)
    target_face = mtcnn(target_img)
    # detach means: set <requires_grad> field to False
    target_embedding = resnet(target_face.unsqueeze(0)).detach()

    group_data = load_data(group_data_path)
    return find_closest_match(group_data, target_embedding, distance_metric)


def find_closest_match(group_data, target_embedding, distance_metric=0):
    member_embeddings, member_names = group_data
    distances = get_dist_by_metric(
        target_embedding,
        member_embeddings,
        distance_metric
    )
    # minimum distance is used to identify the person
    min_distance = min(distances)
    idx_of_min = distances.index(min_distance)
    return {'name': member_names[idx_of_min], 'dist': min_distance}


def load_data(data_path):
    saved_data = torch.load(data_path)
    saved_embeddings = saved_data[0]
    saved_names = saved_data[1]
    return saved_embeddings, saved_names


def get_distance_report(result):
    return 'with distance: %.3f' % result['dist']


def get_result_message(result, distance_threshold=constants.DISTANCE_THRESHOLD):
    found_message = 'is in the image\n'
    not_found_message = 'is not in the given image\n'
    return found_message if result['dist'] <= distance_threshold else not_found_message


def generate_metadata(cropped_images_path):
    cropped_images_dataset = datasets.ImageFolder(cropped_images_path)
    cropped_embedding_list, cropped_name_list = prepare_metadata(cropped_images_dataset)
    cropped_images_data = [cropped_embedding_list, cropped_name_list]
    return cropped_images_data


def save_data(cropped_images_data, data_file_name=constants.DEFAULT_DATA_FILE_NAME):
    output_path = Path(
        add_file_extension(data_file_name, constants.PYTORCH_FILE_EXTENSION)
    )

    torch.save(cropped_images_data, output_path)
    return output_path


def generate_data_of(family_photo):
    extract_faces_from_family_photo(family_photo)

    make_image_folder(DEFAULT_CROPPED_IMAGES_PATH)

    return generate_metadata(
        DEFAULT_CROPPED_IMAGES_PATH
    )


def search_person_in_photo(target_photo, family_photo):
    data_path = generate_data_of(family_photo)

    result = face_match(
        target_photo,
        data_path,
        distance_metric=0
    )

    response_message = f'This Person {get_result_message(result)}' \
        # f' {get_distance_report(result)}'
    return response_message


def extract_faces_from_family_photo(family_photo):
    mtcnn.keep_all = True
    img = Image.open(family_photo)
    mtcnn(img, DEFAULT_CROPPED_IMAGES_PATH + '/kisi.png')
    mtcnn.keep_all = False


def download_search_person_photos(request_body):
    target_photo = MinioService.download_file(request_body[constants.TARGET_IMAGE_KEY])
    family_photo = MinioService.download_file(request_body[constants.FAMILY_PHOTO_KEY])
    return io.BytesIO(family_photo.data), io.BytesIO(target_photo.data)


@app.route('/searchPerson', methods=['POST'])
def search_person():
    request_body = request.get_json()
    family_photo, target_photo = \
        download_search_person_photos(request_body)

    response = search_person_in_photo(target_photo, family_photo)

    shutil.rmtree(DEFAULT_CROPPED_IMAGES_PATH)
    return jsonify(response=response, message='OK', success=True)


@app.route('/group/listMembers', methods=['POST'])
def list_group_members():
    member_list = []

    family_photo = download_file(
        request.get_json(),
        constants.GROUP_PHOTO_KEY
    )

    family_photo_data, cached_group_data = \
        extract_metadata(family_photo)

    family_photo_embeddings = family_photo_data[0]

    for fp_embedding in family_photo_embeddings:
        closest_match = find_closest_match(cached_group_data, fp_embedding)
        if closest_match['dist'] <= constants.DISTANCE_THRESHOLD:
            member_list.append(closest_match['name'])

    shutil.rmtree(DEFAULT_CROPPED_IMAGES_PATH)
    return jsonify(response=str(member_list), message='OK', success=True)


@app.route('/initializeGroup', methods=['POST'])
def initialize_group():
    file = download_file(request.get_json(), constants.GROUP_PHOTO_KEY)

    member_images_parent = extract_zip(file)
    crop_images_in(Path(member_images_parent))

    group_metadata = generate_metadata(
        cropped_images_path=member_images_parent
    )

    save_data(
        group_metadata,
        constants.GROUP_DATA_PATH
    )

    shutil.rmtree(member_images_parent)

    response = "OK"
    return jsonify(response=response, message='OK', success=True)


def extract_zip(file):
    zip_file = zipfile.ZipFile(file, 'r')
    zip_file.extractall(constants.ZIP_EXTRACTION_PATH)
    return f'{constants.ZIP_EXTRACTION_PATH}'


def extract_metadata(family_photo):
    family_photo_data = generate_data_of(family_photo)
    group_data = load_data(
        add_file_extension(
            constants.GROUP_DATA_PATH,
            constants.PYTORCH_FILE_EXTENSION
        )
    )
    return family_photo_data, group_data


def add_random_suffix(file_name):
    random_suffix = random.randint(0, 1000)
    return f'{file_name}_{random_suffix}'


def add_file_extension(file_name, extension):
    return f'{file_name}.{extension}'


def download_file(request_body, param_key):
    file = MinioService.download_file(request_body[param_key])
    return io.BytesIO(file.data)


if __name__ == '__main__':
    app.run(debug=True)
