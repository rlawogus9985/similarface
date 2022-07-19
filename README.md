# similarface

import os
import face_recognition
import matplotlib.pyplot as plt
import numpy as np



dir_path = os.getenv('HOME')+'/aiffel/celebrities/images'
file_list = os.listdir(dir_path)

print ("file_list: {}".format(file_list))

# 작업 환경에 따라 혹시나 뜰 수 있는 ".ipynb_checkpoints"를 제거한 리스트를 만들어줄게요.
file_list = list(filter(lambda file: not file.startswith("."), file_list))

print(file_list)


def get_cropped_face(image_file):
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)
    a, b, c, d = face_locations[0]
    cropped_face = image[a:c, d:b, :]

    return cropped_face

image_path = os.path.join(dir_path, file_list[0])
face = get_cropped_face(image_path)
print(plt.imshow(face))


def get_face_embedding(face):
    return face_recognition.face_encodings(face)


def get_face_embedding_dict(dir_path):
    file_list = os.listdir(dir_path)
    file_list = list(filter(lambda file: not file.startswith("."), file_list))
    embedding_dict = {}

    for file in file_list:
        img_path = os.path.join(dir_path, file)

        #  face_recognition.face_locations를 이용했을때 어떤 이미지는 얼굴영역을 제대로 찾지 못하는 경우도 있습니다.이 경우를 예외처리 해줍니다.
        try:
            face = get_cropped_face(img_path)
            embedding = get_face_embedding(face)
        except:
            face = ""
            embedding = ""

        # 얼굴영역 face가 제대로 detect된 경우에면 dict에 embedding값을 저장해 줍니다.
        if len(embedding) > 0:
            embedding_dict[os.path.splitext(file)[0]] = embedding[
                0]  # os.path.splitext(file)[0]에는 이미지파일명에서 확장자를 제거한 이름이 담깁니다.

    return embedding_dict

embedding_dict = get_face_embedding_dict(dir_path)
print(embedding_dict['김재현'])


def get_distance(name1, name2):
    return np.linalg.norm(embedding_dict[name1] - embedding_dict[name2], ord=2)


print(get_distance('김재현', 'biden'))

my_faces = ['김재현', '김재현2']  # 내 얼굴들이 서로 비교될 필요 없게 내 사진 이름들을 넣어주세요


def get_nearest_face(name, top=7):
    sort_key_func = get_sort_key_func(name)

    # 거리에 따라 오름차순으로 소팅된 얼굴들
    sorted_faces = sorted(embedding_dict.items(), key=lambda x: sort_key_func(x[0]))
    # my_faces 에 포함된 얼굴은 filter 함수를 통해 걸러주세요
    sorted_faces = list(filter(lambda face: face[0] not in my_faces, sorted_faces))
    num_faces = len(sorted_faces)

    for i in range(min(num_faces, top + 1)):
        if sorted_faces[i]:
            print('순위 {} : 이름({}), 거리({})'.format(i, sorted_faces[i][0], sort_key_func(sorted_faces[i][0])))

print(get_nearest_face(my_faces[0]))
