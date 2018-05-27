import random

def tf_corrupt_img(image,percent):
    img=image.copy()
    for num in range(img.shape[0]):
        if img.shape[3]==1:   #black-white
            for row in range(img.shape[1]):
                tmp=[i for i in range(img.shape[2])]
                random.shuffle(tmp)
                tmp=tmp[0:int(img.shape[2]*percent)]
                for i in tmp:
                    img[num][row][i][0]=0
        if img.shape[3]==3:
            for row in range(img.shape[0]):
                for channel in range(3):
                    tmp=[i for i in range(img.shape[2])]
                    random.shuffle(tmp)
                    tmp=tmp[0:int(img.shape[2]*percent)]
                    for i in tmp:
                        img[num][row][i][channel]=0
    return img