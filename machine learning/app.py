import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
BOUNDRY_INC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
IMAGESAVE = False
MODEL = load_model("bestmodel.keras")

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize pygame
pygame.init()

# Set up font
FONT = pygame.font.Font("freesansbold.ttf", 18)

# Set up the display
DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Digit Board")

# Variables
is_writing = False
number_xcord = []
number_ycord = []
image_cnt = 1
PREDICT = True

while True:
    for event in pygame.event.get():  # Pygame olaylarını dinler
        if event.type == QUIT:  # Eğer pencere kapatılmak istenirse
            pygame.quit()  # Pygame'i kapatır
            sys.exit()  # Sistemi sonlandırır

        if event.type == MOUSEMOTION and is_writing:  # Kullanıcı fareyle hareket ediyorsa ve yazıyorsa
            xcord, ycord = event.pos  # Fare pozisyonunu al
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)  # Ekranda beyaz bir daire çizer
            number_xcord.append(xcord)  # X koordinatını kaydeder
            number_ycord.append(ycord)  # Y koordinatını kaydeder

        if event.type == MOUSEBUTTONDOWN:  # Fareye tıklanırsa
            is_writing = True  # Yazmaya başlandığını belirtir

        if event.type == MOUSEBUTTONUP:  # Fare tuşu bırakılırsa
            is_writing = False  # Yazmayı durdurur
            if len(number_xcord) > 0 and len(number_ycord) > 0:
                number_xcord = sorted(number_xcord)  # X koordinatlarını sıralar
                number_ycord = sorted(number_ycord)  # Y koordinatlarını sıralar

                # Yazının çevresini kapsayacak dikdörtgenin minimum ve maksimum koordinatlarını hesaplar
                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRY_INC, 0), min(WINDOW_WIDTH, number_xcord[-1] + BOUNDRY_INC)
                rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRY_INC, 0), min(WINDOW_HEIGHT, number_ycord[-1] + BOUNDRY_INC)

                # Çerçeve çizimi
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

                number_xcord = []  # Koordinatları temizler
                number_ycord = []  # Koordinatları temizler
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)  # Ekrandaki yazıyı numpy array'ine dönüştürür

                if IMAGESAVE:  # Eğer resim kaydedilmesi gerekiyorsa
                    cv2.imwrite(f"image_{image_cnt}.png")  # Resmi kaydeder
                    image_cnt += 1  # Resim sayacını artırır

                if PREDICT:  # Eğer tahmin yapılacaksa
                    image = cv2.resize(img_arr, (28, 28))  # Görüntüyü 28x28 boyutuna küçültür
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)  # Görüntüyü sıfırlarla doldurur (padding işlemi)
                    image = cv2.resize(image, (28, 28)) / 255  # Görüntüyü tekrar 28x28 boyutuna getirir ve piksel değerlerini 0-1 aralığına normalize eder
                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])  # Modelin tahmin ettiği etiketi alır

                    # Eğer yazı gösterilecekse
                    text_surface = FONT.render(label, True, RED, WHITE)  # Etiketi kırmızı renkte beyaz zemin üzerine render eder
                    text_rec_obj = text_surface.get_rect()  # Yazının dikdörtgen sınırlarını alır
                    text_rec_obj.left, text_rec_obj.bottom = rect_min_x, rect_max_y  # Yazıyı belirli bir pozisyonda konumlandırır
                    DISPLAYSURF.blit(text_surface, text_rec_obj)  # Yazıyı ekrana çizer

        if event.type == KEYDOWN:  # Bir tuşa basıldığında
            if event.unicode == "n":  # Eğer basılan tuş "n" ise
                DISPLAYSURF.fill(BLACK)  # Ekranı siyaha boyar (temizler)

    pygame.display.update()  # Ekranı günceller
