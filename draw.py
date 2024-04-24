import pygame
import tensorflow as tf
import numpy as np

pygame.init()

w, h = 784, 784+100
sc = pygame.display.set_mode((w, h))
pygame.display.set_caption('MNIST visualization')
pygame.display.set_icon(pygame.image.load('img/predict.png'))
clock = pygame.time.Clock()

model = tf.keras.models.load_model('mnist.h5')

running = True

font = pygame.font.SysFont('Arial', 50, bold=True)

number_map = [[0 for _ in range(28)] for _ in range(28)]

pencil_is_draw = True

pencil = pygame.image.load('img/pencil.png')
pencil = pygame.transform.scale(pencil, (100, 100))

eraser = pygame.image.load('img/eraser.png')
eraser = pygame.transform.scale(eraser, (100, 100))

clean = pygame.image.load('img/clean.png')
clean = pygame.transform.scale(clean, (100, 100))

rect = pygame.Rect(0, 784, 100, 100)
rect1 = pygame.Rect(100, 784, 100, 100)

fr = font.render("Result: ?", True, (255, 255, 255))
fr1 = font.render("", True, (255, 255, 255))

while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    for i in pygame.event.get():
        
        if i.type == pygame.QUIT:
            running = False
        elif i.type == pygame.MOUSEBUTTONUP:
            if number_map != [[0 for _ in range(28)] for _ in range(28)]:
                array = np.array(number_map)
                img = array.reshape((1, 784))
                predict = model.predict(img)[0]
                res = predict.argmax()
                fr = font.render(f"Result: {res}", True, (255, 255, 255))
                fr1 = font.render(f"{round(predict[res]*100, 2)}%", True, (255, 255, 255))
            else:
                fr = font.render("Result: ?", True, (255, 255, 255))
                fr1 = font.render("", True, (255, 255, 255))
        elif i.type == pygame.MOUSEBUTTONDOWN:
            if i.button == 1:
                if rect.collidepoint(mouse_x, mouse_y):
                    pencil_is_draw = not pencil_is_draw
                if rect1.collidepoint(mouse_x, mouse_y):
                    number_map = [[0 for _ in range(28)] for _ in range(28)]

    nmx = max(min(27, mouse_x // 28), 0)
    nmy = max(min(27, mouse_y // 28), 0)

    if pygame.mouse.get_pressed()[0] and mouse_y // 28 < 28:
        set_color = 1 if pencil_is_draw else 0 
        number_map[nmy][nmx] = set_color

    sc.fill('black')
    pygame.draw.rect(sc, (255, 255, 255), (nmx*28, nmy*28, 28, 28))
    pygame.draw.rect(sc, (255, 0, 255), (0, 784, 784, 100))

    if pencil_is_draw:
        change = pencil
    else:
        change = eraser
        
    sc.blit(change, (0, 784))
    sc.blit(clean, (100, 784))
    sc.blit(fr, (250, 784+20))
    sc.blit(fr1, (500, 784+20))

    for y in range(28):
        for x in range(28):
            if number_map[y][x] == 1:
                pygame.draw.rect(sc, (255, 255, 255), (x*28, y*28, 28, 28))

    keys = pygame.key.get_pressed()
    if keys[pygame.K_c]:
        number_map = [[0 for _ in range(28)] for _ in range(28)]
    
    
    pygame.display.flip()
