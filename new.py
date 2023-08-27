import pygame
import keras
from keras.models import load_model
import cv2
import numpy as np

# Load pre-trained model
model = load_model('oldbestmodel.h5')

# Initialize Pygame
pygame.init()

# Set screen size
size = (700, 500)
screen = pygame.display.set_mode(size)

# Set title of window
pygame.display.set_caption("drawing board")

# Set color of background
bg_color = (255, 255, 255)

# Initialize variables for drawing
drawing = False
last_pos = (0, 0)
color = pygame.Color(255, 255, 255)
# Main loop
running = True
i=0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        if event.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.line(screen, color, last_pos, event.pos, 7)
                last_pos = event.pos
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Get image of drawn number
                img = pygame.surfarray.array3d(pygame.display.get_surface())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (28,28))
                img = img.reshape(28,28,1)

                # Preprocess image for model
                #img = cv2.resize(img, (28, 28))
                

                img = img.astype('float32') / 255
                img = np.expand_dims(img, axis=0)

               # Display the image
               # cv2.imshow("Image", img)
                #cv2.waitKey(0)
                # Predict number using model
                prediction = model.predict(img)
                number = np.argmax(prediction)
                screen.fill((0, 0, 0))
                # Display recognized number on screen
                font = pygame.font.Font(None, 36)
                text = font.render(str(number), True, (255, 0, 0))
                screen.blit(i, (50, 50))
                i=i+1
                pygame.display.update()
            if event.key == pygame.K_s:
                # Save image of drawn number to disk
                pygame.image.save(screen, "drawn_number.png")
            # Ask user for correct label
            correct_label = int(input("Enter the correct label for the drawn number (0-9): "))

            # Tell model about wrong prediction
            if number != correct_label:
                # Create one-hot encoded label
                label = keras.utils.to_categorical(correct_label, 10)
                label = label.reshape(1, 10)

                 # Train model on the wrong prediction
                model.fit(img, label, epochs=1)
    pygame.display.flip()

# Quit Pygame
pygame.quit()
