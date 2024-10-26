import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Button Example")

# Define colors
WHITE = (255, 255, 255)
BLUE = (100, 100, 250)
DARK_BLUE = (70, 70, 180)
GREEN = (0, 255, 0)

# Button properties
button_rect = pygame.Rect(300, 250, 200, 100)  # x, y, width, height
button_color = BLUE
button_hover_color = DARK_BLUE
button_clicked_color = GREEN

# Set up the font for rendering text
font = pygame.font.SysFont(None, 48)

def draw_button(rect, color, text):
    """Draw a button with text."""
    pygame.draw.rect(screen, color, rect)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_pos = (mouse_x, mouse_y)

    # Check if the mouse is over the button
    if button_rect.collidepoint(mouse_pos):
        button_color = button_hover_color  # Change color on hover
        if pygame.mouse.get_pressed()[0]:  # Check if left mouse button is clicked
            button_color = button_clicked_color  # Change color when clicked
    else:
        button_color = BLUE  # Default color

    # Fill the screen with a background color
    screen.fill(WHITE)

    # Draw the button
    draw_button(button_rect, button_color, "Click Me!")

    # Update the display
    pygame.display.flip()

    # Limit the frame rate for smoother movement
    pygame.time.Clock().tick(60)
