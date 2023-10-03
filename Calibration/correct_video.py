import turtle as t
import random
#
#
# def random_color():
#     return random.choice(['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'white', 'cyan'])
#
#
# t.setup(500, 500)
# t.bgcolor('black')
# t.speed(25)
#
# a = 0
# while True:
#     t.color(random_color())  # Set random color
#     t.fd(a)
#     t.rt(118)
#     a += 3
#
#     x, y = t.xcor(), t.ycor()
#     if abs(x) > 250 or abs(y) > 250:  #
#         t.penup()  # Lift the pen up before moving
#         t.goto(0, 0)  # Go to the center
#         t.clear()  # Clear the screen
#         t.pendown()  # Put the pen back down
#         a = 0  # Reset 'a'


# Simple L-System to draw a tree
# def draw_tree(branch_len, t):
#     if branch_len > 5:
#         angle = random.randint(22, 30)
#         sf = random.uniform(0.6, 0.8)
#         t.fd(branch_len)
#         t.lt(angle)
#         draw_tree(branch_len * sf, t)
#         t.rt(angle * 2)
#         draw_tree(branch_len * sf, t)
#         t.lt(angle)
#         t.bk(branch_len)
#
# t.lt(90)
# t.up()
# t.bk(100)
# t.down()
# draw_tree(100, t)

# Random Walk
# directions = [0, 90, 180, 270]
#
# for _ in range(1000):
#     t.setheading(random.choice(directions))
#     t.fd(20)
import turtle as t

def draw_circle(x, y, radius):
    t.penup()
    t.goto(x, y - radius)
    t.pendown()
    t.circle(radius)

def draw_line(x1, y1, x2, y2):
    t.penup()
    t.goto(x1, y1)
    t.pendown()
    t.goto(x2, y2)

# Initialize turtle
t.speed(0)
t.hideturtle()

# Draw neurons
neuron_radius = 20
layer_distance = 200
neuron_distance = 100

# Input layer
input_neurons = 3
x = -layer_distance
for i in range(input_neurons):
    y = neuron_distance * (i - (input_neurons - 1) / 2)
    draw_circle(x, y, neuron_radius)

# Hidden layer
hidden_neurons = 4
x = 0
for i in range(hidden_neurons):
    y = neuron_distance * (i - (hidden_neurons - 1) / 2)
    draw_circle(x, y, neuron_radius)

# Output layer
output_neurons = 2
x = layer_distance
for i in range(output_neurons):
    y = neuron_distance * (i - (output_neurons - 1) / 2)
    draw_circle(x, y, neuron_radius)

# Draw synapses (connections)
# Connect input layer to hidden layer
for i1 in range(input_neurons):
    y1 = neuron_distance * (i1 - (input_neurons - 1) / 2)
    for i2 in range(hidden_neurons):
        y2 = neuron_distance * (i2 - (hidden_neurons - 1) / 2)
        draw_line(-layer_distance, y1, 0, y2)

# Connect hidden layer to output layer
for i1 in range(hidden_neurons):
    y1 = neuron_distance * (i1 - (hidden_neurons - 1) / 2)
    for i2 in range(output_neurons):
        y2 = neuron_distance * (i2 - (output_neurons - 1) / 2)
        draw_line(0, y1, layer_distance, y2)

t.done()

#TODO: Convert to Executable script
