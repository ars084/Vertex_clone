import tkinter as tk
import numpy as np
import copy
import random
import os

# Change to folder of puzzle to complete
image_folder = '/Users/andersonscott/Desktop/Vertex_game/Puzzles/penguin'


def undo():
	if app.found_connections:
		last_connection = app.found_connections.pop(-1)
		last_line = app.connectionline.pop(-1)
		canvas.delete(last_line)

def fill_triangle(triangle):
	pt1 = triangle[0]
	pt2 = triangle[1]
	pt3 = triangle[2]

	index = np.where(np.all(triangle == triangulations, axis=1))[0]
	canvas.create_polygon(Xdots[pt1], Ydots[pt1],
		Xdots[pt2], Ydots[pt2],
		Xdots[pt3], Ydots[pt3], fill=facecolors[index[0]])

def checkfortriangle():
	starting_point_spoke = []
	ending_point_spoke = []

	if app.found_triangles:
		for found_triangle in app.found_triangles:	
			triangle_reference = copy.copy(found_triangle)
			if (app.starting_point in found_triangle) and (app.ending_point in found_triangle):
				triangle_reference.remove(app.starting_point)
				triangle_reference.remove(app.ending_point)
				starting_point_spoke.append(triangle_reference[0])
				ending_point_spoke.append(triangle_reference[0])

			elif app.starting_point in found_triangle:
				triangle_reference.remove(app.starting_point)
				starting_point_spoke.append(triangle_reference[0])
				starting_point_spoke.append(triangle_reference[1])
			elif app.ending_point in found_triangle:
				triangle_reference.remove(app.ending_point)
				ending_point_spoke.append(triangle_reference[0])
				ending_point_spoke.append(triangle_reference[1])


	for found_connection in app.found_connections:
		if app.starting_point in found_connection:
			if app.starting_point == found_connection[0]:
				starting_point_spoke.append(found_connection[1])
			else:
				starting_point_spoke.append(found_connection[0])


		if app.ending_point in found_connection:
			if app.ending_point == found_connection[0]:
				ending_point_spoke.append(found_connection[1])
			else:
				ending_point_spoke.append(found_connection[0])

	common_points = set(starting_point_spoke) & set(ending_point_spoke)

	if common_points:

		for common_point in common_points:
			triangle = [app.starting_point, app.ending_point, common_point]
			triangle.sort()
			if np.any(np.all(triangle == triangulations, axis=1)):
				app.found_triangles.append(triangle)
				fill_triangle(triangle)

				try:
					app.found_connections.remove([triangle[0],triangle[1]])
				except:
					pass
				try:
					app.found_connections.remove([triangle[0],triangle[2]])
				except:
					pass
				try:
					app.found_connections.remove([triangle[1],triangle[2]])
				except:
					pass


def drawdots(Xdots,Ydots):
	for i,Xdot in enumerate(Xdots):
		Ydot = Ydots[i]
		canvas.create_oval(Xdot-5,Ydot-5,Xdot+5,Ydot+5, fill='white')
	canvas.pack()

def startline(event):

	x = event.x
	y = event.y

	dist = np.sqrt((Xdots - x)**2 + (Ydots - y)**2)

	if np.any(dist < 20):
		app.starting_point = np.argmin(dist)
		
		startingX = Xdots[app.starting_point]
		startingY = Ydots[app.starting_point]

		app.line = canvas.create_line(startingX, startingY, startingX+5, startingY+5, 
			dash = (5,1), fill='yellow')

def paint(event):
	
	x = event.x
	y = event.y
	if app.starting_point+1:
		canvas.coords(app.line, Xdots[app.starting_point], Ydots[app.starting_point], x, y)

def connect(event):

	if hasattr(app, 'line'):
		canvas.delete(app.line)

	x = event.x
	y = event.y

	dist = np.sqrt((Xdots - x)**2 + (Ydots - y)**2)

	if np.any(dist < 20):
		
		app.ending_point = np.argmin(dist)

		if app.ending_point != app.starting_point:

			connection = [app.starting_point, app.ending_point]
			connection.sort()


			if connection not in app.found_connections:

				app.found_connections.append(connection)

				holder = canvas.create_line(Xdots[app.starting_point],
					Ydots[app.starting_point],
					Xdots[app.ending_point], 
					Ydots[app.ending_point])

				app.connectionline.append(holder)

				checkfortriangle()

				if len(triangulations) == len(app.found_triangles):
					print('Picture Complete!')

				app.starting_point = None
				app.ending_point = None

def load_puzzle(image_folder):
	Xdots = []
	Ydots = []

	point_filename = os.path.join(image_folder,'points.npy')
	tri_filename = os.path.join(image_folder,'triangulation.npy')
	facecolor_filename = os.path.join(image_folder,'facecolors.txt')

	points = np.load(point_filename)
	for point in points:

		rescale_factor = 500/np.max(points.flatten())

		Xdots.append(int(point[1]*rescale_factor))
		Ydots.append(int(point[0]*rescale_factor))

	Xdots = np.array(Xdots)
	Ydots = np.array(Ydots)

	triangulations = np.load(tri_filename)
	triangulations = np.sort(triangulations)

	facecolors = []
	with open(facecolor_filename, 'r') as fp:
	    for line in fp:

	        x = line[:-1]

	        facecolors.append(x)

	return Xdots, Ydots, triangulations, facecolors


app = tk.Tk()
app.geometry("700x800")

Xdots, Ydots, triangulations, facecolors = load_puzzle(image_folder)

app.found_connections = []
app.found_triangles = []
app.connectionline = []

frame = tk.Frame(app, width=600, height=600)
frame.pack(padx = 30, pady = 10, fill='both')

canvas = tk.Canvas(frame, width = 600, height = 600, bg='#c4c4c4')
canvas.pack()

undobutton = tk.Button(app,text='undo',command=undo)
undobutton.pack(padx=50,anchor='w')

drawdots(Xdots,Ydots)

canvas.bind('<ButtonRelease-1>',connect,add="+")
canvas.bind( "<B1-Motion>", paint,add="+")
canvas.bind( "<ButtonPress>", startline,add="+")

app.mainloop()
