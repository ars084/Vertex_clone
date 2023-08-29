import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
import copy
import os

def normalize(image):
    normalized_image = (image-np.mean(image))/np.std(image)
    return normalized_image

def preprocess_image(image,max_size):
    if max(image.size) > max_size:
        rescale_factor = max_size/max(image.size)
        newsize = (int(image.size[0] * rescale_factor), int(image.size[1] * rescale_factor))
        image = np.array(image.resize(newsize))
    else:
        image = np.array(image)
    
    alpha = []

    if image.shape[-1] > 3:
        alpha = image[:,:,3]

    return image, alpha

def filter_image(image,alpha,stridex,stridey):

    final_image = np.zeros((image.shape))
    final_outline = np.zeros((image[:,:,0].shape))
    filtersize = (3,3)

    for i in range(2):

        edge_filter = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]])
        edge_filter = np.rot90(edge_filter,i)

        if len(alpha)>0:
            full_filter = np.stack((edge_filter,edge_filter, edge_filter,edge_filter))
        else:
            full_filter = np.stack((edge_filter,edge_filter, edge_filter))

        full_filter = np.swapaxes(full_filter,0,1)
        full_filter = np.swapaxes(full_filter,1,2)
        
        
        filted_image = np.zeros((image.shape))

        if len(alpha)>0:
            outline = np.zeros((alpha.shape))
        
        example = normalize(image)
           
        for row in range(0,image.shape[0] - edge_filter.shape[0],stridex):
            for column in range(0,image.shape[1]- edge_filter.shape[1],stridey):
                reference = example[row:row+edge_filter.shape[0],column:column+edge_filter.shape[1],:]
                conv = np.sum(np.multiply(full_filter,reference),axis=0)
                if len(alpha)>0:
                    outline_ref = alpha[row:row+edge_filter.shape[0],column:column+edge_filter.shape[1]]
                    outline_conv = np.sum(np.multiply(edge_filter,outline_ref))
                    outline[row,column] = abs(outline_conv)
                #keep_channels
                filted_image[row,column,:] = abs(np.sum(conv,axis=0))
        final_image = final_image + filted_image
        if len(alpha)>0:
            final_outline = final_outline + outline


    if len(alpha) > 1:
        collapsed_img = np.multiply(np.sum(final_image,axis=2),alpha)
        collapsed_img = normalize(collapsed_img)
        full_image = (collapsed_img>np.max(collapsed_img)*0.05)|(final_outline>(np.max(final_outline)*.7))
    else:
        collapsed_img = normalize(np.sum(final_image,axis=2))
        full_image = (collapsed_img>np.max(collapsed_img)*0.05)

    return full_image

def triangulate_image(full_image, sparcity):

    possible_pts = []

    rowlength = full_image.shape[1]
    outline = copy.copy(full_image)

    for i in range(outline.size):
        element_index_x = int(np.floor(i/rowlength))
        element_index_y = i%rowlength
        if outline[element_index_x, element_index_y]:
            possible_pts.append([element_index_x, element_index_y])
            MinX = max(0,element_index_x-sparcity)
            MaxX = min(full_image.shape[0],element_index_x+sparcity)
            
            MinY = max(0,element_index_y-sparcity)
            MaxY = min(full_image.shape[1],element_index_y+sparcity)

            outline[MinX:MaxX, MinY:MaxY] = False
            
    chosen_pts = copy.copy(possible_pts)
    chosen_pts = np.array(chosen_pts)
    tri = Delaunay(np.array(chosen_pts),qhull_options='QJ')

    return tri, chosen_pts

def clean_triangulation(outline,chosen_pts,tri,alpha):
    new_triangulation = []
    midpoint_reduction = False
    reduce_isosceles = False
    add_corners = False

    if add_corners:
        height = outline.shape[0]
        width = outline.shape[1]
        for j in np.linspace(0,width-1,2):
            chosen_pts = np.concatenate((chosen_pts,[[0,int(j)]]))
            chosen_pts = np.concatenate((chosen_pts,[[height-1,int(j)]]))
        tri = Delaunay(np.array(chosen_pts),qhull_options='QJ')
        
    for triangulation in tri.simplices:    

        if reduce_isosceles:
            x1 = chosen_pts[triangulation[0],1]
            x2 = chosen_pts[triangulation[1],1]
            x3 = chosen_pts[triangulation[2],1]
            y1 = chosen_pts[triangulation[0],0]
            y2 = chosen_pts[triangulation[1],0]
            y3 = chosen_pts[triangulation[2],0]

            ab = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            bc = np.sqrt((x2-x3)**2 + (y2-y3)**2)
            ca = np.sqrt((x3-x1)**2 + (y3-y1)**2)

            longest_side = max([ab,bc,ca])
            shortest_side = min([ab,bc,ca])

            if longest_side > 3*shortest_side:
                chosen_pts = np.concatenate((chosen_pts,[[int(np.mean([y1,y2,y3])), int(np.mean([x1,x2,x3]))]]))

        if midpoint_reduction:
            x1_mean = np.mean(chosen_pts[triangulation[[0,1]],1])
            x2_mean = np.mean(chosen_pts[triangulation[[2,1]],1])
            x3_mean = np.mean(chosen_pts[triangulation[[0,2]],1])
            y1_mean = np.mean(chosen_pts[triangulation[[0,1]],0])
            y2_mean = np.mean(chosen_pts[triangulation[[0,2]],0])
            y3_mean = np.mean(chosen_pts[triangulation[[2,1]],0])

            midpoint1 = (alpha[int(y1_mean),int(x1_mean)] > 0)
            midpoint2 = (alpha[int(y2_mean),int(x2_mean)] > 0)
            midpoint3 = (alpha[int(y3_mean),int(x3_mean)] > 0)
            if np.sum([midpoint1,midpoint2,midpoint3]) > 1:
                new_triangulation.append(triangulation)
                
        else:
            x = np.mean(chosen_pts[triangulation,1])
            y = np.mean(chosen_pts[triangulation,0])        
            if len(alpha) > 1:
                if alpha[int(y),int(x)] > 0:
                    new_triangulation.append(triangulation)
            else:
                new_triangulation.append(triangulation)

    if reduce_isosceles:
        tri = Delaunay(np.array(chosen_pts),qhull_options='QJ')
        new_triangulation = tri.simplices

    new_triangulation = np.array(new_triangulation)

    return new_triangulation

def get_triangulation_colors(image,new_triangulation,chosen_pts):

    tri_image = copy.copy(image)
    full_mask = np.zeros((image.shape[0],image.shape[1]))
    facecolors = []

    for triangulation in new_triangulation:
        tupVerts=[(chosen_pts[point,1], chosen_pts[point,0]) for point in triangulation]
        verts = np.array([[chosen_pts[point,1], chosen_pts[point,0]] for point in triangulation])

        x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1])) # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((y,x)).T 

        p = Path(tupVerts) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(image.shape[1],image.shape[0])
        mask = np.swapaxes(mask,0,1)
        
        full_mask = full_mask + mask
        
        center_valx = int(np.mean(verts[:,1]))
        center_valy = int(np.mean(verts[:,0]))
        
        
        r = image[center_valx,center_valy,0]
        g = image[center_valx,center_valy,1]
        b = image[center_valx,center_valy,2]
        
        facecolor = f"#{r:02x}{g:02x}{b:02x}"
        facecolors.append(facecolor)
        
        for i in range(image.shape[2]-1):
            index_to_pick = (np.sum(mask)/2)+1
            values = np.multiply(mask,image[:,:,i])
            sorted_vals = np.sort(values.flatten())
            median_val = sorted_vals[-1*int(index_to_pick)]
            
            indices = np.argwhere(mask)
            tri_image[indices[:,0],indices[:,1],i] = image[center_valx,center_valy,i]
            #tri_image[indices[:,0],indices[:,1],i] = median_val
    return facecolors

def save_analysis(image_folder,facecolors,chosen_pts,new_triangulation):

    outpath = os.path.join(image_folder,'facecolors.txt')
    with open(outpath, 'w') as fp:
        for item in facecolors:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Colors saved')
        
    outpath = os.path.join(image_folder,'points.npy')
    with open(outpath, 'wb') as f:
        np.save(f, chosen_pts)
        print('Points saved')

    outpath = os.path.join(image_folder,'triangulation.npy')
    with open(outpath, 'wb') as f:
        np.save(f, new_triangulation)
        print('Triangulation saved')

def create_puzzle(image_name,sparcity=10,max_size=250,stridex=1,stridey=1):
    image = Image.open(image_name)
    image_folder = os.path.dirname(image_name)

    image, alpha = preprocess_image(image,max_size)
    full_image = filter_image(image,alpha,stridex,stridey)
    tri, chosen_pts = triangulate_image(full_image, sparcity)
    new_triangulation = clean_triangulation(full_image,chosen_pts,tri,alpha)
    facecolors = get_triangulation_colors(image,new_triangulation,chosen_pts)
    save_analysis(image_folder,facecolors,chosen_pts,new_triangulation)

image_name = '/Users/andersonscott/Desktop/Image Triangulation/dino/dino.png'

create_puzzle(image_name)


