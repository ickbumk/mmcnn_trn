import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random

def create_dataset(cropped_img, intensities, p1, min_sample = 12000, desired_level = 3):

    #same with realistic dataset but saves different name
    
    img_size = int(cropped_img.shape[0])
    window_range = np.array([p1*4,p1*2,p1])
    mid_point_range = np.array([int(img_size/2),int(img_size/4),int(img_size/8)])
    multiplier = 0    
    
    for window_size in window_range:
        v_sample = np.ceil(np.sqrt(min_sample)/2).astype(int)*2*2**(multiplier)
        multiplier+=1

        if multiplier > desired_level:
            continue

        print('window size:', window_size)
        print('number of segments in one direction:',v_sample)
        
        increments = np.floor((cropped_img.shape[0])/v_sample)
        sampling_locations = np.linspace(0,cropped_img.shape[0]-window_size-1,v_sample).astype(int)
        
        labels = []
        imgs = []
        
        for x in sampling_locations:
            for y in sampling_locations:
                for i in range(intensities.shape[0]):

                    intensity = intensities[i,:]
                    intensity = intensity.reshape(img_size,img_size)
        
                    img_iter = cv2.resize(cropped_img[y:y+window_size,x:x+window_size],(p1,p1))
                    int_iter = cv2.resize(intensity[y:y+window_size,x:x+window_size],(p1,p1))
            
                    
                    imgs.append(np.column_stack((img_iter.flatten(), int_iter.flatten())))
                    labels.append([x+int(window_size/2),y+int(window_size/2)])
        
        imgs_array = np.array(imgs)
        labels_array = np.array(labels)

        
        quads_array = np.zeros((labels_array.shape[0],3))
        
        count = 0
        
        
        
        for quad_size in mid_point_range:
            
            xmask = np.floor(labels_array[:,0]/quad_size)%2==0
            ymask = np.floor(labels_array[:,1]/quad_size)%2==0
            x_reverse = np.array([not value for value in xmask])
            y_reverse = np.array([not value for value in ymask])
            
            quads_array[xmask*ymask,count] = 0
            quads_array[x_reverse*ymask,count] = 1
            quads_array[xmask*y_reverse,count] = 2
            quads_array[x_reverse*y_reverse,count] = 3
        
            count+=1
    
        np.save(f'features_with_intensity_{window_size}_increments{increments}.npy', imgs_array)
        print('features saved')
        np.save(f'coords__with_intensity_{window_size}_increments{increments}.npy', labels_array)
        print('coordinates saved')
        np.save(f'labels__with_intensity_{window_size}_increments{increments}.npy', quads_array)
        print('labels saved')
                
        
def create_random_dataset(cropped_img, intensities, p1, min_sample = 12000, desired_level = 3):

    # dataset generated at random positions
    img_size = int(cropped_img.shape[0])
    window_range = np.array([p1*4,p1*2,p1])
    mid_point_range = np.array([int(img_size/2),int(img_size/4),int(img_size/8)])
    multiplier = 0    
    
    for window_size in window_range:
        v_sample = min_sample*4**(multiplier)
        multiplier+=1

        if multiplier > desired_level:
            continue

        print('window size:', window_size)

        
        labels = []
        imgs = []
        
        for _ in range(v_sample):
            
                
            for i in range(intensities.shape[0]):

                x = random.randint(0,img_size-window_size)
                y = random.randint(0,img_size-window_size)
    
                intensity = intensities[i,:]
                intensity = intensity.reshape(img_size,img_size)
    
                img_iter = cv2.resize(cropped_img[y:y+window_size,x:x+window_size],(p1,p1))
                int_iter = cv2.resize(intensity[y:y+window_size,x:x+window_size],(p1,p1))
        
                
                imgs.append(np.column_stack((img_iter.flatten(), int_iter.flatten())))
                labels.append([x+int(window_size/2),y+int(window_size/2)])

        imgs_array = np.array(imgs)
        labels_array = np.array(labels)
        quads_array = np.zeros((labels_array.shape[0],3))
        
        count = 0
        
        
        
        for quad_size in mid_point_range:
            
            xmask = np.floor(labels_array[:,0]/quad_size)%2==0
            ymask = np.floor(labels_array[:,1]/quad_size)%2==0
            x_reverse = np.array([not value for value in xmask])
            y_reverse = np.array([not value for value in ymask])
            
            quads_array[xmask*ymask,count] = 0
            quads_array[x_reverse*ymask,count] = 1
            quads_array[xmask*y_reverse,count] = 2
            quads_array[x_reverse*y_reverse,count] = 3
        
            count+=1
    
    
                
        np.save(f'random_features_with_intensity_{window_size}_samples_{v_sample}.npy', imgs_array)
        print('features saved')
        np.save(f'random_coords__with_intensity_{window_size}_samples_{v_sample}.npy', labels_array)
        print('coordinates saved')
        np.save(f'random_labels__with_intensity_{window_size}_samples_{v_sample}.npy', quads_array)
        print('labels saved')


def create_realistic_dataset(cropped_img, intensities, p1, min_sample = 12000, desired_level = 3):

    # dataset at specific locations, specified by the number of samples
    
    img_size = int(cropped_img.shape[0])
    window_range = np.array([p1*4,p1*2,p1])
    mid_point_range = np.array([int(img_size/2),int(img_size/4),int(img_size/8)])
    multiplier = 0    
    
    for window_size in window_range:
        v_sample = np.ceil(np.sqrt(min_sample)/2).astype(int)*2*2**(multiplier)
        multiplier+=1

        if multiplier > desired_level:
            continue

        print('window size:', window_size)
        print('number of segments in one direction:',v_sample)
        
        increments = np.floor((cropped_img.shape[0])/v_sample)
        sampling_locations = (np.linspace(0,v_sample-1,v_sample)*increments).astype(int)
        
        labels = []
        imgs = []
        
        for x in sampling_locations:
            for y in sampling_locations:
                for i in range(intensities.shape[0]):

                    intensity = intensities[i,:]
                    intensity = intensity.reshape(img_size,img_size)
        
                    img_iter = cv2.resize(cropped_img[y:y+window_size,x:x+window_size],(p1,p1))
                    int_iter = cv2.resize(intensity[y:y+window_size,x:x+window_size],(p1,p1))
            
                    
                    imgs.append(np.column_stack((img_iter.flatten(), int_iter.flatten())))
                    labels.append([x+int(window_size/2),y+int(window_size/2)])
        
        imgs_array = np.array(imgs)
        labels_array = np.array(labels)
        quads_array = np.zeros((labels_array.shape[0],3))
        
        count = 0
        
        
        
        for quad_size in mid_point_range:
            
            xmask = np.floor(labels_array[:,0]/quad_size)%2==0
            ymask = np.floor(labels_array[:,1]/quad_size)%2==0
            x_reverse = np.array([not value for value in xmask])
            y_reverse = np.array([not value for value in ymask])
            
            quads_array[xmask*ymask,count] = 0
            quads_array[x_reverse*ymask,count] = 1
            quads_array[xmask*y_reverse,count] = 2
            quads_array[x_reverse*y_reverse,count] = 3
        
            count+=1
    
    
                
        np.save(f'realistic_features_with_intensity_{window_size}_increments{increments}.npy', imgs_array)
        print('features saved')
        np.save(f'realistic_coords__with_intensity_{window_size}_increments{increments}.npy', labels_array)
        print('coordinates saved')
        np.save(f'realistic_labels__with_intensity_{window_size}_increments{increments}.npy', quads_array)
        print('labels saved')


def calculate_normal(depth_map):
    gradient_x = np.gradient(depth_map, axis=1)
    gradient_y = np.gradient(depth_map, axis=0)

    normal_x = gradient_x
    normal_y = gradient_y
    normal_z = np.ones_like(depth_map)

    normal_magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal = np.stack((normal_x / normal_magnitude, normal_y / normal_magnitude, normal_z / normal_magnitude), axis=-1)

    return normal

def generate_random_normals(num_samples, azimuth_range=(0, 360), elevation_range=(30, 90)):
    # Generate random azimuth and elevation angles
    azimuth = np.random.uniform(*azimuth_range, size=num_samples)
    elevation = np.random.uniform(*elevation_range, size=num_samples)

    # Convert spherical coordinates to Cartesian coordinates (xyz)
    x = np.sin(np.radians(elevation)) * np.cos(np.radians(azimuth))
    y = np.sin(np.radians(elevation)) * np.sin(np.radians(azimuth))
    z = np.cos(np.radians(elevation))

    # Stack the coordinates to form the normal vectors
    normals = np.stack((x, y, z), axis=-1)

    return normals

def generate_random_intensity(n_intensity, cropped_img, visualize = False):

    img_size = cropped_img.shape[0]
    ls = []
    for _ in range(n_intensity):
        
        elev = random.uniform(0,150)
        az = random.uniform(0,90)
        ls_iter = generate_xyz_from_aze(az,elev)
        ls.append(ls_iter)
    
    # ls = generate_random_normals(n_intensity)
    
    normal = calculate_normal(cropped_img)
    
    various_intensity = []
    for l_2 in ls:
        column_normal = normal.reshape(img_size*img_size,3)
        ln = np.dot(l_2, column_normal.T)
        ln[ln<0] = 0
    
        various_intensity.append(ln)
    
        if visualize == True:
            
            plt.imshow(ln.reshape(cropped_img.shape[0],cropped_img.shape[0]))
            plt.scatter(np.array([64,64,256+64,256+64]),np.array([64,256+64,64,256+64]))
            plt.show()

    return np.array(various_intensity)

def euler_to_azimuth_elevation(roll, pitch, yaw):
    # Convert angles from degrees to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    # Calculate azimuth and elevation
    azimuth = math.atan2(math.sin(yaw_rad), math.cos(yaw_rad) * math.sin(roll_rad) - math.sin(pitch_rad) * math.cos(roll_rad))
    elevation = math.asin(math.sin(pitch_rad) * math.sin(roll_rad) + math.cos(pitch_rad) * math.cos(roll_rad) * math.cos(yaw_rad))

    # Convert angles from radians to degrees
    azimuth_deg = math.degrees(azimuth)
    elevation_deg = math.degrees(elevation)

    return azimuth_deg, elevation_deg

def generate_xyz_from_aze(azimuth, elevation):
    x = np.sin(np.radians(elevation)) * np.cos(np.radians(azimuth))
    y = np.sin(np.radians(elevation)) * np.sin(np.radians(azimuth))
    z = np.cos(np.radians(elevation))

    # Stack the coordinates to form the normal vectors
    normals = np.stack((x, y, z), axis=-1)

    return normals