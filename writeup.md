## Project: Search and Sample Return

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[image4]: ./output/read_example_images.png
[image5]: ./output/mapping_obstacles.png
[image6]: ./output/find_rocks.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images.
```python
example_grid = '../calibration_images/example_grid1.jpg'
example_rock = '../calibration_images/example_rock1.jpg'
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(grid_img)
plt.subplot(122)
plt.imshow(rock_img)
```
Show the images as below
![alt text][image4]

#### 2. Modify functions to allow for color selection of obstacles.
I modified the function `perspect_transform` to output mask which refresents the color selection of obstacles by mapping a all-white image using the same matrix as mapping to world.
```python
# Define a function to perform a perspective transform
# I've used the example grid image above to choose source points for the
# grid cell in front of the rover (each grid cell is 1 square meter in the sim)
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))
    return warped, mask


# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
warped, mask = perspect_transform(grid_img, source, destination)
fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(warped)
plt.subplot(122)
plt.imshow(mask, cmap='gray')
#scipy.misc.imsave('../output/warped_example.jpg', warped)
```
![alt text][image5]

#### 3. Add functiuons to allow for color selection of rock samples.
To select rocks, the function `find_rock` is added.
```python
def find_rocks(img, levels=(110, 110, 50)):
    rockpix = ((img[:, :, 0] > levels[0]) \
               & (img[:, :, 1] > levels[1]) \
               & (img[:, :, 2] < levels[2]))
    color_select = np.zeros_like(img[:, :, 0])
    color_select[rockpix] = 1
    return color_select

rock_thred = find_rocks(rock_img)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(rock_img)
plt.subplot(122)
plt.imshow(rock_thred, cmap='gray')
```
![alt text][image6]

#### 4. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap. 
```python
def process_image(img):
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                      [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      ])

    # 2) Apply perspective transform
    warped, mask = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable_threshed = color_thresh(warped)
    rock_threshed = find_rocks(warped)

    # 4) Convert thresholded image pixel values to rover-centric coords
    navi_xpix, navi_ypix = rover_coords(navigable_threshed)
    mask_xpix, mask_ypix = rover_coords(mask)
    rock_xpix, rock_ypix = rover_coords(rock_threshed)

    # 5) Convert rover-centric pixel values to world coords
    scale = dst_size * 2
    navi_x_world, navi_y_world = pix_to_world(navi_xpix, navi_ypix, data.xpos[data.count], 
                                    data.ypos[data.count], data.yaw[data.count], 
                                    data.worldmap.shape[0], scale)
    mask_x_world, mask_y_world = pix_to_world(mask_xpix, mask_ypix, data.xpos[data.count], 
                                    data.ypos[data.count], data.yaw[data.count], 
                                    data.worldmap.shape[0], scale)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, data.xpos[data.count], 
                                    data.ypos[data.count], data.yaw[data.count], 
                                    data.worldmap.shape[0], scale)

    data.rock_world[rock_y_world, rock_x_world] = 1
    data.worldmap[rock_y_world, rock_x_world, :] = 255 

    data.worldmap[navi_y_world, navi_x_world, 2] = 255
    data.worldmap[mask_y_world, mask_x_world, 0] = 255

    navi_pix = (data.worldmap[:, :, 2] > 0) & (data.rock_world[:, :, 0] < 1)
    data.worldmap[navi_pix, 0] = 0

    # 7) Make a mosaic image, below is some example code
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
    output_image[0:img.shape[0], 0:img.shape[1]] = img
    output_image[0:img.shape[0], img.shape[1]:] = warped

    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
```

#### 5. Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
Please checkout `output/test_mapping.mp4`


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) function in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
```python
def perception_step(rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img

    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32(
        [[rover.img.shape[1] / 2 - dst_size, rover.img.shape[0] - bottom_offset],
         [rover.img.shape[1] / 2 + dst_size, rover.img.shape[0] - bottom_offset],
         [rover.img.shape[1] / 2 + dst_size, rover.img.shape[0] - 2 * dst_size - bottom_offset], 
         [rover.img.shape[1] / 2 - dst_size, rover.img.shape[0] - 2 * dst_size - bottom_offset]])

    # 2) Apply perspective transform
    warped, mask = perspect_transform(rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obstacle_map = np.absolute(np.float32(threshed) - 1) * mask

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    rover.vision_image[:, :, 0] = obstacle_map * 255
    rover.vision_image[:, :, 2] = threshed * 255

    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)

    # 6) Convert rover-centric pixel values to world coordinates
    world_size = rover.worldmap.shape[0]
    scale = 2 * dst_size
    x_world, y_world = pix_to_world(xpix, ypix, rover.pos[0], rover.pos[1],
                                    rover.yaw, world_size, scale)
    obs_xpix, obs_ypix = rover_coords(obstacle_map)
    obs_x_world, obs_y_world = pix_to_world(obs_xpix, obs_ypix,
                                            rover.pos[0], rover.pos[1],
                                            rover.yaw, world_size, scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    rover.worldmap[y_world, x_world, 2] += 10
    rover.worldmap[obs_y_world, obs_x_world, 0] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(xpix, ypix)

    # Update Rover pixel distances and angles
    rover.nav_angles = angles

    rock_map = find_rocks(warped)
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, rover.pos[0], rover.pos[1],
                                                  rover.yaw, world_size, scale)
        rock_dist, rock_angle = to_polar_coords(rock_x, rock_y)
        rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_x_workd[rock_idx]
        rock_ycen = rock_y_workd[rock_idx]
        rover.worldmap[rock_ycen, rock_xcen, 1] = 255
        rover.vision_image[:, :, 1] = rock_map * 255
    else:
        rover.vision_image[:, :, 1] = 0

    return rover
```

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup. 
The results are as expected. But need to improve the fidelity. Maybe Kalman filter helps.
