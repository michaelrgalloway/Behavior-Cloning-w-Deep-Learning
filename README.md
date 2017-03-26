## Behavior Cloning Project - SDCND

###Data Analysis
I attempted to drive around the track on my computer however I didnt have a joystick, this resulted in getting training data that was not very balanced and all efforts to train off of the keyboard driven car data didnt turn out well. I then found that Udacity provided data to download for the students to use so I used that instead. 

While studying the data I found that there was much much more images/angles that represented a ~0.0 angle which I didn't see obvious at first and only when I tried my model on the simulator and it would go straight when it should have made a turn. After this realization, I implemented code to only try to pull from the entire dataset only 10% of the time. 

I also noticed immediately that far more turns were making a left turn and just like the bias going straight, there would also be a bias going left. I got past this bias by pulling left and right turn angles on a 50% chance of each. I also implemented flipping of images with angles randomly to fabricate turns of the opposite direction of a mirrored turn. 

Another aspect of the sample images was that there was unneeded data inside the image. For instance a good portion of the top of the images was just of the sky or treetops and doesn't necessarily add to the data needed to calculate what angle to turn on and would be considered as noise. I cropped the upper portion of the images off to try to get rid of the noise. 


###Preprocessing Images
I would read the rows of the data csv file (all columns) via pandas and feed this to a data generator function.

My model uses a data generator and inside of my handler function for this generator I have implemented the following operations per data row:

####Randomnly select an image orientation
I would uniformly select an image from the right, left or center images of this row. If the left image was selected I would add .25 to the angle and of the right image was selected I would subtracted -.25. This angle augmenation is done so if the car were to drift to the edges of the road, the model can recognize this and reshift its steering back towards the center of the lane. 

####Crop the image
Since the top portion of the image is mostly noise, I crop this portion out and try to reduce the noise as much as possible this results in more contributing data to the steering angle calculations.

####Resize the image
I found that reducing the size of the image reduces the input size to the model. This also makes it possible to rely on weights heavily catered to the input plane and also made the model generalize better on the track. I resized the image to 32 x 16 instead of the original image of 320 x 160. This was a dramatic decrease in inputs to tune in the model outout sacrificing performance (actually gaining performance)

###Keras built in functions preprocessing
While reading and researching Keras documentation I found that Keras has a wide array of functionality to hepl the implementer with preprocessing and augmentation of data. For this project I have used the following:

ImageEnhance.Brightness - I used this to either darken or brighten the image. I put 30% chance that the image brightness will get adjusted.

ImageOps.posterize - This function reduces the amount of information that a a color channel represents. Since our images do not needs an exact color representation, I have halved the bit representation from 8 to 4. This will increase training performance.

ImageOps.mirror - This will flip the image horizontally. I put a 40% chance to flip the image. If the image did get flipped I also flipped the angle accordingly. 

ImageDataGenerator - This function is actually an iterator and can act as a data generator in itself. I used this within my own generator because I wanted to flip the images as well as other operations that this iterator doesnt have. The features I did use in this image manipulator are:

rotation_range- will randoming rotate the image to a maximum of of the supplied value. I supplied 5 degrees.

width_shift_range - this will shift the image left or right by the given percent and fill the empty space with black (supplied by the val parameter). I supplied a value of .1 for this parameter for a random shift to upward max of 10% of the image. 

height_shift_range - same as the parameter above but a vertical shift range instead of horizontal. 

channel_shift_range - this adjusts the color channel by a percentage applied, randomly. I supplied a max 20% color channel random shift. 

horizontal_flip - I did NOT use this in the image augmentation generator. The reason is because if the image were to be flipped, I would also need to change the angle and at runtime I would not know what functions were applied to the image and could adjust the angle accordingly. I decided to flip the image manually. Set this paramter to False. 

There are more paramters to this image generator, however I limited the parameters the paramters above. 



 


###Building Model

Building the model I used a Keras Sequential model. 

The first layer I added was a lambda function layer. This was to normalize the data to closer to a unit norm. This layer take an input size of the preprocessed images of 16 x 32 x 3. H x W x C. This layer is implicitly followed by an activation of ELU as supplied by the activation parameter.

The remaining layers do not take an input size as a parameter since they are inferred by the previous layer leading into it. 

2nd layer used was a convolutionalD2 layer of a hidden layer depth of 32 and a filter size of 5 x 5. This will filter the input data from a 5 x 5 dimension for 32 patterns. There is a subsample of 2 x 2 which acts as a pooling output. This will decrease the dimenstion from 32 x 16 to 16 x 8 to the next layer with 32 filters. This layer is implicitly followed by an activation of ELU as supplied by the activation parameter.

3rd layer is another convolutional layer to take 16 convolutoions per filter of the previous layer. This layer uses a filter size of 3 x 3. I left the subsample at constant so it will keep the same dimensions. This layer is implicitly followed by an activation of ELU as supplied by the activation parameter.

I now perform a dropout of 20% of nodes of the graph thus far. This will help to combat against overfitting. 

4th layer is another convolutional layer which adds a complexity of another x16 layers to the current convolutions with another 3 x 3 filter size. Again I leave the pooling at a constant. 

I then flatten the the graph to result in a flat array of 2048 elements. I feed this data into a fully connected layer of size 1024, reducing the elements by half. I then add another dropout layer to again help to combat against overfitting following by an excplicit ELU activation function. 

The next two layers are fully connected layers. First of size 512 with a activation of ELU and then with the size of 1. The output of the last dense FCL is our predicted steering angle for the given image. 

At compile time, this model is supplied an Adam optimizer with learning rate of 0.001 and a loss function of mean squared error since we are targeting a regression problem.

###Running Model
First I shuffle the data that is read from the csv file so we can have a uniformly random dataset to start off with. I then split the data into a training and validation set. 

I create an instance of my generator with the training dataset. The iterator will yeild a batch of augmented images and angles at training time. 

I also create another instance of the generator for the validation set. This ensures that our validation set is not static and validation scores are represented with dynamically augmented and generated images. 

I provide these iterator instances to Keras' fit_generator method along with precomputed constants like number of epochs and batch size. I chose 20 epochs and a batch size of 32.
###Results
Before the decision to be biased against 0.0 angles in my data generator, by car would consistantly drive straight when it shoud have made a turn. However after the adjustment my car started to make turns but not reliably enough to get around the track. 

I then iteratively started to introduce preprocessing into my images. The first largest improvement was when I reduced the size of the images, first to 64 x 64 then to 32 x 16. This showed a major improvement but not quite enough to get reliably around the track full circle. Analyzing the resulting images and introducing more preoprocssing operations, the car improved. Another big discovery that helped is the bult in Keras' function ImageDataGenerator which gives a powerful tool to generate augmented images. This improved the result quite a bit.

I then experimented with other paramters such as learning rate, epochs, batch size and various model layers until I found a combination that made the car drive itself around the tracks by itself. 


###Lessons Learned

Quality data is key, quality data is life. A big lesson I learned with this project is that no matter how good my model is or what I try to tune paramterwise, I will always get subpar results of what I am feeding into is garbage. I learned in this project that I should especially pay attention to what is regarded as good data and that include sinherited bias from the data, noise, the ability to create new data from existing data, the toolsets out there to help with this, etc.. 

Before this project I was planning on focusing on only the Keras part of the project, however now i realize that no matter what model or training tools I use, I need to know the data inside and out and make sure what I am feeding into my models has gotten the attention it deserves. It feels that I should be paying a considerable more attention to data preparation and data generation than the actual training model itself. 

Lesson: respect the data. 

Another lesson I learned was to record every step along the way. There has been multiple times where I would get the car to a reasonable state and would make it around the track but I wanted to reduce the swerving. I would make some very minor adjsutments to my code only to come back after training and test and find that what I did made the performance worse. Thats fine, Ill just revert what I did however I had forgetten what I adjusted and my undo didnt work because I reloaded the IDE. Big lesson, record every possible change I make to my model and my data prep in case I need to revert. I lost hours because of this. 

Lesson: record everything

I also learned a deal about how correlated data can be combined and reduced into a smaller dataset without losing crucial information. Reducing the size of the input did 'lose' information as it now has dramatically less pixels however the take away is that sometimes that its ok because these pixels that were merged together in the process of resizing actuall had positive correlation with each other so we actually retained alot of information. In looking back I probably even could have merged the the color channels into a single channel and still able to get good results. 

Lesson: can reduce correlated data without losing much information


All in all, I think this project was a very valuable project in terms of overall domain. Looking forwad to the next. 
