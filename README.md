# Productivity State Detection Model
The productivity state detection module is intended to identify productivity states of the workers, which are defined as follows: (i) value-adding work; any state that involves movements directly leading to completion of the activity (e.g., painting a wall with a paint roller); (ii) non-value-adding work: any state that involves movements indirectly leading to completion of the activity (e.g., mixing paint in a container); (iii) walking: the participant is traveling; and (iv) idling: the participant is not working (e.g., talking, resting, etc.). In this study, the productivity state is considered value-adding work once a worker performs the main task(s), and it is considered non-value-adding work once the worker performs the secondary task(s). The workers' body movements for the non- and value-adding work states vary based on each crew's main and secondary task(s), whereas all workers have the same body movements for the waking and idling states. The productivity state detection module comprises six models, and each is individually trained for a specific worker. Every model consists of (i) a frequency image generator which segments the tri-axial accelerometer data from the receiving beacons and converts them into frequency images; and (ii) a productivity state detection model, which is a two-dimensional Convolutional Neural Network (CNN) trained to detect the workers' productivity state by taking the generated frequency images as input.
