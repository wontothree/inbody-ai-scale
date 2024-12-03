# InBody Scale AI

![](./icons/inbody-ai-scale.png)

- Height estimation solution using a low performance monocular camera and limied computing resources.
- Face Identification solution along with the pipeline and database

Existing height estimation algorithms and papers using vision assume that the user’s entire body is within the camera’s field of view. However, due to the mechanical characteristics of the InBody device, the user’s full body does not fit within the camera’s field of view. Our attempt to estimate height under these conditions is unprecedented. In our first experiment, we achieved an average error of 0.6% and a maximum error of 1%.

We tackled the ultimate problem of height estimation by breaking it down into sub-problems where computer vision AI can excel: Object Detection and Semantic Segmentation. To achieve this, we designed and utilized a structure that allows the camera to move via motors and rails.

- monocular camera
