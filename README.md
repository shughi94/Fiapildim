# Chord detector

This project has a trainer that builds a model that detects guitar chords.
Run `create_training_data.py` to create the training data + model

e.g. output: 
~~~
...
Saved training example for Dm
Added Dm chord to training data.
Get ready to play chord: F
PLAY NOW!
Recording...
Finished recording
Saved training example for F
Added F chord to training data.
Loaded 45 training samples
....
Epoch 43/50
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 0.1389 - loss: 2.1975 - val_accuracy: 0.1111 - val_loss: 2.1972
....
~~~

Then we can run `main.py` that listens and uses the model to predict the guitar chord

## Note
This project does not work at all (keeps saying that everything is an E chord), but I like the idea

Also it's more of a classification model rather than a predicting one, but hey I should have remembered last weekend to do this project :(