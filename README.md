# Pokemon Card Tracker

Keeping track of Pokemon cards in your collection can be time-consuming, and keeping accurate count of all cards is difficult. With this solution, import individual images of each card and simply choose to "add" them to the collection, or "remove" the cards that are imported. A report of the cards saved in the collection can also be generated


# Getting Started

Run main.py to bring the initial prompt, where you can choose to **add cards, remove cards, generate a report, or quit the program.**

If this is your first time running the program, a new empty card collection will be initialized.

To **add/remove** cards to your collection, before running the program, **upload individual images of each card in your card collection to the "input" folder** to be added to the digital collection.

When adding/removing, the cards will be classified and **you will be asked if each card was classified correctly**. If not, you can input "options" to view a list of acceptable cards for the collection.

Choosing to **generate a report** will print a report to the console of the **amount of each card currently in your collection.**

Once finished, you will be prompted to either quit or continue. **Continuing returns back to the main prompt.**

On the first call to add or remove cards, a convolutional neural network will need to be trained to classify cards. Depending on your CPU, the training process could take **30 to 60** minutes.

## Test examples
The "input" folder includes some preprepared testcases to use on the model once the model has been trained. Feel free to experiment and upload your own images to the "input" folder to stress test the model. Make sure the **images you upload are in .bmp format** You should notice that ideal scenarios where an original digital image of the card facing upwards against a colored, saturated background work best.

## Requirements
To run the program, you will need to have the following python packages installed:
**opencv-python**,
**tensorflow**,
**scikit-learn**,
