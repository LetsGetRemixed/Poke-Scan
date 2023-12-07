import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras

from Train_model import train_model
from build_data import build_data

def main():

    #Global variabes

    # The base training data only contains 204 samples. For more, we augment the data, so this variable will
    # determine the amount of additional augmentations should be done per card.
    # Here, 199 implies that there will be 200 samples of each card. 1 from the base image, and 199 additional augmentations for 200 samples total
    num_augment = 199

    # Main program for users to add/remove cards and generate report of their collection
    current_directory = os.path.abspath(os.path.dirname(__file__))
    parent_directory = os.path.dirname(current_directory)
    # Add the parent directory to sys.path
    sys.path.append(parent_directory)
    input_path = os.path.join(current_directory,"input")
    file_list = os.listdir(input_path)

    if os.path.exists('Collection.npy') == False:
        print("Card collection 'collection.npy' not found, creating new card collection...")
        collection = np.zeros(204,np.uint)
        np.save('Collection.npy', collection)
    else:
        print("Card collection 'collection.npy' found, loading collection...")
        collection = np.load('Collection.npy')


    label_names = [
        "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Chameleon", "Charizard", 
        "Squirrel", "Wartortle", "Blastoise", "Coterie", "Metapod", "Butterfree", 
        "Weedle", "Kakuna", "Bedroll", "Pudgy", "Pidgeotto", "Pidgeot", "Rattata", 
        "Raticate", "Sparrow", "Fearow", "Ekans", "Arbor", "Pikachu", "Raichu", 
        "Sandshrew", "Sandslash", "Nidoran", "Nidorina", "Nidoqueen", "Nidoran", 
        "Nidorino", "Nidoking", "Clefairy", "Clefable", "Vulpine", "Nineties", 
        "Jigglypuff", "Wigglytuff", "Zubat", "Golbat", "Oddish", "Gloom", 
        "Vileplume", "Paras", "Parasect", "Venonat", "Venomoth", "Diglett", 
        "Dugtrio", "Meowth", "Persian", "Psyduck", "Golduck", "Mankey", 
        "Primeape", "Growlithe", "Arcane", "Poliwag", "Poliwhirl", "Poliwrath", 
        "Abra", "Kadabra", "Alakazam", "Machop", "Machoke", "Machamp", 
        "Bellsprout", "Weepinbell", "Victreebel", "Tentacool", "Tentacruel", 
        "Geodude", "Graveler", "Golem", "Ponyta", "Rapidash", "Slowpoke", 
        "Slowbro", "Magnemite", "Magneton", "Farfetchd", "Doduo", "Dottie", 
        "Seel", "Dugong", "Grimer", "Muk", "Shellder", "Cloyster", 
        "Gastly", "Haunter", "Gengar", "Onix", "Drowse", "Hypno", 
        "Krabby", "Kingler", "Voltorb", "Electrode", "Execute", "Exeggutor", 
        "Cubone", "Marowak", "Hitmonlee", "Hitmonchan", "Licking", "Koffing", 
        "Weezing", "Rhyhorn", "Rhydon", "Chansey", "Tangela", "Kangaskhan", 
        "Horsea", "Seadra", "Golden", "Speaking", "Staryu", "Starmie", 
        "MrMine", "Scyther", "Jynx", "Electabuzz", "Magmar", "Pincer", 
        "Tauros", "Magikarp", "Gyarados", "Lapras", "Ditto", "Eevee", 
        "Vaporeon", "Jolteon", "Flareon", "Porygon", "Omanyte", "Omastar", 
        "Kabuto", "Kabutops", "Aerodactyl", "Snorlax", "Articuno", "Zapdos", 
        "Moltres", "Dratini", "Draonair", "Dragonite", "Mewtwo", "Mew", 
        "DomeFossil", "HelixFossil", "OldAmber", "BigAirBalloon", "BillsTransfer", 
        "CyclingRoad", "DaisysHelp", "EnergySticker", "ErikasInvitation", 
        "GiovannisCharisma", "Grabber", "Leftovers", "ProtectiveGoggles", 
        "RigidBand", "Bulbasaur", "Ivysaur", "Charmander", "Chameleon", 
        "Squirrel", "Wartortle", "Caterpie", "Pikachu", "Nidoking", "Psyduck", 
        "Poliwhirl", "Machoke", "Tangela", "MrMime", "Omanyte", "Dragonair", 
        "Venusaur", "Charizard", "Blastoise", "Arbor", "Nineties", "Wigglytuff", 
        "Alakazam", "Golem", "Kangaskhan", "Jynx", "Zapdos", "Mew", 
        "BillsTransfer", "DaisysHelp", "ErikasInvitation", "GiovannisCharisma", 
        "Venusaur", "Charizard", "Blastoise", "Alakazam", "Zapdos", 
        "ErikasInvitation", "GiovannisCharisma"
    ]

    model = 0

    print("Initiated pokemon card collection tracker")

    while True:
        choice = input("What would you like to do? ('add' - add cards from input folder, 'remove' - remove cards from input folder, 'report' - generate report, 'quit')")

        if(choice.upper() == "QUIT"):
            break
        else:
            while True:
                if(choice.upper() == "ADD"):
                    break
                elif(choice.upper() == "REMOVE"):
                    break
                elif(choice.upper() == "REPORT"):
                    break
                else:
                    choice = input("Invalid input, try again ('add' - add cards from input folder, 'remove' - remove cards from input folder, 'report' - generate report)")

            if((choice.upper() == "ADD") | (choice.upper() == "REMOVE")):
                if os.path.exists('card_model.keras') == False:
                    labels = []
                    for i in range(1,205):
                        for _ in range(1,num_augment+2):
                            labels.append(i)
                    labels = np.array(labels)
                    print("\nCNN model card_models.keras for recognizing cards not found, creating and training a new model")
                    if os.path.exists('train_data.npy') == False:
                        print("Training data train_data.npy not found, building new training data from uncompiled_train_data.npy")
                        build_data(num_augment)
                    train_data = np.load('train_data.npy')
                    model, _ = train_model(train_data,labels)
                    model.save('card_model.keras')
                    print("CNN model finished training and saved to file")
                else:
                    if(model == 0):
                        print("\nLoading model...")
                        model = keras.models.load_model('card_model.keras')
                
                if(choice.upper() == "ADD"):
                    change = 1
                elif(choice.upper() == "REMOVE"):
                    change = -1


                for file in file_list:
                    print("\nReading and analyzing file '",file, "'")

                    base = np.array(cv.imread( os.path.join(input_path,file), cv.COLOR_RGB2BGR))

                    top_row, bottom_row, width, center = get_card_dimension(base, 9)

                    if(bottom_row == len(base)):
                        top_row, bottom_row, width, center = get_card_dimension(base, 2)

                    capped_left = max(int(center[0] - width/2), 0)
                    capped_right = min(len(base[0]),int(center[0] + width/2))

                    window = cv.cvtColor(base[top_row:bottom_row,capped_left:capped_right], cv.COLOR_BGR2GRAY)
                    window = cv.resize(window, (165,230))
                    window = window[23:109, 14:152]
                    window = np.divide(window,255)
                    window = window.reshape(1,86,138,1)
                    prob = model.predict(window)
                    prob.ravel()
                    collection = change_collection(collection, prob, label_names, change)

                print("Saving collection")
                np.save('Collection.npy', collection)

            if(choice.upper() == "REPORT"):
                print("\nPOKEMON CARDS IN COLLECTION\n---------------------------")
                for i in range(len(collection)):
                    if(collection[i] > 0):
                        print(label_names[i],": ", int(collection[i]))

            choice = input("\nFinished; quit or continue? ('quit','continue')")
            while True:
                if(choice.upper() == "QUIT"):
                    return
                elif(choice.upper() == "CONTINUE"):
                    break
                else:
                    choice = input("invalid input, try again; quit or continue? ('quit','continue')")

    
    
def change_collection(temp_collection, probability, label_names, change):
    is_card = input(("Is this card " + str(label_names[np.argmax(np.array(probability))-1]) + "? (Y/N)"))

    while True:
        if(is_card.upper() == "Y"):
            break
        elif(is_card.upper() == "N"):
            break
        else:
            is_card = input("Invalid input, try again (Y/N): ")

    if(is_card.upper() == "Y"):
        card_idx = np.argmax(np.array(probability))-1

        if(temp_collection[card_idx] + change < 0):
            print("Value for",label_names[card_idx], "is already 0, cannot remove more cards")
            return temp_collection
        else:
            temp_collection[card_idx] = temp_collection[card_idx] + change

        if(change == 1):
            print("Added ",str(label_names[card_idx]), " to collection")
        elif(change == -1):
            print("Removed ",str(label_names[card_idx]), " from collection")

    elif(is_card.upper() == "N"):
        actual_card = input("What was the actual card (input 'options' for list of available inputs)? ")

        while((actual_card in label_names) == False):
            if(actual_card == "options"):
                print("Here are the available card input options: ",label_names)
                actual_card = input("What was the actual card (input 'options' for list of available inputs)? ")
            else:
                actual_card = input("invalid input; what was the actual card (input 'options' for list of available inputs)?")

        temp_collection[label_names.index(actual_card)] = temp_collection[label_names.index(actual_card)] + change

        if(change == 1):
            print("Added ",actual_card, " to collection")
        elif(change == -1):
            print("Removed ",actual_card, " from collection")

    return temp_collection

def get_card_dimension(img, kernel_size):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    lower = np.array([1,1,1])
    upper = np.array([255,50,255])
    mask = cv.inRange(hsv, lower, upper)
    erodeKernel = np.ones((kernel_size,kernel_size),np.uint8)
    mask = cv.dilate(mask,erodeKernel)

    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=4)
    max_label, max_size = max([(i, stats[i,cv.CC_STAT_AREA]) for i in range(1, nb_components)], key= lambda x:x[1])
    mask[output != max_label] = 0

    top_row = int(stats[max_label,cv.CC_STAT_TOP])
    left_column = int(stats[max_label,cv.CC_STAT_LEFT])
    bottom_row = top_row + int(stats[max_label,cv.CC_STAT_HEIGHT])
    right_column = left_column + int(stats[max_label,cv.CC_STAT_WIDTH])
    width = int((bottom_row - top_row) / 1.4)

    return top_row, bottom_row, width, centroids[1]


if __name__ == '__main__':
    main()
