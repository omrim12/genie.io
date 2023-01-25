from termcolor import cprint
from img_utils import analyze_food_img
from cnn_utils import classify_client_input
from edamam_api_utils import get_food_info, get_recipes_info


class CLI:
    """
    This class provides a CLI instance
    for interacting with genie.io food image classifier
    using Edamame API analyzed food type data specs
    """
    def __init__(self, genie_model):
        self.genie_model = genie_model

    def session(self):
        exit_cli = False

        while not exit_cli:
            # Show usage prompt
            cprint(self.usage_prompt, "yellow")

            # Get client input
            command = input("#> ").split(" ")

            # Classifier option
            if command[0] == 'wish':
                if len(command) != 2:
                    cprint("Invalid command. Please try again\n", "red")
                else:
                    # Convert given image to numpy array
                    image_array = analyze_food_img(command[1])
                    if image_array is not None:
                        food_type = classify_client_input(image_array=image_array, cnn_model=self.genie_model)
                        self.__prompt_food_data(food_type=food_type)
                    else:
                        cprint(f"Invalid path to image given. Please try again\n", "red")

            elif command[0] == 'exit':
                if len(command) > 1:
                    cprint("Invalid command. Please try again\n", "red")
                else:
                    exit_cli = True

    @property
    def usage_prompt(self):
        return ("\n~~~~~~ CLI usage options: ~~~~~~\n"
                "#> wish <food_image_path> - classify food input image\n"
                "#> exit                        - exit CLI session\n"
                "#> help / <any-other-command>  - show usage options\n")

    @staticmethod
    def __prompt_food_data(food_type):
        cprint(f"Your image has been classified as {food_type}!\n"
               f"Here are some {food_type} recipes to get started:\n", "green")
        cprint(get_recipes_info(food_type), color="green")
        # TODO: add prompt for food data once table code is created
