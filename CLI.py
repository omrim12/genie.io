from termcolor import cprint
from img_utils import analyze_food_img
from cnn_utils import classify_client_input
from edamam_api_utils import prompt_food_info


class CLI:
    """
    This class provides a CLI instance
    for interacting with ginny.io food image classifier
    using Edamame API analyzed food type data specs
    """
    def __init__(self, ginny_model):
        self.ginny_model = ginny_model

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
                        cprint(classify_client_input(image_array=image_array, cnn_model=self.ginny_model), 'green')
                        # prompt_food_info(classify_client_input(image_array=image_array, cnn_model=self.ginny_model))
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
