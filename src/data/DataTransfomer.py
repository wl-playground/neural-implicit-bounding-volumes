from src.data.ImageDataLoader import ImageDataLoader


class DataTransformer:
    @staticmethod
    def image_to_csv(image_filepath, csv_filepath):
        normalised_image = ImageDataLoader.normalise_train(
            ImageDataLoader.load_binary_image(image_filepath)
        )

        x_train = ImageDataLoader.get_x_train(
            image_width=normalised_image.shape[0],
            image_height=normalised_image.shape[1]
        )

        y_target = ImageDataLoader.get_y_target(normalised_image)

        result = zip(x_train, y_target)

        file = open(csv_filepath, "w")
        file.write("pixel_x_coordinate,pixel_y_coordinate,y_target\n")

        for pixel_coordinate, y_target in result:
            for coordinate in pixel_coordinate:
                file.write("{},".format(coordinate))

            file.write("{}\n".format(y_target))

        file.close()
