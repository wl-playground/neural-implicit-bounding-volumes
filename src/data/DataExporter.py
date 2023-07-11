from google.colab import files
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


# for saving experiment results
class DataExporter:
    def __init__(self):
        self.directory_path = "/content/"
        self.directory_name = ""
        self.csv_file = None
        self.filename = ""

    # create directory to save results in
    def _create_directory(self, directory_name):
        Path("/content/{}".format(directory_name)).mkdir(parents=True, exist_ok=True)

        self.directory_path = "/content/{}/".format(directory_name)
        self.directory_name = directory_name

    # append a single line to the csv file
    def _append_line_to_csv(self, filename, line):
        if self.csv_file is None:
            self.csv_file = open("{}{}".format(self.directory_path, filename), "w")
            self.filename = filename

        self.csv_file.write("{}\n".format(line))

    # save matlibplot figures
    def _save_figure(self, original, reconstruction, iteration, vmin=0, vmax=1):
        fig, axs = plt.subplots(1, 3, figsize=(17, 9))

        axs[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
        axs[1].imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax)
        axs[2].imshow(original, cmap='jet', vmin=vmin, vmax=vmax, interpolation='none')
        axs[2].imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none')

        plt.savefig("{}{}_{}.png".format(self.directory_path, self.directory_name, iteration))

        plt.close()

    # zip and download results folder
    def _download_directory(self):
        if self.csv_file is not None:
            self.csv_file.close()

        path = self.directory_path[:-1]

        shutil.make_archive(path, 'zip', self.directory_path)

        files.download("{}.zip".format(path))

    # wrapper around above methods
    def export_results(self, name, save_results):
        self._create_directory(name)

        self._append_line_to_csv(
            filename="{}.csv".format(name),
            line="class weights,learning rate,iteration,false positives,false negatives,loss"
        )

        for iteration, iteration_results in save_results.items():
            line = ""

            for key, value in iteration_results.items():
                if key != "original" and key != "reconstruction":
                    line += "{},".format(value)

            self._append_line_to_csv(filename="{}.csv".format(name), line=line[:-1])

            self._save_figure(original=iteration_results["original"],
                              reconstruction=iteration_results["reconstruction"], iteration=iteration)

        self._download_directory()
