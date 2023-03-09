import os

# set the path to the directory containing the BibTeX files
directory = "D:/Academic/MSc/BIBA/task2"

# set the path to the output txt file
output_file = "D:/Academic/MSc/BIBA/task2/file.txt"

dir_list = ["scopus1.bib", "scopus2.bib", "scopus3.bib", "scopus4.bib", "scopus5.bib", "scopus6.bib", "scopus7.bib", "scopus8.bib", "scopus9.bib", "scopus10.bib", "scopus11.bib", "scopus12.bib", "scopus13.bib", "scopus14.bib", "scopus15.bib", "scopus16.bib", "scopus17.bib", "scopus18.bib", "scopus19.bib", "scopus20.bib", "scopus21.bib","scopus22.bib", "scopus23.bib", "scopus24.bib", "scopus25.bib", "scopus26.bib", "scopus27.bib", "scopus28.bib", "scopus29.bib", "scopus30.bib", "scopus31.bib", "scopus32.bib", "scopus33.bib", "scopus34.bib", "scopus35.bib", "scopus36.bib", "scopus37.bib", "scopus38.bib", "scopus39.bib"]

# open the output file for writing
with open(output_file, "w", encoding="utf-8") as f_out:

    # loop over the files in the directory
    for filename in dir_list:

        # check if the file is a BibTeX file
        if filename.endswith(".bib"):

            # open the BibTeX file for reading
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f_in:

                # loop over each line in the BibTeX file
                for line in f_in:

                    # write the line to the output file
                    f_out.write(line)

                    # add a blank line after each entry
                    if line.startswith("}"):
                        f_out.write("\n")
