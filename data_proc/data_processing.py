import pandas as pd
import numpy as np
import os
import re
import argparse
from data_utils import (
    clean_label,
    mass_label,
    aggressive_label,
    met_label,
    clean_txt,
    findings_sect,
    impressions_sect,
    n_tokens,
    report_sections,
)
import codecs
import random


def organize_labeled_data(df, met_type, report_path):
    '''
    df: Dataframe that contains the labels for each report (Column Names: File Name[ID_reportType1.txt], Mass, Aggressive) 
    met_info_file: csv file that contains the labels for Metastasis label 
    report_path: path for where the labeled radiology reports are kept 
    '''

     # since two files are combined together, there are two same columns named this

    df = df.fillna('NA')

    # clean the annotations
    df["Mass"] = df.Mass.apply(clean_label)
    # df["N_Mass"] = df.N_Mass.apply(clean_label)
    df["Aggressive"] = df.Aggressive.apply(clean_label)
    if met_type == 'string':
        df["met_label_x"] = df.met_label_x.apply(clean_label)

    # delete all unknowns and also likelys from the annotations df
    # Likelys were deleted because we decided that we only want to use yes/no for primary brain tumour
    # df = df[(df["N_Mass"] != "unknown") & (df["Mass"] != "likely")]
    print(df.shape)

    # generate the true labels
    df["mass_label"] = df.apply(lambda row: mass_label(row), axis=1)
    df["aggressive_label"] = df.apply(lambda row: aggressive_label(row), axis=1)
    df["met_label_x"] = df.apply(lambda row: met_label(row), axis=1)

    # retrieve the radiology report text
    files = os.listdir(report_path)

    # create two new columns to put the findings and impressions into it:
    df["full_report"] = ""
    df["findings"] = ""
    df["impressions"] = ""
    df["sections"] = ""
    df["findings_tokens"] = ""
    df["impressions_tokens"] = ""
    for file in files:
        # get the patient id first from the file name
        # always be the number before first underscore
        # pid = file[: file.find("_")]
        # open and read text file
        with codecs.open(
            os.path.join(report_path, file), "r", "utf-8", errors="replace"
        ) as f:
            text = f.read()
            # concatenate all of the words together and clean text
            full_report = clean_txt(text)
            # split the text into sections:
            findings = findings_sect(full_report)
            impressions = impressions_sect(full_report)
            # find number of tokens each section has:
            findings_tokens = n_tokens(findings)
            impressions_tokens = n_tokens(impressions)
            # determine which sections it has
            sections = report_sections(findings, impressions)
            # add this nospace text to the csv file:
            index = df[
                df["File Name"].str.contains(file)
            ].index  # assume that there is always only one index match...
            df.loc[index, "full_report"] = full_report
            df.loc[index, "findings"] = findings
            df.loc[index, "impressions"] = impressions
            df.loc[index, "sections"] = sections
            df.loc[index, "findings_tokens"] = findings_tokens
            df.loc[index, "impressions_tokens"] = impressions_tokens
    full_df = df
    # full_df.to_csv(met_info_file)
    print(
        "Shape of pre-process full_df_labeled: ", full_df.shape
    )  # should be 1059 -- this is after deleting the ones we need to verify for
    full_df = full_df[
        (full_df["sections"] == "Both")
        & (full_df["impressions_tokens"].apply(pd.to_numeric) <= 512)
    ]
    print("Shape of full_df_labeled: ", full_df.shape)
    return full_df


def organize_unlabeled_data(report_path):
    df = pd.DataFrame(
        columns=[
            "File Name",
            "full_report",
            "findings",
            "impressions",
            "sections",
            "findings_tokens",
            "impressions_tokens",
        ]
    )

    # retrieve the radiology report text
    files = os.listdir(report_path)

    for file in files:
        # open and read text file
        with codecs.open(
            os.path.join(report_path, file), "r", "utf-8", errors="replace"
        ) as f:
            text = f.read()

            # add this nospace text to the csv file:
            file_name = file
            scan_type = file[
                file.find("_") + 1: file.find("_") + 3
            ]  # the scan type is usually 2 letters long
            # concatenate all of the words together and clean text
            full_report = clean_txt(re.sub("\s{1,}", " ", text))
            # split the text into sections:
            findings = findings_sect(full_report)
            impressions = impressions_sect(full_report)
            # find number of tokens each section has:
            findings_tokens = n_tokens(findings)
            impressions_tokens = n_tokens(impressions)
            # determine which sections it has
            sections = report_sections(findings, impressions)
            row = pd.DataFrame(
                {
                    "File Name": file_name,
                    "Scan type": scan_type,
                    "full_report": full_report,
                    "findings": findings,
                    "impressions": impressions,
                    "sections": sections,
                    "findings_tokens": findings_tokens,
                    "impressions_tokens": impressions_tokens,
                },
                index=[0],
            )
            df = pd.concat([df, row])

    # subset out df that has both report sections and impressions section is not that long
    df = df[
        (df["sections"] == "Both")
        & (df["impressions_tokens"].apply(pd.to_numeric) <= 512)
    ]

    # shuffle data:
    df = df.sample(frac=1, random_state=0)
    df = df.reset_index(drop=True)


    # take the first 30000 samples:
    # if df.shape[0] > 30000:
    #     print("more than 30000 rows")
    #     df = df.iloc[30000:, :]

    return df


# two assumptions are made for this to work:
# reports have a findings and impression section
# the impression section immediately follows the findings section
# impression section is the final section


if __name__ == "__main__":
    # parse some arguments that are needed
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output-name", type=str, default="data.csv")
    argparser.add_argument("--generate-freq", action="store_true")
    argparser.add_argument(
        "--folder-path",
        type=str,
        help="Points to where the txt files are stored corresponding to the annotations",
    )
    args = argparser.parse_args()

    # construct the new samples:
    new_annotations = pd.read_csv('Resources/annotations_Eugene0212.csv')
    new_annotations_df = organize_labeled_data(df=new_annotations, met_type='string', report_path="../To_Label")

    # choose 500 samples to move to the training data:
    random.seed(0)

    indices = random.sample(range(0, 1000), 500)

    # modified annotations does not have the N_Mass and Location columns
    annotations_df = pd.read_csv('Resources/annotations_Eugene0623_modified.csv', encoding="cp1252")
    met_df = pd.read_csv('Resources/met_info_cleaned.csv', encoding="cp1252")[["File Name", "met_label_x"]] 
    annotations_df = pd.merge(annotations_df, met_df, on=['File Name'])
    annotations_df = organize_labeled_data(df=annotations_df, met_type='int', report_path="../To_Label")


    new_annotations_train = new_annotations_df.iloc[indices,:]
    new_annotations_test = new_annotations_df[~new_annotations_df.index.isin(indices)]
    print(new_annotations_test.shape)

    annotations_df = pd.concat([annotations_df, new_annotations_train])
    annotations_df = annotations_df.reset_index()

    annotations_df.to_pickle('augemented_training.pkl')
    print(annotations_df.shape)

    new_annotations_test.to_pickle('test.pkl')

    unlabeled_df = organize_unlabeled_data('../unlabeled')

    # we want the new unlabeled data to not be included in the unlabeled 30k we sampled the first time
    sampled_old = pd.read_csv('sampled_full_30.csv')

    unlabeled_df = unlabeled_df[~unlabeled_df['File Name'].isin(sampled_old['File Name'])]

    print(unlabeled_df.shape)
    unlabeled_df.to_pickle('cotrain_new_samples.pkl')