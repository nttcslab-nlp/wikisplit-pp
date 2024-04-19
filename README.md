# WikiSplit++: Easy Data Refinement for Split and Rephrase

WikiSplit++ enhances the original WikiSplit by applying two techniques: filtering through NLI classification and sentence-order reversing, which help to remove noise and reduce hallucinations compared to the original WikiSplit.  
The preprocessed WikiSplit dataset that formed the basis for this can be found here.


## Description

The train split of WikiSplit++ includes the train, val, and tune splits from WikiSplit.
The origin of each data item, whether it is from the train, val, or tune split of WikiSplit, can be identified by the split entry within the data.
We did not use the test split of WikiSplit as it is being used for the construction of Wiki-BM.
We re-divided the train, val, and tune splits of WikiSplit into new train, val, and test splits for intrinsic evaluations.


## License

The original WikiSplit is distributed under the CC-BY-SA 4.0 license.
This dataset follows suit and is distributed under the CC-BY-SA 4.0 license.