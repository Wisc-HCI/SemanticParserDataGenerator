# Social Robotics Command Generator

## Running the generator
Simply type '''./generate.sh''' and the data will be generated and split into training, testing, and validation sets.

## Modifying the generator
Natural-language utterances for small-commands can be easily generated using an off-the-shelf tool such as chatito [1]. The chatito file is included in the chatito directory. Using chatito, a new leaves.json file can be created.

## Using the generated data
This data can be used alongside other datasets, such as the atis, jobs, and geo datasets described in Yin et al. (2018) [2], for learning neural models for semantic parsing. I currently use this data with the TranX semantic parser [2]. An example dataset generated with this code can be seen in 

# References
[1] https://github.com/rodrigopivi/Chatito
[2] Yin, Pengcheng & Neubig, Graham. (2018). TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation. 7-12. 10.18653/v1/D18-2002. 
