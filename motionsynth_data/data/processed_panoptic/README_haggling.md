Preprocessing Haggling DB from the Panoptic Studio Dataset 

1. gathering_panoptic.py 

This file loads all haggling json files from /yourPath/panoptic-toolbox/data_haggling, which was downloaded from domedb, 
and saves them as pkl files.

Similarly, run the following for face, and hands:
```
gathering_panopticDB_face.py
gathering_panopticDB_hand.py
```

2. process_hagglingDB.py 
This code does some additional processing for each haggling sequence, as follows:

1: Align the startFarmes for the people in the same group

2: Separate the name of pkl file, to have a single group only


Similarly, run the following for face, and hands:
```
process_hagglingDB_face.py 
process_hagglingDB_hand.py
```