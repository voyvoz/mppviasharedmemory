## testgen

should only require numpy

```
--size=  Rows of buildtable. Default 500
--htsize=  Size of hashtable (number of buckets). Default 512
--skew=  0:skewed 0to 1:uniform. Default 0.0
--probesize=  Rows of probetable. Default 5120
--probeskew=  0:skewed 0to 1:uniform. Default 1.0
--selectivity=  0 none: 1 all. Default 0.25
```
e.g.
```
python3 testgen.py --size=1000 --htsize=1024 --skew=0.2 --probesize=10000 --selectivity=0.1
```
