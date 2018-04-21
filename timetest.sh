
python cprofile_test.py
python gprof2dot.py -f pstats result.out | dot -Tpng -o result.png
