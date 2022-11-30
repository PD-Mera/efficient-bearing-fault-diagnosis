#!/bin/bash
num=50

for i in {1..$num}
do
   python quantization.py --pos $i
done