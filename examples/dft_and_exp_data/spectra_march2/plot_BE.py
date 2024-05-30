#!/usr/bin/env python

f = open('exp_param.py')
content = f.read()
exec(content)
BE = master['BE']
print(BE)
