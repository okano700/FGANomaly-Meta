import glob
import os
from TSds import TSds
from find_frequency import get_period

#ds = glob.glob("~/datsets/UCR_Anomaly_FullData/*.txt")
ds = glob.glob('/home/eyokano/datsets/UCR_Anomaly_FullData/*.txt')
print('sadsad')
print(ds)

print('aqui')
ds.sort()
for p in ds:
	ds = TSds.read_UCR(p)
	for f in get_period(ds.ts[:ds.train_split],3):
		for n in [2,3,5]:
			for i in range(5):
				#os.system(f'python t.py --path {i} --ds UCR --WL 100 --n 5 --i 1')
				print(f'python t.py --path {p} --ds UCR --WL {f} --n {n} --i {i}')
				#print(f, n, i, p)
