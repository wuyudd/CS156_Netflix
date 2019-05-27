file_1 = open('../um/base.txt','w+');
file_2 = open('../um/valid.txt','w+');
file_3 = open('../um/hidden.txt','w+');
file_4 = open('../um/probe.txt','w+');
file_5 = open('../um/qual.txt','w+');

file_all = open('um/all.dta','r')
file_all_idx = open('um/all.idx','r')
all_line = file_all.readlines()

for i,idx_line in enumerate(file_all_idx):
	if idx_line == '1\n':
		file_1.write(all_line[i])
	if idx_line == '2\n':
		file_2.write(all_line[i])
	if idx_line == '3\n':
		file_3.write(all_line[i])
	if idx_line == '4\n':
		file_4.write(all_line[i])
	if idx_line == '5\n':
		file_5.write(all_line[i])

file_1.close()
file_2.close()
file_3.close()
file_4.close()
file_5.close()
