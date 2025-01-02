from sklearn.metrics import cohen_kappa_score
import pandas as pd

path_to_dir = '/ihome/hdaqing/abg96/llm/check_no_incomplete_reasoning/refined_extraction_@@/'
files = [] # name of files for which you want to calculate kappa


for i in range(len(files)):
	df = pd.read_csv(path_to_dir + files[i], names=columns)
	

	list1 = list(df['human annotation']) #replace with your column names of choice.
	list2 = list(df['rule_llm_annotation'])  #replace with your column names of choice.

	refined_list1 = [str(i).strip() for i in list1]
	refined_list2 = [str(i).strip() for i in list2]



	def convert(l):
		for i in range(len(l)):
			if l[i] == 'Similar':
				l[i] = 'Correct'
			if l[i] == 'Dissimilar':
				l[i] = 'Incorrect'
		return l


	refined_list1 = convert(refined_list1)
	refined_list2 = convert(refined_list2)



	print('kappa score:', files[i], cohen_kappa_score(refined_list1, refined_list2))
	print('linear kappa :', files[i], cohen_kappa_score(refined_list1, refined_list2, weights = 'linear'))
	print('quadratic kappa:', files[i], cohen_kappa_score(refined_list1, refined_list2, weights = 'quadratic'))



	# calculating user agreement score.
	# agree = 0
	# total = len(list1)
	# for i in range(total):
	# 	if str(list1[i]).strip() == str(list2[i]).strip():
	# 		agree = agree + 1

	# print(f"user agreement:{agree/total}")










