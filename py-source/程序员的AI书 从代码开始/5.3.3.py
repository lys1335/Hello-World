import pandas as pd


def getRatinginformation(ratings):
	rates=[]
	for line in ratings:
		rate=line.split("\t")
		rates.append([int(rate[0]),int(rate[1]),int(rate[2])])
	return rates
#
# 生成用户评分的数据结构
#
# 输入:索引数据[[2,1.5]，[2,4,2]...]
# 输出:①用户打分字典，②电影字典
# 使用字典，key是用户ID，value 是用户对电影的评价，
# rate_dic[2]=[(1, 5), (4,2)]...表示用户2对电影1的评分是5，对电影4的评分是2
def createUserRankDic(rates):
	user_rate_dic={}
	item_to_user={}
	for i in rates:
		user_rank=(i[1], i[2])
		if i[0] in user_rate_dic:
			user_rate_dic[i[0]].append(user_rank)
		else:
			user_rate_dic[i[0]]=[user_rank]
		if i[1] in item_to_user:
			item_to_user[i[1]].append(i[0])
		else:
			item_to_user[i[1]]=[i[0]]
	return user_rate_dic,item_to_user
#
# 使用UserFC进行推荐
# 输入:文件名、用户ID、邻居数量
# 输出:推荐的电影ID、输入用户的电影列表、电影对用户的序列表、邻居列表
#
def recommendByUserCF(file_name,userid,k=5):
	# 读取文件数据
	test_contents=readFile(file_name)
	# 将文件数据格式化成二维数组 List[[用户ID,电影ID，电影评分]...]
	test_rates=getRatingInformation(test_contents)
	# 格式化成字典数据
	#     1.用户字典:dic[用户ID]=[(电影ID,电影评分)...]
	#     2.电影字典:dic[电影ID]=[用户ID1，用户ID2...]
	test_dic,test_item_to_user=createUserRankDic(test_rates)
	# 寻找K个相似用户
	neighbors=calcNearestNeighbor(userid,test_dic,test_item_to_user)[:k]
	recommend_dic=()
	for neighbor in neighbors:
		neighbor_user_id=neighbor[1]
		movies=test_dic[neighbor_user_id]
		for movie in movies:
			if movie[0] not in recommend_dic:
				recommend_dic[movie[0]]=neighbor[0]
			else:
				recommend_dic[movie[0]]+=neighbor[0]
	# 建立推荐列表
	recommend_list=[]
	for key in recommend_dic:
		recommend_list.append([recommend_dic[key], key])
        recommend_list.sort(reverser=True)
    user_movies = [i[0] for i in test_dic[userid]]
    return [i[1] for i in recommend_list],user_movies,test_item_to_user,neighbors

if __name__ == '__main__':
    # 数据文件路径
    path = 'dataset/movies.dat'
	userid = ""
    data = recommendByUserCF(path, userid)