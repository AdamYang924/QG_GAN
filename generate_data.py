from lark import Lark, Transformer
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.lexer import LexerState, LexerThread, Token
from random import randint

# grammar = """
# start: "SELECT" col_ref "FROM" table_ref

# col_ref: single_col_ref ("," single_col_ref)*

# single_col_ref: col_a | col_b

# col_a: "A"

# col_b: "B"

# table_ref: table_a | table_b

# table_a: "TA"

# table_b: "TB"

# %import common.NUMBER
# %import common.WS
# %ignore WS
# """

from preprocessing import *
import json

conn = psycopg2.connect(database="tpch",
                        host="127.0.0.1",
                        user="yongtai",
                        password="gaiyi430",
                        port="5432")
cursor = conn.cursor()

# table_column, column_buckets = schema_to_production(cursor)
# with open("table_column.json",'w') as fv:
# 	json.dump(table_column,fv)

with open("table_column.json","r") as fv:
	table_column = json.loads(fv.read())

all_column = []
for key in table_column.keys():
	all_column+= table_column[key]

symbol_dic = {
	"COMMA":",",
	"ALL":"*",
	"__ANON_1":">=",
	"__ANON_2":"<=",
	"__ANON_0":"!=",
	"LESSTHAN":"<",
	"MORETHAN":">",
	"EQUAL":"=",
	"AND":"AND",
	"OR":"OR",
	"SELECT":"SELECT",
	"WHERE":"WHERE",
	"$END":"$END",
}

def generate_queries():
	avaliable_col = ['WHERE', '$END', 'COMMA']
	select_flag = False
	where_flag = False

	with open("lark_grammar.txt","r") as fv:
			grammar = fv.read()

	parser = Lark(grammar, parser='lalr')
	generated_seq = "FROM "

	interactive = parser.parse_interactive(generated_seq)
	interactive.exhaust_lexer()
	accepts = list(interactive.accepts())
	# print("accepts: ",interactive.accepts())




	next_token = accepts[randint(0,len(accepts)-1)].lower()
	# print("next token: ",next_token)

	table_counter = 0

	while next_token != "$END":

		# print(generated_seq)
		if next_token == "SELECT":
			select_flag = True
		elif next_token == "WHERE":
			select_flag = False
			where_flag = True
		if not where_flag and not select_flag and next_token != "," and next_token != "":
			avaliable_col += table_column[next_token]
			table_counter += 1
		
		if not where_flag and not select_flag and table_counter > 2 and next_token == ",":
			# print("trigger")
			next_token = "SELECT"
			select_flag = True
			table_counter = 0

		

		# print("avaliable_col: ",avaliable_col)
		generated_seq += " {}".format(next_token)
		
		interactive = parser.parse_interactive(generated_seq)
		interactive.exhaust_lexer()
		accepts = list(interactive.accepts())
		if select_flag:
			intersection_lst = [value.upper() for value in avaliable_col if value.upper() in accepts]
			next_token = intersection_lst[randint(0,len(intersection_lst)-1)]
		elif where_flag:
			illegal_col = [value.upper() for value in all_column if value not in avaliable_col]
			# print("illegal_col: ",illegal_col)
			filtered_lst = [value.upper() for value in accepts if value not in illegal_col]
			# print("filtered_lst: ",filtered_lst,"\n")
			next_token = filtered_lst[randint(0,len(filtered_lst)-1)]

		else:
			next_token = accepts[randint(0,len(accepts)-1)]
		# print("next token ++: ",next_token)
		if next_token in symbol_dic:
			next_token = symbol_dic[next_token]
		else:
			next_token = next_token.lower()
		
		if next_token != "," and next_token in generated_seq.split(" "):
			if not select_flag and not where_flag:
				next_token = ""
			elif select_flag:
				generated_seq = generated_seq[:-2]
				next_token = "WHERE"
			
	return generated_seq.replace("  "," ")

def write_query_to_file(data_num):
	with open("query.txt",'w') as fv:

		for i in range(data_num):
			query = generate_queries()
			fv.write(query+'\n')
			if i % (data_num/10) == 0:
				print("generate {}/10 query".format(i/data_num))
	return

def query_to_sequence(query_file, grammar_file):
	max_len = 0
	with open(grammar_file,"r") as fv:
		lines = fv.readlines()
	sequence_table, _ = lark_to_sequence_table(lines)

	with open(query_file,"r") as fv_in, open("real.txt","w") as fv_out:
		lines = fv_in.readlines()
		for query in lines:
			sequence = query_to_sequences(query, sequence_table)
			fv_out.write(" ".join([str(i) for i in sequence])+"\n")
			if len(sequence) > max_len:
				max_len = len(sequence)
			
	return max_len

def padding_data(filename, max_len,padding):
	print("max_len: ",max_len)
	with open(filename,"r") as fv:
		lines = fv.readlines()
	with open(filename,"w") as fv:
		for line in lines:
			line = line.strip()
			line += padding *(max_len - len(line.split(" ")))
			fv.write(line+"\n")
	return


write_query_to_file(1024)
max_len = query_to_sequence("query.txt", "lark_grammar.txt")
with open("lark_grammar.txt","r") as fv:
	lines = fv.readlines()
_, vocab_size = lark_to_sequence_table(lines)
print("max_len: ",max_len)
padding_data("real.txt",max_len," {}".format(vocab_size))

# print(generate_queries())





		


