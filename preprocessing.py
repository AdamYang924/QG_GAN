"""
TODO:
1. Get database, automatically generate LARK grammar -> schema_to_production
2. Generate masks based on LARK grammar 
3. Generate sql query based on defined rules
4. Define the extended SQL grammar to handle nested queries
"""

from lark import Lark, Transformer
from lark.lexer import  Token
import re
from random import randint
import psycopg2
from grammar_tree import TreeNode, GTree
import json
from numpy import inf

query = "SELECT n_name FROM nation WHERE r_name = bucket_4 AND r_regionkey = bucket_4"

class LarkGrammar():
    def __init__(self,file_name):
        with open(file_name,"r") as fv:
            fv_content = fv.readlines()

        with open("table_column.json","r") as jfile:
            self.table_column_dict = json.load(jfile)

        self.sequence_table, self.idx = lark_to_sequence_table(fv_content)
        self.reverse_sequence_table = {v: k for k, v in self.sequence_table.items()}
        self.grammar_tree = _build_grammar_tree(fv_content)
        grammar = "\n".join(fv_content)
        self.parser = Lark(grammar, parser='lalr')


class Sequence():
    def __init__(self,start_token,lark_grammar):
        self.generated_sequence = [start_token]
        self.interactive = lark_grammar.parser.parse_interactive("FROM ")
        self.lark_grammar = lark_grammar
        self.interactive.exhaust_lexer()

    
    def add(self,i):
        if i != 0:
            self.generated_sequence.append(i)
            if i < self.lark_grammar.idx:
                token = self.lark_grammar.reverse_sequence_table[i]
                token_node = self.lark_grammar.grammar_tree.find(token)
                if token in special_character_reverse.keys():
                    token = special_character_reverse[token]
                    self.interactive.feed_token(Token(token.upper(), ''))
                elif token_node.children == []:
                    self.interactive.feed_token(Token(token.upper(), ''))



special_character = {
    "COMMA":"comma",
    "SELECT":"select_stmt",
    "WHERE":"where",
    "FROM":"start",
    "STAR":"all",
    "LESSTHAN":"st",
    "MORETHAN":"lt",
    "EQUAL":"eq",
    "__ANON_0":"le",
    "__ANON_1":"se",
    "__ANON_2":"ne",
    "BUCKET_0":"bucket_zero",
    "BUCKET_1":"bucket_one",
    "BUCKET_2":"bucket_two",
    "BUCKET_3":"bucket_three",
    "BUCKET_4":"bucket_four"
}

special_character_reverse = {v: k for k, v in special_character.items()}

def restructure_query(query):
    from_phase, select_phase, where_phase, having_phase, group_phase = [None, None, None, None, None]

    select_phase = query.split("FROM")[0].split("SELECT")[1]
    if "WHERE" in query:
        from_phase = query.split("FROM")[1].split("WHERE")[0]
        if "GROUP" in query:
            where_phase = query.split("WHERE")[1].split("GROUP")[0]
            if "HAVING" in query:
                group_phase = query.split("GROUP")[1].split("HAVING")[0]
                having_phase = query.split("HAVING")[1]
            else:
                group_phase = query.split("GROUP")[1]
        else:
            where_phase = query.split("WHERE")[1]
    else:
        from_phase = query.split("FROM")[1]

    new_query = "FROM {} SELECT {}".format(from_phase, select_phase)
    if where_phase is not None:
        new_query += " WHERE {}".format(where_phase)
    if having_phase is not None:
        new_query += " HAVING {}".format(having_phase)
    if group_phase is not None:
        new_query += " GROUP {}".format(group_phase)

    subquery_pattern = r'\((SELECT.*?FROM.*?\))'
    subqueries = re.findall(subquery_pattern, new_query, re.IGNORECASE | re.DOTALL)
    if len(subqueries) > 0:
        for subquery in subqueries:
            new_sub = restructure_query(subquery[:-1])
            new_query = new_query.replace(subquery[:-1], new_sub)

    return new_query

def reduce_space(query, num_buckets):
    pattern = r'\b\d+\.\d+\b|\b\d+\b|\'[^\']*\''
    
    # Find all matches in the text
    matches = re.findall(pattern, query)
    
    # Process the matches to convert them to appropriate types
    extracted_values = []
    for match in matches:
        if re.match(r'\b\d+\.\d+\b', match):  # If it's a floating-point number
            extracted_values.append(float(match))
        elif re.match(r'\b\d+\b', match):  # If it's an integer
            extracted_values.append(int(match))
        elif re.match(r'\'[^\']*\'', match):  # If it's a string in single quotes
            extracted_values.append(match.strip("'"))
    
    for value in extracted_values:
        bucket_index = hash(value) % num_buckets
        query = query.replace(str(value), "bucket_{}".format(bucket_index))
    query = query.replace("'", "")
    return query

def schema_to_production(cursor):
    with open("lark_backup.txt",'r') as fv_in, open("lark_grammar.txt",'w') as fv_out:
        content = fv_in.read()
        fv_out.write(content)
    

    #get table name
    table_column = {}
    column_bucket = {}
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cursor.fetchall()

    #init
    column_str = "all"
    table_str = ""
    lark_content = ""

    #get column name in each table
    for table in tables:
        column_lst = []
        table_name = table[0]
        lark_content += '\n{}: "{}"'.format(table_name, table_name)
        table_str += "{} | ".format(table_name)
        query = f"""
        SELECT
            column_name
        FROM
            information_schema.columns
        WHERE
            table_name = '{table_name}'
        ORDER BY
            ordinal_position;
        """

        cursor.execute(query)
        schema = cursor.fetchall()
        for column in range(len(schema)-1):
            column_bucket[column] = _column_to_bucket(cursor, schema[column][0], table_name)
            column_lst.append(schema[column][0])
            column_str += " | {}".format(schema[column][0])
            lark_content += '\n{}: "{}"'.format(schema[column][0],schema[column][0])
        table_column[table[0]] = column_lst


    #modify lark grammar file
    with open("lark_grammar.txt","r") as fv:
        lines = fv.readlines()
    

    table_str = table_str[:-2]
    with open("lark_grammar.txt","w") as fv:
        for line in lines:
            if "sel_col_name" in line:
                line = re.sub(r'\bsel_col_name\b',"({})".format(column_str),line)
            elif "where_col_name" in line:
                line = re.sub(r'\bwhere_col_name\b',"({})".format(column_str[5:]),line)
            if "tab_name" in line:
                line = re.sub(r'\btab_name\b',"({})".format(table_str),line)
            fv.write(line)
            if "all:" in line:
                fv.write(lark_content)


    return table_column, column_bucket

def lark_to_sequence_table(lines):
    sequence_table = {}
   
    idx = 0
    for line in lines:
        if ":" in line:
            production = line.strip().split(":")[0]
            if "?" in production:
                production = production.replace("?","")
            sequence_table[production] = idx
            idx += 1
    
    return sequence_table, idx

def query_to_sequences(query, sequence_table):
    sequence = []

    with open("lark_grammar.txt","r") as fv:
        grammar = fv.read()

    # Create the parser
    parser = Lark(grammar, start='start', parser='lalr')

    # Define a Transformer to convert the parse tree into a more useful structure
    class SQLTransformer(Transformer):
        def select(self, items):
            result = {
                'type': 'query',
                'select_stmt': items[1],
                'table_ref': items[0],
                'where': items[2] if len(items) > 2 and isinstance(items[2], dict) else None,
                # 'subquery': items[-1] if len(items) > 2 and isinstance(items[-1], dict) else None
            }
            return result

        def subquery(self, items):
            return {'subquery': items[0]}

        def select_stmt(self, items):
            return list(items)

        def column(self, items):
            return str(items[0])

        def table_ref(self, items):
            return items[0]

        def where(self, items):
            return items[0]

        def where_clause(self, items):
            return {
                'column': items[0],
                'opreator': items[1],
                'value': items[2]
            }

    tree = parser.parse(query)
    for i in tree.iter_subtrees_topdown():
        # print("data: ",i.data)
        sequence.append(sequence_table[i.data])

    return sequence


def sequence_to_query(sequence,lark_grammar):
    tokens = []
    for i in sequence:
            if i < 95:
                token = lark_grammar.reverse_sequence_table[i]
                # print(token)
                token_node = lark_grammar.grammar_tree.find(token)
                if token in ["st","lt","eq","le","se","ne","comma","all"]:
                    op_dic = {"st":"<","lt":">","eq":"=","le":">=","se":"<=","ne":"!=","comma":",","all":"*"}
                    token = op_dic[token]
                    tokens.append(token)
                    # print(token)
                elif token in special_character_reverse.keys():
                    token = special_character_reverse[token]
                    tokens.append(token)
                    # print(token.upper())
                elif token_node.children == []:
                    tokens.append(token)
                    # print(token.upper())
    select_idx = tokens.index("SELECT")
    from_statement = tokens[:select_idx]
    table_lst = [value for value in from_statement if value not in ["FROM",","]]
    if "WHERE" in tokens:
        where_idx = tokens.index("WHERE")
        select_statement = tokens[select_idx:where_idx]

        for table_name in table_lst:

            for col in lark_grammar.table_column_dict[table_name]:
                if col in select_statement:
                    idx = select_statement.index(col)
                    select_statement[idx] = table_name+"."+col
        tokens = select_statement+from_statement+tokens[where_idx:]
    else:
        select_statement = tokens[select_idx:]
        for table_name in table_lst:
            for col in lark_grammar.table_column_dict[table_name]:
                if col in select_statement:
                    idx = select_statement.index(col)
                    select_statement[idx] = table_name+"."+col
        tokens = select_statement+from_statement
    
    return " ".join(tokens)+";"

def _build_grammar_tree(lines):
    
    grammar_tree = GTree("start")
    for line in lines:
        if ":" in line:
            parent = re.sub(r'\?', '', line.strip().split(":")[0])
            children = re.sub(r'"[^"]*"|[()*?|]', '', line.strip().split(":")[1]).replace("  "," ").split(" ")
            # print(children)
            for child in children:
                if child != "":
                    grammar_tree.add(parent,child)
    # print(grammar_tree)
    return grammar_tree

def _hash_to_buckets(values, num_buckets):
       
    buckets = {i: [] for i in range(num_buckets)}

    for value in values:
        bucket_index = hash(value) % num_buckets
        buckets[bucket_index].append(value)
    
    return buckets

def _column_to_bucket(cursor, column_name, table_name):
    query = "SELECT DISTINCT {} FROM {};".format(column_name, table_name)
    cursor.execute(query)
    values = [i[0] for i in cursor.fetchall()]
    buckets = _hash_to_buckets(values,5)
    
    return buckets

def generate_test_query():
    with open("lark_grammar.txt","r") as fv:
        grammar = fv.read()

    parser = Lark(grammar, parser='lalr')
    generated_seq = "FROM "

    interactive = parser.parse_interactive(generated_seq)
    interactive.exhaust_lexer()
    accepts = interactive.accepts()
    print("accepts: ",interactive.accepts())
    next_token = accepts[randint(0,len(accepts)-1)]

    while next_token != "$END":
        print("next token: ",next_token)
        generated_seq += " {}".format(next_token)
        interactive = parser.parse_interactive(generated_seq)
        interactive.exhaust_lexer()
        accepts = list(interactive.accepts())
        next_token = accepts[randint(0,len(accepts)-1)]
    print(generated_seq)

def _next_possible_tokens(grammar_tree,node,lst):
    parent = grammar_tree.find_parent(node.data)
    node_before_parent = grammar_tree.find_parent(parent.data)
    node_index = parent.children.index(node)
    # print("node data: ",node.data,"parent data: ",parent.data, "node before parent: ",node_before_parent.data)
    if node_before_parent != None:
        parent_index = node_before_parent.children.index(parent)
    else:
        lst.append("$END")
        return lst
    if node_index < len(parent.children)-1:
        next_token = parent.children[node_index+1]
        # print("next token: ",next_token.data)
        lst.append(next_token)
        idx = node_index
        while len(next_token.children)==0 and idx < len(parent.children)-1:
            next_token = parent.children[idx+1]
            if next_token.data != node.data:
                lst.append(next_token)
            idx += 1
        return lst
    elif parent_index < len(node_before_parent.children)-1 :
        next_token = node_before_parent.children[parent_index+1]
        # print("next token: ",next_token.data)
        lst.append(next_token)
        idx = parent_index
        while len(next_token.children)==0 and idx < len(node_before_parent.children)-1:
            next_token = node_before_parent.children[idx+1]
            if next_token.data != parent.data:
                lst.append(next_token)
            idx += 1
        return lst
    else:
        lst = _next_possible_tokens(grammar_tree,parent,lst)
        return lst


def syntax_mask(seq_obj:Sequence,lark_grammar:LarkGrammar):

    # parser = Lark(lark_grammar.grammar, parser='lalr')
    # interactive = parser.parse_interactive("")
    sequence = seq_obj.generated_sequence
    
    if sequence[-1] == lark_grammar.idx:
        return [-inf]*(lark_grammar.idx) + [0]

    last_token = lark_grammar.reverse_sequence_table[sequence[-1]]
    last_token_node = lark_grammar.grammar_tree.find(last_token)
    next_tokens = []

    table_column_dict = lark_grammar.table_column_dict

    # If the last token is non-terminal
    if last_token_node.children != []:
        # If the last token is non-terminal, but its children are terminal, append all its children
        if last_token_node.calculate_height() == 1:
            for child in last_token_node.children:
                next_tokens.append(child.data)
        else:
            # If the last token is non-terminal, append the first token of its child
            next_tokens.append(last_token_node.children[0].data)
    # If the last token is terminal
    else:
        # Feed Terminal tokens in the sequence to parser, get its choices
        choices = seq_obj.interactive.choices()
        # print("choices: ",choices.keys())

        # If there is non-terminal in choices, choose the heighest non-terminal node in tree. 
        # Else, append terminals to next_tokens
        highest_node = None
        for token in choices:
            if token.islower():
                if highest_node is None:
                    highest_node = lark_grammar.grammar_tree.find(token)
                elif lark_grammar.grammar_tree.find(token).calculate_height() > highest_node.calculate_height():
                    highest_node = lark_grammar.grammar_tree.find(token)
    
        if highest_node != None:
            next_tokens.append(highest_node.data)
        else:
            last_token_parent_node = lark_grammar.grammar_tree.find_parent(last_token)
            # print("last p data: ",last_token_parent_node.data)
            parent_lst = []
            last_token_parent_parent_node = lark_grammar.grammar_tree.find_parent(last_token_parent_node.data)
            # print("last p p data: ",last_token_parent_parent_node.data)
            parent_idx = last_token_parent_parent_node.children.index(last_token_parent_node)
            if parent_idx < len(last_token_parent_parent_node.children)-1:
                next_tokens.append(last_token_parent_parent_node.children[parent_idx+1].data)
            for token in choices:
                if token in special_character.keys():
                    token = special_character[token]
                elif token == "$END":
                    next_tokens.append(token)
                else:
                    token = token.lower()
                    token_parent_node = lark_grammar.grammar_tree.find_parent(token)
                    while token_parent_node.calculate_height() < last_token_parent_node.calculate_height():
                        token_parent_node = lark_grammar.grammar_tree.find_parent(token_parent_node.data)
                    # print("token {} parent node height: ".format(token_parent_node.data),token_parent_node.calculate_height())
                    # print("last token {} parent node height".format(last_token),last_token_parent_node.calculate_height())
                    token = token_parent_node.data
                if token not in next_tokens:
                    next_tokens.append(token)
    # print("next tokens",next_tokens)
    # Fillter the illegal col_name
    table_name_lst = [lark_grammar.sequence_table[table_name] for table_name in table_column_dict.keys()]
    # print("table name lst: ",table_name_lst)
    table_name_not_in_seq = [value for value in table_name_lst if value not in sequence]
    table_name_in_seq = [lark_grammar.reverse_sequence_table[value] for value in table_name_lst if value in sequence]
    # print("table name in sequence: ",table_name_in_seq)
    illegal_col_name = []
    for i in table_name_not_in_seq:
        table_name = lark_grammar.reverse_sequence_table[i]
        col_name_lst = table_column_dict[table_name]
        illegal_col_name += col_name_lst
    
    illegal_col_name += table_name_in_seq
    if len(table_name_in_seq) == len(table_name_lst):
        illegal_col_name.append("comma")
    # print("illegal: ",illegal_col_name)
    next_tokens = [token for token in next_tokens if token not in illegal_col_name]

    if lark_grammar.sequence_table["where"] not in sequence:
        illegal_token = ['lt', 'eq', 'se', 'le', 'ne', 'st']
        next_tokens = [token for token in next_tokens if token not in illegal_token]
    else:
        illegal_token = ['comma', 'where', '$END','sel_column']
        next_tokens = [token for token in next_tokens if token not in illegal_token]
    if last_token == "comma":
        next_tokens = [lark_grammar.reverse_sequence_table[sequence[-3]]]
            
    # print("next tokens: ",next_tokens)
    mask = [-inf]*(lark_grammar.idx+1)
    for token in next_tokens:
        if token != "$END":
            mask_idx = lark_grammar.sequence_table[token]
            mask[mask_idx] = 0
        else:
            mask[lark_grammar.idx] = 0
    
    return mask



if __name__ == "__main__":

    query = restructure_query(query)
    # print(query)
    # seq = query_to_sequences(query)
    # print("seq: ",seq)

    # q = sequence_to_query(seq)
    # print(q)

    conn = psycopg2.connect(database="tpch",
                        host="127.0.0.1",
                        user="yongtai",
                        password="gaiyi430",
                        port="5432")
    cursor = conn.cursor()
    # table_column, column_buckets = schema_to_production(cursor)
    # query = restructure_query(query)
    # query = reduce_space(query, 5)
    fv = open("lark_grammar.txt","r")
    sequence_table,idx = lark_to_sequence_table(fv)
    reverse_sequence_table = {v: k for k, v in sequence_table.items()}
    fv.close()
    # print(sequence_table[0])
    sequence = query_to_sequences(query, sequence_table)
    print("sequence: ", sequence)

  

    # print(lark_to_sequence_table("lark_grammar.txt"))
    # tree = _build_grammar_tree("lark_grammar.txt")
    # node = tree.find("nation")
    # print(node.calculate_height())
    # _column_to_bucket(cursor, "n_nationkey", "nation")
    # test_seq_lst = [0,2,6,19,1,7,20,5,7,21,3,4,8,20,17,93]
    # for i in range(1,len(test_seq_lst)+1):
    #     print("test seq: {}\nmask: {}".format(test_seq_lst[:i],syntax_mask(test_seq_lst[:i])))

    #     print("len mask: ",len(syntax_mask(test_seq_lst[:i])))
    lark_grammar = LarkGrammar("lark_grammar.txt")
    seq_obj = Sequence(0,lark_grammar)
    # debug_lst = [0, 2, 6, 29, 1, 7, 35, 3, 4, 8, 35, 16, 94, 10, 11, 4, 8, 32]
    debug_lst = [0, 2, 6, 29,5,6,20,5,6,25,5,6,39,5,6,47,5,6,53,5,6,62,5,6,72]
    real_seq = "0 2 6 25 1 7 19 95".split(" ")
    real_seq = [int(i) for i in real_seq]
    # for i in range(1,len(debug_lst)):
    #     seq_obj.add(debug_lst[i])
    # print([reverse_sequence_table[i] for i in debug_lst])
    # print(syntax_mask(seq_obj,lark_grammar))
    print(sequence_to_query(real_seq,lark_grammar))


    

