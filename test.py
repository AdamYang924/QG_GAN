def syntax_mask(sequence,lark_grammar):
    fv = open("lark_grammar.txt","r")
    sequence_table, idx = lark_to_sequence_table(fv)
    # reverse_sequence_table = {v: k for k, v in sequence_table.items()}
    # grammar_tree = _build_grammar_tree(fv)
    # grammar = fv.read()
    parser = Lark(lark_grammar.grammar, parser='lalr')
    interactive = parser.parse_interactive("")
    
    if sequence[-1] == lark_grammar.idx:
        return [-inf]*(lark_grammar.idx) + [0]

    last_token = lark_grammar.reverse_sequence_table[sequence[-1]]
    # print(last_token)
    last_token_node = lark_grammar.grammar_tree.find(last_token)
    next_tokens = []

    with open("table_column.json","r") as jfile:
        table_column_dict = json.load(jfile)

    # print(table_name_lst)

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
        for i in sequence:
            token = lark_grammar.reverse_sequence_table[i]
            token_node = lark_grammar.grammar_tree.find(token)
            if token in special_character_reverse.keys():
                token = special_character_reverse[token]
                # print(token.upper())
                interactive.feed_token(Token(token.upper(), ''))
            elif token_node.children == []:
                interactive.feed_token(Token(token.upper(), ''))
                # print(token.upper())
        choices = interactive.choices()
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
            last_token_parent_node = lark_grammar.grammar_tree.find_parent(last_token,lark_grammar.grammar_tree.root)[0]
            # print("last p data: ",last_token_parent_node.data)
            parent_lst = []
            last_token_parent_parent_node = lark_grammar.grammar_tree.find_parent(last_token_parent_node.data,lark_grammar.grammar_tree.root)[0]
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
                    token_parent_node = lark_grammar.grammar_tree.find_parent(token,lark_grammar.grammar_tree.root)[0]
                    while token_parent_node.calculate_height() < last_token_parent_node.calculate_height():
                        token_parent_node = lark_grammar.grammar_tree.find_parent(token_parent_node.data,lark_grammar.grammar_tree.root)[0]
                    # print("token {} parent node height: ".format(token_parent_node.data),token_parent_node.calculate_height())
                    # print("last token {} parent node height".format(last_token),last_token_parent_node.calculate_height())
                    token = token_parent_node.data
                if token not in next_tokens:
                    next_tokens.append(token)
    # print("next tokens",next_tokens)
    # Fillter the illegal col_name
    table_name_lst = [lark_grammar.sequence_table[table_name] for table_name in table_column_dict.keys()]
    table_name_not_in_seq = [value for value in table_name_lst if value not in sequence]
    illegal_col_name = []
    for i in table_name_not_in_seq:
        table_name = lark_grammar.reverse_sequence_table[i]
        col_name_lst = table_column_dict[table_name]
        illegal_col_name += col_name_lst
    next_tokens = [token for token in next_tokens if token not in illegal_col_name]

    if lark_grammar.sequence_table["where"] not in sequence:
        illegal_token = ['lt', 'eq', 'se', 'le', 'ne', 'st']
        next_tokens = [token for token in next_tokens if token not in illegal_token]
    else:
        illegal_token = ['comma', 'where', '$END','sel_column']
        next_tokens = [token for token in next_tokens if token not in illegal_token]
    if last_token == "comma":
        next_tokens = [lark_grammar.reverse_sequence_table[sequence[-3]]]
            
    # print(next_tokens)
    mask = [-inf]*(lark_grammar.idx+1)
    for token in next_tokens:
        if token != "$END":
            mask_idx = lark_grammar.sequence_table[token]
            mask[mask_idx] = 0
        else:
            mask[lark_grammar.idx] = 0
    
    return mask


