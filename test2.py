from lark import Lark, Transformer
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.lexer import LexerState, LexerThread, Token

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

fv = open("lark_grammar.txt","r")
grammar = fv.read()

parser = Lark(grammar, parser='lalr')
# print(parser.parse(query))
interactive = parser.parse_interactive("FROM nation SELECT n_nationkey, n_name")

# Process the input interactively
interactive.exhaust_lexer()
print(interactive.choices().keys())
# print(interactive.accepts())

# parser = Lark(grammar, parser='lalr')

# Create an InteractiveParser instance
# interactive = parser.parse_interactive("From nation SELECT n_nationkey, n_name")

# Manually feeding tokens
interactive.feed_token(Token('$END', ''))
# interactive.feed_token(Token('NATION', ''))
# interactive.feed_token(Token('SELECT', ''))
# interactive.feed_token(Token('L_LINESTATUS', ''))
# interactive.feed_token(Token('COMMA', ''))
# interactive.feed_token(Token('WHERE', ''))
# interactive.feed_token(Token('C_PHONE', ''))


# print(interactive_parser.accepts())
# # Complete the parsing
# # interactive_parser.feed_token(Token('$END', ''))

# # Get the result
# result = interactive_parser.result
# print(result)
# interactive.exhaust_lexer()
print(interactive.choices().keys())
# print(interactive.accepts())