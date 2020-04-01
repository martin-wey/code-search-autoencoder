'''
Get all function calls from a python file

The MIT License (MIT)
Copyright (c) 2016 Suhas S G <jargnar@gmail.com>
'''
import ast
from collections import deque


class FuncCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self._name = deque()

    @property
    def name(self):
        return '.'.join(self._name)

    @name.deleter
    def name(self):
        self._name.clear()

    def visit_Name(self, node):
        self._name.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            self._name.appendleft(node.attr)
            self._name.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)


def get_func_calls(tree):
    """
    Retrieve all function calls from a code snippets.
    """
    func_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callvisitor = FuncCallVisitor()
            callvisitor.visit(node.func)
            func_calls.append(callvisitor.name)

    return func_calls


def get_func_def(tree):
    """
    Retrieve function declaration from a code snippets.
    This function takes as input the AST tree of a function code snippet.
    Consequently, func_def contains only one item that is returned by get_func_def
    """
    func_def = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)]
    return func_def[0] if len(func_def) > 0 else None
