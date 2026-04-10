"""
贝叶斯网络实现
支持离散变量和精确推断（枚举法）
"""

import numpy as np


class SimpleBayesianNetwork:
    """
    简单贝叶斯网络实现
    支持离散变量和精确推断（枚举法）
    """

    def __init__(self):
        self.nodes = {}  # 节点信息：{name: {'parents': [], 'values': []}}
        self.cpts = {}   # 条件概率表：{name: {parent_values: {value: prob}}}
        self.topo_order = []  # 拓扑排序

    def add_node(self, name, values, parents=None):
        """添加节点"""
        if parents is None:
            parents = []
        self.nodes[name] = {'parents': parents, 'values': values}
        self._update_topo_order()

    def set_cpt(self, name, cpt):
        """
        设置条件概率表

        cpt格式：{parent_value_tuple: {value: prob}}
        对于无父节点的变量：{(): {value: prob}}
        """
        self.cpts[name] = cpt

    def _update_topo_order(self):
        """计算拓扑排序"""
        visited = set()
        order = []

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in self.nodes[node]['parents']:
                visit(parent)
            order.append(node)

        for node in self.nodes:
            visit(node)

        self.topo_order = order

    def get_prob(self, name, value, parent_values):
        """获取条件概率 P(name=value | parent_values)"""
        parent_key = tuple(parent_values) if parent_values else ()
        return self.cpts[name].get(parent_key, {}).get(value, 0)

    def joint_prob(self, assignment):
        """计算联合概率 P(X1, X2, ...)"""
        prob = 1.0
        for node in self.topo_order:
            parents = self.nodes[node]['parents']
            parent_values = [assignment[p] for p in parents]
            value = assignment[node]
            prob *= self.get_prob(node, value, parent_values)
        return prob

    def enumerate_inference(self, query, evidence):
        """
        枚举推断：计算 P(query | evidence)

        query: {node: '?'} 返回分布
        evidence: {node: value}
        """
        query_nodes = list(query.keys())
        hidden = [n for n in self.nodes if n not in query_nodes and n not in evidence]

        def enumerate_assignments(variables, current):
            if not variables:
                yield current.copy()
                return
            var = variables[0]
            for value in self.nodes[var]['values']:
                current[var] = value
                yield from enumerate_assignments(variables[1:], current)
            del current[var]

        query_values = {}
        total = 0.0

        query_node = query_nodes[0]
        for qv in self.nodes[query_node]['values']:
            prob_sum = 0.0
            for assignment in enumerate_assignments(hidden, {}):
                assignment.update(evidence)
                assignment[query_node] = qv
                prob_sum += self.joint_prob(assignment)
            query_values[qv] = prob_sum
            total += prob_sum

        # 归一化
        for k in query_values:
            query_values[k] /= total
        return query_values