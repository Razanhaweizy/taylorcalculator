from flask import Flask, request, jsonify, render_template
import sympy as sp
from typing import List, Tuple, Dict

app = Flask(__name__)

class TaylorCalculator:
    @staticmethod
    def parse_inputs(function_expr: str, variables: str, expansions: str) -> Tuple[sp.Expr, List[sp.Symbol], List[Tuple[sp.Symbol, float]]]:
        """Parse user inputs into SymPy objects."""
        try:
            # Convert the function expression to a SymPy object
            f = sp.sympify(function_expr)
            
            # Parse the variables into a list of SymPy symbols
            vars_list = [sp.symbols(var.strip()) for var in variables.split(',')]
            
            # Parse expansion points into a list of (variable, value) pairs
            exp_points = [
                (sp.symbols(var.strip()), float(value.strip()))
                for var, value in [exp.split('=') for exp in expansions.split(';')]
            ]

            return f, vars_list, exp_points
        except Exception as e:
            raise ValueError(f"Input parsing error: {str(e)}")

    @staticmethod
    def decompose_function(function_expr: str) -> List[Tuple[sp.Expr, List[sp.Symbol]]]:
        """Decompose a composite function into its components"""
        try:
            f = sp.sympify(function_expr)
            components = []
            
            # Get the full expression tree
            expr_tree = sp.preorder_traversal(f)
            
            # Find composite sub-expressions
            for expr in expr_tree:
                if isinstance(expr, sp.Basic) and not expr.is_Atom:
                    free_symbols = list(expr.free_symbols)
                    if free_symbols:  # Only consider expressions with variables
                        components.append((expr, free_symbols))
            
            # Sort components by complexity (number of operations)
            components.sort(key=lambda x: len(str(x[0])))
            return components
            
        except Exception as e:
            raise ValueError(f"Function decomposition error: {str(e)}")

    @staticmethod
    def compute_univariate_taylor(f: sp.Expr, var: sp.Symbol, 
                                point: float, order: int) -> sp.Expr:
        """Compute Taylor expansion for a single variable function"""
        expansion = 0
        term = f
        fact = 1
        
        for n in range(order + 1):
            if n > 0:
                term = sp.diff(term, var)
                fact *= n
            term_at_point = term.subs(var, point)
            expansion += term_at_point * (var - point)**n / fact
            
        return expansion

    @staticmethod
    def substitute_expansions(original: sp.Expr, expansions: Dict[sp.Expr, sp.Expr]) -> sp.Expr:
        """Substitute Taylor expansions back into the original expression"""
        result = original
        
        # Sort expansions by expression size (largest first) to handle nested substitutions
        sorted_expansions = sorted(expansions.items(), key=lambda x: len(str(x[0])), reverse=True)
        
        for expr, expansion in sorted_expansions:
            result = result.subs(expr, expansion)
            
        return result

    @staticmethod
    def calculate_series(f: sp.Expr, variables: List[sp.Symbol], 
                        expansion_points: List[Tuple[sp.Symbol, float]], 
                        order: int) -> Tuple[sp.Expr, List[Dict]]:
        """Calculate Taylor series using component-wise expansion"""
        steps = []
        
        # Step 1: Decompose the function
        components = TaylorCalculator.decompose_function(str(f))
        steps.append({
            'step': 'decomposition',
            'components': [str(comp[0]) for comp in components]
        })
        
        # Step 2: Calculate Taylor expansions for each component
        expansions = {}
        for comp, comp_vars in components:
            comp_expansion = comp
            for var, point in expansion_points:
                if var in comp_vars:
                    comp_expansion = TaylorCalculator.compute_univariate_taylor(
                        comp_expansion, var, point, order
                    )
            expansions[comp] = comp_expansion
            steps.append({
                'component': str(comp),
                'expansion': str(comp_expansion)
            })
        
        # Step 3: Substitute back into original expression
        result = TaylorCalculator.substitute_expansions(f, expansions)
        
        return result, steps

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        function_expr = data['function']
        variables = data['variables']
        expansions = data['expansions']
        order = int(data['order'])

        calculator = TaylorCalculator()
        
        # Parse inputs
        f, vars_list, expansion_points = calculator.parse_inputs(function_expr, variables, expansions)
        
        # Calculate Taylor series using component-wise approach
        result, steps = calculator.calculate_series(f, vars_list, expansion_points, order)
        
        # Simplify final result
        final_result = str(result.simplify())
        
        return jsonify({
            'success': True,
            'steps': steps,
            'result': final_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
