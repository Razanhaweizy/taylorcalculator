from flask import Flask, request, jsonify, render_template
import sympy as sp
from typing import List, Tuple, Dict

app = Flask(__name__)

class TaylorCalculator:
    @staticmethod
    def parse_inputs(function_expr: str, variables: str, expansions: str) -> Tuple[sp.Expr, List[sp.Symbol], List[Tuple[sp.Symbol, float]]]:
        """Parse and validate all user inputs"""
        try:
            # Parse function
            if not function_expr:
                raise ValueError("Function expression cannot be empty")
            f = sp.sympify(function_expr)
            
            # Parse variables
            if not variables:
                raise ValueError("Variables cannot be empty")
            variables = [sp.symbols(var.strip()) for var in variables.split(',')]
            
            # Parse expansion points
            if not expansions:
                raise ValueError("Expansion points cannot be empty")
            expansion_points = []
            for pt in expansions.split(';'):
                var_name, value = pt.split('=')
                var = sp.symbols(var_name.strip())
                if var not in variables:
                    raise ValueError(f"Expansion point variable {var} not in declared variables")
                expansion_points.append((var, float(value)))
                
            # Validate that all variables have expansion points
            if len(expansion_points) != len(variables):
                raise ValueError("Number of expansion points must match number of variables")
                
            return f, variables, expansion_points
            
        except Exception as e:
            raise ValueError(f"Input parsing error: {str(e)}")

    @staticmethod
    def compute_taylor_expansion(f: sp.Expr, var: sp.Symbol, 
                               point: float, order: int) -> sp.Expr:
        """Compute Taylor expansion for a single variable"""
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
    def calculate_series(f: sp.Expr, variables: List[sp.Symbol], 
                        expansion_points: List[Tuple[sp.Symbol, float]], 
                        order: int) -> Tuple[sp.Expr, List[Dict]]:
        """Calculate the complete Taylor series expansion"""
        result = f
        steps = []
        
        # Calculate Taylor expansion for each variable
        for var, point in expansion_points:
            result = TaylorCalculator.compute_taylor_expansion(result, var, point, order)
            steps.append({
                'variable': str(var),
                'point': point,
                'expansion': str(result)
            })
            
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
        f, variables, expansion_points = calculator.parse_inputs(function_expr, variables, expansions)
        
        # Calculate complete Taylor series
        result, steps = calculator.calculate_series(f, variables, expansion_points, order)
        
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