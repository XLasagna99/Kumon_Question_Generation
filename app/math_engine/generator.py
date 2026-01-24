def generate_random_math_question():
    import random

    operations = ['+', '-', '*', '/']
    operation = random.choice(operations)

    if operation == '+':
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        output = {
            "operation": operation,
            "a": num1,
            "b": num2,
            "answer": num1 + num2
        }
    elif operation == '-':
        num1 = random.randint(2, 10)
        num2 = random.randint(1, num1) # For multiplication, we keep it simple so num1 >= num2
        output = {
            "operation": operation,
            "a": num1,
            "b": num2,
            "answer": num1 - num2
        }
    elif operation == '*':
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        output = {
            "operation": operation,
            "a": num1,
            "b": num2,
            "answer": num1 * num2
        }
    elif operation == '/':
        # Ensure no division by zero and integer result
        num2 = random.randint(1, 10)
        num1 = num2 * random.randint(1, 10)
        output = {
            "operation": operation,
            "a": num1,
            "b": num2,
            "answer": int(num1 / num2)
        }

    return output