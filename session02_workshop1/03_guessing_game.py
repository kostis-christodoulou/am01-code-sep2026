'''
Use the random package:
We'll be using the randint function from the random package for this exercise. Remember that "functions" are repeatable operations that make it easier to apply the same process to varying inputs. A "package" is a collection of "functions" that someone else wrote, and which you can import into your own script.

import random adds the random package to our program. To use the randint function, we "call" it with the following syntax: 
random.randint(a,b). To understand what a and b arguments are, take a look at the function's documentation here. Developers of packages document their code to make it easier for other developers, such as yourself, to know how to use their functions!

Get user inputs and initialize variables:
Get two int inputs from the user: one of which is the lower bound we'll supply to random.randint, and the other is the upper bound.


Use the given inputs to generate a random number from random.randint, and store it in a variable called random_number.
Create two variables to help us keep track of the game:
guess stores the value of the user's current guess. Set this to None initially.
num_guesses keeps track of how many guesses the user has made. Set this to 0 initially.
Game logic:
while the guess is not equal to the random_number:
Prompt the user for a new guess
if the guess is greater/lesser than the random_number, tell the user they're too high/low
Increase the num_guesses by 1

When the guess is equal to the random_number, tell the user they've won, and how many guesses it took them to do it!
Bonus (optional): Can you modify your code such that the random number is found by the code itself? In other words, instead of prompting the user for repeated input, can you write code that provides guesses?
'''

import random

# Part 1: User Guesses the Number

# Get user inputs for the range
try:
    lower_bound = int(input("Enter the lower bound: "))
    upper_bound = int(input("Enter the upper bound: "))

    # Ensure the lower bound is less than the upper bound
    if lower_bound >= upper_bound:
        print("The lower bound must be less than the upper bound. Please run the script again.")
    else:
        # Generate a random number within the user's specified range
        random_number = random.randint(lower_bound, upper_bound)

        # Initialize variables
        guess = None
        num_guesses = 0

        print(f"\nI'm thinking of a number between {lower_bound} and {upper_bound}.")

        # Game logic
        while guess != random_number:
            # Prompt the user for a guess
            try:
                guess = int(input("What is your guess? "))
                num_guesses += 1

                # Provide feedback on the guess
                if guess < random_number:
                    print("Too low!")
                elif guess > random_number:
                    print("Too high!")
                else:
                    print(f"Congratulations! You've guessed the number {random_number} correctly.")
                    print(f"It took you {num_guesses} guesses to win!")
            except ValueError:
                print("Invalid input. Please enter a whole number.")

except ValueError:
    print("Invalid input. Please enter whole numbers for the bounds.")


# Bonus Part 2: Computer Guesses the Number

print("\n--- Bonus: Let's see the computer guess! ---")

# Use the same bounds from the user's game or ask for new ones
# For simplicity, we'll reuse the bounds if they were valid
if 'lower_bound' in locals() and 'upper_bound' in locals() and lower_bound < upper_bound:
    # Generate a new random number for the computer to guess
    random_number_computer = random.randint(lower_bound, upper_bound)
    print(f"\nThe computer will now try to guess a new number between {lower_bound} and {upper_bound}.")
    print(f"(For demonstration, the secret number is: {random_number_computer})")

    # Initialize variables for the computer's guessing strategy
    computer_guess = None
    num_computer_guesses = 0
    low_bound_search = lower_bound
    high_bound_search = upper_bound

    # Game logic for the computer
    while computer_guess != random_number_computer:
        # A simple strategy: guess the midpoint of the current range
        computer_guess = (low_bound_search + high_bound_search) // 2
        num_computer_guesses += 1
        print(f"\nComputer's guess #{num_computer_guesses}: {computer_guess}")

        # Adjust the search range based on feedback
        if computer_guess < random_number_computer:
            print("Feedback: Too low!")
            low_bound_search = computer_guess + 1
        elif computer_guess > random_number_computer:
            print("Feedback: Too high!")
            high_bound_search = computer_guess - 1
        else:
            print(f"The computer guessed the number {random_number_computer} correctly!")
            print(f"It took the computer {num_computer_guesses} guesses.")
else:
    print("\nCould not run the bonus part due to invalid initial bounds.")