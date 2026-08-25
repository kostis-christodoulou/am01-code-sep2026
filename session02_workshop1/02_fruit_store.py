'''
In this exercise, you'll write the software to manage a fruit store that sells only 3 fruits: apples, bananas, and avocados.

In this store, apples cost \$3, bananas are \$5, and avocados are \$12.

## Instructions
Write a script that does the following:

### Get inputs
* Store the prices of items in constants. Remember, Python has no explicit concept of constants. Every variable can be overridden, hence why in cases where it matters that a variable is not changed, that is perhaps the time to make that obvious by using the convention of capitalizing the variable name.
* Ask the user for their budget, and which fruit they'd like to buy. For now, assume a user only buys one type of fruit. Remember that you can use `\n` for the print/input function to skip to a new line; this will be useful for making your user prompts cleaner.
  * The `input()` function always returns a string, even if the user gives a number. The `int()` and `float()` functions can be used to convert strings to a numeric type.
* Ask the user which fruit they would like to buy, and store this in a variable called `item_name`.

### Shop logic: Setting variables for the transaction   
* Depending on the fruit selected, show the user a message saying "The price of each `FRUIT NAME` is `FRUIT PRICE`", and ask the user how many of the fruit they'd like to buy.
  * Notice that here, we're "casting" (i.e. converting) the input `string` to an `int` — we're selling _whole_ fruits here!
  * The `if` logic is started out for you by checking if the user's inputted fruit is an apple. Use `elif` (short for "else, if") statements to check for the other types of fruit.
* If the user inputs an invalid product name (i.e. the input is not "apple", "banana", or "lemon"), show them a message telling them the error, and asking them to try again using one of the valid fruit names, and list them.
  * We've used `if` and `elif` to catch the cases where we have a valid fruit name. We can use `else` to deal with remaining possible cases (i.e. invalid fruit names) without specifying the exact cases we're trying to catch; in English, `else` is like saying "otherwise, do this".
  * If the user's input is invalid, make sure to reassign `valid_input` to `False`.

### Shop logic: Closing the transaction
* If the user inputs a valid product name, check that they have enough money in their budget.
  * If the user has enough money, show a message saying "Here are your `QUANTITY` items. You have `REMAINING MONEY FROM BUDGET` left."
  * If the user does not have enough money, tell them so.

  ### Running your code
  To run your code, save your work in `fruit_store.py` and run it using the command `python3 fruit_store.py` in your terminal.
'''


# Store the price of each type of fruit in a constant
APPLE_PRICE = 3
BANANA_PRICE = 5
AVOCADO_PRICE = 12

# Get initial inputs from the user
try:
    budget = float(input("What is your budget?\n"))
    item_name = input("Which fruit would you like to buy? (apple, banana, avocado)\n").lower()
except ValueError:
    print("Invalid input for budget. Please enter a number.")
    exit()

# Use variable valid_input to keep track of whether the input is valid or not. Anything else to keep track of?
valid_input = True

if item_name == "apple":
  print(f"The price of each apple is {APPLE_PRICE}")
  quantity = int(input("How many apples would you like to buy? "))
  item_price = APPLE_PRICE

elif:
  # add conditions for other types of fruit, as we did for the apple above

else:
  # if the user doesn't provide a valid input, track this and show an error message. The program then ends here.


# Proceed to use the inputs given, if they're valid
if valid_input:
  # Check that the items are within the user's budget.
  # If so, give the items and inform the user of the remainder.
  # If not, tell the user that the items are out of budget.