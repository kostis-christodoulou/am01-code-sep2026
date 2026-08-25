x = [1,2,3,4]
y = ["hello", "goodbye"]
z = ["1", 3.5, False]
k = [ [1,2,3], ["A","B","C"], [True, False] ]

print(x)
print(type(y))

print(x+y)

print(y[0])

z.append("another value")

print(len(z))
print(z)


for i in [1,2,3,4,5,6]:
  print(f"the number is {i} and its square is {i**2}")

for i in range(1, 10):
  print(f"the number is {i} and its square is {i**2}")

for i in range(0, 50, 10):
  print(f"the number is {i} and its square is {i**2}")



# outer loop iterates numbers 1 to 10
for i in range(1, 11):
    # inner loop iterates numbers 1 to 10 for each i
    for j in range(1, 11):
        # print product of i and j, end=' ' to keep output in one line
        print(i * j, end=' ')
    # print a newline after inner loop finishes
    print()


       

n = 20
for a in range(1, n):
    for b in range(1, n):
        for c in range(1, n):
            if a**2 + b**2 == c**2:
                print(f"{a}, {b}, {c} are pythagorean triples")