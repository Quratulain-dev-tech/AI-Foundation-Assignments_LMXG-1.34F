import csv
students_data = [
    ["Quratulain", 21, 85],
    ["Hafsa", 20, 78],
    ["Laiba", 21, 91],
    ["Imaan", 20, 88],
    ["Ayesha", 22, 76],
    ["Hamida", 21, 82],
    ["Talha Rao", 22, 89],
    ["Hassan", 20, 94],
    ["Ali", 21, 73],
    ["Umar", 22, 87],
    ["Abbas", 21, 80],
    ["Abu Bakar", 20, 92],
    ["Muhammad", 22, 84],
    ["Eesa", 21, 79],
    ["Abrar", 23, 71],
    ["Ihtisham", 20, 86],
    ["Ahmed", 21, 90],
    ["Usman", 22, 77],
    ["Hamza", 20, 95],
    ["Huzaifa", 21, 81],
    ["Zainab", 20, 89],
    ["Fatima", 22, 93],
    ["Maryam", 21, 75],
    ["Hira", 20, 84],
    ["Maham", 22, 80],
    ["Sana", 21, 88],
    ["Sidra", 20, 72],
    ["Anaya", 21, 91],
    ["Alina", 22, 86],
    ["Rabia", 20, 79],
    ["Bilal", 23, 83],
    ["Saad", 21, 90],
    ["Hamdan", 20, 87],
    ["Abdullah", 22, 96],
    ["Ibrahim", 21, 82],
    ["Ismail", 23, 74],
    ["Yahya", 20, 89],
    ["Rayyan", 21, 93],
    ["Fahad", 22, 78],
    ["Arham", 20, 85],
    ["Sameer", 21, 81],
    ["Salman", 23, 76],
    ["Danish", 22, 88],
    ["Areeba", 20, 92],
    ["Madiha", 21, 83],
    ["Noor", 22, 90],
    ["Mehwish", 20, 77],
    ["Komal", 21, 86],
    ["Eman", 22, 94],
    ["Saba", 20, 80],
    ["Mariam", 21, 87],
    ["Rida", 22, 73],
    ["Asad", 20, 91],
    ["Taha", 21, 84],
    ["Adeel", 23, 79],
    ["Waleed", 22, 88],
    ["Junaid", 21, 82],
    ["Kashif", 20, 75],
    ["Farhan", 22, 90],
    ["Mustafa", 21, 95],
    ["Abdul Rehman", 23, 86],
    ["Abdul Hadi", 20, 81],
    ["Saima", 21, 89],
    ["Bushra", 22, 78],
    ["Iqra", 20, 93],
    ["Amna", 21, 85],
    ["Sumaiya", 22, 88],
    ["Muneeba", 20, 76],
    ["Hareem", 21, 91],
    ["Zoya", 22, 84]
]

students = []

for data in students_data:

    student = {
        "name": data[0],
        "age": data[1],
        "AI_marks": data[2]
    }

    percentage = (student["AI_marks"] / 100) * 100

    student["percentage"] = str(percentage) + "%"

    students.append(student)


# Har student ko alag line mein display karna
for student in students:
    print(student)

with open("students_data.txt", "w") as file:

    for student in students:
        file.write(str(student) + "\n")


with open("AI_students.csv", "w", newline="") as file:

    fieldnames = ["name", "age", "AI_marks", "percentage"]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    writer.writerows(students)
