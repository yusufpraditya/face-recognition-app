class Cat:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __str__(self) -> str:
        return f"{self.name} is {self.age} years old."