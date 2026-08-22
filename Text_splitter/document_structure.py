from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Car:
    wheels = 4

    def __init__(self, brand, model, year):
        self.brand = brand      # Instance attribute
        self.model = model      # Instance attribute
        self.year = year        # Instance attribute
        self.odometer = 0       # Default instance attribute

    def drive(self, miles):
        self.odometer += miles
        return f"The {self.brand} {self.model} drove {miles} miles."

    def get_info(self):
        return f"{self.year} {self.brand} {self.model} (Odometer: {self.odometer} miles)"

my_car = Car("Tesla", "Model 3", 2024)
friend_car = Car("Toyota", "RAV4", 2022)

print(my_car.drive(50))        # Output: The Tesla Model 3 drove 50 miles.
print(my_car.get_info())       # Output: 2024 Tesla Model 3 (Odometer: 50 miles)
print(friend_car.get_info())   # Output: 2022 Toyota RAV4 (Odometer: 0 miles)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=600,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(chunks[0])
print(chunks[1])