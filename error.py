class ErrorHandler():
	def generic(self):
		print("Error: An error occurred.")
		exit(69)
	def fileNotFound(self, _exit):
		print("Error: File not found.")
		if (_exit):
			exit(2)
	def os(self):
		print("Error: System call failed.")
		exit(3)
	def syntax(self):
		print("Error: Invalid Syntax.")
		exit(4)
	def arithmeticError(self):
		print("Error: Calculation Failed.")
		exit(5)
	def attributeError(self):
		print("Error: Failed to assign value or reference.")
		exit(6)
	def endOfFile(self):
		print("Error: EOF Reached.")
		exit(7)
	def floatingPoint(self):
		print("Error: Floating Point Calculation Failed.")
		exit(8)
	def importation(self):
		print("Error: Imported Module Doesn't exist.")
		exit(9)
	def indentation(self):
		print("Error: Invalid Line Indentation.")
		exit(10)
	def index(self):
		print("Error: Specified Index Doesn't Exist.")
		exit(11)
	def sql(self):
		print("\nError: A SQL error occurred.")
		exit(12)
	def keyboard(self):
		print("Error: Keyboard Interrupt.")
		exit(13)
	def name(self):
		print("Error: Variable doesn't exist.")
		exit(14)
	def runtime(self):
		print("Error: Fatal Runtime Error.")
		exit(15)
	def bsod(self):
		print("Error: BSOD Occurred.")
		exit(420)
	def unicode(self):
		print("Error: A unicode error occurred.")
		exit(16)
	def value(self):
		print("Error: A value error occurred")
		exit(17)
	def zero(self):
		print("Error: Divide by zero.")
		exit(18)
