# Use a base Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Expose the port on which the Flask app will run (change this if needed)
EXPOSE 5003

# Start the Flask app when the container starts
CMD ["python", "lapredict.py"]