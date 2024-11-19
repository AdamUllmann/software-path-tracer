CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm

TARGET = noise_calculator

SRC = noise_calculator.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

