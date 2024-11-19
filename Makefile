CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm

TARGET = pathtracer

SRC = pathtracer.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

