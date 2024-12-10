CC = nvcc
CFLAGS = -O2 -arch=sm_60 
LDFLAGS = -lm

TARGET = pathtracer
SRC = pathtracer.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

debug: CFLAGS += -g -G
debug: $(TARGET)

