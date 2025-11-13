CC = gcc
CFLAGS = -Wall -Wextra
LIBS = -lm
INCLUDES =
SRCS = xnn.c
OBJS = $(SRCS:.c=.o)
MAIN = xnn

$(MAIN): $(OBJS)
	$(CC) $(CFALGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN)
