ifdef GCC
CC = g++
else
CC = clang++
endif

CPPFLAGS = -std=c++17 -Wall -Wextra

ifdef BREAKDOWN
CPPFLAGS += -DBREAKDOWN
endif

ifdef CILKPLUS
CPPFLAGS += -DPARLAY_CILKPLUS -DCILK -fcilkplus
else ifdef OPENCILK
CPPFLAGS += -DPARLAY_OPENCILK -DCILK -fopencilk
else ifdef SERIAL
CPPFLAGS += -DPARLAY_SEQUENTIAL
else
CPPFLAGS += -pthread
endif

ifdef DEBUG
CPPFLAGS += -Og -mcx16 -DDEBUG
else ifdef PERF
CC = g++
CPPFLAGS += -Og -mcx16 -march=native -g
else ifdef MEMCHECK
CPPFLAGS += -Og -mcx16 -DPARLAY_SEQUENTIAL
else
CPPFLAGS += -O3 -mcx16 -march=native
endif

ifdef STDALLOC
CPPFLAGS += -DPARLAY_USE_STD_ALLOC
endif

ifdef NGRAM
CPPFLAGS += -DNGRAM
endif

all: semisort histogram collect_reduce

semisort: semisort.h semisort.cpp internal/integer_sort.h internal/counting_sort.h
	$(CC) $(CPPFLAGS) semisort.cpp -o semisort

histogram: collect_reduce.h histogram.cpp 
	$(CC) $(CPPFLAGS) histogram.cpp -o histogram

collect_reduce: collect_reduce.h collect_reduce.cpp 
	$(CC) $(CPPFLAGS) collect_reduce.cpp -o collect_reduce

statistics: statistics.cpp
	$(CC) $(CPPFLAGS) statistics.cpp -o statistics
