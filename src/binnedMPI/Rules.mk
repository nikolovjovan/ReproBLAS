TARGETS :=
ifeq ($(BUILD_MPI),true)
TARGETS += libbinnedmpi.a
endif
SUBDIRS :=

INSTALL_LIB := $(TARGETS)

LIBBINNEDMPI := $(OBJPATH)/libbinnedmpi.a

LDFLAGS += $(MPILDFLAGS)
CFLAGS += $(MPICFLAGS)

COGGED = DBDBADD.ccog \
         ZBZBADD.ccog \
         DBDBADDSQ.ccog \
         SBSBADD.ccog \
         CBCBADD.ccog \
         SBSBADDSQ.ccog

libbinnedmpi.a_DEPS = $$(LIBBINNED) DOUBLE_BINNED.o \
                                DOUBLE_COMPLEX_BINNED.o \
                                DOUBLE_BINNED_SCALED.o \
                                FLOAT_BINNED.o \
                                FLOAT_COMPLEX_BINNED.o \
                                FLOAT_BINNED_SCALED.o \
                                DBDBADD.o \
                                ZBZBADD.o \
                                DBDBADDSQ.o \
                                SBSBADD.o \
                                CBCBADD.o \
                                SBSBADDSQ.o
