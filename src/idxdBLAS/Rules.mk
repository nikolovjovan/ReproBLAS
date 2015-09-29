TARGETS := libidxdblas.a
SUBDIRS :=

INSTALL_LIB := $(TARGETS)

COGGED = damax.ccog damaxm.ccog                                         \
         zamax_sub.ccog zamaxm_sub.ccog                                 \
         samax.ccog samaxm.ccog                                         \
         camax_sub.ccog camaxm_sub.ccog                                 \
         dmdsum.ccog dmdasum.ccog dmdssq.ccog dmddot.ccog               \
         smssum.ccog smsasum.ccog smsssq.ccog smsdot.ccog               \
         cmcsum.ccog smcasum.ccog smcssq.ccog cmcdotu.ccog cmcdotc.ccog \
         zmzsum.ccog dmzasum.ccog dmzssq.ccog zmzdotu.ccog zmzdotc.ccog \
         didgemv.ccog didgemm.ccog                                      \
         sisgemv.ccog sisgemm.ccog                                      \
         cicgemv.ccog cicgemm.ccog                                      \
         zizgemv.ccog zizgemm.ccog                                      \

PRECIOUS = damax.c damaxm.c                                \
           zamax_sub.c zamaxm_sub.c                        \
           samax.c samaxm.c                                \
           camax_sub.c camaxm_sub.c                        \
           dmdsum.c dmdasum.c dmdssq.c dmddot.c            \
           smssum.c smsasum.c smsssq.c smsdot.c            \
           zmzsum.c dmzasum.c dmzssq.c zmzdotu.c zmzdotc.c \
           cmcsum.c smcasum.c smcssq.c cmcdotu.c cmcdotc.c \
           didgemv.c didgemm.c                             \
           sisgemv.c sisgemm.c                             \
           zizgemv.c zizgemm.c                             \
           cicgemv.c cicgemm.c                             \

LIBIDXDBLAS := $(OBJPATH)/libidxdblas.a

libidxdblas.a_DEPS = $$(LIBIDXD)                                     \
                     damax.o damaxm.o                                \
                     zamax_sub.o zamaxm_sub.o                        \
                     samax.o samaxm.o                                \
                     camax_sub.o camaxm_sub.o                        \
                     dmdsum.o dmdasum.o dmdssq.o dmddot.o            \
                     smssum.o smsasum.o smsssq.o smsdot.o            \
                     cmcsum.o smcasum.o smcssq.o cmcdotu.o cmcdotc.o \
                     zmzsum.o dmzasum.o dmzssq.o zmzdotu.o zmzdotc.o \
                     didsum.o didasum.o didssq.o diddot.o            \
                     sissum.o sisasum.o sisssq.o sisdot.o            \
                     cicsum.o sicasum.o sicssq.o cicdotu.o cicdotc.o \
                     zizsum.o dizasum.o dizssq.o zizdotu.o zizdotc.o \
                     didgemv.o didgemm.o                             \
                     sisgemv.o sisgemm.o                             \
                     zizgemv.o zizgemm.o                             \
                     cicgemv.o cicgemm.o                             \

damax.c_DEPS = damax.ccog
damaxm.c_DEPS = damaxm.ccog
zamax_sub.c_DEPS = zamax_sub.ccog
zamaxm_sub.c_DEPS = zamaxm_sub.ccog
samax.c_DEPS = samax.ccog
samaxm.c_DEPS = samaxm.ccog
camax_sub.c_DEPS = camax_sub.ccog
camaxm_sub.c_DEPS = camaxm_sub.ccog
dmdsum.c_DEPS = $$(GETTER) dmdsum.ccog
dmdasum.c_DEPS = $$(GETTER) dmdasum.ccog
dmdssq.c_DEPS = $$(GETTER) dmdssq.ccog
dmddot.c_DEPS = $$(GETTER) dmddot.ccog
smssum.c_DEPS = $$(GETTER) smssum.ccog
smsasum.c_DEPS = $$(GETTER) smsasum.ccog
smsssq.c_DEPS = $$(GETTER) smsssq.ccog
smsdot.c_DEPS = $$(GETTER) smsdot.ccog
cmcsum.c_DEPS = $$(GETTER) cmcsum.ccog
smcasum.c_DEPS = $$(GETTER) smcasum.ccog
smcssq.c_DEPS = $$(GETTER) smcssq.ccog
cmcdotu.c_DEPS = $$(GETTER) cmcdotu.ccog
cmcdotc.c_DEPS = $$(GETTER) cmcdotc.ccog
zmzsum.c_DEPS = $$(GETTER) zmzsum.ccog
dmzasum.c_DEPS = $$(GETTER) dmzasum.ccog
dmzssq.c_DEPS = $$(GETTER) dmzssq.ccog
zmzdotu.c_DEPS = $$(GETTER) zmzdotu.ccog
zmzdotc.c_DEPS = $$(GETTER) zmzdotc.ccog
didgemv.c_DEPS = $$(GETTER) didgemv.ccog
didgemm.c_DEPS = $$(GETTER) didgemm.ccog
sisgemv.c_DEPS = $$(GETTER) sisgemv.ccog
sisgemm.c_DEPS = $$(GETTER) sisgemm.ccog
zizgemv.c_DEPS = $$(GETTER) zizgemv.ccog
zizgemm.c_DEPS = $$(GETTER) zizgemm.ccog
cicgemv.c_DEPS = $$(GETTER) cicgemv.ccog
cicgemm.c_DEPS = $$(GETTER) cicgemm.ccog
