#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _AMPA_NMDA_STP_reg(void);
extern void _CaDynamics_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVA_reg(void);
extern void _GABA_A_reg(void);
extern void _GABA_A_STP_reg(void);
extern void _Ih_reg(void);
extern void _Im_v2_reg(void);
extern void _Kd_reg(void);
extern void _K_T_reg(void);
extern void _Kv2like_reg(void);
extern void _Kv3_1_reg(void);
extern void _Nap_reg(void);
extern void _NaTa_reg(void);
extern void _SK_reg(void);
extern void _vecstim_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"modfiles/AMPA_NMDA_STP.mod\"");
    fprintf(stderr, " \"modfiles/CaDynamics.mod\"");
    fprintf(stderr, " \"modfiles/Ca_HVA.mod\"");
    fprintf(stderr, " \"modfiles/Ca_LVA.mod\"");
    fprintf(stderr, " \"modfiles/GABA_A.mod\"");
    fprintf(stderr, " \"modfiles/GABA_A_STP.mod\"");
    fprintf(stderr, " \"modfiles/Ih.mod\"");
    fprintf(stderr, " \"modfiles/Im_v2.mod\"");
    fprintf(stderr, " \"modfiles/Kd.mod\"");
    fprintf(stderr, " \"modfiles/K_T.mod\"");
    fprintf(stderr, " \"modfiles/Kv2like.mod\"");
    fprintf(stderr, " \"modfiles/Kv3_1.mod\"");
    fprintf(stderr, " \"modfiles/Nap.mod\"");
    fprintf(stderr, " \"modfiles/NaTa.mod\"");
    fprintf(stderr, " \"modfiles/SK.mod\"");
    fprintf(stderr, " \"modfiles/vecstim.mod\"");
    fprintf(stderr, "\n");
  }
  _AMPA_NMDA_STP_reg();
  _CaDynamics_reg();
  _Ca_HVA_reg();
  _Ca_LVA_reg();
  _GABA_A_reg();
  _GABA_A_STP_reg();
  _Ih_reg();
  _Im_v2_reg();
  _Kd_reg();
  _K_T_reg();
  _Kv2like_reg();
  _Kv3_1_reg();
  _Nap_reg();
  _NaTa_reg();
  _SK_reg();
  _vecstim_reg();
}

#if defined(__cplusplus)
}
#endif
