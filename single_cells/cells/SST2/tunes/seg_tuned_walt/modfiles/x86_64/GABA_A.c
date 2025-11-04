/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__GABA_A
#define _nrn_initial _nrn_initial__GABA_A
#define nrn_cur _nrn_cur__GABA_A
#define _nrn_current _nrn_current__GABA_A
#define nrn_jacob _nrn_jacob__GABA_A
#define nrn_state _nrn_state__GABA_A
#define _net_receive _net_receive__GABA_A 
#define state state__GABA_A 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define initW _p[0]
#define initW_columnindex 0
#define tau_r_GABAA _p[1]
#define tau_r_GABAA_columnindex 1
#define tau_d_GABAA _p[2]
#define tau_d_GABAA_columnindex 2
#define e_GABAA _p[3]
#define e_GABAA_columnindex 3
#define gmax _p[4]
#define gmax_columnindex 4
#define synapseID _p[5]
#define synapseID_columnindex 5
#define verboseLevel _p[6]
#define verboseLevel_columnindex 6
#define conductance _p[7]
#define conductance_columnindex 7
#define i _p[8]
#define i_columnindex 8
#define g _p[9]
#define g_columnindex 9
#define gmax_GABAA _p[10]
#define gmax_GABAA_columnindex 10
#define A_GABAA _p[11]
#define A_GABAA_columnindex 11
#define B_GABAA _p[12]
#define B_GABAA_columnindex 12
#define factor_GABAA _p[13]
#define factor_GABAA_columnindex 13
#define DA_GABAA _p[14]
#define DA_GABAA_columnindex 14
#define DB_GABAA _p[15]
#define DB_GABAA_columnindex 15
#define v _p[16]
#define v_columnindex 16
#define _g _p[17]
#define _g_columnindex 17
#define _tsav _p[18]
#define _tsav_columnindex 18
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_toggleVerbose(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "toggleVerbose", _hoc_toggleVerbose,
 0, 0
};
#define toggleVerbose toggleVerbose_GABA_A
 extern double toggleVerbose( _threadargsproto_ );
 /* declare global and static user variables */
#define nc_type_param nc_type_param_GABA_A
 double nc_type_param = 7;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau_r_GABAA", "ms",
 "tau_d_GABAA", "ms",
 "e_GABAA", "mV",
 "gmax", "uS",
 "i", "nA",
 "g", "uS",
 "gmax_GABAA", "uS",
 0,0
};
 static double A_GABAA0 = 0;
 static double B_GABAA0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "nc_type_param_GABA_A", &nc_type_param_GABA_A,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[2]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GABA_A",
 "initW",
 "tau_r_GABAA",
 "tau_d_GABAA",
 "e_GABAA",
 "gmax",
 "synapseID",
 "verboseLevel",
 "conductance",
 0,
 "i",
 "g",
 "gmax_GABAA",
 0,
 "A_GABAA",
 "B_GABAA",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	initW = 1;
 	tau_r_GABAA = 0.2;
 	tau_d_GABAA = 8;
 	e_GABAA = -75;
 	gmax = 0.001;
 	synapseID = 0;
 	verboseLevel = 0;
 	conductance = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 19;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _GABA_A_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 19, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GABA_A /home/hrbncv/PV-SST/single_cells/cells/SST2/tunes/seg_tuned_walt/modfiles/GABA_A.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "GABAA receptor";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DA_GABAA = - A_GABAA / tau_r_GABAA ;
   DB_GABAA = - B_GABAA / tau_d_GABAA ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DA_GABAA = DA_GABAA  / (1. - dt*( ( - 1.0 ) / tau_r_GABAA )) ;
 DB_GABAA = DB_GABAA  / (1. - dt*( ( - 1.0 ) / tau_d_GABAA )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    A_GABAA = A_GABAA + (1. - exp(dt*(( - 1.0 ) / tau_r_GABAA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_r_GABAA ) - A_GABAA) ;
    B_GABAA = B_GABAA + (1. - exp(dt*(( - 1.0 ) / tau_d_GABAA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_d_GABAA ) - B_GABAA) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   double _lweight_GABAA ;
 if ( t < 0.0 ) {
     
/*VERBATIM*/
        return;
 }
   if ( verboseLevel > 0.0 ) {
     printf ( "Synapse %f at time %g: weight = %g\n" , synapseID , t , _args[0] ) ;
     }
   _lweight_GABAA = _args[0] * factor_GABAA ;
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = A_GABAA;
    double __primary = (A_GABAA + _lweight_GABAA) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_r_GABAA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_r_GABAA ) - __primary );
    A_GABAA += __primary;
  } else {
 A_GABAA = A_GABAA + _lweight_GABAA ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = B_GABAA;
    double __primary = (B_GABAA + _lweight_GABAA) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_d_GABAA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_d_GABAA ) - __primary );
    B_GABAA += __primary;
  } else {
 B_GABAA = B_GABAA + _lweight_GABAA ;
     }
 if ( verboseLevel > 0.0 ) {
     printf ( " vals %g %g %g %g\n" , A_GABAA , _lweight_GABAA , factor_GABAA , _args[0] ) ;
     }
   if ( _lflag  == 1.0 ) {
     _args[0] = conductance ;
     }
   } }
 
double toggleVerbose ( _threadargsproto_ ) {
   double _ltoggleVerbose;
 verboseLevel = 1.0 - verboseLevel ;
   
return _ltoggleVerbose;
 }
 
static double _hoc_toggleVerbose(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  toggleVerbose ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  A_GABAA = A_GABAA0;
  B_GABAA = B_GABAA0;
 {
   double _ltp_GABAA ;
 A_GABAA = 0.0 ;
   B_GABAA = 0.0 ;
   _ltp_GABAA = ( tau_r_GABAA * tau_d_GABAA ) / ( tau_d_GABAA - tau_r_GABAA ) * log ( tau_d_GABAA / tau_r_GABAA ) ;
   factor_GABAA = - exp ( - _ltp_GABAA / tau_r_GABAA ) + exp ( - _ltp_GABAA / tau_d_GABAA ) ;
   factor_GABAA = 1.0 / factor_GABAA ;
   gmax_GABAA = initW * gmax ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gmax_GABAA * ( B_GABAA - A_GABAA ) ;
   i = g * ( v - e_GABAA ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {   state(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = A_GABAA_columnindex;  _dlist1[0] = DA_GABAA_columnindex;
 _slist1[1] = B_GABAA_columnindex;  _dlist1[1] = DB_GABAA_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/hrbncv/PV-SST/single_cells/cells/SST2/tunes/seg_tuned_walt/modfiles/GABA_A.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "/**\n"
  " * @file GABA_A.mod\n"
  " * @brief GABAA receptor without short-term plasticity, adapted from GABA_A_STP.mod\n"
  " * @author chindemi\n"
  " * @date 2014-05-25\n"
  " * @remark Copyright (c) BBP/EPFL 2005-2021. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)\n"
  " * Several changes have been made from the orginal version of this synapse by Greg Glickert to better adapt the model for Large Scale BMTk/Neuron models\n"
  " * STP components removed to create a basic GABAA receptor\n"
  " */\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "TITLE GABAA receptor\n"
  "\n"
  "\n"
  "COMMENT\n"
  "GABAA receptor conductance using a dual-exponential profile\n"
  "without short-term plasticity.\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "\n"
  "    POINT_PROCESS GABA_A\n"
  "    RANGE initW     : synaptic scaler for large scale networks added by Greg\n"
  "    RANGE tau_r_GABAA, tau_d_GABAA\n"
  "    RANGE gmax, gmax_GABAA\n"
  "    RANGE i, g, e_GABAA\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE synapseID, verboseLevel\n"
  "    RANGE conductance\n"
  "    GLOBAL nc_type_param\n"
  "}\n"
  "\n"
  "\n"
  "PARAMETER {\n"
  "    initW        = 1.0        : added by Greg Glickert to scale synaptic weight for large scale modeling\n"
  "    tau_r_GABAA  = 0.2   (ms) : dual-exponential conductance profile\n"
  "    tau_d_GABAA  = 8     (ms) : IMPORTANT: tau_r < tau_d\n"
  "    e_GABAA      = -75   (mV) : GABAA reversal potential was -80mv change to -75 never heard of e_gaba not -75 - Greg\n"
  "    gmax         = .001  (uS) : weight conversion factor (from nS to uS)\n"
  "    synapseID    = 0\n"
  "    verboseLevel = 0\n"
  "    conductance  = 0.0\n"
  "    nc_type_param = 7\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "    v (mV)\n"
  "    i (nA)\n"
  "    g (uS)\n"
  "    gmax_GABAA (uS)\n"
  "    factor_GABAA\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "STATE {\n"
  "    A_GABAA       : GABAA state variable to construct the dual-exponential profile - decays with conductance tau_r_GABAA\n"
  "    B_GABAA       : GABAA state variable to construct the dual-exponential profile - decays with conductance tau_d_GABAA\n"
  "}\n"
  "\n"
  "\n"
  "INITIAL{\n"
  "    LOCAL tp_GABAA\n"
  "\n"
  "    A_GABAA = 0\n"
  "    B_GABAA = 0\n"
  "\n"
  "    tp_GABAA = (tau_r_GABAA*tau_d_GABAA)/(tau_d_GABAA-tau_r_GABAA)*log(tau_d_GABAA/tau_r_GABAA) :time to peak of the conductance\n"
  "\n"
  "    factor_GABAA = -exp(-tp_GABAA/tau_r_GABAA)+exp(-tp_GABAA/tau_d_GABAA) :GABAA Normalization factor - so that when t = tp_GABAA, gsyn = gpeak\n"
  "    factor_GABAA = 1/factor_GABAA\n"
  "\n"
  "    gmax_GABAA = initW * gmax\n"
  "}\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE state METHOD cnexp\n"
  "    g = gmax_GABAA*(B_GABAA-A_GABAA) :compute time varying conductance as the difference of state variables B_GABAA and A_GABAA\n"
  "    i = g*(v-e_GABAA) :compute the GABAA driving force based on the time varying conductance, membrane potential, and GABAA reversal\n"
  "}\n"
  "\n"
  "\n"
  "DERIVATIVE state{\n"
  "    A_GABAA' = -A_GABAA/tau_r_GABAA\n"
  "    B_GABAA' = -B_GABAA/tau_d_GABAA\n"
  "}\n"
  "\n"
  "\n"
  "NET_RECEIVE (weight){\n"
  "    LOCAL weight_GABAA\n"
  "\n"
  "    : Disable in case of t < 0 (in case of ForwardSkip) which causes numerical\n"
  "    : instability if synapses are activated.\n"
  "    if(t < 0 ) {\n"
  "    VERBATIM\n"
  "        return;\n"
  "    ENDVERBATIM\n"
  "    }\n"
  "\n"
  "    if( verboseLevel > 0 ) {\n"
  "        printf(\"Synapse %f at time %g: weight = %g\\n\", synapseID, t, weight )\n"
  "    }\n"
  "\n"
  "    weight_GABAA = weight*factor_GABAA\n"
  "    A_GABAA = A_GABAA + weight_GABAA\n"
  "    B_GABAA = B_GABAA + weight_GABAA\n"
  "\n"
  "    if( verboseLevel > 0 ) {\n"
  "        printf( \" vals %g %g %g %g\\n\", A_GABAA, weight_GABAA, factor_GABAA, weight )\n"
  "    }\n"
  "\n"
  "    if (flag == 1) {\n"
  "        : self event to set next weight at delay\n"
  "          weight = conductance\n"
  "    }\n"
  "    : flag == 0, i.e. a spike has arrived\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION toggleVerbose() {\n"
  "    verboseLevel = 1-verboseLevel\n"
  "}\n"
  ;
#endif
