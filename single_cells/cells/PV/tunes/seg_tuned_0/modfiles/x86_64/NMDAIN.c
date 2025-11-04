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
 
#define nrn_init _nrn_init__NMDAIN
#define _nrn_initial _nrn_initial__NMDAIN
#define nrn_cur _nrn_cur__NMDAIN
#define _nrn_current _nrn_current__NMDAIN
#define nrn_jacob _nrn_jacob__NMDAIN
#define nrn_state _nrn_state__NMDAIN
#define _net_receive _net_receive__NMDAIN 
#define release release__NMDAIN 
 
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
#define Alpha _p[0]
#define Alpha_columnindex 0
#define Beta _p[1]
#define Beta_columnindex 1
#define e _p[2]
#define e_columnindex 2
#define iNMDA _p[3]
#define iNMDA_columnindex 3
#define g _p[4]
#define g_columnindex 4
#define gmax _p[5]
#define gmax_columnindex 5
#define ica _p[6]
#define ica_columnindex 6
#define Ron _p[7]
#define Ron_columnindex 7
#define Roff _p[8]
#define Roff_columnindex 8
#define Rinf _p[9]
#define Rinf_columnindex 9
#define Rtau _p[10]
#define Rtau_columnindex 10
#define synon _p[11]
#define synon_columnindex 11
#define B _p[12]
#define B_columnindex 12
#define DRon _p[13]
#define DRon_columnindex 13
#define DRoff _p[14]
#define DRoff_columnindex 14
#define v _p[15]
#define v_columnindex 15
#define _g _p[16]
#define _g_columnindex 16
#define _tsav _p[17]
#define _tsav_columnindex 17
#define _nd_area  *_ppvar[0]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
 
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
 static double _hoc_mgblock(void*);
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
 "mgblock", _hoc_mgblock,
 0, 0
};
#define _f_mgblock _f_mgblock_NMDAIN
#define mgblock mgblock_NMDAIN
 extern double _f_mgblock( _threadargsprotocomma_ double );
 extern double mgblock( _threadargsprotocomma_ double );
 
static void _check_mgblock(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_mgblock(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define Cmax Cmax_NMDAIN
 double Cmax = 1;
#define Cdur Cdur_NMDAIN
 double Cdur = 1;
#define mg mg_NMDAIN
 double mg = 1;
#define usetable usetable_NMDAIN
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_NMDAIN", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Cmax_NMDAIN", "mM",
 "Cdur_NMDAIN", "ms",
 "mg_NMDAIN", "mM",
 "Alpha", "/ms",
 "Beta", "/ms",
 "e", "mV",
 "iNMDA", "nA",
 "g", "umho",
 0,0
};
 static double Roff0 = 0;
 static double Ron0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Cmax_NMDAIN", &Cmax_NMDAIN,
 "Cdur_NMDAIN", &Cdur_NMDAIN,
 "mg_NMDAIN", &mg_NMDAIN,
 "usetable_NMDAIN", &usetable_NMDAIN,
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
 
#define _cvode_ieq _ppvar[5]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"NMDAIN",
 "Alpha",
 "Beta",
 "e",
 0,
 "iNMDA",
 "g",
 "gmax",
 "ica",
 0,
 "Ron",
 "Roff",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	Alpha = 4;
 	Beta = 0.0125;
 	e = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 18;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 6, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[4]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _NMDAIN_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 18, 6);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "netsend");
  hoc_register_dparam_semantics(_mechtype, 5, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 5;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 NMDAIN /home/hrbncv/PV-SST/single_cells/cells/PV/tunes/seg_tuned/modfiles/NMDAIN.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_mgblock;
static int _reset;
static char *modelname = "simple NMDA receptors";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static double _n_mgblock(_threadargsprotocomma_ double _lv);
 static int _slist1[2], _dlist1[2];
 static int release(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DRon = ( synon * Rinf - Ron ) / Rtau ;
   DRoff = - Beta * Roff ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DRon = DRon  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Rtau )) ;
 DRoff = DRoff  / (1. - dt*( ( - Beta )*( 1.0 ) )) ;
  return 0;
}
 /*END CVODE*/
 static int release (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    Ron = Ron + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Rtau)))*(- ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - Ron) ;
    Roff = Roff + (1. - exp(dt*(( - Beta )*( 1.0 ))))*(- ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - Roff) ;
   }
  return 0;
}
 static double _mfac_mgblock, _tmin_mgblock;
  static void _check_mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_mg;
  if (!usetable) {return;}
  if (_sav_mg != mg) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_mgblock =  - 140.0 ;
   _tmax =  80.0 ;
   _dx = (_tmax - _tmin_mgblock)/1000.; _mfac_mgblock = 1./_dx;
   for (_i=0, _x=_tmin_mgblock; _i < 1001; _x += _dx, _i++) {
    _t_mgblock[_i] = _f_mgblock(_p, _ppvar, _thread, _nt, _x);
   }
   _sav_mg = mg;
  }
 }

 double mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 return _n_mgblock(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_mgblock(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_mgblock * (_lv - _tmin_mgblock);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_mgblock[0];
 }
 if (_xi >= 1000.) {
 return _t_mgblock[1000];
 }
 _i = (int) _xi;
 return _t_mgblock[_i] + (_xi - (double)_i)*(_t_mgblock[_i+1] - _t_mgblock[_i]);
 }

 
double _f_mgblock ( _threadargsprotocomma_ double _lv ) {
   double _lmgblock;
 _lmgblock = 1.0 / ( 1.0 + exp ( 0.072 * - _lv ) * ( mg / 3.57 ) ) ;
   
return _lmgblock;
 }
 
static double _hoc_mgblock(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 
#if 1
 _check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 _r =  mgblock ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 0.0 ) {
     _args[2] = _args[2] + 1.0 ;
     if (  ! _args[1] ) {
       _args[3] = _args[3] * exp ( - Beta * ( t - _args[4] ) ) ;
       _args[4] = t ;
       _args[1] = 1.0 ;
       synon = synon + _args[0] ;
           if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron + _args[3] ) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron + _args[3]  ;
         }
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff - _args[3] ) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff - _args[3]  ;
         }
 }
     net_send ( _tqitem, _args, _pnt, t +  Cdur , _args[2] ) ;
     }
   if ( _lflag  == _args[2] ) {
     _args[3] = _args[0] * Rinf + ( _args[3] - _args[0] * Rinf ) * exp ( - ( t - _args[4] ) / Rtau ) ;
     _args[4] = t ;
     synon = synon - _args[0] ;
         if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron - _args[3] ) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron - _args[3]  ;
       }
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff + _args[3] ) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff + _args[3]  ;
       }
 _args[1] = 0.0 ;
     }
   gmax = _args[0] ;
   } }
 
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
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  Roff = Roff0;
  Ron = Ron0;
 {
   Rinf = Cmax * Alpha / ( Cmax * Alpha + Beta ) ;
   Rtau = 1.0 / ( Cmax * Alpha + Beta ) ;
   synon = 0.0 ;
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

#if 0
 _check_mgblock(_p, _ppvar, _thread, _nt);
#endif
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
   B = mgblock ( _threadargscomma_ v ) ;
   g = ( Ron + Roff ) * 1.0 * B ;
   iNMDA = g * ( v - e ) ;
   ica = 7.0 * iNMDA / 10.0 ;
   iNMDA = 3.0 * iNMDA / 10.0 ;
   }
 _current += ica;
 _current += iNMDA;

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
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 * 1.e2/ (_nd_area);
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica * 1.e2/ (_nd_area);
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
 {   release(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = Ron_columnindex;  _dlist1[0] = DRon_columnindex;
 _slist1[1] = Roff_columnindex;  _dlist1[1] = DRoff_columnindex;
   _t_mgblock = makevector(1001*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/hrbncv/PV-SST/single_cells/cells/PV/tunes/seg_tuned/modfiles/NMDAIN.mod";
static const char* nmodl_file_text = 
  "TITLE simple NMDA receptors\n"
  "\n"
  "COMMENT\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "Essentially the same as /examples/nrniv/netcon/ampa.mod in the NEURON\n"
  "distribution - i.e. Alain Destexhe's simple AMPA model - but with\n"
  "different binding and unbinding rates and with a magnesium block.\n"
  "Modified by Andrew Davison, The Babraham Institute, May 2000\n"
  "\n"
  "\n"
  "	Simple model for glutamate AMPA receptors\n"
  "	=========================================\n"
  "\n"
  "  - FIRST-ORDER KINETICS, FIT TO WHOLE-CELL RECORDINGS\n"
  "\n"
  "    Whole-cell recorded postsynaptic currents mediated by AMPA/Kainate\n"
  "    receptors (Xiang et al., J. Neurophysiol. 71: 2552-2556, 1994) were used\n"
  "    to estimate the parameters of the present model; the fit was performed\n"
  "    using a simplex algorithm (see Destexhe et al., J. Computational Neurosci.\n"
  "    1: 195-230, 1994).\n"
  "\n"
  "  - SHORT PULSES OF TRANSMITTER (0.3 ms, 0.5 mM)\n"
  "\n"
  "    The simplified model was obtained from a detailed synaptic model that \n"
  "    included the release of transmitter in adjacent terminals, its lateral \n"
  "    diffusion and uptake, and its binding on postsynaptic receptors (Destexhe\n"
  "    and Sejnowski, 1995).  Short pulses of transmitter with first-order\n"
  "    kinetics were found to be the best fast alternative to represent the more\n"
  "    detailed models.\n"
  "\n"
  "  - ANALYTIC EXPRESSION\n"
  "\n"
  "    The first-order model can be solved analytically, leading to a very fast\n"
  "    mechanism for simulating synapses, since no differential equation must be\n"
  "    solved (see references below).\n"
  "\n"
  "\n"
  "\n"
  "References\n"
  "\n"
  "   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  An efficient method for\n"
  "   computing synaptic conductances based on a kinetic model of receptor binding\n"
  "   Neural Computation 6: 10-14, 1994.  \n"
  "\n"
  "   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for\n"
  "   excitable membranes, synaptic transmission and neuromodulation using a \n"
  "   common kinetic formalism, Journal of Computational Neuroscience 1: \n"
  "   195-230, 1994.\n"
  "\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS NMDAIN\n"
  "	RANGE g, Alpha, Beta, e, gmax, ica\n"
  "	USEION ca WRITE ica\n"
  "	NONSPECIFIC_CURRENT iNMDA            \n"
  "	GLOBAL Cdur, mg, Cmax\n"
  "}\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	Cmax	= 1	 (mM)           : max transmitter concentration\n"
  "	Cdur	= 1	 (ms)		: transmitter duration (rising phase) \n"
  "	Alpha	= 4	 (/ms /mM)	: forward (binding) rate (4)\n"
  "	Beta	= 0.0125	(/ms)		: backward rate (updated by Y. Zerlaut 2023 for 80ms decay)\n"
  "	e	= 0	 (mV)		: reversal potential\n"
  "        mg      = 1      (mM)           : external magnesium concentration\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v		(mV)		: postsynaptic voltage\n"
  "	iNMDA 		(nA)		: current = g*(v - e)\n"
  "	g 		(umho)		: conductance\n"
  "	Rinf				: steady state channels open\n"
  "	Rtau		(ms)		: time constant of channel binding\n"
  "	synon\n"
  "        B \n"
  "	gmax                              : magnesium block\n"
  "	ica\n"
  "}\n"
  "\n"
  "STATE {Ron Roff}\n"
  "\n"
  "INITIAL {\n"
  "	Rinf = Cmax*Alpha / (Cmax*Alpha + Beta)\n"
  "	Rtau = 1 / (Cmax*Alpha + Beta)\n"
  "	synon = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE release METHOD cnexp\n"
  "        B = mgblock(v)\n"
  "	g = (Ron + Roff)*1(umho) * B\n"
  "	iNMDA = g*(v - e)\n"
  "        ica = 7*iNMDA/10   :(5-10 times more permeable to Ca++ than Na+ or K+, Ascher and Nowak, 1988)\n"
  ":       ica = 0\n"
  "        iNMDA = 3*iNMDA/10\n"
  "\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "	Ron' = (synon*Rinf - Ron)/Rtau\n"
  "	Roff' = -Beta*Roff\n"
  "}\n"
  "\n"
  "FUNCTION mgblock(v(mV)) {\n"
  "        TABLE \n"
  "        DEPEND mg\n"
  "        FROM -140 TO 80 WITH 1000\n"
  "\n"
  "        : from Jahr & Stevens\n"
  "\n"
  "      \n"
  "	 mgblock = 1 / (1 + exp(0.072 (/mV) * -v) * (mg / 3.57 (mM)))  	\n"
  "}\n"
  "\n"
  ": following supports both saturation from single input and\n"
  ": summation from multiple inputs\n"
  ": if spike occurs during CDur then new off time is t + CDur\n"
  ": ie. transmitter concatenates but does not summate\n"
  ": Note: automatic initialization of all reference args to 0 except first\n"
  "\n"
  "			\n"
  "NET_RECEIVE(weight, on, nspike, r0, t0 (ms)) {\n"
  "	: flag is an implicit argument of NET_RECEIVE and  normally 0\n"
  "        if (flag == 0) { : a spike, so turn on if not already in a Cdur pulse\n"
  "		nspike = nspike + 1\n"
  "		if (!on) {\n"
  "			r0 = r0*exp(-Beta*(t - t0))\n"
  "			t0 = t\n"
  "			on = 1\n"
  "			synon = synon + weight\n"
  "			state_discontinuity(Ron, Ron + r0)\n"
  "			state_discontinuity(Roff, Roff - r0)\n"
  "		}\n"
  ":		 come again in Cdur with flag = current value of nspike\n"
  "		net_send(Cdur, nspike)\n"
  "       }\n"
  "	if (flag == nspike) { : if this associated with last spike then turn off\n"
  "		r0 = weight*Rinf + (r0 - weight*Rinf)*exp(-(t - t0)/Rtau)\n"
  "		t0 = t\n"
  "		synon = synon - weight\n"
  "		state_discontinuity(Ron, Ron - r0)\n"
  "		state_discontinuity(Roff, Roff + r0)\n"
  "		on = 0\n"
  "	}\n"
  "gmax = weight\n"
  "}\n"
  "\n"
  "\n"
  ;
#endif
