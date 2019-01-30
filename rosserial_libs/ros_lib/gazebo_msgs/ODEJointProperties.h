#ifndef _ROS_gazebo_msgs_ODEJointProperties_h
#define _ROS_gazebo_msgs_ODEJointProperties_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace gazebo_msgs
{

  class ODEJointProperties : public ros::Msg
  {
    public:
      uint8_t damping_length;
      double st_damping;
      double * damping;
      uint8_t hiStop_length;
      double st_hiStop;
      double * hiStop;
      uint8_t loStop_length;
      double st_loStop;
      double * loStop;
      uint8_t erp_length;
      double st_erp;
      double * erp;
      uint8_t cfm_length;
      double st_cfm;
      double * cfm;
      uint8_t stop_erp_length;
      double st_stop_erp;
      double * stop_erp;
      uint8_t stop_cfm_length;
      double st_stop_cfm;
      double * stop_cfm;
      uint8_t fudge_factor_length;
      double st_fudge_factor;
      double * fudge_factor;
      uint8_t fmax_length;
      double st_fmax;
      double * fmax;
      uint8_t vel_length;
      double st_vel;
      double * vel;

    ODEJointProperties():
      damping_length(0), damping(NULL),
      hiStop_length(0), hiStop(NULL),
      loStop_length(0), loStop(NULL),
      erp_length(0), erp(NULL),
      cfm_length(0), cfm(NULL),
      stop_erp_length(0), stop_erp(NULL),
      stop_cfm_length(0), stop_cfm(NULL),
      fudge_factor_length(0), fudge_factor(NULL),
      fmax_length(0), fmax(NULL),
      vel_length(0), vel(NULL)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset++) = damping_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < damping_length; i++){
      union {
        double real;
        uint64_t base;
      } u_dampingi;
      u_dampingi.real = this->damping[i];
      *(outbuffer + offset + 0) = (u_dampingi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_dampingi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_dampingi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_dampingi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_dampingi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_dampingi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_dampingi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_dampingi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->damping[i]);
      }
      *(outbuffer + offset++) = hiStop_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < hiStop_length; i++){
      union {
        double real;
        uint64_t base;
      } u_hiStopi;
      u_hiStopi.real = this->hiStop[i];
      *(outbuffer + offset + 0) = (u_hiStopi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_hiStopi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_hiStopi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_hiStopi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_hiStopi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_hiStopi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_hiStopi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_hiStopi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->hiStop[i]);
      }
      *(outbuffer + offset++) = loStop_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < loStop_length; i++){
      union {
        double real;
        uint64_t base;
      } u_loStopi;
      u_loStopi.real = this->loStop[i];
      *(outbuffer + offset + 0) = (u_loStopi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_loStopi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_loStopi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_loStopi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_loStopi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_loStopi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_loStopi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_loStopi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->loStop[i]);
      }
      *(outbuffer + offset++) = erp_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < erp_length; i++){
      union {
        double real;
        uint64_t base;
      } u_erpi;
      u_erpi.real = this->erp[i];
      *(outbuffer + offset + 0) = (u_erpi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_erpi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_erpi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_erpi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_erpi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_erpi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_erpi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_erpi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->erp[i]);
      }
      *(outbuffer + offset++) = cfm_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < cfm_length; i++){
      union {
        double real;
        uint64_t base;
      } u_cfmi;
      u_cfmi.real = this->cfm[i];
      *(outbuffer + offset + 0) = (u_cfmi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_cfmi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_cfmi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_cfmi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_cfmi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_cfmi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_cfmi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_cfmi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->cfm[i]);
      }
      *(outbuffer + offset++) = stop_erp_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < stop_erp_length; i++){
      union {
        double real;
        uint64_t base;
      } u_stop_erpi;
      u_stop_erpi.real = this->stop_erp[i];
      *(outbuffer + offset + 0) = (u_stop_erpi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_stop_erpi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_stop_erpi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_stop_erpi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_stop_erpi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_stop_erpi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_stop_erpi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_stop_erpi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->stop_erp[i]);
      }
      *(outbuffer + offset++) = stop_cfm_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < stop_cfm_length; i++){
      union {
        double real;
        uint64_t base;
      } u_stop_cfmi;
      u_stop_cfmi.real = this->stop_cfm[i];
      *(outbuffer + offset + 0) = (u_stop_cfmi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_stop_cfmi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_stop_cfmi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_stop_cfmi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_stop_cfmi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_stop_cfmi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_stop_cfmi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_stop_cfmi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->stop_cfm[i]);
      }
      *(outbuffer + offset++) = fudge_factor_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < fudge_factor_length; i++){
      union {
        double real;
        uint64_t base;
      } u_fudge_factori;
      u_fudge_factori.real = this->fudge_factor[i];
      *(outbuffer + offset + 0) = (u_fudge_factori.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_fudge_factori.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_fudge_factori.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_fudge_factori.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_fudge_factori.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_fudge_factori.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_fudge_factori.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_fudge_factori.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->fudge_factor[i]);
      }
      *(outbuffer + offset++) = fmax_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < fmax_length; i++){
      union {
        double real;
        uint64_t base;
      } u_fmaxi;
      u_fmaxi.real = this->fmax[i];
      *(outbuffer + offset + 0) = (u_fmaxi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_fmaxi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_fmaxi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_fmaxi.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_fmaxi.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_fmaxi.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_fmaxi.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_fmaxi.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->fmax[i]);
      }
      *(outbuffer + offset++) = vel_length;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      *(outbuffer + offset++) = 0;
      for( uint8_t i = 0; i < vel_length; i++){
      union {
        double real;
        uint64_t base;
      } u_veli;
      u_veli.real = this->vel[i];
      *(outbuffer + offset + 0) = (u_veli.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_veli.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_veli.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_veli.base >> (8 * 3)) & 0xFF;
      *(outbuffer + offset + 4) = (u_veli.base >> (8 * 4)) & 0xFF;
      *(outbuffer + offset + 5) = (u_veli.base >> (8 * 5)) & 0xFF;
      *(outbuffer + offset + 6) = (u_veli.base >> (8 * 6)) & 0xFF;
      *(outbuffer + offset + 7) = (u_veli.base >> (8 * 7)) & 0xFF;
      offset += sizeof(this->vel[i]);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint8_t damping_lengthT = *(inbuffer + offset++);
      if(damping_lengthT > damping_length)
        this->damping = (double*)realloc(this->damping, damping_lengthT * sizeof(double));
      offset += 3;
      damping_length = damping_lengthT;
      for( uint8_t i = 0; i < damping_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_damping;
      u_st_damping.base = 0;
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_damping.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_damping = u_st_damping.real;
      offset += sizeof(this->st_damping);
        memcpy( &(this->damping[i]), &(this->st_damping), sizeof(double));
      }
      uint8_t hiStop_lengthT = *(inbuffer + offset++);
      if(hiStop_lengthT > hiStop_length)
        this->hiStop = (double*)realloc(this->hiStop, hiStop_lengthT * sizeof(double));
      offset += 3;
      hiStop_length = hiStop_lengthT;
      for( uint8_t i = 0; i < hiStop_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_hiStop;
      u_st_hiStop.base = 0;
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_hiStop.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_hiStop = u_st_hiStop.real;
      offset += sizeof(this->st_hiStop);
        memcpy( &(this->hiStop[i]), &(this->st_hiStop), sizeof(double));
      }
      uint8_t loStop_lengthT = *(inbuffer + offset++);
      if(loStop_lengthT > loStop_length)
        this->loStop = (double*)realloc(this->loStop, loStop_lengthT * sizeof(double));
      offset += 3;
      loStop_length = loStop_lengthT;
      for( uint8_t i = 0; i < loStop_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_loStop;
      u_st_loStop.base = 0;
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_loStop.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_loStop = u_st_loStop.real;
      offset += sizeof(this->st_loStop);
        memcpy( &(this->loStop[i]), &(this->st_loStop), sizeof(double));
      }
      uint8_t erp_lengthT = *(inbuffer + offset++);
      if(erp_lengthT > erp_length)
        this->erp = (double*)realloc(this->erp, erp_lengthT * sizeof(double));
      offset += 3;
      erp_length = erp_lengthT;
      for( uint8_t i = 0; i < erp_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_erp;
      u_st_erp.base = 0;
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_erp.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_erp = u_st_erp.real;
      offset += sizeof(this->st_erp);
        memcpy( &(this->erp[i]), &(this->st_erp), sizeof(double));
      }
      uint8_t cfm_lengthT = *(inbuffer + offset++);
      if(cfm_lengthT > cfm_length)
        this->cfm = (double*)realloc(this->cfm, cfm_lengthT * sizeof(double));
      offset += 3;
      cfm_length = cfm_lengthT;
      for( uint8_t i = 0; i < cfm_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_cfm;
      u_st_cfm.base = 0;
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_cfm.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_cfm = u_st_cfm.real;
      offset += sizeof(this->st_cfm);
        memcpy( &(this->cfm[i]), &(this->st_cfm), sizeof(double));
      }
      uint8_t stop_erp_lengthT = *(inbuffer + offset++);
      if(stop_erp_lengthT > stop_erp_length)
        this->stop_erp = (double*)realloc(this->stop_erp, stop_erp_lengthT * sizeof(double));
      offset += 3;
      stop_erp_length = stop_erp_lengthT;
      for( uint8_t i = 0; i < stop_erp_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_stop_erp;
      u_st_stop_erp.base = 0;
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_stop_erp.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_stop_erp = u_st_stop_erp.real;
      offset += sizeof(this->st_stop_erp);
        memcpy( &(this->stop_erp[i]), &(this->st_stop_erp), sizeof(double));
      }
      uint8_t stop_cfm_lengthT = *(inbuffer + offset++);
      if(stop_cfm_lengthT > stop_cfm_length)
        this->stop_cfm = (double*)realloc(this->stop_cfm, stop_cfm_lengthT * sizeof(double));
      offset += 3;
      stop_cfm_length = stop_cfm_lengthT;
      for( uint8_t i = 0; i < stop_cfm_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_stop_cfm;
      u_st_stop_cfm.base = 0;
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_stop_cfm.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_stop_cfm = u_st_stop_cfm.real;
      offset += sizeof(this->st_stop_cfm);
        memcpy( &(this->stop_cfm[i]), &(this->st_stop_cfm), sizeof(double));
      }
      uint8_t fudge_factor_lengthT = *(inbuffer + offset++);
      if(fudge_factor_lengthT > fudge_factor_length)
        this->fudge_factor = (double*)realloc(this->fudge_factor, fudge_factor_lengthT * sizeof(double));
      offset += 3;
      fudge_factor_length = fudge_factor_lengthT;
      for( uint8_t i = 0; i < fudge_factor_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_fudge_factor;
      u_st_fudge_factor.base = 0;
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_fudge_factor.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_fudge_factor = u_st_fudge_factor.real;
      offset += sizeof(this->st_fudge_factor);
        memcpy( &(this->fudge_factor[i]), &(this->st_fudge_factor), sizeof(double));
      }
      uint8_t fmax_lengthT = *(inbuffer + offset++);
      if(fmax_lengthT > fmax_length)
        this->fmax = (double*)realloc(this->fmax, fmax_lengthT * sizeof(double));
      offset += 3;
      fmax_length = fmax_lengthT;
      for( uint8_t i = 0; i < fmax_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_fmax;
      u_st_fmax.base = 0;
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_fmax.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_fmax = u_st_fmax.real;
      offset += sizeof(this->st_fmax);
        memcpy( &(this->fmax[i]), &(this->st_fmax), sizeof(double));
      }
      uint8_t vel_lengthT = *(inbuffer + offset++);
      if(vel_lengthT > vel_length)
        this->vel = (double*)realloc(this->vel, vel_lengthT * sizeof(double));
      offset += 3;
      vel_length = vel_lengthT;
      for( uint8_t i = 0; i < vel_length; i++){
      union {
        double real;
        uint64_t base;
      } u_st_vel;
      u_st_vel.base = 0;
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 3))) << (8 * 3);
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 4))) << (8 * 4);
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 5))) << (8 * 5);
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 6))) << (8 * 6);
      u_st_vel.base |= ((uint64_t) (*(inbuffer + offset + 7))) << (8 * 7);
      this->st_vel = u_st_vel.real;
      offset += sizeof(this->st_vel);
        memcpy( &(this->vel[i]), &(this->st_vel), sizeof(double));
      }
     return offset;
    }

    const char * getType(){ return "gazebo_msgs/ODEJointProperties"; };
    const char * getMD5(){ return "1b744c32a920af979f53afe2f9c3511f"; };

  };

}
#endif