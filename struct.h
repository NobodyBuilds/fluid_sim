#pragma once
#include<vector>
#include <unordered_map>
struct Vec3 {
    float x, y, z;
};
struct Body {

    float* posx,posy,posz;
    
   
    float* aclx,acly,aclz;
    

    float* old_aclx,old_acly,old_aclz;
   
   
    float* velx,vely,velz;
    
    
    float* forcex,forcey,forcez;
    

    
    float* Size;
    float* Mass;

    int* Iscenter;
    int* r,b,g;
   
    int* br,bb,bg;
   
   
    float* Heat ;
   
    float* Density;
    
    float* Pressure;


};


