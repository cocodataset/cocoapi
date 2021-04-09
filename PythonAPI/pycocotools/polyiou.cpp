
#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cmath>
#include <vector>
using namespace std;
#define maxn 51
const double eps=1E-8;
int sig(double d){
    return(d>eps)-(d<-eps);
}
struct Point{
    double x,y; Point(){}
    Point(double x,double y):x(x),y(y){}
    bool operator==(const Point&p)const{
        return sig(x-p.x)==0&&sig(y-p.y)==0;
    }
};
double cross(Point o,Point a,Point b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}
double area(Point* ps,int n){
    ps[n]=ps[0];
    double res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
int lineCross(Point a,Point b,Point c,Point d,Point&p){
    double s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}
//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
//void polygon_cut(Point*p,int&n,Point a,Point b){
//    static Point pp[maxn];
//    int m=0;p[n]=p[0];
//    for(int i=0;i<n;i++){
//        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
//        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
//            lineCross(a,b,p[i],p[i+1],pp[m++]);
//    }
//    n=0;
//    for(int i=0;i<m;i++)
//        if(!i||!(pp[i]==pp[i-1]))
//            p[n++]=pp[i];
//    while(n>1&&p[n-1]==p[0])n--;
//}
void polygon_cut(Point*p,int&n,Point a,Point b, Point* pp){
//    static Point pp[maxn];
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;
    for(int i=0;i<m;i++)
        if(!i||!(pp[i]==pp[i-1]))
            p[n++]=pp[i];
    while(n>1&&p[n-1]==p[0])n--;
}
//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
double intersectArea(Point a,Point b,Point c,Point d){
    Point o(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    if(s1==-1) swap(a,b);
    if(s2==-1) swap(c,d);
    Point p[10]={o,a,b};
    int n=3;
    Point pp[maxn];
    polygon_cut(p,n,o,c, pp);
    polygon_cut(p,n,c,d, pp);
    polygon_cut(p,n,d,o, pp);
    double res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;return res;
}
//求两多边形的交面积
double intersectArea(Point*ps1,int n1,Point*ps2,int n2){
    if(area(ps1,n1)<0) reverse(ps1,ps1+n1);
    if(area(ps2,n2)<0) reverse(ps2,ps2+n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    double res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;//assumeresispositive!
}




double iou_poly(vector<double> p, vector<double> q) {
    Point ps1[maxn],ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    double inter_area = intersectArea(ps1, n1, ps2, n2);
    double union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    double iou = 0;
    if (union_area == 0) {
        iou = (inter_area + 1) / (union_area + 1);
    } else {
        iou = inter_area / union_area;
    }

//    cout << "inter_area:" << inter_area << endl;
//    cout << "union_area:" << union_area << endl;
//    cout << "iou:" << iou << endl;

    return iou;
}
//
int main(){

    double p[8] = {6.86000000e+02,   2.97600000e+03,   7.09000000e+02,   2.97600000e+03,
              7.24000000e+02,   2.97600000e+03,   7.01000000e+02,   2.97600000e+03};
    double q[8] = { 6.86000000e+02,   2.97600000e+03,   7.09000000e+02,   2.97600000e+03,
   7.24000000e+02,   2.97600000e+03,   7.01000000e+02,   2.97600000e+03};
    vector<double> P(p, p + 8);
    vector<double> Q(q, q + 8);
    double iou = iou_poly(P, Q);
    printf("iou_poly: %f\n", iou);
    return 0;
}

//int main(){
//    double p[8] = {0, 0, 1, 0, 1, 1, 0, 1};
//    double q[8] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
//    iou_poly(p, q);
//    return 0;
//}