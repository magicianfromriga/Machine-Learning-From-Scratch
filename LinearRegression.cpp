#include<bits/stdc++.h>
using namespace std;

// First a simple function that computes the equation of a line y = mx + c. This is the basic equation of the linear regression
float simpleLineEquation(float x, float m, float c)
{
    float y = (m*x) + c;
    return y;
}

// Extend the above function to compute y_hat for given predictions x.

vector<float> predictY_hat(vector<float> x, float beta_1, float beta_0)
{
    int len = x.size();
    vector<float> y (len,0);
    for(int i=0; i<len; i++)
        y[i]=beta_1*x[i] + beta_0;
    return y;
}

// Next extend this line equation to include multiple x variables. Have an array for both m and x.

float complexLineEquation(vector<float> x, vector<float> m, float c)
{
    int len = x.size();
    float res = 0;
    for(int i=0; i<len;i++)
    {
        res+=x[i]*m[i];
    }
    return res+=c;
}

// Write the formula for simple linear regression - compute the slope and intercept given a number of points (x,y)

vector<float> simpleLinearRegression(vector<float> x, vector<float> y)
{
    int len = x.size();
    float x_bar = accumulate(x.begin(),x.end(), 0.0)/len;
    float y_bar = accumulate(y.begin(), y.end(), 0.0)/len;
    float numerator = 0.0;
    float denominator = 0.0;
    for(int i=0; i<len; i++)
    {
        numerator+=(x[i]-x_bar)*(y[i]-y_bar);
        denominator+=pow((x[i]-x_bar),2);
    }
   float beta_1 = numerator/denominator;
   float beta_0 = y_bar - (beta_1*x_bar);
   vector<float> retVal = {beta_0,beta_1};
   return retVal;
}

// Function to calculate RSE and RSS:

vector<float> calculateErrors(vector<float> y, vector<float> y_hat)
{
    float rse=0.0,rss=0.0,tss=0.0;
    int len = y.size();
    float y_bar = accumulate(y.begin(), y.end(), 0.0)/len;
    for(int i=0; i<len; i++)
    {
      rss+=pow(y[i]-y_hat[i],2);
      tss+=pow(y[i]-y_bar,2);
    }
    float r_squared = 1 - (rss/tss);
    rse=sqrt((1/(len-2))*rss);
    vector<float> retVal = {rss,rse,r_squared};
    return retVal;
}

int main()
{
    int num_x;
    cout<<"Enter number of x and y variables - it should be greater than two"<<endl;
    cin>>num_x;
    if(num_x<=2)
    {
        cout<<"Error - regression can't be performed with less than two points";
        return 0;
    }
    vector<float> x(num_x), y(num_x);
       cout<<"Enter x values"<<endl;
    for(int i=0; i<num_x; i++ )
        cin>>x[i];
    cout<<"Enter y values"<<endl;
    for(int i=0; i<num_x; i++ )
        cin>>y[i];
    vector<float> betaVal = simpleLinearRegression(x,y);
    vector<float> y_hat = predictY_hat(x,betaVal[1],betaVal[0]);
    cout<<"Slope is "<<betaVal[1]<<" and intercept is "<<betaVal[0] <<endl;
    cout<<"Equation of the line is y = "<<betaVal[1]<<"x + "<<betaVal[0]<<endl;
    cout<<"Predicted y values are as follows: "<<endl;
    for(int i=0 ; i<num_x;i++)
        cout<<y_hat[i]<<endl;
    vector<float> errors = calculateErrors(y,y_hat);
    cout<<"Residual Sum of Squares is: "<<errors[0]<<endl;
    cout<<"Residual Standard Error is: "<<errors[1]<<endl;
    cout<<"R Squared Error is: "<<errors[2]<<endl;
    return 0;
}
