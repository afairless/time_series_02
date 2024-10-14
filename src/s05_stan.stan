data {
  int<lower=0> N;        // number of cases/data points
  vector[N] y;           // outcome/response variable
}
parameters {
  //real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  //y[2:N] ~ normal(alpha + beta * y[1:(N-1)], sigma);
  y[2:N] ~ normal(beta * y[1:(N-1)], sigma);
}
