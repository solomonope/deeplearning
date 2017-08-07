import math as Math;

a = [-1, 1, 5];

l0 = Math.exp(a[0]);
l1 = Math.exp(a[1]);
l2 = Math.exp(a[2]);

print(l0);
print(l1);
print(l2);

sigma = l0 +l1 + l2;

print(sigma)

p0 = l0/ sigma;

p1 = l1/sigma

p2 = l2 /sigma;

print(p0);
print(p1);
print(p2);

