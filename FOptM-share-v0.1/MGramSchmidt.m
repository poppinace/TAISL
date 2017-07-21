function V = MGramSchmidt(V)
[n,k] = size(V);

for dj = 1:k
    for di = 1:dj-1
        V(:,dj) = V(:,dj) - proj(V(:,di), V(:,dj));
    end
    V(:,dj) = V(:,dj)/norm(V(:,dj));
end
end


%project v onto u
function v = proj(u,v)
v = (dot(v,u)/dot(u,u))*u;
end
