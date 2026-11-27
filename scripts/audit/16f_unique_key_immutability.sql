-- Read-only: are the functions of the proposed unique-key expression IMMUTABLE (required for an expression index)?
select p.oid::regprocedure as fn, case p.provolatile when 'i' then 'immutable' when 's' then 'stable' else 'volatile' end as volatility
from pg_proc p where p.oid in ('md5(text)'::regprocedure, 'lower(text)'::regprocedure, 'btrim(text)'::regprocedure,
                               'regexp_replace(text,text,text,text)'::regprocedure);
