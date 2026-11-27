-- Read-only: per-role settings (statement_timeout etc.) for the API roles.
select rolname, rolconfig from pg_roles where rolname in ('anon', 'authenticated', 'service_role', 'authenticator') order by 1;
