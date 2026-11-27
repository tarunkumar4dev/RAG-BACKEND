-- Read-only: the CLI login role's expiry and a4ai's own a4_backup schemas.
select rolname, rolcanlogin, rolvaliduntil from pg_roles where rolname = 'cli_login_postgres';
