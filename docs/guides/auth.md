# Authentication

FinThesis uses [Supabase Auth](https://supabase.com/docs/guides/auth) with Google as the identity provider. There are no passwords to manage and no separate sign-up step.

## Signing in to the app

Click **Continue with Google** on the login screen. After the OAuth redirect you land back in the app with your email shown in the header, next to a **Sign out** button. Sessions refresh automatically in the background; you stay signed in across visits until you sign out.

## Data isolation

Every thesis belongs to the account that generated it, enforced at the database layer with row-level security. Your library, resume picker, and related-theses recall only ever operate on your own theses. The one exception is the [admin role](admin.md), which can list and delete theses across accounts.

## Calling the API directly

All API endpoints except `GET /api/health` require a Supabase JWT in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

The token is your Supabase session access token - the same one the web app uses. The most direct way to obtain one for experimentation is to sign in to the app and copy the current session's `access_token` from the browser (it lives in the Supabase auth entry in local storage). Tokens expire and refresh on the normal Supabase schedule, so treat a copied token as short-lived.

With a token in hand:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://fintechmarketthesisgenerator-prod.onrender.com/api/theses?limit=5"
```

Requests without a valid token receive `401`; requests for admin-only resources without the admin role receive `403`. See the [API reference](../reference/api.md) for the full surface and error codes.
