# Vercel Test Copy

This is a Vercel-oriented test version of the manga automation dashboard.

## Purpose

- Test a serverless-friendly deployment on Vercel
- Keep the main working Flask/Playwright project untouched
- Process one selected manga per request

## Notes

- This copy is intentionally simplified for Vercel
- It is not the same as the main long-running worker version
- Thumbnail fallback still depends on wp-admin automation and may be limited on Vercel

## Deploy

1. Push this folder to GitHub
2. Import the repo into Vercel
3. Set required environment variables in Vercel if needed
