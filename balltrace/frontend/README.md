# BallTrace frontend

React 19 + TypeScript + Vite + Tailwind. Talks to the FastAPI backend at `http://localhost:8000/api` (see [`../API_CONTRACT.md`](../API_CONTRACT.md)).

```bash
npm install
npm run dev      # http://localhost:5173
npm run lint     # oxlint
npm test         # vitest + React Testing Library (jsdom)
npm run build    # tsc -b && vite build -> dist/
```

Layout: `src/api/client.ts` (HTTP client), `src/components/{analysis,layout,player,settings}`, `src/hooks` (`useVideoPlayer`, `useAnalysisProgress` SSE), `src/lib` (trajectory and status helpers), `src/types`.

Every failure shown on screen goes through `src/lib/friendlyError.ts`, which maps the backend's worded errors and known HTTP statuses to plain language and logs the raw error to the console. Tests live next to the code as `*.test.ts(x)`; `src/test/` holds the setup and fixtures.
