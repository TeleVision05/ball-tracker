# BallTrace frontend

React 19 + TypeScript + Vite + Tailwind. Talks to the FastAPI backend at `http://localhost:8000/api` (see [`../API_CONTRACT.md`](../API_CONTRACT.md)).

```bash
npm install
npm run dev      # http://localhost:5173
npm run lint     # oxlint
npm run build    # tsc -b && vite build -> dist/
```

Layout: `src/api/client.ts` (HTTP client), `src/components/{analysis,layout,player,settings}`, `src/hooks` (`useVideoPlayer`, `useAnalysisProgress` SSE), `src/lib` (trajectory and status helpers), `src/types`.

There is no test runner yet; see the roadmap at the repo root.
