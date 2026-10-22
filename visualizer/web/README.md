# Epsilon visualizer (web)

Vite + React UI for Condition **A+B** grouped JSON and derived artifacts.

Both `npm run dev` and `npm run build` regenerate the A+B data first. `dist/` is build output and is not versioned.

`npm run build` checks TypeScript before bundling; `npm test` checks the shared
graph geometry used by the screen and PNG exporter. Use `npm run typecheck` for
the type check alone.

See **[../README.md](../README.md)** for the canonical data inputs, build commands, and local serving or GitHub Pages deployment.
