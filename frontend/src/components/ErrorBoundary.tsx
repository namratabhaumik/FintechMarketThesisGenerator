import { Component, type ReactNode } from "react";

// Guards JobView the way app.ts's try/catch around renderJob did: an unexpected
// response shape shows a message instead of crashing the app. Reset by keying
// the boundary on the job id so each new job gets a fresh attempt.
interface Props {
  fallback: ReactNode;
  children: ReactNode;
}

export class ErrorBoundary extends Component<Props, { hasError: boolean }> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(err: unknown) {
    console.error("Failed to render job", err);
  }

  render() {
    return this.state.hasError ? this.props.fallback : this.props.children;
  }
}
