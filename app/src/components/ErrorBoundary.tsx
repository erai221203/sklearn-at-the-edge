import { Component, type ErrorInfo, type ReactNode } from "react";

interface State {
  error: Error | null;
}

/**
 * Without this, a render error anywhere unmounts the whole tree and leaves a
 * blank white page with the reason visible only in the console.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("render failed", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="notice" data-tone="error" style={{ margin: 24 }}>
        <strong>Something broke while rendering this page.</strong>
        <p className="small" style={{ marginTop: 6 }}>
          {error.message}
        </p>
        <button
          className="btn btn--ghost"
          style={{ marginTop: 12 }}
          onClick={() => this.setState({ error: null })}
        >
          Try again
        </button>
      </div>
    );
  }
}
