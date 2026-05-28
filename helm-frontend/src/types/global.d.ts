interface Window {
  RELEASE: string | undefined;
  SUITE: string | undefined;
  BENCHMARK_OUTPUT_BASE_URL: string;
  PROJECT_ID: string;
  /** HELM hub base URL, no trailing slash. Default: https://crfm.stanford.edu/helm */
  HELM_PORTAL_BASE_URL?: string;
  /** Full URL to project_metadata.json */
  HELM_PROJECT_METADATA_URL?: string;
  /** CRFM logo link; if portal is custom and unset, uses location.origin */
  HELM_LOGO_HREF?: string;
  /** Release tags for navbar (newest first) when project_metadata is unavailable */
  HELM_RELEASES?: string[];
}
