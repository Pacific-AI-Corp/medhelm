/** Default public HELM hub (Stanford). Override with window.HELM_PORTAL_BASE_URL in config.js. */
const DEFAULT_HELM_PORTAL_BASE_URL = "https://crfm.stanford.edu/helm";

export function getHelmPortalBaseUrl(): string {
  const raw = window.HELM_PORTAL_BASE_URL ?? DEFAULT_HELM_PORTAL_BASE_URL;
  return raw.replace(/\/$/, "");
}

export function getProjectMetadataUrl(): string {
  if (window.HELM_PROJECT_METADATA_URL) {
    return window.HELM_PROJECT_METADATA_URL;
  }
  return `${getHelmPortalBaseUrl()}/project_metadata.json`;
}

export function getHelmLogoHref(): string {
  if (window.HELM_LOGO_HREF !== undefined && window.HELM_LOGO_HREF !== "") {
    return window.HELM_LOGO_HREF;
  }
  if (getHelmPortalBaseUrl() !== DEFAULT_HELM_PORTAL_BASE_URL) {
    return window.location.origin;
  }
  return "https://crfm.stanford.edu/";
}

/** Navbar release list (newest first). Used when project_metadata is missing or fails to load. */
export function getHelmReleases(): string[] {
  const configured = window.HELM_RELEASES;
  if (configured !== undefined && configured.length > 0) {
    return configured;
  }
  if (window.RELEASE) {
    return [window.RELEASE];
  }
  if (window.SUITE) {
    return [window.SUITE];
  }
  return ["v1.0.0"];
}
