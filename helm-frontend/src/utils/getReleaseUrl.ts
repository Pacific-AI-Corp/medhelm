import { getHelmPortalBaseUrl } from "@/utils/helmPortalConfig";

/** Project / release links (navbar, release menu, home cards). */
export default function getReleaseUrl(
  version: string | undefined,
  currProjectId: string | undefined,
): string {
  if (!currProjectId) {
    return "#";
  }

  const base = getHelmPortalBaseUrl();
  if (currProjectId === "home") {
    return `${base}/`;
  }
  if (!version) {
    return `${base}/${currProjectId}/latest/`;
  }
  return `${base}/${currProjectId}/${version}/`;
}
