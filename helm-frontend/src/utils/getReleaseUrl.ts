import { getHelmPortalBaseUrl } from "@/utils/helmPortalConfig";

/** Project / release links (navbar, release menu, home cards). */
export default function getReleaseUrl(
  version: string | undefined,
  currProjectId: string | undefined,
): string {
  const base = getHelmPortalBaseUrl();

  const project = currProjectId ?? "medhelm";

  if (project === "home") {
    return `${base}/`;
  }

  if (!version) {
    return `${base}/${project}/latest/`;
  }

  return `${base}/${project}/${version}/`;
}
