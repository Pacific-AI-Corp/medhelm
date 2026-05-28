import { Badge } from "@tremor/react";
import { useEffect, useState } from "react";
import getReleaseSummary from "@/services/getReleaseSummary";
import type ReleaseSummary from "@/types/ReleaseSummary";
import type ProjectMetadata from "@/types/ProjectMetadata";
import { ChevronDownIcon } from "@heroicons/react/24/solid";
import getReleaseUrl from "@/utils/getReleaseUrl";
import {
  getHelmReleases,
  getProjectMetadataUrl,
} from "@/utils/helmPortalConfig";

function ReleaseDropdown() {
  const [summary, setSummary] = useState<ReleaseSummary>({
    release: undefined,
    suites: undefined,
    suite: undefined,
    date: "",
  });

  const [metadataReleases, setMetadataReleases] = useState<
    string[] | undefined
  >();

  const [currProjectMetadata, setCurrProjectMetadata] = useState<
    ProjectMetadata | undefined
  >();

  useEffect(() => {
    fetch(getProjectMetadataUrl())
      .then((response) => response.json())
      .then((data: ProjectMetadata[]) => {
        const projectId = window.PROJECT_ID ?? "medhelm";

        const currentEntry =
          data.find((entry) => entry.id === projectId) ||
          data.find((entry) => entry.id === "lite");

        setCurrProjectMetadata(currentEntry);

        if (currentEntry?.releases?.length) {
          setMetadataReleases(currentEntry.releases);
        }
      })
      .catch((error) => {
        console.error("Error fetching JSON:", error);
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      const summ = await getReleaseSummary(controller.signal);
      setSummary(summ);
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  const configuredReleases = getHelmReleases();

  const releases =
    configuredReleases.length > 1
      ? configuredReleases
      : metadataReleases?.length
        ? metadataReleases
        : ["latest"];

  const currentVersion = summary.release || summary.suite || null;

  if (!currentVersion) {
    return null;
  }

  const releaseInfo = `Release ${currentVersion} (${summary.date})`;

  if (releases.length <= 1) {
    return <div>{releaseInfo}</div>;
  }

  const indexOfCurrentVersion = releases.indexOf(currentVersion);

  const badge =
    indexOfCurrentVersion === 0 ? (
      <Badge color="blue">latest</Badge>
    ) : indexOfCurrentVersion > 0 ? (
      <Badge color="yellow">stale</Badge>
    ) : (
      <Badge color="blue">preview</Badge>
    );

  // ✅ FIXED: safe projectId (no broken fallback chain)
  const projectId = currProjectMetadata?.id || "medhelm";

  const menuVersions = ["latest"].concat(
    releases.filter((release) => release !== "latest"),
  );

  return (
    <div className="dropdown">
      <div
        tabIndex={0}
        role="button"
        className="normal-case bg-white border-0 block whitespace-nowrap"
        aria-haspopup="true"
        aria-controls="menu"
      >
        {releaseInfo}&nbsp;{badge}&nbsp;
        <ChevronDownIcon
          fill="black"
          color="black"
          className="inline text w-4 h-4"
        />
      </div>

      <ul
        tabIndex={0}
        className="dropdown-content z-[50] menu p-1 shadow-lg bg-base-100 rounded-box w-max text-base"
        role="menu"
      >
        {menuVersions.map((release) => (
          <li key={release}>
            <a
              href={`${getReleaseUrl(release, projectId)}#/leaderboard`}
              className="block"
              role="menuitem"
            >
              {release}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ReleaseDropdown;
