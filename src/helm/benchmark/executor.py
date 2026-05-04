import os
from typing import List, Optional
from dataclasses import dataclass, replace

from helm.common.context import Context
from helm.common.local_context import LocalContext
from helm.common.remote_context import RemoteContext
from helm.common.cache_backend_config import (
    CacheBackendConfig,
    BlackHoleCacheBackendConfig,
    MongoCacheBackendConfig,
    SqliteCacheBackendConfig,
)
from helm.common.general import parallel_map
from helm.common.hierarchical_logger import htrack, hlog, hwarn
from helm.common.request import RequestResult, GeneratedOutput
from helm.common.authentication import Authentication
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState


class ExecutorError(Exception):
    pass


@dataclass(frozen=True)
class ExecutionSpec:

    url: Optional[str]
    """If non-empty, URL of the proxy server we send requests to (e.g., http://localhost:1959)."""

    auth: Authentication
    """Authentication that will be passed into the remote service, if using the remote context."""

    local_path: Optional[str]
    """Path where API credentials and cache is stored.

    This path is the same as `--base-path` when launching the proxy server (see server.py).
    Required when url is not set."""

    parallelism: int
    """How many threads to have at once"""

    dry_run: bool = False
    """Whether to skip execution"""

    sqlite_cache_backend_config: Optional[SqliteCacheBackendConfig] = None
    """If set, SQLite will be used for the cache.

    This specifies the directory in which the SQLite cache will store files.
    At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set."""

    mongo_cache_backend_config: Optional[MongoCacheBackendConfig] = None
    """If set, MongoDB will be used for the cache.

    This specifies the MongoDB database to be used by the MongoDB cache.
    At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set."""

    batch_size: Optional[int] = None
    """Batch size for batch requests. Only applicable if the context supports batch requests."""


class Executor:
    """
    An `Executor` takes a `ScenarioState` which has a bunch of requests.
    Issue them to the API and return the results.
    """

    def __init__(self, execution_spec: ExecutionSpec):
        self.execution_spec = execution_spec
        self.batch_size = execution_spec.batch_size

        cache_backend_config: CacheBackendConfig
        if execution_spec.sqlite_cache_backend_config and execution_spec.mongo_cache_backend_config:
            raise ExecutorError("At most one of sqlite_cache_backend_config and mongo_cache_backend_config can be set.")
        elif execution_spec.sqlite_cache_backend_config:
            cache_backend_config = execution_spec.sqlite_cache_backend_config
        elif execution_spec.mongo_cache_backend_config:
            cache_backend_config = execution_spec.mongo_cache_backend_config
        else:
            cache_backend_config = BlackHoleCacheBackendConfig()

        self.context: Context
        if execution_spec.url:
            hlog(f"Running using remote API proxy server: {execution_spec.url}")
            self.context = RemoteContext(execution_spec.url, execution_spec.auth)
        elif execution_spec.local_path:
            hlog(f"Running in local mode with base path: {execution_spec.local_path}")
            self.context = LocalContext(
                base_path=execution_spec.local_path,
                cache_backend_config=cache_backend_config,
            )
        else:
            raise ValueError("Either the proxy server URL or the local path must be set")

    @htrack(None)
    def execute(self, scenario_state: ScenarioState) -> ScenarioState:
        if self.execution_spec.dry_run:
            hlog("Skipped execution.")
            return scenario_state

        if self.batch_size:
            hlog(f"Processing requests in batches of {self.batch_size}...")
            batches = [
                scenario_state.request_states[i : i + self.batch_size]
                for i in range(0, len(scenario_state.request_states), self.batch_size)
            ]
            batch_results = parallel_map(
                self.process_batch,
                batches,
                parallelism=self.execution_spec.parallelism,
            )
            request_states = [state for batch in batch_results for state in batch]
        else:
            # Do it!
            request_states = parallel_map(
                self.process,
                scenario_state.request_states,
                parallelism=self.execution_spec.parallelism,
            )

        hlog(f"Processed {len(request_states)} requests")
        return ScenarioState(
            adapter_spec=scenario_state.adapter_spec,
            request_states=request_states,
            annotator_specs=scenario_state.annotator_specs,
        )

    def process(self, state: RequestState) -> RequestState:
        try:
            result: RequestResult = self.context.make_request(state.request)
        except Exception as e:
            raise ExecutorError(f"{str(e)} Request: {state.request}") from e
        if not result.success:
            if result.error_flags and not result.error_flags.is_fatal:
                hwarn(f"Non-fatal error treated as empty completion: {result.error}")
                result.completions = [GeneratedOutput(text="", logprob=0, tokens=[])]
            else:
                raise ExecutorError(f"{str(result.error)} Request: {state.request}")
        return replace(state, result=result)

    def process_batch(self, states: List[RequestState]) -> List[RequestState]:
        try:
            local_path = self.execution_spec.local_path + "/batches" if self.execution_spec.local_path else "./batches"
            os.makedirs(local_path, exist_ok=True)
            if not os.path.exists(local_path):
                hlog(f"Creating local path for batch requests: {local_path}")
            results: List[RequestResult] = self.context.make_batch_request(
                requests=[state.request for state in states],
                local_path=local_path,
            )
        except Exception as e:
            raise ExecutorError(f"{str(e)} Requests: {[state.request for state in states[:5]]}") from e
        if len(results) != len(states):
            raise ExecutorError(f"Batch request returned {len(results)} results but expected {len(states)}.")
        new_states = []
        for state, result in zip(states, results):
            if result is None or not result.success:
                if result.error_flags and not result.error_flags.is_fatal:
                    hwarn(f"Non-fatal error treated as empty completion: {result.error}")
                    result.completions = [GeneratedOutput(text="", logprob=0, tokens=[])]
                else:
                    raise ExecutorError(f"{str(result.error)} Request: {state.request}")
            new_states.append(replace(state, result=result))

        return new_states
