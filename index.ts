/**
 * pi-kilo — Kilo Code Gateway provider for pi, rebuilt for fast startup.
 *
 * What it does:
 *   - Registers the `kilo` provider against Kilo's OpenRouter-compatible
 *     gateway (chat completions + Responses API routing per model).
 *   - Caches the model catalog on disk, so startup never blocks on the
 *     network after the first run; stale entries revalidate in background.
 *   - Registers only the models you actually need by default: the free tier
 *     plus models you have used recently (sticky). Registering 367 catalog
 *     models cost ~580ms of every launch; 12 cost ~20ms.
 *   - Full catalog stays reachable: `KILO_MODELS=all` restores it, and
 *     `KILO_MODELS=pattern1,pattern2` (exact ids or globs) adds more.
 *   - OAuth device-code login via `/login kilo`, with organization selection.
 *     Credentials live in the same auth.json entry ("kilo") the official
 *     provider uses, so an existing login carries over.
 *
 * Environment:
 *   KILO_API_KEY        API key / token (else the OAuth credential is used)
 *   KILO_ORG_ID         Organization id (else the one stored at login)
 *   KILO_API_URL        API base (default https://api.kilo.ai)
 *   KILO_MODELS         "all", or comma-separated ids/globs to register
 *   PI_KILO_DEBUG=1     trace logging to stderr
 */

import type { ExtensionAPI, ProviderConfig, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type { Api, Context, Model, OAuthCredentials, OAuthLoginCallbacks, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { normalizeContext } from "@earendil-works/pi-ai";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// =============================================================================
// Constants
// =============================================================================

const KILO_API_BASE = process.env.KILO_API_URL || "https://api.kilo.ai";
const KILO_GATEWAY_BASE = `${KILO_API_BASE}/api/gateway`;
const KILO_OPENROUTER_BASE = `${KILO_API_BASE}/api/openrouter`;
const KILO_PROFILE_ENDPOINT = `${KILO_API_BASE}/api/profile`;
const KILO_DEVICE_AUTH_ENDPOINT = `${KILO_API_BASE}/api/device-auth/codes`;
const KILO_ORG_HEADER = "X-KiloCode-OrganizationId";
const KILO_TOS_URL = "https://kilo.ai/terms";
const MODELS_FETCH_TIMEOUT_MS = 10_000;
const POLL_INTERVAL_MS = 3000;
const TOKEN_EXPIRATION_MS = 365 * 24 * 60 * 60 * 1000; // 1 year

// Reuse the catalog cache of the original provider so the first launch of
// pi-kilo does not refetch anything.
const CACHE_DIR = join(getAgentDir(), "cache", "kilo-pi-provider");
const MODELS_CACHE_MAX_AGE_MS = 6 * 60 * 60 * 1000;
const STICKY_FILE = join(CACHE_DIR, "used-models.json");
const STICKY_MAX = 20;

function getAgentDir(): string {
	return process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
}

function debugLog(event: string, data: Record<string, unknown> = {}): void {
	if (process.env.PI_KILO_DEBUG !== "1") return;
	try {
		console.error("[pi-kilo] " + JSON.stringify({ event, ...data }));
	} catch {
		// Debug logging must never break the provider.
	}
}

// =============================================================================
// Access (stored OAuth credential or KILO_API_KEY)
// =============================================================================

interface KiloAccess {
	token: string;
	organizationId?: string;
}

function getEnvOrganizationId(): string | undefined {
	return process.env.KILO_ORG_ID || process.env.KILOCODE_ORGANIZATION_ID;
}

function readStoredKiloCredentials(): OAuthCredentials | undefined {
	try {
		const authPath = join(getAgentDir(), "auth.json");
		if (!existsSync(authPath)) return undefined;
		const auth = JSON.parse(readFileSync(authPath, "utf8")) as {
			kilo?: { type?: string } & OAuthCredentials;
		};
		const cred = auth.kilo;
		if (cred?.type !== "oauth" || !cred.access) return undefined;
		return cred;
	} catch {
		return undefined;
	}
}

function getEffectiveOrganizationId(credentials?: OAuthCredentials): string | undefined {
	const accountId = credentials?.accountId;
	return typeof accountId === "string" && accountId.trim() ? accountId : getEnvOrganizationId();
}

function getKiloAccess(): KiloAccess | undefined {
	const credentials = readStoredKiloCredentials();
	const token = credentials?.access ?? process.env.KILO_API_KEY;
	if (!token) return undefined;
	return { token, organizationId: getEffectiveOrganizationId(credentials) };
}

// =============================================================================
// Device-code OAuth login (same flow and credential shape as the official provider)
// =============================================================================

interface DeviceAuthResponse {
	code: string;
	verificationUrl: string;
	expiresIn: number;
}

interface DeviceAuthPollResponse {
	status: "pending" | "approved" | "denied" | "expired";
	token?: string;
	userEmail?: string;
}

interface KiloOrganization {
	id: string;
	name: string;
	role?: string;
}

interface KiloProfile {
	user?: { email?: string; name?: string };
	organizations?: KiloOrganization[];
}

function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(new Error("Login cancelled"));
			return;
		}
		const timeout = setTimeout(resolve, ms);
		signal?.addEventListener("abort", () => {
			clearTimeout(timeout);
			reject(new Error("Login cancelled"));
		}, { once: true });
	});
}

async function initiateDeviceAuth(): Promise<DeviceAuthResponse> {
	const response = await fetch(KILO_DEVICE_AUTH_ENDPOINT, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
	});
	if (response.status === 429) {
		throw new Error("Too many pending authorization requests. Please try again later.");
	}
	if (!response.ok) {
		throw new Error(`Failed to initiate device authorization: ${response.status}`);
	}
	return (await response.json()) as DeviceAuthResponse;
}

async function pollDeviceAuth(code: string): Promise<DeviceAuthPollResponse> {
	const response = await fetch(`${KILO_DEVICE_AUTH_ENDPOINT}/${code}`);
	if (response.status === 202) return { status: "pending" };
	if (response.status === 403) return { status: "denied" };
	if (response.status === 410) return { status: "expired" };
	if (!response.ok) {
		throw new Error(`Failed to poll device authorization: ${response.status}`);
	}
	return (await response.json()) as DeviceAuthPollResponse;
}

async function fetchKiloProfile(token: string): Promise<KiloProfile> {
	const response = await fetch(KILO_PROFILE_ENDPOINT, {
		headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
	});
	if (!response.ok) {
		throw new Error(`Failed to fetch Kilo profile: ${response.status}`);
	}
	return (await response.json()) as KiloProfile;
}

async function selectKiloOrganization(
	token: string,
	callbacks: Partial<Pick<OAuthLoginCallbacks, "onProgress" | "onSelect">>,
): Promise<string | undefined> {
	let profile: KiloProfile;
	try {
		callbacks.onProgress?.("Fetching Kilo profile...");
		profile = await fetchKiloProfile(token);
	} catch (error) {
		console.warn("[pi-kilo] Failed to fetch profile for organization selection:", error instanceof Error ? error.message : error);
		return getEnvOrganizationId();
	}

	const organizations = profile.organizations ?? [];
	const envOrganizationId = getEnvOrganizationId();
	if (envOrganizationId && organizations.some((org) => org.id === envOrganizationId)) {
		return envOrganizationId;
	}
	if (!callbacks.onSelect || organizations.length === 0) {
		return envOrganizationId;
	}

	const selected = await callbacks.onSelect({
		message: "Select Kilo account",
		options: [
			{ id: "personal", label: "Personal Account" },
			...organizations.map((org) => ({ id: org.id, label: `${org.name}${org.role ? ` (${org.role})` : ""}` })),
		],
	});

	if (!selected || selected === "personal") return undefined;
	return selected;
}

async function loginKilo(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
	callbacks.onProgress?.("Initiating device authorization...");
	const { code, verificationUrl, expiresIn } = await initiateDeviceAuth();

	callbacks.onAuth({
		url: verificationUrl,
		instructions: `Enter code: ${code}`,
	});
	callbacks.onProgress?.("Waiting for browser authorization...");

	const deadline = Date.now() + expiresIn * 1000;
	while (Date.now() < deadline) {
		if (callbacks.signal?.aborted) {
			throw new Error("Login cancelled");
		}
		await abortableSleep(POLL_INTERVAL_MS, callbacks.signal);
		const result = await pollDeviceAuth(code);

		if (result.status === "approved") {
			if (!result.token) {
				throw new Error("Authorization approved but no token received");
			}
			callbacks.onProgress?.("Login successful!");
			const organizationId = await selectKiloOrganization(result.token, callbacks);
			return {
				refresh: result.token,
				access: result.token,
				expires: Date.now() + TOKEN_EXPIRATION_MS,
				...(organizationId ? { accountId: organizationId } : {}),
			};
		}
		if (result.status === "denied") {
			throw new Error("Authorization denied by user.");
		}
		if (result.status === "expired") {
			throw new Error("Authorization code expired. Please try again.");
		}
		const remaining = Math.ceil((deadline - Date.now()) / 1000);
		callbacks.onProgress?.(`Waiting for browser authorization... (${remaining}s remaining)`);
	}

	throw new Error("Authentication timed out. Please try again.");
}

function refreshKiloToken(credentials: OAuthCredentials): OAuthCredentials {
	if (credentials.expires > Date.now()) {
		return credentials;
	}
	throw new Error("Kilo token expired. Please run /login kilo to re-authenticate.");
}

// =============================================================================
// Catalog fetch
// =============================================================================

interface OpenRouterModel {
	id: string;
	name: string;
	context_length: number;
	max_completion_tokens?: number | null;
	pricing?: {
		prompt?: string | null;
		completion?: string | null;
		input_cache_write?: string | null;
		input_cache_read?: string | null;
	};
	architecture?: {
		input_modalities?: string[] | null;
		output_modalities?: string[] | null;
	};
	top_provider?: { max_completion_tokens?: number | null };
	supported_parameters?: string[];
	opencode?: {
		variants?: Record<string, { reasoning?: { enabled?: boolean; effort?: string }; verbosity?: string }>;
		ai_sdk_provider?: string;
	};
}

async function fetchKiloModelCatalog(request: { token?: string; organizationId?: string } = {}): Promise<OpenRouterModel[]> {
	const headers: Record<string, string> = {
		"Content-Type": "application/json",
		"User-Agent": "pi-kilo",
	};
	if (request.token) {
		headers.Authorization = `Bearer ${request.token}`;
	}
	const modelsUrl = request.organizationId
		? `${KILO_API_BASE}/api/organizations/${encodeURIComponent(request.organizationId)}/models`
		: `${KILO_GATEWAY_BASE}/models`;

	const response = await fetch(modelsUrl, {
		headers: request.organizationId ? { ...headers, [KILO_ORG_HEADER]: request.organizationId } : headers,
		signal: AbortSignal.timeout(MODELS_FETCH_TIMEOUT_MS),
	});
	if (!response.ok) {
		throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
	}
	const json = (await response.json()) as { data?: OpenRouterModel[] };
	if (!json.data || !Array.isArray(json.data)) {
		throw new Error("Invalid models response: missing data array");
	}
	return json.data;
}

// =============================================================================
// Model mapping (OpenRouter catalog entry -> pi model config)
// =============================================================================

type KiloChatModelConfig = Extract<ProviderModelConfig, { type?: "chat" }>;

function parsePrice(price: string | null | undefined): number {
	if (!price) return 0;
	const parsed = parseFloat(price);
	if (Number.isNaN(parsed)) return 0;
	// OpenRouter prices are per-token; pi expects per-million-token.
	return parsed * 1_000_000;
}

function isFreeModelId(id: string): boolean {
	if (id === "kilo-auto/free") return true;
	if (id.includes(":free")) return true;
	if (!id.includes("/")) return true;
	if (id.startsWith("kilo/") || id.startsWith("openrouter/")) return true;
	return false;
}

function isFreeModel(m: KiloChatModelConfig): boolean {
	return m.cost.input === 0 && m.cost.output === 0 && isFreeModelId(m.id);
}

function shouldUseResponsesApi(m: OpenRouterModel): boolean {
	if (m.opencode?.ai_sdk_provider === "openai") return true;
	// Some catalog entries arrive before ai_sdk_provider is populated. Kilo
	// routes current OpenAI reasoning/frontier models through the Responses
	// API; chat completions for these yield: "please use any of: responses".
	const id = m.id.toLowerCase();
	const shortId = id.includes("/") ? (id.split("/").pop() ?? id) : id;
	return (
		shortId === "gpt-5" ||
		shortId.startsWith("gpt-5.") ||
		shortId.startsWith("gpt-5-") ||
		shortId.startsWith("o1") ||
		shortId.startsWith("o3") ||
		shortId.startsWith("o4")
	);
}

type PiThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

type KiloModelCompat = {
	thinkingFormat?: "openrouter";
	cacheControlFormat?: "anthropic";
	requiresReasoningContentOnAssistantMessages?: boolean;
	supportsStore?: boolean;
};

function getKiloModelCompat(m: OpenRouterModel, api: Api | undefined): KiloChatModelConfig["compat"] {
	if (api === "openai-responses") {
		return {
			// Kilo/OpenRouter-compatible responses endpoints do not need OpenAI's
			// session_id header, and long prompt-cache retention is provider-specific.
			sessionAffinityFormat: "openai-nosession",
			supportsLongCacheRetention: false,
		};
	}
	const compat: KiloModelCompat = {
		// Kilo's gateway is OpenRouter-compatible but sits on api.kilo.ai, so
		// pi-ai's URL/provider auto-detection cannot infer OpenRouter quirks.
		thinkingFormat: "openrouter",
		supportsStore: false,
	};
	if (m.id.startsWith("anthropic/")) {
		compat.cacheControlFormat = "anthropic";
	}
	if (m.id === "deepseek/deepseek-v4-flash" || m.id === "deepseek/deepseek-v4-pro") {
		compat.requiresReasoningContentOnAssistantMessages = true;
	}
	return compat;
}

function mapVariantEffort(
	variants: NonNullable<OpenRouterModel["opencode"]>["variants"],
	key: string,
): string | undefined {
	const variant = variants?.[key];
	if (!variant) return undefined;
	const reasoning = variant.reasoning;
	if (!reasoning) return key;
	if (reasoning.enabled === false || reasoning.effort === "none") return "none";
	return reasoning.effort ?? key;
}

function thinkingLevelMapFromVariants(
	variants: NonNullable<OpenRouterModel["opencode"]>["variants"],
): KiloChatModelConfig["thinkingLevelMap"] | undefined {
	if (!variants || Object.keys(variants).length === 0) return undefined;

	const map: Partial<Record<PiThinkingLevel, string | null>> = {};
	const off = mapVariantEffort(variants, "none") ?? mapVariantEffort(variants, "instant");
	if (off !== undefined) map.off = off;

	for (const level of ["minimal", "low", "medium", "high", "xhigh", "max"] as const) {
		const effort = mapVariantEffort(variants, level);
		map[level] = effort === undefined ? null : effort;
	}

	return map;
}

function getKiloThinkingLevelMap(m: OpenRouterModel): KiloChatModelConfig["thinkingLevelMap"] | undefined {
	const fromVariants = thinkingLevelMapFromVariants(m.opencode?.variants);
	if (/^deepseek\/deepseek-v4-(flash|pro)(?:-|$)/.test(m.id)) {
		return { ...fromVariants, max: "max" };
	}
	if (fromVariants) return fromVariants;
	// Safety net for current frontier OpenAI models while metadata catches up.
	if (m.id.includes("gpt-5.5")) {
		return { off: "none", minimal: null, low: "low", medium: "medium", high: "high", xhigh: "xhigh" };
	}
	return undefined;
}

function mapOpenRouterModel(m: OpenRouterModel): KiloChatModelConfig {
	const inputModalities = m.architecture?.input_modalities ?? ["text"];
	const supportsImages = inputModalities.includes("image");
	const supportsReasoning = m.supported_parameters?.includes("reasoning") ?? false;
	const maxTokens =
		m.top_provider?.max_completion_tokens ?? m.max_completion_tokens ?? Math.ceil(m.context_length * 0.2);
	const api = shouldUseResponsesApi(m) ? ("openai-responses" as const) : ("openai-completions" as const);

	return {
		id: m.id,
		name: m.name,
		api,
		...(api === "openai-responses" ? { baseUrl: KILO_OPENROUTER_BASE } : {}),
		reasoning: supportsReasoning,
		input: supportsImages ? ["text", "image"] : ["text"],
		cost: {
			input: parsePrice(m.pricing?.prompt),
			output: parsePrice(m.pricing?.completion),
			cacheRead: parsePrice(m.pricing?.input_cache_read),
			cacheWrite: parsePrice(m.pricing?.input_cache_write),
		},
		contextWindow: m.context_length,
		maxTokens: maxTokens,
		thinkingLevelMap: getKiloThinkingLevelMap(m),
		compat: getKiloModelCompat(m, api),
	};
}

/** Map the raw catalog and drop image-generation models. */
function mapCatalog(raw: OpenRouterModel[]): KiloChatModelConfig[] {
	return raw
		.filter((m) => !(m.architecture?.output_modalities ?? []).includes("image"))
		.map(mapOpenRouterModel);
}

// =============================================================================
// Catalog disk cache (same directory and format as the original provider)
// =============================================================================

interface ModelsCacheEntry {
	fetchedAt: number;
	models: KiloChatModelConfig[];
}

function modelsCachePath(key: string): string {
	return join(CACHE_DIR, `models-${key}.json`);
}

function readModelsCache(key: string): ModelsCacheEntry | null {
	try {
		const raw = JSON.parse(readFileSync(modelsCachePath(key), "utf-8")) as ModelsCacheEntry;
		if (!Array.isArray(raw.models) || raw.models.length === 0) return null;
		if (typeof raw.fetchedAt !== "number") return null;
		return raw;
	} catch {
		return null;
	}
}

function writeModelsCache(key: string, models: KiloChatModelConfig[]): void {
	try {
		mkdirSync(CACHE_DIR, { recursive: true });
		const entry: ModelsCacheEntry = { fetchedAt: Date.now(), models };
		writeFileSync(modelsCachePath(key), JSON.stringify(entry));
	} catch {
		// Cache writes must never break the provider.
	}
}

function refreshModelsCacheInBackground(key: string, request: { token?: string; organizationId?: string }): void {
	void fetchKiloModelCatalog(request)
		.then((raw) => {
			const mapped = mapCatalog(raw);
			if (mapped.length > 0) writeModelsCache(key, mapped);
		})
		.catch(() => {
			// Stale cache keeps serving; the next launch retries.
		});
}

// =============================================================================
// Which models get registered
// =============================================================================

/** Recently used kilo models, so a model you picked once stays registered. */
function readStickyModels(): string[] {
	try {
		const raw = JSON.parse(readFileSync(STICKY_FILE, "utf-8")) as { models?: string[] };
		return Array.isArray(raw.models) ? raw.models.filter((id) => typeof id === "string") : [];
	} catch {
		return [];
	}
}

function rememberStickyModel(id: string): void {
	try {
		const models = readStickyModels().filter((m) => m !== id);
		models.unshift(id);
		mkdirSync(CACHE_DIR, { recursive: true });
		writeFileSync(STICKY_FILE, JSON.stringify({ models: models.slice(0, STICKY_MAX) }));
	} catch {
		// Sticky persistence must never break the provider.
	}
}

function globToRegex(pattern: string): RegExp {
	const escaped = pattern
		.trim()
		.replace(/[.+?^${}()|[\]\\]/g, "\\$&")
		.replace(/\*/g, ".*");
	return new RegExp(`^${escaped}$`, "i");
}

/** Apply the registration filter: free models + sticky used + KILO_MODELS. */
function selectRegisteredModels(catalog: KiloChatModelConfig[]): KiloChatModelConfig[] {
	const mode = (process.env.KILO_MODELS ?? "").trim();
	if (mode.toLowerCase() === "all") return catalog;

	const wanted = new Set<string>();
	for (const m of catalog) {
		if (isFreeModel(m)) wanted.add(m.id);
	}
	for (const id of readStickyModels()) {
		if (catalog.some((m) => m.id === id)) wanted.add(id);
	}
	if (mode) {
		for (const pattern of mode.split(",").map((p) => p.trim()).filter(Boolean)) {
			const re = globToRegex(pattern);
			for (const m of catalog) {
				if (re.test(m.id)) wanted.add(m.id);
			}
		}
	}
	const picked = catalog.filter((m) => wanted.has(m.id));
	debugLog("registered-models", { count: picked.length, total: catalog.length, mode: mode || "default" });
	return picked;
}

// =============================================================================
// Responses streaming (Kilo gateway quirk normalization)
// =============================================================================

const openAIResponses = openAIResponsesApi();

type JsonObject = Record<string, unknown>;

function isJsonObject(value: unknown): value is JsonObject {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Kilo may emit response.failed with error.type but no error.code, which the
 *  OpenAI Responses parser requires. Supply the field, keep the failure. */
function normalizeResponseFailedEvent(event: unknown): unknown {
	if (!isJsonObject(event) || event.type !== "response.failed") return event;
	const response = event.response;
	if (!isJsonObject(response)) return event;
	const error = response.error;
	if (!isJsonObject(error) || (error.code !== undefined && error.code !== null)) return event;
	if (typeof error.type !== "string" || error.type.length === 0) return event;
	return { ...event, response: { ...response, error: { ...error, code: error.type } } };
}

function sseData(line: string): string | undefined {
	if (!line.startsWith("data:")) return undefined;
	const value = line.slice("data:".length);
	return value.startsWith(" ") ? value.slice(1) : value;
}

function normalizeSseEvent(eventText: string): string {
	if (!eventText.includes("response.failed")) return eventText;

	const lines = eventText.split(/\r\n|\r|\n/u);
	const payload = lines.flatMap((line) => {
		const data = sseData(line);
		return data === undefined ? [] : [data];
	});
	if (payload.length === 0) return eventText;

	try {
		const event = JSON.parse(payload.join("\n")) as unknown;
		const normalized = normalizeResponseFailedEvent(event);
		if (normalized === event) return eventText;

		let wroteData = false;
		return lines
			.flatMap((line) => {
				if (sseData(line) === undefined) return [line];
				if (wroteData) return [];
				wroteData = true;
				return [`data: ${JSON.stringify(normalized)}`];
			})
			.join("\n");
	} catch {
		return eventText;
	}
}

function normalizeSseBody(body: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
	const decoder = new TextDecoder();
	const encoder = new TextEncoder();
	// Keep CRLF atomic so regex backtracking cannot mistake one line ending for a blank line.
	const eventSeparator = /(?:\r\n|\r(?!\n)|(?<!\r)\n)(?:\r\n|\r(?!\n)|(?<!\r)\n)/u;
	let buffered = "";

	function emitEvents(controller: TransformStreamDefaultController<Uint8Array>): void {
		for (;;) {
			const separator = eventSeparator.exec(buffered);
			if (!separator) return;
			const eventText = buffered.slice(0, separator.index);
			controller.enqueue(encoder.encode(`${normalizeSseEvent(eventText)}${separator[0]}`));
			buffered = buffered.slice(separator.index + separator[0].length);
		}
	}

	return body.pipeThrough(
		new TransformStream<Uint8Array, Uint8Array>({
			transform(chunk, controller) {
				buffered += decoder.decode(chunk, { stream: true });
				emitEvents(controller);
			},
			flush(controller) {
				buffered += decoder.decode();
				emitEvents(controller);
				if (buffered.length > 0) controller.enqueue(encoder.encode(buffered));
			},
		}),
	);
}

function normalizeResponsesFetch(fetchImplementation: typeof fetch): typeof fetch {
	return async (input, init) => {
		const response = await fetchImplementation(input, init);
		if (!response.body || !response.headers.get("content-type")?.toLowerCase().includes("text/event-stream")) {
			return response;
		}
		return new Response(normalizeSseBody(response.body), {
			status: response.status,
			statusText: response.statusText,
			headers: response.headers,
		});
	};
}

function streamKiloResponses(model: Model<Api>, context: Context, options?: SimpleStreamOptions) {
	return openAIResponses.streamSimple(model, normalizeContext(context), {
		...options,
		fetch: normalizeResponsesFetch(options?.fetch ?? globalThis.fetch),
	});
}

// =============================================================================
// Extension entry point
// =============================================================================

export default async function (pi: ExtensionAPI): Promise<void> {
	const access = getKiloAccess();
	let catalog: KiloChatModelConfig[] = [];

	// Load the catalog: cache first, one blocking fetch only when it is empty.
	if (access) {
		const key = `org-${access.organizationId || "default"}`;
		const hit = readModelsCache(key);
		if (hit) {
			catalog = hit.models;
			if (Date.now() - hit.fetchedAt > MODELS_CACHE_MAX_AGE_MS) {
				refreshModelsCacheInBackground(key, { token: access.token, organizationId: access.organizationId });
			}
		} else {
			try {
				catalog = mapCatalog(
					await fetchKiloModelCatalog({ token: access.token, organizationId: access.organizationId }),
				);
				writeModelsCache(key, catalog);
			} catch (error) {
				console.warn("[pi-kilo] Failed to fetch models at startup:", error instanceof Error ? error.message : error);
			}
		}
	} else {
		const hit = readModelsCache("free");
		if (hit) {
			catalog = hit.models;
			if (Date.now() - hit.fetchedAt > MODELS_CACHE_MAX_AGE_MS) {
				refreshModelsCacheInBackground("free", {});
			}
		} else {
			try {
				catalog = mapCatalog(await fetchKiloModelCatalog({}));
				writeModelsCache("free", catalog);
			} catch (error) {
				console.warn("[pi-kilo] Failed to fetch free models at startup:", error instanceof Error ? error.message : error);
			}
		}
	}

	// Last resort: an empty catalog (fetch failed, no cache) still leaves the
	// provider usable by falling back to the free catalog.
	if (catalog.length === 0) {
		const freeHit = readModelsCache("free");
		if (freeHit) {
			catalog = freeHit.models;
		} else {
			try {
				catalog = mapCatalog(await fetchKiloModelCatalog({}));
				writeModelsCache("free", catalog);
			} catch {
				// Nothing more to try; the provider registers modelless and
				// modifyModels populates it after login.
			}
		}
	}

	const registeredModels = selectRegisteredModels(catalog);
	function makeProviderConfig(organizationId?: string): ProviderConfig {
		return {
			name: "Kilo",
			baseUrl: KILO_GATEWAY_BASE,
			apiKey: "$KILO_API_KEY",
			// pi applies streamSimple only to models whose API matches this one;
			// mapOpenRouterModel assigns the Responses API per model explicitly.
			api: "openai-responses",
			streamSimple: streamKiloResponses,
			headers: {
				"X-KILOCODE-EDITORNAME": "Pi",
				"User-Agent": "pi-kilo",
				...(organizationId ? { [KILO_ORG_HEADER]: organizationId } : {}),
			},
		};
	}

	// modifyModels upgrades the registered free set to the filtered full
	// catalog whenever credentials exist, and falls back after logout.
	function makeOAuthConfig() {
		return {
			name: "Kilo",
			login: async (callbacks: OAuthLoginCallbacks) => {
				const cred = await loginKilo(callbacks);
				// Cache the full catalog so modifyModels can use it during the
				// modelRegistry.refresh() that runs right after login returns.
				try {
					const organizationId = getEffectiveOrganizationId(cred);
					const raw = await fetchKiloModelCatalog({ token: cred.access, organizationId });
					catalog = mapCatalog(raw);
					writeModelsCache(`org-${organizationId || "default"}`, catalog);
				} catch (error) {
					console.warn("[pi-kilo] Failed to fetch models after login:", error instanceof Error ? error.message : error);
				}
				return cred;
			},
			refreshToken: refreshKiloToken,
			getApiKey: (cred: OAuthCredentials) => cred.access,
			modifyModels: (models: Model<Api>[], cred: OAuthCredentials) => {
				const organizationId = getEffectiveOrganizationId(cred);
				const filtered = selectRegisteredModels(catalog);
				if (filtered.length === 0) return models;
				const template = models.find((m) => m.provider === "kilo");
				if (!template) return models;
				const headers = organizationId ? { [KILO_ORG_HEADER]: organizationId } : undefined;
				const nonKilo = models.filter((m) => m.provider !== "kilo");
				const full = filtered.map((m) => ({
					...template,
					id: m.id,
					name: m.name,
					api: m.api ?? template.api,
					baseUrl: m.baseUrl ?? template.baseUrl,
					reasoning: m.reasoning,
					input: m.input,
					cost: m.cost,
					contextWindow: m.contextWindow,
					maxTokens: m.maxTokens,
					thinkingLevelMap: m.thinkingLevelMap,
					headers,
					compat: m.compat,
				}));
				return [...nonKilo, ...full];
			},
		};
	}

	pi.registerProvider("kilo", {
		...makeProviderConfig(getEnvOrganizationId()),
		models: registeredModels,
		oauth: makeOAuthConfig(),
	});

	// Keep recently used kilo models registered across sessions.
	pi.on("model_select", (event) => {
		if (event.model.provider === "kilo") rememberStickyModel(event.model.id);
	});

	// On first use of a Kilo model, surface the Terms of Service.
	let tosShown = false;
	pi.on("before_agent_start", (_event, ctx) => {
		if (tosShown) return;
		if (ctx.model?.provider !== "kilo") return;
		tosShown = true;
		if (getKiloAccess()) return;
		return {
			message: {
				customType: "kilo",
				content: `By using Kilo, you agree to the Terms of Service: ${KILO_TOS_URL}`,
				display: true,
			},
		};
	});

	debugLog("loaded", { registered: registeredModels.length, catalogSize: catalog.length, loggedIn: !!access });
}
