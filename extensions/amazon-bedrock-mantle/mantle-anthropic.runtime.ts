import Anthropic from "@anthropic-ai/sdk";
import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { Api, Model, SimpleStreamOptions } from "@mariozechner/pi-ai";
import { streamAnthropic } from "@mariozechner/pi-ai/anthropic";

const MANTLE_ANTHROPIC_BETA = "fine-grained-tool-streaming-2025-05-14";
type AnthropicOptions = ConstructorParameters<typeof Anthropic>[0];
type AnthropicStreamOptions = NonNullable<Parameters<typeof streamAnthropic>[2]>;
type AnthropicStreamClient = NonNullable<AnthropicStreamOptions["client"]>;

export function resolveMantleAnthropicBaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  if (trimmed.endsWith("/anthropic")) {
    return trimmed;
  }
  if (trimmed.endsWith("/v1")) {
    return `${trimmed.slice(0, -"/v1".length)}/anthropic`;
  }
  return `${trimmed}/anthropic`;
}

function requiresDefaultSampling(modelId: string): boolean {
  return modelId.includes("claude-opus-4-7");
}

/**
 * Models that use Anthropic's adaptive-thinking path (thinking.type='adaptive'
 * + output_config.effort) instead of the legacy budget_tokens mode.
 *
 * AWS Bedrock rejects the legacy budget_tokens form for Opus 4.7 with a 400
 * ValidationException, so we must emit the adaptive payload for any of these
 * model IDs. Matches pi-ai's own whitelist in streamSimpleAnthropic.
 */
function supportsAdaptiveThinking(modelId: string): boolean {
  return (
    modelId.includes("opus-4-7") ||
    modelId.includes("opus-4.7") ||
    modelId.includes("opus-4-6") ||
    modelId.includes("opus-4.6") ||
    modelId.includes("sonnet-4-6") ||
    modelId.includes("sonnet-4.6")
  );
}

/**
 * Map pi-ai ThinkingLevel to Anthropic adaptive-thinking effort value.
 * Mirrors pi-ai's mapThinkingLevelToEffort so Mantle stays in sync with the
 * direct Anthropic path:
 *   - xhigh -> "xhigh" on Opus 4.7, "max" on Opus 4.6, else "high"
 *   - high -> "high", medium -> "medium", low/minimal -> "low"
 */
function mapThinkingLevelToEffort(
  level: NonNullable<SimpleStreamOptions["reasoning"]>,
  modelId: string,
): "low" | "medium" | "high" | "xhigh" | "max" {
  switch (level) {
    case "minimal":
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
      return "high";
    case "xhigh":
      if (modelId.includes("opus-4-6") || modelId.includes("opus-4.6")) {
        return "max";
      }
      if (modelId.includes("opus-4-7") || modelId.includes("opus-4.7")) {
        return "xhigh";
      }
      return "high";
    default:
      return "high";
  }
}

function mergeHeaders(
  ...headerSources: Array<Record<string, string> | undefined>
): Record<string, string> {
  const merged: Record<string, string> = {};
  for (const headers of headerSources) {
    if (headers) {
      Object.assign(merged, headers);
    }
  }
  return merged;
}

function buildMantleAnthropicBaseOptions(
  model: Model<Api>,
  options: SimpleStreamOptions | undefined,
  apiKey: string,
) {
  return {
    temperature: requiresDefaultSampling(model.id) ? undefined : options?.temperature,
    maxTokens: options?.maxTokens || Math.min(model.maxTokens, 32_000),
    signal: options?.signal,
    apiKey,
    cacheRetention: options?.cacheRetention,
    sessionId: options?.sessionId,
    onPayload: options?.onPayload,
    maxRetryDelayMs: options?.maxRetryDelayMs,
    metadata: options?.metadata,
  };
}

function adjustMaxTokensForThinking(
  baseMaxTokens: number,
  modelMaxTokens: number,
  reasoningLevel: NonNullable<SimpleStreamOptions["reasoning"]>,
  customBudgets?: SimpleStreamOptions["thinkingBudgets"],
): { maxTokens: number; thinkingBudget: number } {
  const defaultBudgets = {
    minimal: 1024,
    low: 2048,
    medium: 8192,
    high: 16384,
    xhigh: 16384,
  } as const;
  const budgets = { ...defaultBudgets, ...customBudgets };
  const minOutputTokens = 1024;
  let thinkingBudget = budgets[reasoningLevel];
  const maxTokens = Math.min(baseMaxTokens + thinkingBudget, modelMaxTokens);
  if (maxTokens <= thinkingBudget) {
    thinkingBudget = Math.max(0, maxTokens - minOutputTokens);
  }
  return { maxTokens, thinkingBudget };
}

export function createMantleAnthropicStreamFn(deps?: {
  createClient?: (options: AnthropicOptions) => Anthropic;
  stream?: typeof streamAnthropic;
}): StreamFn {
  return (model, context, options) => {
    const apiKey = options?.apiKey ?? "";
    const createClient = deps?.createClient ?? ((clientOptions) => new Anthropic(clientOptions));
    const stream = deps?.stream ?? streamAnthropic;
    const client = createClient({
      apiKey: null,
      authToken: apiKey,
      baseURL: resolveMantleAnthropicBaseUrl(model.baseUrl),
      dangerouslyAllowBrowser: true,
      defaultHeaders: mergeHeaders(
        {
          accept: "application/json",
          "anthropic-dangerous-direct-browser-access": "true",
          "anthropic-beta": MANTLE_ANTHROPIC_BETA,
        },
        model.headers,
        options?.headers,
      ),
    });
    const base = buildMantleAnthropicBaseOptions(model, options, apiKey);
    // Staged plugin runtime deps can give this plugin a distinct physical SDK copy.
    // The client API is the same, but the SDK class private field makes types nominal.
    const streamClient = client as unknown as AnthropicStreamClient;

    // No reasoning requested: disable thinking. Temperature is already cleared
    // by buildMantleAnthropicBaseOptions for models that require default sampling.
    if (!options?.reasoning) {
      return stream(model as Model<"anthropic-messages">, context, {
        ...base,
        client: streamClient,
        thinkingEnabled: false,
      });
    }

    // Adaptive-thinking models (Opus 4.7 / 4.6, Sonnet 4.6) must use
    // thinking.type='adaptive' + output_config.effort. Bedrock rejects the
    // legacy budget_tokens form on these models with 400 ValidationException.
    if (supportsAdaptiveThinking(model.id)) {
      const effort = mapThinkingLevelToEffort(options.reasoning, model.id);
      return stream(model as Model<"anthropic-messages">, context, {
        ...base,
        client: streamClient,
        thinkingEnabled: true,
        effort,
      });
    }

    // Legacy models: keep budget_tokens-based thinking.
    const adjusted = adjustMaxTokensForThinking(
      base.maxTokens || 0,
      model.maxTokens,
      options.reasoning,
      options.thinkingBudgets,
    );
    return stream(model as Model<"anthropic-messages">, context, {
      ...base,
      client: streamClient,
      maxTokens: adjusted.maxTokens,
      thinkingEnabled: true,
      thinkingBudgetTokens: adjusted.thinkingBudget,
    });
  };
}
