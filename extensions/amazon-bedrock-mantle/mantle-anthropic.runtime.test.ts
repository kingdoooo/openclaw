import type { Api, Model } from "@mariozechner/pi-ai";
import { describe, expect, it, vi } from "vitest";
import {
  createMantleAnthropicStreamFn,
  resolveMantleAnthropicBaseUrl,
} from "./mantle-anthropic.runtime.js";

function createTestModel(): Model<Api> {
  return {
    id: "anthropic.claude-opus-4-7",
    name: "Claude Opus 4.7",
    provider: "amazon-bedrock-mantle",
    api: "anthropic-messages",
    baseUrl: "https://bedrock-mantle.us-east-1.api.aws/v1",
    headers: {
      "X-Test": "model-header",
    },
    reasoning: false,
    input: ["text", "image"],
    cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
    contextWindow: 1_000_000,
    maxTokens: 128_000,
  } as Model<Api>;
}

function createModelWithId(id: string, name: string = id): Model<Api> {
  return {
    ...createTestModel(),
    id,
    name,
  } as Model<Api>;
}

function createTestDeps() {
  return {
    createClient: vi.fn((options: unknown) => ({ options }) as never),
    stream: vi.fn(),
  };
}

describe("createMantleAnthropicStreamFn", () => {
  it("uses authToken bearer auth for Mantle Anthropic requests", () => {
    const stream = { kind: "anthropic-stream" };
    const model = createTestModel();
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue(stream as never);

    const result = createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      headers: {
        "X-Caller": "caller-header",
      },
    });

    expect(result).toBe(stream);
    expect(deps.createClient).toHaveBeenCalledWith(
      expect.objectContaining({
        apiKey: null,
        authToken: "bedrock-bearer-token",
        baseURL: "https://bedrock-mantle.us-east-1.api.aws/anthropic",
        defaultHeaders: expect.objectContaining({
          accept: "application/json",
          "anthropic-beta": "fine-grained-tool-streaming-2025-05-14",
          "X-Test": "model-header",
          "X-Caller": "caller-header",
        }),
      }),
    );
    expect(deps.stream).toHaveBeenCalledWith(
      model,
      context,
      expect.objectContaining({
        client: expect.objectContaining({
          options: expect.objectContaining({
            authToken: "bedrock-bearer-token",
          }),
        }),
        thinkingEnabled: false,
      }),
    );
  });

  it("emits adaptive thinking + effort for Opus 4.7 with reasoning", () => {
    const model = createTestModel();
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue({ kind: "anthropic-stream" } as never);

    void createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      temperature: 0.2,
      reasoning: "high",
    });

    const opts = deps.stream.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(opts).toMatchObject({
      temperature: undefined,
      thinkingEnabled: true,
      effort: "high",
    });
    expect(opts).not.toHaveProperty("thinkingBudgetTokens");
  });

  it("maps Opus 4.7 xhigh reasoning to effort:'xhigh'", () => {
    const model = createTestModel();
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue({ kind: "anthropic-stream" } as never);

    void createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      reasoning: "xhigh",
    });

    const opts = deps.stream.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(opts).toMatchObject({
      thinkingEnabled: true,
      effort: "xhigh",
    });
    expect(opts).not.toHaveProperty("thinkingBudgetTokens");
  });

  it("maps Opus 4.6 xhigh reasoning to effort:'max'", () => {
    const model = createModelWithId("anthropic.claude-opus-4-6", "Claude Opus 4.6");
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue({ kind: "anthropic-stream" } as never);

    void createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      reasoning: "xhigh",
    });

    const opts = deps.stream.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(opts).toMatchObject({
      thinkingEnabled: true,
      effort: "max",
    });
    expect(opts).not.toHaveProperty("thinkingBudgetTokens");
  });

  it("maps Sonnet 4.6 medium reasoning to effort:'medium'", () => {
    const model = createModelWithId("anthropic.claude-sonnet-4-6", "Claude Sonnet 4.6");
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue({ kind: "anthropic-stream" } as never);

    void createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      reasoning: "medium",
    });

    const opts = deps.stream.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(opts).toMatchObject({
      thinkingEnabled: true,
      effort: "medium",
    });
    expect(opts).not.toHaveProperty("thinkingBudgetTokens");
  });

  it("maps Opus 4.7 minimal reasoning to effort:'low'", () => {
    const model = createTestModel();
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue({ kind: "anthropic-stream" } as never);

    void createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      reasoning: "minimal",
    });

    const opts = deps.stream.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(opts).toMatchObject({
      thinkingEnabled: true,
      effort: "low",
    });
  });

  it("keeps legacy Claude 3.5 Sonnet on budget_tokens thinking path", () => {
    const model = createModelWithId("anthropic.claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet");
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue({ kind: "anthropic-stream" } as never);

    void createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      reasoning: "high",
    });

    const opts = deps.stream.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(opts.thinkingEnabled).toBe(true);
    expect(opts.thinkingBudgetTokens).toBe(16384);
    expect(opts).not.toHaveProperty("effort");
    // maxTokens should grow to fit the thinking budget (base + budget).
    expect(opts.maxTokens).toBeGreaterThan(16384);
  });

  it("follows pi-ai includes() behavior: 'opus-4-70' substring also hits adaptive path", () => {
    // Document current behavior: supportsAdaptiveThinking uses
    // modelId.includes("opus-4-7") to mirror pi-ai's own whitelist. A
    // hypothetical "opus-4-70" id would therefore also be treated as
    // adaptive-capable. This is a known, intentional limitation kept in sync
    // with pi-ai; revisit only when pi-ai tightens its matcher.
    const model = createModelWithId("anthropic.claude-opus-4-70", "Hypothetical lookalike");
    const context = { messages: [] };
    const deps = createTestDeps();
    deps.stream.mockReturnValue({ kind: "anthropic-stream" } as never);

    void createMantleAnthropicStreamFn(deps)(model, context, {
      apiKey: "bedrock-bearer-token",
      reasoning: "xhigh",
    });

    const opts = deps.stream.mock.calls[0]?.[2] as Record<string, unknown>;
    expect(opts.thinkingEnabled).toBe(true);
    expect(opts.effort).toBe("xhigh");
  });

  it("normalizes Mantle provider URLs to the Anthropic endpoint", () => {
    expect(resolveMantleAnthropicBaseUrl("https://bedrock-mantle.us-east-1.api.aws/v1")).toBe(
      "https://bedrock-mantle.us-east-1.api.aws/anthropic",
    );
    expect(
      resolveMantleAnthropicBaseUrl("https://bedrock-mantle.us-east-1.api.aws/anthropic/"),
    ).toBe("https://bedrock-mantle.us-east-1.api.aws/anthropic");
  });
});
