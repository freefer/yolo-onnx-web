import type { Sam3TokenizerTables } from './types';

const TOKEN_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+/gu;

function unescapeHtml(text: string): string {
  return text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, code) => String.fromCodePoint(parseInt(code, 16)));
}

function whitespaceClean(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function getPairs(word: readonly string[]): Set<string> {
  const pairs = new Set<string>();

  for (let index = 0; index < word.length - 1; index += 1) {
    pairs.add(`${word[index]} ${word[index + 1]}`);
  }

  return pairs;
}

function applyBpe(token: string, tables: Sam3TokenizerTables, cache: Map<string, string>): string {
  const cached = cache.get(token);

  if (cached !== undefined) {
    return cached;
  }

  if (token.length === 0) {
    return token;
  }

  let word = token.slice(0, -1).split('').concat(`${token.slice(-1)}</w>`);
  let pairs = getPairs(word);

  if (pairs.size === 0) {
    const result = `${token}</w>`;
    cache.set(token, result);
    return result;
  }

  while (true) {
    let minRank = Number.POSITIVE_INFINITY;
    let bigram: string | null = null;

    for (const pair of pairs) {
      const rank = tables.bpe_ranks[pair];

      if (rank !== undefined && rank < minRank) {
        minRank = rank;
        bigram = pair;
      }
    }

    if (bigram === null) {
      break;
    }

    const [first, second] = bigram.split(' ');
    const nextWord: string[] = [];
    let index = 0;

    while (index < word.length) {
      const found = word.indexOf(first, index);

      if (found < 0) {
        nextWord.push(...word.slice(index));
        break;
      }

      nextWord.push(...word.slice(index, found));
      index = found;

      if (word[index] === first && index < word.length - 1 && word[index + 1] === second) {
        nextWord.push(`${first}${second}`);
        index += 2;
      } else {
        nextWord.push(word[index] ?? '');
        index += 1;
      }
    }

    word = nextWord;

    if (word.length === 1) {
      break;
    }

    pairs = getPairs(word);
  }

  const result = word.join(' ');
  cache.set(token, result);
  return result;
}

export class ClipBpeTokenizer {
  private readonly tables: Sam3TokenizerTables;
  private readonly cache = new Map<string, string>();
  private readonly utf8 = new TextEncoder();

  constructor(tables: Sam3TokenizerTables) {
    this.tables = tables;
    this.cache.set('<start_of_text>', '<start_of_text>');
    this.cache.set('<end_of_text>', '<end_of_text>');
  }

  get contextLength(): number {
    return this.tables.context_length;
  }

  encode(text: string): number[] {
    const cleaned = whitespaceClean(unescapeHtml(unescapeHtml(text))).toLowerCase();
    const tokens: number[] = [];

    for (const match of cleaned.matchAll(TOKEN_PATTERN)) {
      const mapped = Array.from(this.utf8.encode(match[0] ?? ''), byte => this.tables.byte_encoder[String(byte)] ?? '')
        .join('');

      for (const bpeToken of applyBpe(mapped, this.tables, this.cache).split(' ')) {
        const id = this.tables.encoder[bpeToken];

        if (id !== undefined) {
          tokens.push(id);
        }
      }
    }

    return tokens;
  }

  tokenize(text: string): Int32Array {
    const contextLength = this.tables.context_length;
    const ids = new Int32Array(contextLength);
    const tokens = [this.tables.sot_token_id, ...this.encode(text), this.tables.eot_token_id];

    if (tokens.length > contextLength) {
      tokens.length = contextLength;
      tokens[contextLength - 1] = this.tables.eot_token_id;
    }

    ids.set(tokens);
    return ids;
  }
}

export async function loadClipTokenizer(source: string | Sam3TokenizerTables): Promise<ClipBpeTokenizer> {
  if (typeof source !== 'string') {
    return new ClipBpeTokenizer(source);
  }

  const response = await fetch(source);

  if (!response.ok) {
    throw new Error(`Failed to load SAM3 tokenizer tables from ${source}`);
  }

  return new ClipBpeTokenizer((await response.json()) as Sam3TokenizerTables);
}
