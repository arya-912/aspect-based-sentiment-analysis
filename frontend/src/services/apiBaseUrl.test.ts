/**
 * Resolution of {@link BASE_URL}.
 *
 * `BASE_URL` is computed once when the module is first evaluated, so every
 * case here resets the module registry and re-imports rather than calling a
 * function: that is the behaviour the deployed bundle actually has.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const OVERRIDE_KEY = 'absa.apiUrl'

/** The default when nothing else supplies an origin. */
const LOCAL_DEFAULT = 'http://localhost:8000'

/**
 * Evaluate `api.ts` fresh with a given query string, and report the
 * `BASE_URL` it settles on.
 *
 * Storage is deliberately *not* cleared here -- `beforeEach` handles
 * isolation -- so a test can call this twice and observe what the first load
 * left behind. Pass `stored` to seed a value first.
 */
async function resolve(
  search = '',
  stored?: string | null,
): Promise<string> {
  vi.resetModules()
  if (stored !== undefined && stored !== null) {
    window.localStorage.setItem(OVERRIDE_KEY, stored)
  }
  window.history.replaceState({}, '', `/${search}`)
  const { BASE_URL } = await import('./api')
  return BASE_URL
}

beforeEach(() => {
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
})

describe('BASE_URL', () => {
  it('falls back to the local uvicorn port when nothing overrides it', async () => {
    // VITE_API_URL is unset under vitest, so this exercises the last resort.
    await expect(resolve()).resolves.toBe(LOCAL_DEFAULT)
  })

  it('takes the origin from ?api=', async () => {
    await expect(resolve('?api=https://tunnel.trycloudflare.com')).resolves.toBe(
      'https://tunnel.trycloudflare.com',
    )
  })

  it('remembers ?api= so a reload without it keeps working', async () => {
    await resolve('?api=https://tunnel.trycloudflare.com')
    expect(window.localStorage.getItem(OVERRIDE_KEY)).toBe(
      'https://tunnel.trycloudflare.com',
    )
    // Second load, parameter gone: the remembered value still applies.
    await expect(resolve('')).resolves.toBe('https://tunnel.trycloudflare.com')
  })

  it('uses a previously stored override when no parameter is present', async () => {
    await expect(resolve('', 'https://stored.example')).resolves.toBe(
      'https://stored.example',
    )
  })

  it('an empty ?api= forgets the stored override', async () => {
    await expect(resolve('?api=', 'https://stored.example')).resolves.toBe(
      LOCAL_DEFAULT,
    )
    expect(window.localStorage.getItem(OVERRIDE_KEY)).toBeNull()
  })

  it('strips trailing slashes, which would double up against the path prefix', async () => {
    await expect(resolve('?api=https://tunnel.example//')).resolves.toBe(
      'https://tunnel.example',
    )
  })

  it.each([
    ['not-a-url', 'not-a-url'],
    ['a non-http scheme', 'javascript:alert(1)'],
    ['a bare host', 'tunnel.example'],
  ])('ignores %s rather than breaking the page', async (_label, value) => {
    await expect(resolve(`?api=${encodeURIComponent(value)}`)).resolves.toBe(
      LOCAL_DEFAULT,
    )
  })

  it('ignores a stored value that is not an http(s) URL', async () => {
    await expect(resolve('', 'javascript:alert(1)')).resolves.toBe(LOCAL_DEFAULT)
  })
})
