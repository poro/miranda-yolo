/**
 * SupabaseAuth — Authentication middleware for Express
 *
 * Provides:
 * - Supabase client initialization
 * - requireAuth middleware (blocks unauthenticated requests)
 * - optionalAuth middleware (attaches user if present, continues if not)
 * - Graceful fallback when Supabase is not configured (single-user mode)
 */

let createClient;
try {
    ({ createClient } = require('@supabase/supabase-js'));
} catch {
    // supabase-js not installed — single-user mode
}

class SupabaseAuth {
    constructor() {
        this.supabase = null;
        this.enabled = false;

        const url = process.env.SUPABASE_URL;
        const key = process.env.SUPABASE_ANON_KEY;

        if (url && key && createClient) {
            this.supabase = createClient(url, key);
            this.enabled = true;
            console.log('[Auth] Supabase enabled');
        } else {
            console.log('[Auth] Supabase not configured — single-user mode (no auth required)');
        }
    }

    /**
     * Express middleware: require authenticated user.
     * If Supabase is not configured, passes through (single-user mode).
     */
    requireAuth() {
        return async (req, res, next) => {
            if (!this.enabled) {
                req.user = null;
                return next();
            }

            const token = this._extractToken(req);
            if (!token) {
                return res.status(401).json({ error: 'Authentication required' });
            }

            try {
                const { data: { user }, error } = await this.supabase.auth.getUser(token);
                if (error || !user) {
                    return res.status(401).json({ error: 'Invalid or expired token' });
                }
                req.user = user;
                next();
            } catch (err) {
                return res.status(401).json({ error: 'Authentication failed: ' + err.message });
            }
        };
    }

    /**
     * Express middleware: attach user if token present, continue either way.
     * Useful for routes that work for both authenticated and anonymous users.
     */
    optionalAuth() {
        return async (req, res, next) => {
            req.user = null;

            if (!this.enabled) return next();

            const token = this._extractToken(req);
            if (!token) return next();

            try {
                const { data: { user }, error } = await this.supabase.auth.getUser(token);
                if (!error && user) req.user = user;
            } catch {
                // Ignore auth errors for optional auth
            }

            next();
        };
    }

    /**
     * Extract Bearer token from Authorization header
     */
    _extractToken(req) {
        const header = req.headers.authorization;
        if (!header) return null;
        if (header.startsWith('Bearer ')) return header.slice(7);
        return header;
    }

    /**
     * Get the Supabase client URL for frontend initialization
     */
    getConfig() {
        return {
            enabled: this.enabled,
            url: process.env.SUPABASE_URL || null,
            anonKey: process.env.SUPABASE_ANON_KEY || null
        };
    }
}

module.exports = SupabaseAuth;
