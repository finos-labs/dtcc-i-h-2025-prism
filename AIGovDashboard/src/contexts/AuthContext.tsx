import React, { createContext, useContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import type { User } from '@supabase/supabase-js';

interface AuthContextType {
  user: User | null;
  signInWithGoogle: () => Promise<void>;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check active sessions and sets the user
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      // Store user ID in localStorage when retrieving the session
      if (session?.user) {
        localStorage.setItem('userId', session.user.id);
        console.log('User ID stored in localStorage:', session.user.id);
      } else {
        // Check if we have a direct database user
        const directUserId = localStorage.getItem('userId');
        const directUserEmail = localStorage.getItem('userEmail');
        
        if (directUserId && directUserEmail) {
          console.log('Found direct database user:', directUserId);
          // Create a pseudo-user object for the direct database user
          const pseudoUser = {
            id: directUserId,
            email: directUserEmail,
            user_metadata: {
              full_name: 'Direct User', // Will be replaced with actual data
            },
            app_metadata: {},
            aud: 'direct',
            created_at: new Date().toISOString(),
          } as unknown as User;
          
          setUser(pseudoUser);
          
          // Try to fetch additional user data from profiles
          supabase
            .from('user_profiles')
            .select('*')
            .eq('id', directUserId)
            .single()
            .then(({ data, error }) => {
              if (!error && data) {
                // Update pseudo-user with real data
                const updatedUser = {
                  ...pseudoUser,
                  user_metadata: {
                    full_name: data.full_name,
                    organization: data.organization,
                  }
                } as unknown as User;
                setUser(updatedUser);
              }
            });
        } else {
          localStorage.removeItem('userId');
        }
      }
      setLoading(false);
    });

    // Listen for changes on auth state
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      if (session?.user) {
        localStorage.setItem('userId', session.user.id);
        console.log('User ID updated in localStorage:', session.user.id);
      } else {
        // Don't remove userId here as it might be a direct database user
        // We'll handle that in checkDirectDatabaseUser
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  // Add a direct sign-in function for the workaround
  const signInWithDirectDb = async (email: string, password: string) => {
    try {
      console.log("Attempting direct database login with email:", email);
      
      // Check the user_profiles table for this email and password
      const { data, error } = await supabase
        .from('user_profiles')
        .select('*')
        .eq('email', email)
        .eq('password_hash', password) // In a real app, you would compare hashed passwords
        .single();
      
      if (error) {
        console.error("Direct login error:", error);
        throw new Error("Invalid email or password");
      }
      
      if (data) {
        console.log("Direct login successful for user:", data);
        // Store user info in localStorage
        localStorage.setItem('userId', data.id);
        localStorage.setItem('userEmail', data.email);
        
        // Create a pseudo-user
        const pseudoUser = {
          id: data.id,
          email: data.email,
          user_metadata: {
            full_name: data.full_name,
            organization: data.organization,
          },
          app_metadata: {},
          aud: 'direct',
          created_at: data.created_at,
        } as unknown as User;
        
        setUser(pseudoUser);
        navigate('/home');
      } else {
        throw new Error("User not found");
      }
    } catch (error) {
      console.error('Error signing in with direct database:', error);
      throw error;
    }
  };

  const signInWithGoogle = async () => {
    try {
      console.log("Attempting to sign in with Google...");
      console.log("Fetching token from production endpoint");

      
        // Second token endpoint (localhost)
        
      // Make sure the callback URL is correctly defined
      const redirectUrl = `${window.location.origin}/auth/callback`;
      console.log("Redirect URL set to:", redirectUrl);
      
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: redirectUrl,
          queryParams: {
            access_type: 'offline',
            prompt: 'consent',
          }
        }
      });
      
      if (error) {
        console.error("OAuth error:", error);
        throw error;
      }
      
      // The user ID will be stored via the onAuthStateChange listener after successful OAuth
      
      // Check if there's a URL to redirect to
      if (data && data.url) {
        console.log("Redirecting to:", data.url);
        window.location.href = data.url;
      } else {
        console.error("No redirect URL returned from supabase");
      }
    } catch (error) {
      console.error('Error signing in with Google:', error);
      alert("Failed to sign in with Google. Please try again.");
      throw error;
    }
  };

  const signInWithEmail = async (email: string, password: string) => {
    try {
      // Try Supabase auth first
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      console.log("Sign in data:", data);
      if (error) {
        console.error("Supabase auth error:", error);
        // If Supabase auth fails, try direct database auth
        try {
          return await signInWithDirectDb(email, password);
        } catch (directError) {
          // Both methods failed, throw the original error
          throw error;
        }
      }
      
      if (data.user) {
        localStorage.setItem('userId', data.user.id);
        navigate('/home');
      }
    } catch (error) {
      console.error('Error signing in with email:', error);
      throw error;
    }
  };

  const signOut = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      
      // Clear both auth and direct database user info
      localStorage.removeItem('userId');
      localStorage.removeItem('userEmail');
      
      setUser(null);
      navigate('/login');
    } catch (error) {
      console.error('Error signing out:', error);
      throw error;
    }
  };

  return (
    <AuthContext.Provider value={{ user, signInWithGoogle, signInWithEmail, signOut, loading }}>
      {children}
    </AuthContext.Provider>
  );
}