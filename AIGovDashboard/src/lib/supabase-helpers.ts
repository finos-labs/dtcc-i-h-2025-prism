import { supabase } from './supabase';

/**
 * Create a user directly in the database, bypassing Supabase Auth
 * This is used as a fallback method when Supabase Auth is not working
 */
export const createUserDirectly = async (userData: {
  id: string;
  email: string;
  password_hash: string;
  full_name: string;
  organization?: string | null;
  user_type?: string;
}) => {
  try {
    // Check if user_profiles table exists
    console.log("Checking user_profiles table...");
    const { data: tableExists, error: tableCheckError } = await supabase
      .from('information_schema.tables')
      .select('table_name')
      .eq('table_schema', 'public')
      .eq('table_name', 'user_profiles');
      
    if (tableCheckError) {
      console.error("Error checking for table:", tableCheckError);
      return { success: false, error: tableCheckError };
    }
    
    // Create table if it doesn't exist
    if (!tableExists || tableExists.length === 0) {
      console.log("Creating user_profiles table...");
      
      // SQL to create the table
      const createTableSQL = `
        CREATE TABLE IF NOT EXISTS user_profiles (
          id UUID PRIMARY KEY,
          email TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          full_name TEXT NOT NULL,
          organization TEXT,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          user_type TEXT,
          ai_proficiency TEXT,
          usage_frequency TEXT,
          usage_context TEXT,
          primary_goal TEXT,
          industry_focus TEXT
        );
      `;
      
      try {
        // Using raw SQL query through Supabase
        const { error: createError } = await supabase.rpc('exec_sql', { 
          sql: createTableSQL 
        });
        
        if (createError) {
          console.error("Failed to create table:", createError);
          return { success: false, error: createError };
        }
      } catch (e) {
        console.error("Exception creating table:", e);
        // If RPC fails, try direct creation (may not work with default permissions)
        try {
          await supabase.from('user_profiles').insert([{}]).select();
        } catch (innerE) {
          console.error("Could not create table:", innerE);
        }
      }
    }
    
    // First, check if a user with this email already exists
    console.log("Checking if user already exists...");
    const { data: existingUser, error: existingError } = await supabase
      .from('user_profiles')
      .select('id')
      .eq('email', userData.email);
      
    if (existingError) {
      console.error("Error checking for existing user:", existingError);
      return { success: false, error: existingError };
    }
    
    if (existingUser && existingUser.length > 0) {
      console.log("User with this email already exists");
      return { success: false, error: new Error("Email address already in use") };
    }
    
    // Insert the user
    console.log("Inserting new user...");
    const { data, error } = await supabase
      .from('user_profiles')
      .insert([
        {
          id: userData.id,
          email: userData.email,
          password_hash: userData.password_hash,
          full_name: userData.full_name,
          organization: userData.organization || null,
          created_at: new Date().toISOString(),
          user_type: userData.user_type || 'direct_signup',
        }
      ]);
      
    if (error) {
      console.error("Error inserting user:", error);
      return { success: false, error };
    }
    
    return { success: true, data };
  } catch (error) {
    console.error("Unexpected error in createUserDirectly:", error);
    return { success: false, error };
  }
}; 