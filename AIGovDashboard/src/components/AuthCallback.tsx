import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';

export default function AuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    // Handle the OAuth callback
    const handleAuthCallback = async () => {
      try {
        console.log("AuthCallback component mounted, processing authentication...");
        
        // Get the current session
        const { data, error } = await supabase.auth.getSession();
        
        if (error) {
          console.error('Error getting session:', error);
          navigate('/home');
          return;
        }

        console.log("Session data:", data);

        if (data.session) {
          console.log('Session established successfully');
          
          // Store user ID in localStorage
          if (data.session.user) {
            localStorage.setItem('userId', data.session.user.id);
            console.log('User ID stored in localStorage:', data.session.user.id);
            
            try {
              // Check if user exists in the users table
              const { data: userData, error: userError } = await supabase
                .from('users')
                .select('id')
                .eq('id', data.session.user.id)
                .single();
                
              if (userError || !userData) {
                console.log('Creating new user record');
                // Create user record if it doesn't exist
                await supabase.from('users').insert([
                  {
                    id: data.session.user.id,
                    email: data.session.user.email,
                    full_name: data.session.user.user_metadata?.full_name || data.session.user.user_metadata?.name || '',
                    created_at: new Date().toISOString()
                  }
                ]);
              }
              
              // Check if user exists in the userData table
              const { data: userDataRecord, error: userDataError } = await supabase
                .from('userData')
                .select('user_id')
                .eq('user_id', data.session.user.id)
                .single();
                
              if (userDataError || !userDataRecord) {
                console.log('Creating new userData record');
                // Create userData record if it doesn't exist
                const { error: insertError } = await supabase.from('userData').insert([
                  {
                    user_id: data.session.user.id
                    // No need to specify other fields as they have defaults
                  }
                ]);
                
                if (insertError) {
                  console.error('Error creating userData record:', insertError);
                } else {
                  console.log('userData record created successfully');
                }
              } else {
                console.log('userData record already exists');
              }
            } catch (err) {
              console.error('Error checking/creating user records:', err);
            }
          }

         
          
          // Redirect to home
          navigate('/home');
        } else {
          console.error('No session found in auth callback');
          navigate('/login');
        }
      } catch (error) {
        console.error('Unexpected error during auth callback:', error);
        navigate('/login');
      }
    };

    handleAuthCallback();
  }, [navigate]);

  return (
    <div className="flex justify-center items-center h-screen">
      <div className="text-center">
        <h2 className="text-xl mb-4">Processing authentication...</h2>
        <p>Please wait while we complete your sign-in.</p>
        <div className="mt-4 animate-spin h-8 w-8 border-4 border-indigo-500 rounded-full border-t-transparent mx-auto"></div>
      </div>
    </div>
  );
} 