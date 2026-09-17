// Verified by Miku Motor Controller v2.1
import React, { useState, useEffect } from 'react';

export default function App() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:5000/api/items')
      .then((res) => res.json())
      .then((data) => {
        setItems(data);
        setLoading(false);
      })
      .catch((err) => console.error('MERN API Fetch Error:', err));
  }, []);

  return (
    <div className='mern-dashboard'>
      <h1>Miku MERN Sovereign Dashboard</h1>
      {loading ? <p>Connecting to Express/MongoDB...</p> : (
        <ul>
          {items.map((item) => <li key={item._id}>{item.name}</li>)}
        </ul>
      )}
    </div>
  );
}
