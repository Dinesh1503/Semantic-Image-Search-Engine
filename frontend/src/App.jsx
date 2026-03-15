// import { useState } from 'react'
// import axios from "axios"
// import './App.css'

// function App() {
//   const [caption, setCaption] = useState("")
//   const [results, setResults] = useState(null)

//   const getImageUrl = (filePath) => {
//     const filename = filePath.split("/").pop()
//     return `http://localhost:8000/images/${filename}`
//   }

//   const handleSearch = async () => {
//     const url = "http://localhost:8000/search"
//     const body = { caption: caption }
//     axios.post(url, body).then((response) => {
//       setResults(response.data)
//     })
//   }

//   return (
//     <div>
//       <h1>Semantic Image Search</h1>

//       <input
//         type="text"
//         value={caption}
//         onChange={(e) => setCaption(e.target.value)}
//         placeholder="Describe an image..."
//       />
//       <button onClick={handleSearch}>Search</button>

//       {results && (
//         <div>
//           <h3>Good matches</h3>
//           <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
//             {results.good.map((item, i) => (
//               <img key={i} src={getImageUrl(item[2])} alt={item[3]} width={200} />
//             ))}
//           </div>

//           <h3>Other results</h3>
//           <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
//             {results.bad.map((item, i) => (
//               <img key={i} src={getImageUrl(item[2])} alt={item[3]} width={200} />
//             ))}
//           </div>
//         </div>
//       )}
//     </div>
//   )
// }

// export default App


import { useState } from 'react'

function App() {
  const [caption, setCaption] = useState("")
  console.log(caption)
  return (
    <div>
      <input 
        value={caption}
        onChange={(e) => setCaption(e.target.value)}
        placeholder="Describe an image..."
      />
      <button>Search</button>
    </div>
  )
}

export default App