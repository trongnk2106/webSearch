import React, { useState } from 'react';

function Search() {
  const [query, setQuery] = useState('');
  const [results_fromapi, setResults] = useState([]);

  const [modelSelect, setModelSelect] = useState('bm25');
  const [number, setNumber] = useState(20);



  const handleSearch = async (event) => {
    console.log('handleSearch');
    event.preventDefault();
    if (modelSelect === 'bm25')
    {
      fetch(`http://localhost:8000/bm25?q=${query}&k=${number}`)
      .then(response => response.json())
      .then(data => {
        console.log(data.result)
        setResults(data.result)}
        )
      .catch(error => console.log(error));
    }

    if (modelSelect === 'vec'){
      fetch(`http://localhost:8000/vec?q=${query}&k=${number}`)
      .then(response => response.json())
      .then(data => {
        console.log(data.result)
        setResults(data.result)}
        )
      .catch(error => console.log(error));
    }
   

  }


  return (
    <>
    <h1 style={{textAlign : 'center'}}> Information Retrieval</h1>
      <form onSubmit={handleSearch}>
        <input
          type="text"
          placeholder="Tìm kiếm..."
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          style={{marginLeft : '35%', marginRight:'10px', marginTop : '1%',width: '500px', height: '30px', borderRadius: '5px'}}
        />
        <button type="submit">Tìm kiếm</button>
      </form>
      {/* <div style={{marginLeft : '25%', marginRight:'25%'}}>
        <button onClick={() => setModelSelect('bm25')}>BM25</button>
        <button onClick={() => setModelSelect('vec')}>VEC</button>
      </div> */}
      {/* create radio button for select model */}

      <div style={{ marginLeft: "35%", marginRight: "25%" , marginBottom : '30px'}}>
        <input
          type="radio"
          id="bm25"
          name="model"
          value="bm25"
          checked={modelSelect === "bm25"}
          onChange={() => setModelSelect("bm25")}
        />
        <label for="bm25">BM25</label>
        <input
          type="radio"
          id="vec"
          name="model"
          value="vec"
          checked={modelSelect === "vec"}
          onChange={() => setModelSelect("vec")}
        />
        <label for="vec">VEC</label>
      </div>

      <div style={{display :'flex', marginBottom : '100px', marginLeft: '20%'}}>
        <p style={{marginTop: '2px'}}> Top K </p>
        <div>
          <input 
          type="number" 
          style={{marginLeft : "20px", width : '50px', height : '20px', borderRadius : '5px'}}
          value={number} 
          placeholder = 'Top K'
          onChange={(e) => setNumber(e.target.value)}
          />
        </div>
        
      </div>
      <div>
        <h3 style={{textAlign:'center'}}>Kết quả tìm kiếm</h3>
        <ol>
          {results_fromapi.map((result, index) => (
            <li key={index}  style={{
              border: "1px solid black",
              padding: "10px",
              margin: "5px",
            }}>{result}</li>
          ))}
        </ol>
      </div>
     
    </>
  );
}

export default Search;
