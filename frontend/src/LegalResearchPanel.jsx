import {useState} from "react";

// Completely optional/offline PDF lookup for the university capstone.
// It never offers a made-up monetary penalty or announces legal commands.
export default function LegalResearchPanel({apiBase=""}) {
  const [query,setQuery]=useState("quy tắc chấp hành báo hiệu đường bộ");
  const [result,setResult]=useState(null);
  const [error,setError]=useState("");
  const [pending,setPending]=useState(false);
  async function search(event) {
    event.preventDefault();
    setPending(true);setError("");setResult(null);
    try {
      const response=await fetch(apiBase+"/pipeline/legal-search",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({query,top_k:3})
      });
      if(!response.ok)throw new Error("Không tra cứu được tài liệu ("+response.status+")");
      setResult(await response.json());
    } catch(e){setError(e.message||"Không kết nối được kho luật offline");}
    finally{setPending(false);}
  }
  return <section className="panel" style={{marginTop:20}}>
    <h2>Tra cứu pháp luật Việt Nam — offline RAG</h2>
    <p>Tra cứu Luật 36, Nghị định 168, các bản sửa đổi 118/2025, 238/2026 và QCVN 41 theo văn bản nguồn. Kết quả phục vụ nghiên cứu; hiệu lực và điều kiện áp dụng cần được xác minh.</p>
    <form onSubmit={search} style={{display:"flex",gap:8,flexWrap:"wrap"}}>
      <input style={{flex:"1 1 330px",minHeight:42}} value={query} onChange={e=>setQuery(e.target.value)}
        aria-label="Nhập nội dung cần tra cứu" minLength={2} maxLength={180}/>
      <button type="submit" className="run-button" disabled={pending||query.trim().length<2}>
        {pending?"Đang tìm…":"Tra cứu có nguồn"}
      </button>
    </form>
    {error&&<p role="alert">{error}</p>}
    {result&&<div>
      <p><b>{result.retrieved_count} đoạn trích phù hợp.</b> {result.warning}</p>
      {result.hits.map((hit,i)=><details key={hit.source_file+"-"+hit.page+"-"+i} style={{marginTop:10}} open={i===0}>
        <summary><b>{hit.document}</b> — {hit.locator||(hit.page?'trang PDF '+hit.page:'đoạn nguồn')}</summary>
        {hit.amendments?.length>0&&<p role="status">Văn bản gốc có sửa đổi: {hit.amendments.map(x=>x.document_number).join(", ")}. Cần đọc các bản sửa đổi liên quan.</p>}
        {hit.extraction_method?.includes("ocr")&&<p>Văn bản OCR tự động: đối chiếu bản PDF khi sử dụng điều khoản.</p>}
        <p style={{whiteSpace:"pre-wrap",lineHeight:1.5,overflowWrap:"anywhere"}}>{hit.excerpt}</p>
        <a href={hit.source_url+(hit.page?'#page='+hit.page:'')} target="_blank" rel="noreferrer">
          Mở văn bản nguồn{hit.page?' (trang '+hit.page+')':''}
        </a>
      </details>)}
      {!result.hits.length&&<p>Không có đoạn nguồn đủ phù hợp. Không suy đoán câu trả lời.</p>}
    </div>}
  </section>;
}
