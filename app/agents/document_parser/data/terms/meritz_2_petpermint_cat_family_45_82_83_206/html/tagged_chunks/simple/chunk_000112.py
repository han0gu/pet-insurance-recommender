from langchain_core.documents import Document

chunk = Document(
    page_content=("각각의 부활(효력회복)계약을<br>최초계약으로 봅니다)</p><h1 id='55' "
 "style='font-size:20px'>제18조(사기에 의한 계약)</h1><br><p id='56' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자 또는 피보험자가 대리진단, "
 '약물사용을 수단으로<br>진단절차를 통과하거나 진단서 위·변조 또는 청약일 이전<br>에 암 또는 인간면역결핍바이러스(HIV) 감염의 '
 '진단 확정을<br>받은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000112',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
