from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이<br>경우 계약자의 답변과 확인내용을 음성 녹음함으로써<br>약관의 중요한 내용을 설명한 것으로 봅니다.</h1><br><h1 '
 "id='91' style='font-size:20px'>【통신판매계약】</h1><br><p id='92' "
 "data-category='paragraph' style='font-size:20px'>전화·우편·인터넷 등 통신수단을 이용하여 "
 "체결하는<br>계약을 말합니다.</p><br><p id='93' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사가"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000137',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
