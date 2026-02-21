from langchain_core.documents import Document

chunk = Document(
    page_content=('. 만약, 회사가 전자우편 및 전자적<br>의사표시로 제공한 경우 계약자 또는 그 대리인이 약관 및 계약자 보관용 청약서<br>등을 '
 "수신하였을 때에는 해당 문서를 드린 것으로 봅니다.</p><br><p id='206' data-category='list' "
 "style='font-size:16px'>1. 서면교부<br>2. 우편 또는 전자우편<br>3"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000166',
              'chunk_char_len': 195,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
