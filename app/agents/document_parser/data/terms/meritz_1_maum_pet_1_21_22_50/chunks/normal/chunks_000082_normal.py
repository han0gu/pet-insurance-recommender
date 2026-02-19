from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사가 제1항에 따라 제공될 약관 및 계약자 보관용 청약서를 청약할 때 계약자에게 전달하지 않거나 약관의 중요한 내용을 설명하지 않은 '
 '때 또는 계약을 체결할 때 계약 자가 청약서에 자필서명(날인(도장을 찍음) 및 ⌜전자서명법⌟ 제2조 제2호에 따른 전 자서명을 '
 '포함합니다)을 하지 않은 때에는 계약자는 계약이 성립한 날부터 3개월 이내 에 계약을 취소할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 206,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
