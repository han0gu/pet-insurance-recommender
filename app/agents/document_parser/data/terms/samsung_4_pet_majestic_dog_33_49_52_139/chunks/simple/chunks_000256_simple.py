from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[약관의 중요한 내용] 금융소비자 보호에 관한 법률 제19조(설명의무) 등에서 정한 다음의 내용을 말합니다.\n'
 '- 보험금 지급제한 사유 및 지급절차 - 청약의 철회에 관한 사항 - 계약의 해지 및 해제'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 59},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000256',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
