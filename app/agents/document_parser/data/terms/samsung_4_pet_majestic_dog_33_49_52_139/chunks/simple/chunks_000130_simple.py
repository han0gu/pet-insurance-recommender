from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 회사는 자동대출납입이 종료된 날부터 15일 이내에 자동대출납입이 종료되었음을 서 면, 전화(음성녹음) 또는 전자문서(SMS 포함) '
 '등으로 계약자에게 안내하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 44},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 97,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
