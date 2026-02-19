from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[위법계약]\n'
 "금융상품판매업자등이 '금융소비자보호에 관한 법률' 제47조에서 정한 적합성원칙, 적정성원칙, 설 명의무, 불공정영업행위의 금지 또는 "
 '부당권유행위 금지를 위반한 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 63},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000303',
              'chunk_char_len': 112,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
