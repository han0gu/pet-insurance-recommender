from langchain_core.documents import Document

chunk = Document(
    page_content=('제1관 목적 및 용어의 정의\n'
 '제 1조 (목적)\n'
 '이 보험계약(이하「계약」이라 합니다)은 보험계약자(이하「계약자」라 합니다)와 보험회 사(이하「회사」라 합니다) 사이에 피보험자의 질병이나 '
 '상해에 대한 위험을 보장하기 위 하여 체결됩니다.\n'
 '제 2조 (용어의 정의)\n'
 '이 계약에서 사용되는 용어의 정의는, 이 계약의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '① 계약관계 관련 용어'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000000',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
