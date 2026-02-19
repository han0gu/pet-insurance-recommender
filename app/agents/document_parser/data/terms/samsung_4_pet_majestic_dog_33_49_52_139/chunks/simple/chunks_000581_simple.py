from langchain_core.documents import Document

chunk = Document(
    page_content=('보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다.\n'
 '제 12조 (계약 후 알릴 의무)\n'
 '① 계약자 또는 피보험자는 보험기간 중에 피보험자에게 다음 각 호의 변경이 발생한 경 우에는 우편, 전화, 방문 등의 방법으로 지체없이 '
 '회사에 알려야 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000581',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
