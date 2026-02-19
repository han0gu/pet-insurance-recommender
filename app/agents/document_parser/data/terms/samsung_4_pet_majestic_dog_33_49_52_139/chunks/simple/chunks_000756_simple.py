from langchain_core.documents import Document

chunk = Document(
    page_content=('. 가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임 13. 가입동물이 질병을 전염시켜 발생한 배상책임 14. 동물보호법 '
 '시행규칙 제1조의 3에 따른 맹견의 경우 동법 시행규칙 제12조 제2항'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 121},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000756',
              'chunk_char_len': 112,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
