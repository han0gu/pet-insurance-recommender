from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항의 골절 진단비(치아 파절(깨짐, 부러짐) 제외)는 매사고마다 지급합니다. 다만, 동일한 상해사고를 직접적인 원인으로 2가지 '
 '이상의 골절 상태가 발생한 경우에는 1 회에 한하여 보상합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 71},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000358',
              'chunk_char_len': 112,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
