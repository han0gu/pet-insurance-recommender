from langchain_core.documents import Document

chunk = Document(
    page_content=('. 사시교정, 안와격리증(양쪽 눈을 감싸고 있는 뼈와 뼈 사이의 거리가 넓은 증 상)의 교정 등 시각계 수술로서 시력개선 목적이 아닌 '
 '외모개선 목적의 수술 다. 안경, 콘텍트렌즈 등을 대체하기 위한 시력교정술(국민건강보험 요양급여 대상 수술방법 또는 치료재료가 사용되지 '
 '않은 부분은 시력교정술로 봅니다) 라. 외모개선 목적의 다리 정맥류 수술'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 76},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['eye', 'skin', 'joint']},
 'indexing': {'chunk_id': 'chunk_000398',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
