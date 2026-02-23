from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서 그 일련의 과정으로 시행한 지방흡입술은 보상합니다), 주름살 제거술 등\n'
 '- 나. 사시교정, 안와격리증(양쪽 눈을 감싸고 있는 뼈와 뼈 사이의 거리가 넓은 증\n'
 '- 상)의 교정 등 시각계 수술로서 시력개선 목적이 아닌 외모개선 목적의 수술\n'
 '- 다. 안경, 콘텍트렌즈 등을 대체하기 위한 시력교정술(국민건강보험 요양급여 대상\n'
 '- 수술방법 또는 치료재료가 사용되지 않은 부분은 시력교정술로 봅니다)\n'
 '- 라. 외모개선 목적의 다리 정맥류 수술\n'
 '- 4. 위생관리, 미모를 위한 성형수술. 다만, 사고전 상태로의 회복을 위한 수술은 보상'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000333',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
