from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나. 사시교정, 안와격리증(양쪽 눈을 감싸고 있는 뼈와 뼈 사이의 거리가 넓\n'
 '- 76 -| 은 증상)의 교정 등 시각계 수술로서 시력개선 목적이 아닌 외모개선 목 적의 수술 다. 안경, 콘택트렌즈 등을 대체하기 '
 '위한 시력교정술(국민건강보험 요양급여 대상 수술방법 또는 치료재료가 사용되지 않은 부분은 시력교정술로 봅니 다) 라. 외모개선 목적의 '
 '다리정맥류 수술 4. 위생관리, 미모를 위한 성형수술(다만, 사고전 상태로의 회복을 위한 수술은 보장합니다) |\n'
 '| --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000267',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
