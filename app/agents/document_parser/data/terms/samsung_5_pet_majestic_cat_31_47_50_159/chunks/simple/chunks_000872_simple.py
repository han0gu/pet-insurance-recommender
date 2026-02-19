from langchain_core.documents import Document

chunk = Document(
    page_content=('2) "교정시력" 이라 함은 안경(콘택트렌즈를 포함한 모든 종류의 시력 교정수단) 으로 교정한 원거리 최대교정시력을 말한다. 다만, '
 '각막이식술을 받은 환자인 경우 각막이식술 이전의 시력상태를 기준으로 평가한다. 3) "한 눈이 멀었을 때" 라 함은 안구의 적출은 물론 '
 '명암을 가리지 못하거나 ( "광각무" ) 겨우 가릴 수 있는 경우( "광각유" )를 말한다. 4) "한눈의 교정시력이 0.02 이하로 된 '
 '때" 라 함은 안전수동(Hand Movement)주 1) 안전수지(Finger Counting)주2) 상태를 포함한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000872',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
