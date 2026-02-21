from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보장하는 다른 계약 중 회사가 유사하다고 판단한 보험계약의 보험기간이 만료되어 종료일\n'
 '- 부터 7일 이내에 재가입하는 것을 말합니다.\n'
 '# 2. 지급사유 관련 용어가. 상해: 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 반려동물에 입은 상해를 말\n'
 '하며, 유독 가스 또는 유독 물질을 반려동물이 우연히 일시적으로 흡입, 흡수 또는 섭취한\n'
 '결과로 생긴 중독 증상을 포함합니다. 그러나 음식물 섭취로 인한 증상, 세균성 음식물 중'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000003',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
