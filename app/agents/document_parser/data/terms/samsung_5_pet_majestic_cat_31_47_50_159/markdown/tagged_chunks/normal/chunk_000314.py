from langchain_core.documents import Document

chunk = Document(
    page_content=('- 원 또는 국외의 의료 관련법에 정한 의료기관의 의사(한의사, 치과의사는 제외합니다)\n'
 '- 의 면허를 가진 자에 의하여 내려져야 하며, 이 진단은 임상적 특징 또는 혈액, 항체,\n'
 '- 항원검사, 유발검사 및 피부시험 등을 기초로 내려져야 합니다.\n'
 '# 제4조 (응급실의 정의)① 이 특별약관에서 「응급실」 이라 함은 응급의료에 관한 법률 제2조(정의) 제5호에서\n'
 '정하는 응급의료기관(권역응급의료센터, 전문응급의료센터, 지역응급의료센터, 지역응\n'
 '급의료기관) 또는 응급의료에 관한 법률 제35조의2(응급의료기관 외의 의료기관)에서'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000314',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
