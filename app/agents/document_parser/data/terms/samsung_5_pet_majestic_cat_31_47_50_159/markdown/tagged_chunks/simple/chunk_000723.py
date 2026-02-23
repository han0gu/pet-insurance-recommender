from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제7조 (준용규정)- \n'
 '이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.- - 132 -\n'
 '별표 및 참고\n'
 '별표\n'
 '약관에서 인용된 법·규정\n'
 '특별약관 색인별표[별표1] 보험금을 지급할 때의 적립이율 계산| 구분 | 기간 | 지급이자 |\n'
 '| --- | --- | --- |\n'
 '| 보장관련 보험금 | 지급기일의 다음날부터 30일이내 기간 | 보험계약대출이율 |\n'
 '| 보장관련 보험금 | 지급기일의 31일 이후부터 60일 이내 기간 | 보험계약대출이율 +가산이율(4.0%) |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000723',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
