from langchain_core.documents import Document

chunk = Document(
    page_content=('- ④ 이 특약의 계약자는 전환대상계약의 계약자와 동일하여야 합니다.\n'
 '# 제2조(제출서류)- ① 이 특약에 가입하고자 하는 계약자는 모든 피보험자 또는 모든 보험수익자의 「소득세법 시행규칙\n'
 '- 별지 제38호 서식에 의한 장애인증명서의 원본 또는 사본」 (이하, "장애인증명서"라 합니다)을 제\n'
 '- 출하여 제1조(특약의 적용범위) 제1항 제2호에서 정한 조건에 해당함을 회사에 알려야 합니다.\n'
 '- ② 제1항에도 불구하고 「국가유공자 등 예우 및 지원에 관한 법률」 에 따른 상이자의 증명을 받은'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000145',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
