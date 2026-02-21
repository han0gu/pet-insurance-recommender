from langchain_core.documents import Document

chunk = Document(
    page_content=('| 8 | 전신성 질환 |  |  |\n'
 '| 8 | 전신성 질환 | PAA018 | 고양이 전염성복막염(FIP) |\n'
 '173Ⅲ. 별표# 【별표1】보험금을 지급할 때의 적립이율 계산\n'
 '(제8조 제5항, 제10조 제3항 및 제35조 제2항 관련)| 구 분 | 기 간 | 지 급 이 자 |\n'
 '| --- | --- | --- |\n'
 '| 보장관련 보험금 (보통약관 제3조) (특별약관이 부가된 경우 특별약관의 보험금 포함) | 지급기일의 다음 날부터 30일 이내 기간 | '
 '보험계약대출이율 |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000504',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
