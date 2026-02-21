from langchain_core.documents import Document

chunk = Document(
    page_content=('- ·보험금 지급금액\n'
 '= [(13만원 - 3만원)×50%, 10만원] 중 적은금액\n'
 '= 5만원② 입원 중 수술을 한 경우(보상비율 50%)- ·피보험자가 부담한 수술당일 치료비 400만원\n'
 '- ·보험금 지급금액\n'
 '- = [(400만원-3만원)×50%, 150만원] 중 적은금액\n'
 '- = 150만원\n'
 '\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여\n'
 '30일 이내에 발생한 질병은 보상하지 않습니다. 단,「반려\n'
 '동물 비용손해 관련 특별약관 일반조항」제15조(재가입) 제\n'
 '6항에 따라 보험계약이 연장된 경우에는 적용하지 않습니\n'
 '다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000275',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
