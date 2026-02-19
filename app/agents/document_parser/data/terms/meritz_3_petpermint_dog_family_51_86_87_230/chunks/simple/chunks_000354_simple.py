from langchain_core.documents import Document

chunk = Document(
    page_content=('= [(400만원-3만원)×50%, 150만원] 중 적은금액 = 150만원\n'
 '\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여 30일 이내에 발생한 질병은 보상하지 않습니다. 단,「반려 동물 '
 '비용손해 관련 특별약관 일반조항」제15조(재가입) 제 6항에 따라 보험계약이 연장된 경우에는 적용하지 않습니 다. \uf000 제1항에도 '
 '불구하고 보장개시일로부터 그 날을 포함하여 1년 이내에 발생한 슬관절탈구, 고관절탈구, 슬관절형성부 전, 고관절형성부전(대퇴 골두 허혈성 '
 '괴사 포함) 또는 기'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 123},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000354',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
