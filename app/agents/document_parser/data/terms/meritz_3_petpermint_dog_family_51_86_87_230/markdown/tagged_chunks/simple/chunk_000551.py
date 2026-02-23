from langchain_core.documents import Document

chunk = Document(
    page_content=('는 보험계약의 부활(효력회복)을 승낙한 경우에 한하여 보\n'
 '통약관 제30조(보험료의 납입을 연체하여 해지된 계약의 부193활(효력회복))를 준용합니다.제4조(준용규정)이 특별약관에 정하지 않은 '
 '사항은 보통약관 및 해당 특별\n'
 '약관을 따릅니다.194# 【 별첨 】특정질병 분류표(반려견)보험계약을 체결할 때 반려동물의 건강상태가 회사가 정한 기준에 적합\n'
 '하지 않은 경우 또는 보험계약을 체결한 후 계약 전 알릴 의무 위반의\n'
 '효과 등으로 보장을 제한할 경우에 한하여 보상하지 않는 질병을 분류'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000551',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
