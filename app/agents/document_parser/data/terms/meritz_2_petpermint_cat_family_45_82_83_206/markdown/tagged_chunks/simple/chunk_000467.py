from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금의 지급사유인지 아닌지는 수의사의 진단서와 의견을\n'
 '주된 판단자료로 하여 결정합니다.# 제3조(특별약관의 부활(효력회복))회사는 이 특별약관의 부활(효력회복) 청약을 받은 경우에167는 '
 '보험계약의 부활(효력회복)을 승낙한 경우에 한하여 보\n'
 '통약관 제30조(보험료의 납입을 연체하여 해지된 계약의 부\n'
 '활(효력회복))를 준용합니다.제4조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관 및 해당 특별\n'
 '약관을 따릅니다.168# 【 별첨 】특정질병 분류표(반려묘)보험계약을 체결할 때 반려동물의 건강상태가 회사가 정한 기준에 적합'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000467',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
