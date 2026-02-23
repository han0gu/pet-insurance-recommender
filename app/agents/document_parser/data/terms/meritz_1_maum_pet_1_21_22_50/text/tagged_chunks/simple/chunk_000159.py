from langchain_core.documents import Document

chunk = Document(
    page_content=('발생할 수 없는 경우에는 이 특별약관은 그 때부터 효력이 없습니다.- 29 -제20조(계약자의 임의해지)계약자는 손해가 발생하기 전에는 '
 '언제든지 계약을 해지할 수 있습니다. 다만, 타인을 위\n'
 '한 계약의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여\n'
 '계약을 해지할 수 있습니다.제21조(준용규정)이 특별약관에서 정하지 않은 사항은 보통약관을 따릅니다.- 30 -반려견 슬관절·고관절 '
 '치료비 보장 특별약관제1조(보상하는 손해)① 회사는 보통약관 제5조(보험금을 지급하지 않은 사유) 제2항 제3호에도 불구하고'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000159',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
