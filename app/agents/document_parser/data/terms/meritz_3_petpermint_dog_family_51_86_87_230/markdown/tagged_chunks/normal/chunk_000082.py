from langchain_core.documents import Document

chunk = Document(
    page_content=('- 따라 단체보험의 피보험자가 될 때에 의사능력이 있는\n'
 '- 경우에는 계약이 유효합니다.\n'
 '- ③ 계약을 체결할 때 계약에서 정한 피보험자의 나이에\n'
 '- 미달되었거나 초과되었을 경우. 다만, 회사가 나이의\n'
 '- 착오를 발견하였을 때 이미 계약나이에 도달한 경우에\n'
 '- 는 유효한 계약으로 보나, 제2호의 만15세 미만자에\n'
 '- 관한 예외가 인정되는 것은 아닙니다.\n'
 '# 【상법 제731조(타인의 생명의 보험)】① 타인의 사망을 보험사고로 하는 보험계약에는 보험계\n'
 '약 체결시에 그 타인의 서면(｢전자서명법｣제2조제2호에'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
