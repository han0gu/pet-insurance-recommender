from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 때 단체보험 의 보험수익자를 피보험자 또는 그 상속인이 아닌 자 로 지정할 때에는 단체의 규약에서 명시적으로 정한 경우가 아니면 '
 '이를 적용합니다. ② 만15세 미만자, 심신상실자 또는 심신박약자를 피보험 자로 하여 사망을 보험금 지급사유로 한 경우. 다만, '
 '심신박약자가 계약을 체결하거나 소속 단체의 규약에 따라 단체보험의 피보험자가 될 때에 의사능력이 있는 경우에는 계약이 유효합니다. ③ '
 '계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을 경우'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 67},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
