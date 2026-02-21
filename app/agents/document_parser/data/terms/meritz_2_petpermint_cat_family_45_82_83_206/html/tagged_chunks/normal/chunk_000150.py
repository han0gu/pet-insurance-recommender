from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 단체가 규약에 따라 구성원<br>의 전부 또는 일부를 피보험자로 하는 계약을 체결하<br>는 경우에는 이를 적용하지 않습니다. '
 '이 때 단체보험<br>의 보험수익자를 피보험자 또는 그 상속인이 아닌 자<br>로 지정할 때에는 단체의 규약에서 명시적으로 '
 '정한<br>경우가 아니면 이를 적용합니다.<br>② 만15세 미만자, 심신상실자 또는 심신박약자를 피보험<br>자로 하여 사망을 보험금 '
 '지급사유로 한 경우'),
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
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
