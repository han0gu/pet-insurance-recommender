from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 전부 또는 일부를 피보험자로 하는 계약을 체결하\n'
 '- 는 경우에는 이를 적용하지 않습니다. 이 때 단체보험\n'
 '- 의 보험수익자를 피보험자 또는 그 상속인이 아닌 자\n'
 '- 로 지정할 때에는 단체의 규약에서 명시적으로 정한\n'
 '- 경우가 아니면 이를 적용합니다.\n'
 '- ② 만15세 미만자, 심신상실자 또는 심신박약자를 피보험\n'
 '- 자로 하여 사망을 보험금 지급사유로 한 경우. 다만,\n'
 '- 심신박약자가 계약을 체결하거나 소속 단체의 규약에\n'
 '- 따라 단체보험의 피보험자가 될 때에 의사능력이 있는\n'
 '- 경우에는 계약이 유효합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
