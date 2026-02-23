from langchain_core.documents import Document

chunk = Document(
    page_content=('- 충·중재 또는 소송(확인의 소를 포함합니다)에 대하여 협조하거나, 피보험자를 위하여 이러한 절차\n'
 '- 를 대행할 수 있습니다.\n'
 '- ② 회사는 피보험자에 대하여 보상책임을 지는 한도 내에서 제1항의 절차에 협조하거나 대행합니다.\n'
 '【보상책임을 지는 한도】 동일한 사고로 이미 지급한 보험금이나 가지급보험금이 있는 경우에는 그 금액\n'
 '을 공제한 액수를 말합니다.- 회사가 제1항의 절차에 협조하거나 대행하는 경우에는 피보험자는 회사의 요청에 따라 협력해야'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000122',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
