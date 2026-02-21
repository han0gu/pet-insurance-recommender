from langchain_core.documents import Document

chunk = Document(
    page_content=('는 합의·절충·중재 또는 소송(확인의 소를 포함합니다)에 대하여 협조하거나, 피보험자\n'
 '를 위하여 이러한 절차를 대행할 수 있습니다.- 26 -② 회사는 피보험자에 대하여 보상책임을 지는 한도 내에서 제1항의 절차에 '
 '협조하거나\n'
 '대행합니다.【보상책임을 지는 한도】동일한 사고로 이미 지급한 보험금이나 가지급보험금이 있는 경우에는 그 금액을 공제\n'
 '한 액수를 말합니다.③ 회사가 제1항의 절차에 협조하거나 대행하는 경우에는 피보험자는 회사의 요청에 따라\n'
 '협력해야 하며, 피보험자가 정당한 이유없이 협력하지 않는 경우에는 그로 말미암아 늘'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
