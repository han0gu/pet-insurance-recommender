from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 피보험자에 대하여 보상책임을 지는 한도 내에서 제1항의 절차에 협조하거나 대행합니다.\n'
 '【보상책임을 지는 한도】\n'
 '동일한 사고로 이미 지급한 보험금이나 가지급보험금이 있는 경우에는 그 금액을 공제 한 액수를 말합니다.\n'
 '③ 회사가 제1항의 절차에 협조하거나 대행하는 경우에는 피보험자는 회사의 요청에 따라 협력해야 하며, 피보험자가 정당한 이유없이 협력하지 '
 '않는 경우에는 그로 말미암아 늘 어난 손해에 대해서 보상하지 않습니다. ④ 회사는 다음의 경우에는 제1항의 절차를 대행하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
