from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조(합의. 절충. 중재. 소송의 협조. 대행 등)\n'
 '① 회사는 피보험자의 법률상 손해배상책임을 확정하기 위하여 피보험자가 피해자와 행하는 합의·절 충·중재 또는 소송(확인의 소를 '
 '포함합니다)에 대하여 협조하거나, 피보험자를 위하여 이러한 절차 를 대행할 수 있습니다. ② 회사는 피보험자에 대하여 보상책임을 지는 '
 '한도 내에서 제1항의 절차에 협조하거나 대행합니다.\n'
 '【보상책임을 지는 한도】 동일한 사고로 이미 지급한 보험금이나 가지급보험금이 있는 경우에는 그 금액 을 공제한 액수를 말합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
