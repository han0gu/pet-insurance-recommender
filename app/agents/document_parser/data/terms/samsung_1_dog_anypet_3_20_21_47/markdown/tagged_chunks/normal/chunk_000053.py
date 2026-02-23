from langchain_core.documents import Document

chunk = Document(
    page_content=('- 유로 회사에 이의를 제기할 수 없습니다.\n'
 '- ② 타인을 위한 계약에서 보험사고가 발생한 경우에 계약자가 그 타인에게 보험사고의 발생으로 생긴\n'
 '- 손해를 배상한 때에는 계약자는 그 타인의 권리를 해하지 않는 범위 안에서 회사에 보험금의 지급\n'
 '- 을 청구할 수 있습니다.\n'
 '【타인을 위한 계약】 계약자가 다른 사람의 이익을 위하여 자기의 이름으로 체결하는 보험계약을 말합니'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 207,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
