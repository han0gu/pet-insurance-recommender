from langchain_core.documents import Document

chunk = Document(
    page_content=('【계약 후 알릴 의무】\n'
 '상법 제652조에서 정하고 있는 의무. 보험기간 중에 보험계약자 또는 피보험자가 사고발생 위험이 현저하게 변경 또는 증가된 사실을 안 '
 '때에는 지체없이 보험자에게 통지하여야 하며, 위반 시 보험계약이 해지되거나 보험금 지급이 제한될 수 있습니 다. (이하 같습니다.)\n'
 '제16조(알릴 의무 위반의 효과)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 그 사실을 안 날부터 1개월 이내에 이 계약을 해지할 수 '
 '있습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000174',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
