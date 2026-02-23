from langchain_core.documents import Document

chunk = Document(
    page_content=('사고발생 위험이 현저하게 변경 또는 증가된 사실을 안 때에는 지체없이 보험자에게\n'
 '통지하여야 하며, 위반 시 보험계약이 해지되거나 보험금 지급이 제한될 수 있습니\n'
 '다. (이하 같습니다.)제16조(알릴 의무 위반의 효과)① 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 그 사실을 '
 '안\n'
 '날부터 1개월 이내에 이 계약을 해지할 수 있습니다.- 1. 계약자, 피보험자 또는 이들의 대리인이 고의 또는 중대한 과실로 보통약관 '
 '제15조\n'
 '- (계약 전 알릴 의무)를 위반하고 그 의무가 중요한 사항에 해당하는 경우'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
