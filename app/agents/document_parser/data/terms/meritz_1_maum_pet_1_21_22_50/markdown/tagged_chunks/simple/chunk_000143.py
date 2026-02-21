from langchain_core.documents import Document

chunk = Document(
    page_content=('- 과 회사의 동의를 받지 않은 행위로 증가된 손해\n'
 '# 제12조(손해배상청구에 대한 회사의 해결)- ① 피보험자가 피해자에게 손해배상책임을 지는 사고가 생긴 때에는 피해자는 이 특별약관\n'
 '- 에 따라 회사가 피보험자에게 지급책임을 지는 금액 한도내에서 회사에 대하여 보험금\n'
 '- 의 지급을 직접 청구할 수 있습니다. 그러나 회사는 피보험자가 그 사고에 관하여 가지\n'
 '- 는 항변으로써 피해자에게 대항할 수 있습니다.\n'
 '- ② 회사는 제1항의 청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며, 회사의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
