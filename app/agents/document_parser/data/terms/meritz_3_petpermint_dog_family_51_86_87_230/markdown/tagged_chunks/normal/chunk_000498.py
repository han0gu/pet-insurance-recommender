from langchain_core.documents import Document

chunk = Document(
    page_content=('의 지급을 직접 청구할 수 있습니다. 그러나 회사는 피보험\n'
 '자가 그 사고에 관하여 가지는 항변으로써 피해자에게 대항\n'
 '할 수 있습니다.\n'
 '\uf000 회사는 제1항의 청구를 받았을 때에는 지체없이 피보험\n'
 '자에게 통지하여야 하며, 회사의 요구가 있으면 피보험자\n'
 '및 계약자는 필요한 서류·증거의 제출, 증언 또는 증인 출\n'
 '석에 협조하여야 합니다.\n'
 '\uf000 피보험자가 피해자로부터 손해배상의 청구를 받았을 경\n'
 '우에 회사가 필요하다고 인정할 때에는 피보험자를 대신하\n'
 '여 회사의 비용으로 이를 해결할 수 있습니다. 이 경우 회'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000498',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
