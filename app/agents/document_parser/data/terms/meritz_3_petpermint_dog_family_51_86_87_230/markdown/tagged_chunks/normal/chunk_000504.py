from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 피보험자가 손해배상을 함으로써 대위 취득하는 것이\n'
 '- 있을 경우에는 그 대위권\n'
 '\uf000 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한\n'
 '권리를 행사하거나 지키는 것에 관하여 조치를 하여야 하\n'
 '며, 또한 회사가 요구하는 증거 및 서류를 제출하여야 합니\n'
 '다. 이에 필요한 비용은 회사가 지급합니다.\n'
 '\uf000 회사는 제1항 및 제2항에도 불구하고 타인을 위한 보험\n'
 '계약의 경우에는 계약자에 대한 대위권을 포기합니다.\n'
 '\uf000 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생\n'
 '계를 같이 하는 가족에 대한 것인 경우에는 그 권리를 취득'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000504',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
