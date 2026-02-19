from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한 권리를 행사하거나 지키는 것에 관하여 조치를 하여야 하 며, 또한 '
 '회사가 요구하는 증거 및 서류를 제출하여야 합니 다. 이에 필요한 비용은 회사가 지급합니다. \uf000 회사는 제1항 및 제2항에도 '
 '불구하고 타인을 위한 보험 계약의 경우에는 계약자에 대한 대위권을 포기합니다. \uf000 회사는 제1항에 따른 권리가 계약자 또는 '
 '피보험자와 생 계를 같이 하는 가족에 대한 것인 경우에는 그 권리를 취득 하지 못합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 181},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000608',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
