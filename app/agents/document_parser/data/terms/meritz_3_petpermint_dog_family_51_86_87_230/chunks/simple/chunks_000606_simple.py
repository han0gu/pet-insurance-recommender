from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사가 제1항의 절차를 대행하는 경우에는, 피보험자에 대하여 보상책임을 지는 한도 내에서, 가압류나 가집행을 면하기 위한 '
 '공탁금을 피보험자에게 대부할 수 있으며 이에 소요되는 비용을 보상합니다. 이 경우 대부금의 이자는 공 탁금에 붙여지는 것과 같은 이율로 '
 '하며, 피보험자는 공탁 금(이자를 포함합니다)의 회수청구권을 회사에 양도하여야 합니다.\n'
 '제11조(대위권)\n'
 '\uf000 회사가 보험금을 지급한 때(현물보상한 경우를 포함합니'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 180},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000606',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
