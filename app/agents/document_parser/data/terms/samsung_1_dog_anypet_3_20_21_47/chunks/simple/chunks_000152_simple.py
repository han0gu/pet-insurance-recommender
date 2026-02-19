from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 회사가 제1항의 절차를 대행하는 경우에는, 피보험자에 대하여 보상책임을 지는 한도 내에서, 가압 류나 가집행을 면하기 위한 공탁금을 '
 '피보험자에게 대부할 수 있으며 이에 소요되는 비용을 보상 합니다. 이 경우 대부금의 이자는 공탁금에 붙여지는 것과 같은 이율로 하며, '
 '피보험자는 공탁금 (이자를 포함합니다)의 회수청구권을 회사에 양도하여야 합니다.\n'
 '제10조(대위권)'),
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
 'indexing': {'chunk_id': 'chunk_000152',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
