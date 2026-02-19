from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 단체의 내규에 의한 복지제도로서 노사합의에 의하며, 보험료의 일부를 단체 또는 단체의 대표 자가 부담하여야 합니다. 2. 제1항 '
 '제2호 및 제3호에 해당하는 단체는 내규에 의해 단체의 대표자와 회사가 협정에 의해 체 결하여야 합니다.\n'
 '제2조(상법 제735조3의 적용)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000174',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
