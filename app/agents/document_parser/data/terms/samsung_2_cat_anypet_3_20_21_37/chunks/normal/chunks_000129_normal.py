from langchain_core.documents import Document

chunk = Document(
    page_content=('단체계약 특별약관\n'
 '제1조(계약의 적용 범위)\n'
 '① 피보험자가 다음 중 한가지의 단체에 소속되어야 하며, 단체를 대표하여 계약자로 된 자가 단체보 험 계약상의 모든 권리, 의무를 행사할 '
 '수 있어야 합니다.\n'
 '1. 제1종 단체'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 122,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
