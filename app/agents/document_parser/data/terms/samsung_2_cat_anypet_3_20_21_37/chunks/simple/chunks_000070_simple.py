from langchain_core.documents import Document

chunk = Document(
    page_content=('을 실시할 것\n'
 '4. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것 5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 '
 '안내할 것\n'
 '제1항에 따라 계약이 해지된 경우에는 제30조(보험료의 환급)에 따라 보험료를 계약자에게 지급합 니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 147,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
