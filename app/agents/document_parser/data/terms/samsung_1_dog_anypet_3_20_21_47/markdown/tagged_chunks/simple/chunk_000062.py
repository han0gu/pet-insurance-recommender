from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 실시할 것\n'
 '- 4. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것\n'
 '- 14 -당신에게 좋은보험 삼성화재5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것⑤ 제1항에 따라 계약이 해지된 '
 '경우에는 제30조(보험료의 환급)에 따라 보험료를 계약자에게 지급합\n'
 '니다.제24조[보험료의 납입연체로 인한 해지계약의 부활(효력회복)]① 제23조[보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 '
 '해지]에 따라 계약이 해지되었으'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
