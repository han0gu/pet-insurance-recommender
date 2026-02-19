from langchain_core.documents import Document

chunk = Document(
    page_content=('【파보바이러스 감염증】 파보바이러스에 감염되어 구토와 설사 등의 증상을 일으킴 【디스템퍼바이러스 감염증】 디스템퍼바이러스에 감염되어 '
 '호흡기 질환과 신경증상을 일으킴 【코로나바이러스 감염증】 코로나바이러스성 장염으로 불리며, 소화계통의 바이러스 감염으로 인해 구토, 설사 '
 '등의 증상을 일으킴\n'
 '제2조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 47},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000231',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
