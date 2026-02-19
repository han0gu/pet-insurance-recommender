from langchain_core.documents import Document

chunk = Document(
    page_content=('【공제계약】 공제사업을 실시하는 경영주체(협동조합 등)와 공제계약자(일반적으로 조합원) 사이 에 체결되는 계약으로, 공제계약자들이 단체에 '
 '일정금액을 적립해두고 우연한 사고가 발생한 경우 적립금에서 이를 구제함으로써 상호부조를 도모하는 계약을 말합니다.\n'
 '라. 대위권: 회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합니다.\n'
 '4. 이자율 관련 용어\n'
 '가. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로 하는 '
 '이자 계산방법을 말합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000007',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
