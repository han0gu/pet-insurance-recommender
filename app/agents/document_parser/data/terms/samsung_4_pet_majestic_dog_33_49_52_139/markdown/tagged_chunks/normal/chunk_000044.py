from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 제2항에도 불구하고 회사는 「보험금의 지급사유」에서 정한 나누어 지급하는 보험금\n'
 '- 에 대해서 일시에 지급하는 경우에 한하여 평균공시이율을 반영하여 연단위 복리로\n'
 '- 할인한 금액과 보장부분 적용이율을 반영하여 연단위 복리로 할인한 금액 중 큰 금액\n'
 '- 을 지급합니다.\n'
 '# <예시안내>- [보험금을 나누어 지급받을 경우]\n'
 '보험금: 6천만원, 보험금 지급일자: 2024년 4월 1일 일때 보험금을 일시에 지급받지 않고 3년간 매\n'
 '년 동일한 금액으로 나누어 지급받는 경우| 지급일 | 지급액 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
