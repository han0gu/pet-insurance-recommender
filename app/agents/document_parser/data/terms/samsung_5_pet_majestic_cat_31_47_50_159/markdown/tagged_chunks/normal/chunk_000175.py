from langchain_core.documents import Document

chunk = Document(
    page_content=('- 따라 보험금의 전부 또는 일부에 대하여 나누어 지급받거나 일시에 지급받는 방법으\n'
 '- 로 변경할 수 있습니다.\n'
 '- ② 회사는 제1항에 따라 일시에 지급할 금액을 나누어 지급하는 경우에는 나중에 지급할\n'
 '- 금액에 대하여 평균공시이율을 연단위 복리로 계산한 금액을 더하며, 나누어 지급할\n'
 '- 금액을 일시에 지급하는 경우에는 평균공시이율을 연단위 복리로 할인한 금액을 지급\n'
 '- 합니다.\n'
 '<예시안내>[보험금을 나누어 지급받을 경우]\n'
 '보험금: 6천만원, 보험금 지급일자: 2024년 4월 1일 일때 보험금을 일시에 지급받지 않고 3년간 매'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
