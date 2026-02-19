from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조 (공시이율의 적용 및 공시)\n'
 '① 이 보험의 적립부분 순보험료(적립보험료에서 정해진 계약체결비용 및 계약관리비용 을 공제한 금액을 말합니다. 이하 같습니다)에 대한 '
 '적립이율은 매월 1일 회사가 정한 보장성 공시이율Ⅴ(이하「공시이율」이라 합니다)로 하며, 공시이율은 매월 1일부터\n'
 '해당월 마지막 날까지 1개월간 확정 적용합니다.\n'
 '<용어풀이>\n'
 '[공시이율]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000039',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
