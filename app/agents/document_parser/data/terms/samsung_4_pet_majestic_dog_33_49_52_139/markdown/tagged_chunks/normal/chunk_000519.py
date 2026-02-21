from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 진단계약에서 보험금 지급사유가 발생할 때까지 진단을 받지 않은 경우. 다만, 진\n'
 '- 단계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 발생하는 경우\n'
 '- 에는 보장을 해드립니다.\n'
 '# 제21조 (보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)① 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 '
 '않아 보험료 납입이 연체중\n'
 '인경우에 회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고\n'
 '(독촉)기간으로 정하여 계약자에게 다음 각 호의 내용을 서면(등기우편 등), 전화(음성'),
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
 'indexing': {'chunk_id': 'chunk_000519',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
