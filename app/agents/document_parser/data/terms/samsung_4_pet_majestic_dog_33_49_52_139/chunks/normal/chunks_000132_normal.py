from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 않아 보험료 납입이 연체중 인 경우에 회사는 14일(보험기간이 1년 미만인 '
 '경우에는 7일) 이상의 기간을 납입최 고(독촉)기간으로 정하여 계약자에게 다음 각 호의 내용을 서면(등기우편 등), 전화(음 성녹음) '
 '또는 전자문서 등으로 알려 드립니다. 1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용 2. 납입최고(독촉)기간이 끝나는 '
 '날까지 보험료를 납입하지 않을 경우 납입최고(독촉)기 간이 끝나는 날의 다음날에 계약이 해지된다는 내용 3'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 45},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
