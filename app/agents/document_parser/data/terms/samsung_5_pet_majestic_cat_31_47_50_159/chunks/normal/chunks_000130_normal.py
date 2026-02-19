from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 않아 보험료 납입이 연체중 인 경우에 회사는 14일(보험기간이 1년 미만인 '
 '경우에는 7일) 이상의 기간을 납입최 고(독촉)기간으로 정하여 계약자에게 다음 각 호의 내용을 서면(등기우편 등), 전화(음 성녹음) '
 '또는 전자문서 등으로 알려 드립니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 43},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 169,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
